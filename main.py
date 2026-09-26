import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI

load_dotenv()

# =====================================================================================
# CONFIG  (check these against examples/api-call-examples.md in the challenge pack)
# =====================================================================================
TEAM_NAME = os.getenv("TEAM_NAME", "Team Vera")
MEMBER_NAME = os.getenv("MEMBER_NAME", "Your Name")
BOT_VERSION = "1.0.0"

CTA_TYPES = ["yes_no", "open_ended"]      # allowed values for "cta"
SEND_AS_TYPES = ["vera", "merchant"]      # allowed values for "send_as"

PRIMARY = {
    "name": "primary",
    "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    "api_key": os.getenv("LLM_API_KEY", ""),
    "model": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
}
BACKUP = {  # optional second provider, used when the first one is rate-limited or down
    "name": "backup",
    "base_url": os.getenv("BACKUP_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "api_key": os.getenv("BACKUP_API_KEY", ""),
    "model": os.getenv("BACKUP_MODEL", "gemini-2.5-flash-lite"),
}

LLM_TIMEOUT = 8        # seconds allowed for one LLM call
TICK_BUDGET = 11       # seconds allowed for a whole /tick (the local judge gives up at 15s)
MAX_ACTIONS = 20       # judge limit per tick
MAX_TURNS = 8          # end a conversation after this many turns
CACHE_FILE = "cache.json"

# =====================================================================================
# MEMORY (kept in RAM while the server runs)
# =====================================================================================
STORE = {"category": {}, "merchant": {}, "customer": {}, "trigger": {}}
SENT_KEYS = set()          # suppression keys already used, so nothing is sent twice
SENT_BODIES = {}           # merchant_id -> last few messages we sent them
CONVOS = {}                # conversation_id -> {"merchant_id", "customer_id", "history", counters}
WAITS = {}                 # merchant (or conversation) -> how many times in a row we chose "wait"
LOCK = threading.Lock()
START_TIME = time.time()
POOL = ThreadPoolExecutor(max_workers=4)

try:
    with open(CACHE_FILE, encoding="utf-8") as f:
        CACHE = json.load(f)
except Exception:
    CACHE = {}

app = FastAPI(title="Vera bot")


# =====================================================================================
# SMALL HELPERS
# =====================================================================================
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def find(obj, keys, depth=0):
    """Look for the first non-empty value of any of `keys`, searching nested dicts/lists.
    The pack's JSON shapes can vary, so we search instead of hard-coding paths."""
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if v not in (None, "", [], {}):
                return v
        for v in obj.values():
            r = find(v, keys, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v, keys, depth + 1)
            if r is not None:
                return r
    return None


def get(scope, cid):
    item = STORE.get(scope, {}).get(cid)
    return item["payload"] if item else None


def shrink(obj, depth=0):
    """Make big JSON smaller before sending it to the LLM (saves tokens and time)."""
    if depth > 6:
        return "..."
    if isinstance(obj, dict):
        return {k: shrink(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shrink(v, depth + 1) for v in obj[:6]]
    if isinstance(obj, str) and len(obj) > 400:
        return obj[:400] + "..."
    return obj


def as_text(obj, limit=3500):
    if obj is None:
        return "none"
    s = json.dumps(shrink(obj), ensure_ascii=False, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...(cut)"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower()).rstrip("s")


def match_category(value):
    """Turn 'dentist' / 'Dentists' / {'id': 'dentists'} into a stored category id."""
    if isinstance(value, dict):
        value = find(value, ["id", "slug", "category_id", "name"])
    if not value:
        return None
    n = norm(value)
    for cid in STORE["category"]:
        if norm(cid) == n or n in norm(cid) or norm(cid) in n:
            return cid
    return None


def category_of(merchant):
    c = match_category(find(merchant, ["category", "category_id", "category_slug", "vertical"]))
    if c:
        return c
    text = json.dumps(merchant).lower()          # last resort: look for a category id in the text
    for cid in STORE["category"]:
        if norm(cid) and norm(cid) in re.sub(r"[^a-z0-9]", "", text):
            return cid
    return None


def merchant_name(merchant):
    return find(merchant, ["name", "display_name", "business_name", "merchant_name", "owner_name"]) or "there"


# =====================================================================================
# THE CHECKER  (the most important part: never send a made-up fact)
# =====================================================================================
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
SRC_NUM = re.compile(r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?(?![A-Za-z_\d])")   # skips digits inside IDs like m_001
LINK = re.compile(r"(https?://|www\.|\.com\b|\.in\b|bit\.ly)", re.I)


def source_numbers(source_text):
    """All numbers in the data, plus the same values written as percentages (0.021 -> 2.1)."""
    every, decimals = set(), set()
    for n in SRC_NUM.findall(source_text):
        n = n.replace(",", "").rstrip(".")
        try:
            f = float(n)
        except ValueError:
            continue
        every.add(round(f, 4))
        if "." in n:
            decimals.add(round(f, 4))
        if f < 1:
            for pct in (round(f * 100, 2), round(f * 100, 1), float(round(f * 100))):
                every.add(pct)
                decimals.add(pct)
    return every, decimals


def check_message(body, source_text, taboos=()):
    """Return a list of problems. Empty list = message is safe to send."""
    problems = []
    if not body or len(body.strip()) < 15:
        problems.append("message is empty or too short")
        return problems
    if len(body) > 700:
        problems.append("message is too long")
    if LINK.search(body):
        problems.append("contains a link or URL")
    every, decimals = source_numbers(source_text)
    for n in NUM.findall(body):
        clean = n.replace(",", "").rstrip(".")
        try:
            v = round(float(clean), 4)
        except ValueError:
            continue
        if "." in clean:                                  # decimals must match a real decimal or percentage
            if v in decimals:
                continue
        elif v <= 10 or v in every:                       # small whole counts like '2 minutes' are fine
            continue
        problems.append(f"number {n} is not in the input data")
    low = body.lower()
    for t in taboos:
        if isinstance(t, str) and 2 < len(t) < 40 and t.lower() in low:
            problems.append(f"uses a word the category says to avoid: {t}")
    return problems


def taboos_of(category):
    t = find(category or {}, ["taboos", "taboo_words", "avoid", "words_to_avoid", "do_not_say", "banned_phrases"])
    if isinstance(t, str):
        return [t]
    return [x for x in (t or []) if isinstance(x, str)]


# =====================================================================================
# LLM CALL  (OpenAI-compatible: works with Groq, Gemini, OpenRouter, OpenAI...)
# =====================================================================================
def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def call_llm(messages):
    """Ask the LLM for a JSON answer. Cached so the same input always gives the same output."""
    key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    if key in CACHE:
        return CACHE[key]
    for prov in (PRIMARY, BACKUP):
        if not prov["api_key"]:
            continue
        payload = {
            "model": prov["model"],
            "messages": messages,
            "temperature": 0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
        }
        if "gpt-oss" in prov["model"]:
            payload["reasoning_effort"] = "low"
        try:
            r = httpx.post(
                f"{prov['base_url'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {prov['api_key']}"},
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            if r.status_code == 400 and "response_format" in r.text:
                payload.pop("response_format")
                r = httpx.post(f"{prov['base_url'].rstrip('/')}/chat/completions",
                               headers={"Authorization": f"Bearer {prov['api_key']}"},
                               json=payload, timeout=LLM_TIMEOUT)
            if r.status_code != 200:
                log(f"LLM {prov['name']} error {r.status_code}: {r.text[:200]}")
                continue
            data = parse_json(r.json()["choices"][0]["message"]["content"])
            if data is None:
                log(f"LLM {prov['name']} returned non-JSON")
                continue
            with LOCK:
                CACHE[key] = data
                try:
                    with open(CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(CACHE, f)
                except Exception:
                    pass
            return data
        except Exception as e:
            log(f"LLM {prov['name']} failed: {type(e).__name__} {e}")
    return None


# =====================================================================================
# COMPOSE  — write one proactive message
# =====================================================================================
COMPOSE_RULES = f"""You are Vera, magicpin's WhatsApp assistant for Indian local businesses
(dentists, salons, restaurants, gyms, pharmacies). Write ONE message for the situation in the INPUT.

Rules:
1. Build the message from KEY FACTS: that is what the reviewer sees. Every number, price, date and name
   should come from KEY FACTS. Use a fact that is only in FULL DATA just when the trigger is about it,
   and then name its source in the same sentence ("per the DCI circular...", "your recall list shows...").
   Never invent a number, offer, statistic, review, name or claim. If a fact is missing, write around it.
   Never present a number from a study, digest or another business as the merchant's own figure.
1a. If the TRIGGER payload gives no concrete number, price or date to lead with, do NOT invent one to
   satisfy rule 5. Instead lead with whatever is real and specific in KEY FACTS: the merchant's own
   performance number (views/calls/ctr), an active offer's exact title, or their locality tied plainly
   to the trigger's topic (e.g. "Hi {{name}}, {{locality}} sees a lot of {{topic}} interest this time of
   year — your {{offer}} could be worth surfacing for it."). A message with no invented stat but one real,
   specific detail scores far better than one with a fabricated number.
2. The TRIGGER is why this message goes out now. Pick the single strongest signal and lead with it.
   Do not list every fact you have.
3. Personalise from KEY FACTS: greet the owner by first name (Dr. for dentists), mention the locality,
   and use one of their own performance numbers, signals or an active offer by its exact title.
   If their languages include Hindi, one short natural Hinglish phrase is welcome.
4. Match the CATEGORY voice (e.g. clinical for dentists and pharmacies, visual for salons,
   timely for restaurants, motivating for gyms) and obey anything the category says to avoid.
5. Give one concrete reason to reply now. Prefer a fact from KEY FACTS: a deadline, a customer about to
   lapse, a competitor or peer benchmark, or demand that is happening this week. If none of those exist
   for this trigger, timeliness itself is a valid reason ("this is peaking right now", "before the week is
   out") — do not manufacture a statistic just to have one. Then end with exactly ONE easy next step the
   reader can answer in a word or two (YES/NO, or pick 1 or 2).
5a. Make replying feel necessary, not optional. Instead of a generic offer ("Ready to draft the post?",
   "Want me to share details?"), frame the CTA around something specific that depends on their answer:
   a slot or window that needs holding ("Should I book the 6pm slot?"), a decision that's already teed
   up ("Reply 1 for the ₹299 offer or 2 to skip"), or a short window before it's less useful ("This
   works best if sent today — go ahead?"). The reader should feel that answering NOW, even briefly,
   moves something forward — not that they're being asked for permission to do more work later.
6. No links or URLs. No hype words (amazing, guaranteed, best ever). At most one emoji.
7. Short: 2-4 sentences, usually under 320 characters.
8. If a CUSTOMER is given and the trigger is about that customer, write to the customer on behalf of
   the merchant (send_as "merchant"). If that customer has not consented or has opted out, set skip true.
9. If the trigger clearly does not fit this merchant (wrong business type, expired, or the data needed
   is missing), set skip true.
10. Do not repeat any message listed under ALREADY SENT.

Reply with JSON only:
{{"skip": false, "body": "the message", "cta": one of {CTA_TYPES}, "send_as": one of {SEND_AS_TYPES},
 "rationale": "one line: which signal you chose and why"}}"""

def key_facts(category, merchant, trigger, customer):
    """The fields magicpin's scorer shows its reviewer. Messages built from these score best."""
    category, merchant, trigger = category or {}, merchant or {}, trigger or {}
    ident = merchant.get("identity") or {}
    perf = merchant.get("performance") or {}
    voice = category.get("voice") or {}
    facts = {
        "business_name": ident.get("name"),
        "owner_first_name": ident.get("owner_first_name"),
        "locality": ident.get("locality"),
        "languages": ident.get("languages"),
        "performance": {k: perf.get(k) for k in ("views", "calls", "ctr") if perf.get(k) is not None},
        "signals": merchant.get("signals"),
        "active_offers": [o.get("title") for o in merchant.get("offers") or []
                          if isinstance(o, dict) and o.get("status") == "active"],
        "category": category.get("slug"),
        "voice_tone": voice.get("tone") if isinstance(voice, dict) else voice,
        "trigger_kind": trigger.get("kind"),
        "trigger_payload": trigger.get("payload", trigger),
        "trigger_urgency": trigger.get("urgency"),
        "customer": (customer or {}).get("identity") if customer else None,
    }
    return {k: v for k, v in facts.items() if v not in (None, "", [], {})}


def fallback_message(plan):
    """Safe message used when the LLM keeps fabricating. Uses only verified fields, no invented numbers."""
    if plan["customer_id"]:
        return None
    merchant, trigger, category = plan["merchant"], plan["trigger"], plan["category"]
    facts = key_facts(category, merchant, trigger, plan.get("customer"))
    kind = str(find(trigger, ["kind", "type", "trigger_type", "name"]) or "update").replace("_", " ")
    name = merchant_name(merchant)
    locality = facts.get("locality")
    offers = facts.get("active_offers") or []

    if offers:
        lead = f"Hi {name}, your {offers[0]} offer is a good fit for this {kind}."
    elif locality:
        lead = f"Hi {name} in {locality}, this {kind} update is worth a look for your business."
    else:
        lead = f"Hi {name}, this {kind} update is relevant to your business right now."

    return {
        "skip": False,
        "body": lead + " Want me to put together a short draft to send out? Reply YES.",
        "cta": CTA_TYPES[0],
        "send_as": SEND_AS_TYPES[0],
        "rationale": "fallback: draft kept citing facts outside the source data, used verified fields only",
    }


def compose(plan, now):
    category, merchant, trigger, customer = plan["category"], plan["merchant"], plan["trigger"], plan["customer"]
    already = SENT_BODIES.get(plan["merchant_id"], [])[-3:]
    facts = key_facts(category, merchant, trigger, customer)
    source = json.dumps([category, merchant, trigger, customer], ensure_ascii=False) + " " + str(now)
    visible = json.dumps(facts, ensure_ascii=False) + " " + str(now)
    user = (
        f"NOW: {now}\n\nKEY FACTS (the reviewer sees these; build the message from them):\n"
        f"{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"FULL DATA (background only; see rule 1):\nCATEGORY: {as_text(category, 2000)}\n"
        f"MERCHANT: {as_text(merchant, 2500)}\nTRIGGER: {as_text(trigger, 2000)}\nCUSTOMER: {as_text(customer, 1000)}\n\n"
        f"ALREADY SENT:\n{json.dumps(already, ensure_ascii=False)}"
    )
    messages = [{"role": "system", "content": COMPOSE_RULES}, {"role": "user", "content": user}]
    taboos = taboos_of(category)

    safe_draft = None                       # a draft that is true but leans on background facts
    draft = call_llm(messages)
    for attempt in range(2):
        if draft is None:
            break
        if draft.get("skip"):
            return None
        body = str(draft.get("body", ""))
        problems = check_message(body, source, taboos)
        if not problems and body in already:
            problems = ["repeated an old message"]
        if not problems:
            off_view = [p for p in check_message(body, visible) if p.startswith("number")]
            if not off_view or attempt == 1:
                return draft
            safe_draft = draft
            problems = [p.replace("is not in the input data", "is not in KEY FACTS: use a KEY FACTS number, "
                                  "or name exactly where it comes from") for p in off_view]
        if attempt == 0:
            log(f"draft for {plan['trigger_id']} sent back: {problems}")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(draft, ensure_ascii=False)},
                {"role": "user", "content": "Fix this draft: " + "; ".join(problems)
                 + ". Keep everything true to the INPUT. JSON only."},
            ]
            draft = call_llm(messages)
    return safe_draft or fallback_message(plan)

def make_plan(trigger_id, trigger):
    """Plain-code decisions: who is this for, does it fit, was it already sent?"""
    customer_id = find(trigger, ["customer_id"])
    customer = get("customer", customer_id) if customer_id else None
    merchant_id = find(trigger, ["merchant_id"]) or (find(customer, ["merchant_id"]) if customer else None)
    if not merchant_id:                                   # last resort: a merchant id mentioned in the trigger
        text = json.dumps(trigger)
        merchant_id = next((m for m in STORE["merchant"] if m in text), None)
    merchant = get("merchant", merchant_id)
    if not merchant:
        log(f"skip {trigger_id}: no merchant found")
        return None
    category_id = category_of(merchant)
    trig_cat = match_category(find(trigger, ["category", "category_id", "vertical"]))
    if trig_cat and category_id and trig_cat != category_id:
        log(f"skip {trigger_id}: trigger is for {trig_cat}, merchant is {category_id}")
        return None
    if customer and str(find(customer, ["consent", "opted_in", "opt_in"])).lower() in ("false", "no", "revoked"):
        log(f"skip {trigger_id}: customer has not consented")
        return None
    key = find(trigger, ["suppression_key"]) or f"{trigger_id}:{merchant_id}" + (f":{customer_id}" if customer_id else "")
    if key in SENT_KEYS:
        return None
    return {
        "trigger_id": trigger_id, "trigger": trigger, "merchant_id": merchant_id, "merchant": merchant,
        "customer_id": customer_id if customer else None, "customer": customer,
        "category": get("category", category_id) if category_id else None, "suppression_key": key,
    }


# =====================================================================================
# ENDPOINTS
# =====================================================================================
@app.get("/")
def home():
    return {"ok": True, "message": "Vera bot is running. Try /v1/healthz"}


@app.get("/v1/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": {s: len(STORE.get(s, {})) for s in ("category", "merchant", "customer", "trigger")},
    }


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": [MEMBER_NAME],
        "model": PRIMARY["model"],
        "approach": "rule-based decision layer + LLM writer at temperature 0 + fact checker + input-hash cache",
        "version": BOT_VERSION,
    }


@app.post("/v1/context")
def context(body: dict = Body(default={})):
    scope, cid = body.get("scope"), body.get("context_id")
    if not scope or not cid:
        return {"accepted": False, "reason": "scope and context_id are required"}
    try:
        version = float(body.get("version", 0))
    except (TypeError, ValueError):
        version = 0.0
    with LOCK:
        bucket = STORE.setdefault(scope, {})
        current = bucket.get(cid)
        if current and version < current["version"]:
            return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
        if not current or version > current["version"]:
            bucket[cid] = {"version": version, "payload": body.get("payload") or {}}
    ack = "ack_" + hashlib.md5(f"{scope}:{cid}:{version}".encode()).hexdigest()[:10]
    return {"accepted": True, "ack_id": ack, "stored_at": now_iso()}


@app.post("/v1/tick")
def tick(body: dict = Body(default={})):
    started = time.time()
    now = body.get("now") or now_iso()
    plans = []
    for item in body.get("available_triggers") or []:
        if isinstance(item, dict):                       # trigger sent inline instead of by id
            tid = item.get("trigger_id") or item.get("id") or item.get("context_id")
            trigger = get("trigger", tid) or item            # keep kind, urgency and payload together
        else:
            tid, trigger = item, get("trigger", item)
        if not trigger:
            log(f"skip {tid}: trigger not loaded")
            continue
        plan = make_plan(tid, trigger)
        if plan and plan["suppression_key"] not in {p["suppression_key"] for p in plans}:
            plans.append(plan)
    plans = plans[:MAX_ACTIONS]

    futures = {POOL.submit(compose, p, now): p for p in plans}
    done, _ = wait(futures, timeout=max(1, TICK_BUDGET - (time.time() - started)))

    actions = []
    for fut, plan in futures.items():
        msg = fut.result() if fut in done else fallback_message(plan)
        if not msg:
            continue
        action = {
            "merchant_id": plan["merchant_id"],
            "trigger_id": plan["trigger_id"],
            "body": msg["body"],
            "cta": msg.get("cta") if msg.get("cta") in CTA_TYPES else CTA_TYPES[0],
            "send_as": msg.get("send_as") if msg.get("send_as") in SEND_AS_TYPES else SEND_AS_TYPES[0],
            "suppression_key": plan["suppression_key"],
            "rationale": msg.get("rationale", ""),
        }
        if plan["customer_id"]:
            action["customer_id"] = plan["customer_id"]
        actions.append(action)
        with LOCK:
            SENT_KEYS.add(plan["suppression_key"])
            SENT_BODIES.setdefault(plan["merchant_id"], []).append(msg["body"])
    log(f"tick: {len(actions)} actions in {time.time() - started:.1f}s")
    return {"actions": actions}


# ---------- replies ----------
OPT_OUT = re.compile(r"\b(stop|unsubscribe|not interested|don'?t (message|text|contact)|leave me alone|remove me|"
                     r"band karo|mat bhejo|no more messages)\b", re.I)
AUTO = re.compile(r"(thank you for (contacting|reaching|your message)|thanks for (contacting|reaching)|"
                  r"will get back to you|get back to you shortly|auto[- ]?reply|out of office|"
                  r"currently (closed|unavailable|away)|business hours|this is an automated)", re.I)
HOSTILE = re.compile(r"\b(fuck\w*|shit|idiot|stupid|scam\w*|fraud|bakwas|bewakoof|spam\w*|useless)\b", re.I)
LATER = re.compile(r"\b(later|not now|busy|baad mein|kal|tomorrow)\b", re.I)

REPLY_RULES = """You are Vera, magicpin's WhatsApp assistant for Indian local businesses, in the middle of a
conversation. Decide the next step and write the reply.

Rules:
1. Use ONLY facts from CONTEXT and the conversation. Never invent numbers, offers or claims.
2. If they said yes / agreed: do exactly what was offered, confirm it in one line, and give ONE small next step.
3. If they asked a question: answer it briefly from the facts, then ONE easy next step.
4. If it is off-topic: reply politely in one line and steer back to their business, or end if clearly unrelated.
5. Short: 1-3 sentences. No links. One question at most.
6. action is "send" (reply now), "wait" (they need time, send nothing) or "end" (conversation is over).

Reply with JSON only: {"action": "send", "body": "the reply (empty if wait)", "rationale": "one line why"}"""


@app.post("/v1/reply")
def reply(body: dict = Body(default={})):
    cid = str(body.get("conversation_id") or "unknown")
    text = str(body.get("message") or "").strip()
    role = body.get("from_role") or "merchant"
    try:
        turn = int(body.get("turn_number") or 1)
    except (TypeError, ValueError):
        turn = 1

    with LOCK:
        convo = CONVOS.setdefault(cid, {"merchant_id": None, "customer_id": None, "history": [], "auto": 0, "hostile": 0})
    convo["merchant_id"] = body.get("merchant_id") or convo["merchant_id"] or find(body, ["merchant_id"])
    convo["customer_id"] = body.get("customer_id") or convo["customer_id"] or find(body, ["customer_id"])
    convo["history"].append({"from": role, "text": text})

    def answer(action, reply_body, why):
        who = convo["merchant_id"] or cid
        if action == "wait":
            WAITS[who] = WAITS.get(who, 0) + 1
            if WAITS[who] >= 3:        # waited twice already and still no real reply: stop
                action, reply_body, why = "end", "", "still no real reply after repeated waits: stop to avoid spamming"
        else:
            WAITS[who] = 0
        out = {"action": action, "rationale": why}
        if reply_body:
            out["body"] = reply_body
            convo["history"].append({"from": "vera", "text": reply_body})
        log(f"reply {cid} turn {turn}: {action} ({why})")
        return out

    # --- rules first: fast, predictable, no LLM needed ---
    if OPT_OUT.search(text):
        return answer("end", "Understood, I won't message you about this again. Just say hi if you ever need help.",
                      "opt-out detected: respect it and close")
    if AUTO.search(text):
        convo["auto"] += 1
        if convo["auto"] >= 2:
            return answer("end", "", "repeated auto-replies: stop to avoid spamming")
        return answer("wait", "", "auto-reply detected: wait for a real person")
    if HOSTILE.search(text):
        convo["hostile"] += 1
        if convo["hostile"] >= 2:
            return answer("end", "Sorry for the trouble. I'll stop here.", "hostile twice: close politely")
        return answer("send", "Sorry if this came at a bad time. I only share updates that can help your business on magicpin. "
                              "Should I stop these messages? Reply STOP.", "hostile: de-escalate and offer opt-out")
    if turn >= MAX_TURNS:
        return answer("end", "Thanks for your time! I'm here whenever you need help.", "turn limit reached")
    if LATER.search(text) and len(text) < 40:
        return answer("wait", "", "merchant asked for later: wait")

    # --- everything else: LLM with context + history ---
    merchant = get("merchant", convo["merchant_id"]) if convo["merchant_id"] else None
    customer = get("customer", convo["customer_id"]) if convo["customer_id"] else None
    category = get("category", category_of(merchant)) if merchant else None
    source = json.dumps([category, merchant, customer, convo["history"]], ensure_ascii=False)
    user = (f"CONTEXT:\nCATEGORY: {as_text(category, 1500)}\nMERCHANT: {as_text(merchant, 3000)}\n"
            f"CUSTOMER: {as_text(customer, 1000)}\n\nCONVERSATION (latest last):\n"
            + "\n".join(f"{h['from']}: {h['text']}" for h in convo["history"][-10:]))
    out = call_llm([{"role": "system", "content": REPLY_RULES}, {"role": "user", "content": user}])

    if out and out.get("action") in ("send", "wait", "end"):
        reply_body = str(out.get("body") or "")
        if out["action"] != "send" or not check_message(reply_body, source, taboos_of(category)):
            return answer(out["action"], reply_body if out["action"] != "wait" else "", out.get("rationale", ""))
        log(f"reply draft rejected: {check_message(reply_body, source)}")

    return answer("send", "Got it, thanks! I'll prepare this and share it here for your OK.",
                  "fallback: acknowledge and keep one clear next step")
