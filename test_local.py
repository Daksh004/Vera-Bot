"""
Quick self-test for the Vera bot. Sends a tiny fake dentist scenario and prints what the bot answers.

Usage:
  python test_local.py                                   (tests http://localhost:8000)
  python test_local.py https://your-bot.onrender.com     (tests the live bot)
"""
import json
import sys

import httpx

BOT = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
c = httpx.Client(timeout=40)


def show(title, r):
    ok = "OK " if r.status_code == 200 else "ERR"
    print(f"\n[{ok}] {title}  (HTTP {r.status_code}, {r.elapsed.total_seconds():.1f}s)")
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text[:500])


show("healthz", c.get(f"{BOT}/v1/healthz"))
show("metadata", c.get(f"{BOT}/v1/metadata"))

contexts = [
    ("category", "dentists", {"name": "Dentists", "voice": "clinical, calm, trustworthy",
                              "avoid": ["cheap", "guaranteed"]}),
    ("merchant", "m_001_drmeera", {"identity": {"name": "Dr. Meera's Dental Clinic", "category": "dentists",
                                                "locality": "South Delhi"},
                                   "performance": {"ctr": 0.021, "peer_median_ctr": 0.030, "views_30d": 1840},
                                   "offers": [{"title": "Dental Cleaning", "price": 299}]}),
    ("customer", "c_001_rahul", {"name": "Rahul", "merchant_id": "m_001_drmeera", "last_visit": "2026-03-10",
                                 "consent": True}),
    ("trigger", "trg_research_digest_dentists", {"kind": "research_digest", "merchant_id": "m_001_drmeera",
                                                  "category": "dentists",
                                                  "summary": "New study: 6-monthly cleanings cut gum disease by 40%"}),
    ("trigger", "trg_recall_rahul", {"kind": "recall_due", "merchant_id": "m_001_drmeera",
                                     "customer_id": "c_001_rahul", "months_since_visit": 6}),
]
for scope, cid, payload in contexts:
    show(f"context {scope}/{cid}", c.post(f"{BOT}/v1/context", json={
        "scope": scope, "context_id": cid, "version": 1, "payload": payload,
        "delivered_at": "2026-09-26T10:00:00Z"}))

show("context again (same version = no-op)", c.post(f"{BOT}/v1/context", json={
    "scope": "merchant", "context_id": "m_001_drmeera", "version": 1, "payload": {}}))

show("tick", c.post(f"{BOT}/v1/tick", json={
    "now": "2026-09-26T10:30:00Z",
    "available_triggers": ["trg_research_digest_dentists", "trg_recall_rahul"]}))
show("tick again (should send nothing new)", c.post(f"{BOT}/v1/tick", json={
    "now": "2026-09-26T10:35:00Z", "available_triggers": ["trg_research_digest_dentists"]}))

for i, msg in enumerate(["Yes, send me the details", "Thank you for contacting us, we will get back to you shortly",
                         "what is the price of cleaning?", "stop messaging me"], start=2):
    show(f"reply: '{msg}'", c.post(f"{BOT}/v1/reply", json={
        "conversation_id": "conv_001", "merchant_id": "m_001_drmeera", "from_role": "merchant",
        "message": msg, "turn_number": i}))

show("healthz (should now show loaded contexts)", c.get(f"{BOT}/v1/healthz"))
