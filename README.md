# Vera Bot — magicpin AI Challenge

## Approach
Vera Bot decides what WhatsApp message to send a merchant and when, using a
two-step pipeline:
1. **Decide (code, not AI):** for each trigger, deterministic Python logic
   picks the single most relevant signal (e.g. a CTR gap vs peers, an
   overdue-patient recall, a regulation change) and assembles a "key facts"
   sheet from only the data the merchant/category/trigger context provides.
2. **Write (LLM, temp 0):** the key facts sheet + category tone rules are
   sent to the LLM, which returns body/cta/rationale as JSON. A post-check
   verifies every number in the message exists in the key facts sheet before
   it's sent — if not, the LLM is asked to redo it once, citing a source.
3. **Fallback (no LLM available or all drafts rejected):** rather than a
   generic template, the fallback is built from the same verified fields —
   the merchant's active offer, locality, or category — so even a failed
   LLM call never produces a message with fabricated content.

Every number, price or date must either appear in the merchant/category/
trigger data given to the bot, or (if it's a background fact, like a
regulation or industry study) the message must name its exact source in
the same sentence. A message is rejected and retried once if it cites a
number that isn't traceable either way.

Replies are handled with simple rules first (yes/stop/auto-reply/hostile),
falling back to the same fact-checked LLM path for anything else. Auto-reply
loops end automatically after repeated non-answers.

## Model
- Primary: Groq (openai/gpt-oss-20b)
- Backup: Gemini (gemini-3.5-flash-lite), used if Groq is rate-limited

## Trade-offs
- Deterministic decision logic over letting the LLM choose the trigger,
  to guarantee decision quality and avoid hallucinated reasoning.
- Strict fact-checking on numbers costs a small amount of latency (occasional
  one retry) but eliminates fabricated figures.
- Caches identical inputs so outputs are deterministic across repeated ticks.

## Endpoints
/v1/context, /v1/tick, /v1/reply, /v1/healthz, /v1/metadata — implemented per spec.
