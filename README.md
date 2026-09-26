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
