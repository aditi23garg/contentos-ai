# ContentOS AI — Phase 1, first runnable slice

This is the smallest slice of the ContentOS AI v2.3 spec that runs end-to-end:

```
Research Agent -> Content Producer Agent -> Brand Guardian Agent
```

with a bounded regenerate loop when the Guardian rejects the content. It intentionally
does **not** yet include: content batching, the dashboard, dedup/vector-DB, scheduling,
publishing, or analytics. Those are additive once this slice is proven with real daily
use — building them now would be planning ahead of data, which the spec itself argues
against.

## What's here

```
contentos_ai/
├── main.py                          # entry point — runs the pipeline once, prints result
├── requirements.txt
├── .env.example
├── data/
│   └── brand_profile.json           # the "Life Out Loud" brand from the spec
└── app/
    ├── core/
    │   ├── schemas.py                # Pydantic models: BrandProfile, Idea, GeneratedContent,
    │   │                              #   RubricScores, BrandGuardianResult, AgentDecisionLog
    │   └── config.py                 # env-driven config, brand profile loader
    ├── providers/
    │   └── llm.py                    # Groq/Ollama provider abstraction (OpenAI-SDK-compatible)
    ├── agents/
    │   ├── research_agent.py
    │   ├── content_producer_agent.py
    │   └── brand_guardian_agent.py
    └── graph/
        └── pipeline.py               # the LangGraph state machine wiring it together
```

This follows the Repository Pattern / Service Layer / Provider Abstraction standards
from the spec's Engineering Standards section — each agent is a plain function taking
a Pydantic model and returning one, the LLM provider is swappable via `.env` alone, and
nothing in `app/agents` knows or cares whether it's talking to Groq or Ollama.

## Setup

```bash
cd contentos_ai
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your Groq API key (free, no credit card — https://console.groq.com),
or set `LLM_PROVIDER=ollama` if you'd rather run fully local/offline.

## Run

```bash
python main.py
```

This runs one full cycle: the Research Agent proposes a few niche-locked ideas, the
highest-confidence one goes to the Content Producer, and the Brand Guardian scores it
against the six-dimension rubric. If it fails, it's regenerated once before the run
ends either "approved" or "rejected". The full decision log (what was tried, what
scored what, why) prints at the end — this is the console version of the
`agent_decisions` table the fuller spec calls for.

## What's verified

This scaffold was tested with a mocked LLM provider (no API calls) to confirm:
- the LangGraph wiring runs Research → Produce → Guardian → Finalize correctly
- the reject → regenerate → re-guardian retry loop works and respects
  `MAX_GUARDIAN_RETRIES`
- every step's Pydantic schema round-trips correctly

It has **not** been run against a live Groq/Ollama endpoint from this environment —
you'll want to run it locally with a real API key first, and sanity-check the actual
model output before trusting the rubric scores.

## Natural next steps, in order

1. Wire this same pipeline to SQLite so `agent_decisions`, `Post`, and `Idea` persist
   instead of only printing to console.
2. Swap the single-idea flow for real Content Batching (~20 ideas → best ~10 → batch
   produce/score), per the spec.
3. Add the ChromaDB dedup + knowledge-retrieval step to the Research Agent.
4. Build the Streamlit dashboard for review/approve/edit, replacing the console print.
5. Only then: Scheduler, Publisher (Phase 2), and the weekly Performance Reviewer
   (Phase 3).
