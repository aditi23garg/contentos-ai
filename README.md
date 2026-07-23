# ContentOS AI — Phase 1, Content Batching Pipeline

An AI-powered content factory for the **"Life Out Loud"** motivational brand.
Three specialised agents research ideas, write publish-ready posts, and quality-score
them against a six-dimension brand rubric — all wired together as a LangGraph
state machine that runs one full weekly-style batch end-to-end.

```
Research Agent -> dedup filter -> Content Producer Agent -> Brand Guardian Agent -> SQLite + ChromaDB
```

---

## Architecture

```
contentos_ai_batching/
├── main.py                          # Entry point — runs one full batch cycle
├── probe_guardian.py                # Discrimination probe: tests Guardian scoring range
├── test_json_repair.py              # Smoke tests for the hardened JSON parser
├── test_backlog_topup.py            # Smoke tests for backlog read-back / archive-on-stale
├── inspect_history.py               # CLI tool to inspect persisted ideas/posts/decisions
├── requirements.txt
├── .env.example
├── data/
│   ├── brand_profile.json           # "Life Out Loud" brand — versioned, never mutated in place
│   ├── contentos.db                 # SQLite — ideas, posts, agent_decisions (auto-created)
│   └── chroma/                      # ChromaDB vector index for idea dedup (auto-created)
└── app/
    ├── core/
    │   ├── schemas.py               # Pydantic models: BrandProfile, Idea, GeneratedContent,
    │   │                            #   RubricScores, BrandGuardianResult, AgentDecisionLog
    │   └── config.py                # Env-driven config — all thresholds & provider settings
    ├── providers/
    │   └── llm.py                   # Groq / Ollama provider abstraction (OpenAI-SDK-compatible)
    │                                # Includes hardened _extract_json with state-machine repair
    ├── agents/
    │   ├── research_agent.py        # Generates niche-locked ideas from LLM knowledge
    │   ├── content_producer_agent.py# Writes caption + image prompt + hashtags + CTA in one call
    │   └── brand_guardian_agent.py  # Scores content on 6-dimension rubric, pass/fail decision
    │                                #   v2: injects real post history for concrete strategic_fit scoring
    ├── graph/
    │   └── pipeline.py              # LangGraph state machine: research -> produce -> guardian ->
    │                                # retry loop -> persist batch
    ├── repositories/
    │   ├── db.py                    # SQLAlchemy models: IdeaRecord, PostRecord, AgentDecisionRecord
    │   ├── repository.py            # Repository functions — only place translating Pydantic <-> ORM
    │   └── vector_store.py          # ChromaDB wrapper for idea similarity / dedup
    └── services/
        └── batching.py              # Dedup filter, near-dup filter, top-N ranking, surplus backlog
```

---

## What's Implemented

### 1. Three-Agent Pipeline (LangGraph)

The pipeline runs as a linear state machine with conditional edges:

```
research -> produce -> guardian ---+
               ^                    +-- passed   -> record_result -> advance -> produce (next item)
           bump_retry <-- retry ----+                             -> finalize -> persist_batch -> END
                                    +-- rejected -> record_result
```

- **Research Agent** — prompts the LLM for `IDEAS_PER_BATCH` (default: 8) niche-locked
  ideas with topic, angle, reasoning, confidence score, and knowledge sources. Only
  evergreen, credibility-weighted sources (books, peer-reviewed research, psychology,
  behavioral science, philosophy).

- **Content Producer Agent** — takes a single idea and produces a complete post in one
  structured LLM call: caption, image prompt, hashtags, CTA, plus Instagram/LinkedIn
  platform variants.

- **Brand Guardian Agent** — scores each post on six dimensions (each 1-5):
  `niche_fit`, `brand_alignment`, `originality`, `value_to_audience`,
  `grammar_clarity`, `strategic_fit`. Pass requires average >= 4.0 and every dimension >= 3.
  If it fails, content is regenerated once (configurable via `MAX_GUARDIAN_RETRIES`)
  before being recorded as rejected.

### 2. Content Batching with Dedup — Two-Tier Threshold

Before any content is produced, the Research Agent's candidate pool goes through two
filtering layers:

1. **History dedup** (ChromaDB) — embeds each candidate idea and checks embedding
   similarity against every previously approved idea, using a **two-tier threshold**:

   | Pair type | Threshold | Config var |
   |-----------|-----------|------------|
   | Different topic labels | `0.85` | `DEDUP_SIMILARITY_THRESHOLD` |
   | Same topic label (case-insensitive match) | `0.62` | `DEDUP_SAME_TOPIC_THRESHOLD` |

   The two-tier design comes from running `calibrate_dedup_threshold.py` against 54
   real approved ideas (1 431 pairs): same-topic duplicate pairs scored 0.58–0.75
   similarity — the general 0.85 threshold caught **0%** of them. An exact topic-label
   match is a free, much stronger signal than embedding similarity alone for short
   motivational copy; gating a lower threshold on that match costs nothing extra (no
   new embedding calls) and cannot introduce false positives on genuinely distinct
   cross-topic pairs. Approved ideas are now indexed in ChromaDB **with their topic
   label as metadata** so the two-tier check can apply on future runs.

   This ensures a rejected attempt on a topic does not permanently block it — only
   *approved* history counts.

2. **In-batch near-duplicate filter** (difflib SequenceMatcher) — drops near-identical
   ideas proposed within the same Research call. Threshold: 0.9 similarity ratio.

After filtering, ideas are ranked by confidence score and the top `BATCH_SIZE` (default: 5)
go through production. The rest are saved to the Idea Library as `status='backlog'`.

The per-idea result of the dedup check is now written to **`IdeaRecord.dedup_note`**
in SQLite (this column existed in the schema but was never populated before). You can
see it in the `inspect_history.py` output.


### 3. Idea Library — Backlog Top-Up

Ideas that survive both dedup filters but do not make the `BATCH_SIZE` cutoff are **no
longer silently discarded**. They are persisted to SQLite with `status='backlog'`.

The next cycle now reads that backlog back in *before* asking the Research Agent for
anything: `research_node` pulls up to `IDEAS_PER_BATCH` backlog ideas (ranked by
confidence, oldest first as tiebreak) and only asks Research for the deficit —
`IDEAS_PER_BATCH - len(backlog_pulled)`. A cycle with a full backlog can run with
**zero** fresh Research calls, which matters on a free-tier LLM budget.

Backlog-sourced ideas that get produced this cycle update their *existing* `ideas`
row in place (approved/rejected) instead of inserting a duplicate. If a backlog idea
gets filtered out this cycle — because a similar idea is now in approved history, or
because a fresh candidate duplicates it — it's archived (`status='archived'`) rather
than left as `backlog` to be re-pulled and re-filtered every cycle indefinitely.

Status lifecycle in the `ideas` table:

| Status     | Meaning |
|------------|---------|
| `approved` | Passed Guardian, full post produced, indexed in ChromaDB |
| `rejected` | Failed Guardian after all retries |
| `backlog`  | Survived dedup but below the batch cutoff — read back in next cycle |
| `archived` | Backlog idea that went stale (dedup/near-dup filtered it on a later cycle) |

### 4. Hardened JSON Parser (`_extract_json`)

The LLM occasionally wraps phrases in unescaped double-quotes inside a JSON string
(e.g. `"reason": "The concept of "growth mindset" is..."`), causing `json.loads`
to fail. Previously this triggered a full retry LLM call on every Guardian invocation.

The parser now has four layered attempts before falling through to a retry:

| Attempt | What it does |
|---------|-------------|
| 1 | `json.loads(raw)` — happy path |
| 2 | `json.loads(repair(raw))` — state-machine repair: strips illegal control chars, escapes bare inner quotes |
| 3 | `json.loads(extract_block(raw))` — pulls first `{...}` block in case of preamble text |
| 4 | `json.loads(repair(extract_block(raw)))` — block extraction + repair combined |

The **state-machine repair** (`_repair_json_string`) walks each character, tracks
whether it is inside a JSON string, and uses a lookahead heuristic — next non-whitespace
char is `:`, `,`, `}`, `]`, or EOF means closing delimiter; otherwise it is a bare
inner quote and gets escaped. A regex-based approach was tried first but cannot work
because it can only match valid string tokens, never the malformed boundary.

**Result:** zero `(after 1 retry)` annotations observed in pipeline runs since the fix.

### 5. Content Diversity Check

Before v2, the Brand Guardian scored `strategic_fit` purely qualitatively —
it had no concrete knowledge of what had already been published and consistently
awarded 4 almost every run.

The Guardian now receives the **last 20 approved post topics+angles** (from
`get_recent_approved_topics()` in the repository) as a numbered history section
appended to its system prompt. Explicit scoring instructions are included:

| strategic_fit | When to award it |
|---------------|------------------|
| 1-2 | Closely matches a topic already in recent history |
| 3 | Thematically adjacent but meaningfully different angle |
| 4-5 | Fresh theme not covered in recent approved posts |

This is a **zero-cost addition**: one read-only SQL query per item, no new
tables, no new agents, no schema changes. Guardian prompt bumped to `v2`.

The decision log entry for each Guardian call now includes a
`diversity_context=N recent posts` annotation confirming how many history
entries were injected.

### 6. Guardian Discrimination Validated

A deliberate probe (`probe_guardian.py`) confirmed the Guardian scores are genuinely
discriminating, not stuck on a safe default:

| Test Case | Avg | niche_fit | originality | grammar_clarity |
|-----------|-----|-----------|-------------|-----------------|
| Strong on-brand content | 4.67 | 5 | 4 | 5 |
| Generic / cliche filler | 3.83 | 5 | **2** | 5 |
| Wrong niche (crypto) | 2.00 | **1** | 2 | 5 |
| Poor grammar + clickbait | 1.17 | 2 | **1** | **1** |

**Score spread: 3.50** — well above the 1.5 threshold for meaningful discrimination.
The consistent 4.67 average in real runs reflects genuine output quality from Groq's
`llama-3.3-70b-versatile`, not a stuck rubric.

---

## Setup

```bash
# 1. Clone and enter the project
cd contentos_ai_batching

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add GROQ_API_KEY (free at https://console.groq.com)
# Or set LLM_PROVIDER=ollama to run fully offline
```

---

## Run

```bash
python main.py
```

One full batch cycle:
1. Research Agent generates 8 candidate ideas
2. Dedup filters against ChromaDB history; near-dup filter removes in-batch siblings
3. Top 5 by confidence go through Content Producer -> Brand Guardian
4. All results (approved / rejected) persist to SQLite; surplus ideas saved as backlog
5. Approved ideas indexed in ChromaDB for future dedup
6. Full decision log printed to console

### Inspect history

```bash
python inspect_history.py
```

Queries SQLite to show all persisted ideas, posts, and agent decisions from past runs.

### Run Guardian discrimination probe

```bash
python probe_guardian.py
```

Sends 4 hand-crafted test cases (strong, generic, wrong-niche, bad grammar) to the
live Guardian and prints a discrimination analysis with a spread verdict.
Re-run this any time the rubric prompt is changed.

### Run JSON parser smoke tests

```bash
python test_json_repair.py
```

8 unit tests covering: clean JSON, embedded quotes, multiple quoted phrases, control
characters, markdown fences, combined fence+quotes, preamble text, truncated JSON.

### Run two-tier dedup smoke test

```bash
python test_two_tier_dedup.py
```

5 unit tests verifying: same-topic candidate caught by lower `same_topic_threshold`;
different-topic candidate at the same similarity **not** caught; backward-compatible
call (no topic arg) works; genuinely distinct content never caught; topic metadata
actually persisted to ChromaDB.

### Calibrate dedup threshold against real data

```bash
python calibrate_dedup_threshold.py        # full report (top 15 closest pairs)
python calibrate_dedup_threshold.py --top 20
```

Computes the full pairwise similarity matrix across every approved idea in ChromaDB,
splits pairs into same-topic vs. different-topic groups, and prints distribution
statistics. Use this whenever you want to re-evaluate `DEDUP_SIMILARITY_THRESHOLD`
or `DEDUP_SAME_TOPIC_THRESHOLD` against accumulated real data.

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | — | From https://console.groq.com (free tier) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq-hosted model |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1` | Any locally pulled Ollama model |
| `LLM_MAX_TOKENS` | `1024` | Response cap; raise for larger Research batches |
| `IDEAS_PER_BATCH` | `8` | Candidates Research generates per cycle |
| `BATCH_SIZE` | `5` | Max ideas sent through production per cycle |
| `MAX_GUARDIAN_RETRIES` | `1` | Retries per item when Guardian fails |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.85` | Similarity cutoff vs. approved history (different topic label) |
| `DEDUP_SAME_TOPIC_THRESHOLD` | `0.62` | Lower cutoff applied only when candidate topic = stored idea's topic |
| `DB_PATH` | `data/contentos.db` | SQLite file path |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `RUBRIC_PASS_AVERAGE` | `4.0` | Minimum average score to pass Guardian |
| `RUBRIC_MIN_DIMENSION` | `3` | Minimum score on any single dimension to pass |

---

## Design Decisions

- **Single LLM call per agent** — the Content Producer consolidates writer, image-prompt,
  SEO, and CTA into one structured-output call. Four separate calls for the same output
  is wasteful.

- **Provider abstraction** — Groq and Ollama are both OpenAI-SDK-compatible. Switching
  providers is a config change, not a code change. Adding OpenAI/Claude/Gemini means
  adding one branch to `llm.py`, not touching any agent.

- **Repository Pattern** — `app/repositories/repository.py` is the only place that
  translates between Pydantic schemas (what agents use) and SQLAlchemy records (what
  the DB stores). Nothing outside it imports the `*Record` classes.

- **Linear graph, not agent mesh** — the pipeline is a LangGraph state machine with
  conditional edges, not a free-form multi-agent mesh. Per the v2.3 spec philosophy:
  the simplest wiring that does the job.

- **Dedup only blocks on approved history** — a rejected attempt on a topic does not
  permanently block it. The topic can come back, get better content produced, and pass
  the Guardian on a subsequent run. This is intentional by design.

---

## What's Next

1. **Streamlit dashboard** — visual review/approve/edit interface replacing console print.
2. **Scheduler + Publisher** — Phase 2: auto-schedule approved posts and publish via
   platform APIs.
3. **Performance Reviewer** — Phase 3: weekly analytics feedback loop.
