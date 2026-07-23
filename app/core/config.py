"""
Environment configuration. Everything provider-related is read from env vars so that
switching LLM/image/search providers is a config change, never a code change
(Provider Abstraction requirement in the spec).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.core.schemas import BrandProfile

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- LLM provider config -----------------------------------------------------------
# Groq is the v2.3 default: free tier, OpenAI-SDK-compatible, no GPU required.
# Ollama is the offline/zero-dependency fallback.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" | "ollama"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Explicit response size cap. Without this, a long `reason` field (Brand Guardian) or
# long `caption`/`platform_variants` (Content Producer) can hit the model's default
# token limit mid-response and produce truncated, unparseable JSON. 1024 is generous
# for this pipeline's field sizes; raise it if you extend any agent's output schema.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- Brand Guardian rubric thresholds (Configuration section of the spec) ----------
RUBRIC_PASS_AVERAGE = float(os.getenv("RUBRIC_PASS_AVERAGE", "4.0"))
RUBRIC_MIN_DIMENSION = int(os.getenv("RUBRIC_MIN_DIMENSION", "3"))

# --- Batching. Research generates IDEAS_PER_BATCH candidates in one call; after dedup
# and in-batch near-duplicate filtering, the top BATCH_SIZE (by confidence) go through
# full production + scoring. Kept modest by default to stay comfortably inside Groq's
# free-tier request budget -- raise toward the spec's ~20/~10 once you've watched a
# few batches run and are comfortable with the cost/time per cycle. -----------------
IDEAS_PER_BATCH = int(os.getenv("IDEAS_PER_BATCH", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
MAX_GUARDIAN_RETRIES = int(os.getenv("MAX_GUARDIAN_RETRIES", "1"))

# --- Persistence (SQLite for records, ChromaDB for idea dedup embeddings) ----------
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "contentos.db")))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma")))
# How similar a new idea can be to a previously approved one before it's filtered out,
# when the two ideas have DIFFERENT topic labels. 0 = never filter, 1 = only filter
# exact duplicates. See vector_store.py for how this similarity score is computed.
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))
# Same idea, but applied only when the candidate's topic label exactly matches
# (case-insensitive) a previously approved idea's topic label. Deliberately much
# lower than DEDUP_SIMILARITY_THRESHOLD -- calibrate_dedup_threshold.py, run against
# 54 real approved ideas (1431 pairs), found that genuinely-duplicate same-topic
# pairs scored 0.58-0.75 similarity (never above 0.75, let alone 0.85 -- the general
# threshold was catching 0% of them), while cross-topic pairs almost never exceeded
# 0.72. An exact topic-label match is a much stronger and free signal here than
# embedding similarity alone can provide for short, thematically-clustered
# motivational copy -- gating a lower threshold on that match costs nothing extra
# (no new embedding calls) and can't introduce new false positives on genuinely
# distinct ideas, since it never applies to a different-topic pair.
DEDUP_SAME_TOPIC_THRESHOLD = float(os.getenv("DEDUP_SAME_TOPIC_THRESHOLD", "0.62"))


def load_brand_profile(path: str | Path | None = None) -> BrandProfile:
    """Load the active (latest) versioned brand profile from data/brand_profile.json."""
    profile_path = Path(path) if path else BASE_DIR / "data" / "brand_profile.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return BrandProfile(**raw)
