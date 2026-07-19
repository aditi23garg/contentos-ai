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

# --- Brand Guardian rubric thresholds (Configuration section of the spec) ----------
RUBRIC_PASS_AVERAGE = float(os.getenv("RUBRIC_PASS_AVERAGE", "4.0"))
RUBRIC_MIN_DIMENSION = int(os.getenv("RUBRIC_MIN_DIMENSION", "3"))

# --- Batch sizing (kept small for this first runnable slice) -----------------------
IDEAS_PER_RUN = int(os.getenv("IDEAS_PER_RUN", "3"))
MAX_GUARDIAN_RETRIES = int(os.getenv("MAX_GUARDIAN_RETRIES", "1"))


def load_brand_profile(path: str | Path | None = None) -> BrandProfile:
    """Load the active (latest) versioned brand profile from data/brand_profile.json."""
    profile_path = Path(path) if path else BASE_DIR / "data" / "brand_profile.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return BrandProfile(**raw)
