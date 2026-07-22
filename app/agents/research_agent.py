"""
Research Agent.

Scope in this first slice: generates niche-locked ideas from the model's own knowledge,
weighted toward the trusted-source categories in the spec's Research Agent Rules. It does
NOT yet call a live search API or the ChromaDB knowledge-retrieval layer -- that's a
direct follow-on addition (Knowledge Refresh Scheduler) once this slice is proven, not
a new design decision.

Recent-history awareness: run_research can be given the same recent-approved-topics
list already computed for the Brand Guardian's strategic_fit score (see
get_recent_approved_topics / Content Diversity Check). Without it, every candidate was
proposed blind and dedup_filter (app/services/batching.py) was the only thing catching
collisions with prior approved history -- fine early on, but as approved-idea history
grows in a narrow niche, an increasing share of a fresh batch gets filtered out after
the fact, sometimes all of it (build_batch_queue's single-item fallback). Feeding the
same history in up front lets Research actively steer away from recently covered
themes, so dedup becomes a safety net again instead of the primary filter. Zero-cost:
reuses an existing query, no new agent or LLM call.
"""

from __future__ import annotations

from app.core.schemas import BrandProfile, Idea
from app.providers.llm import get_llm

SOURCE_CREDIBILITY_NOTE = (
    "When you cite knowledge_sources_used, prefer high-credibility categories in this "
    "order: books, peer-reviewed research, government publications, trusted podcasts, "
    "industry experts. Avoid citing 'general internet' or 'trending topics' as a source."
)


def run_research(brand: BrandProfile, count: int, recent_topics: list[str] | None = None) -> list[Idea]:
    """
    Args:
        brand: The active brand profile.
        count: Number of fresh candidate ideas to request.
        recent_topics: Last N approved post topics+angles (newest-first), from
            get_recent_approved_topics(). Pass an empty list (or omit) when no
            history is available (first run, tests, etc.). When provided, injected
            into the system prompt so Research steers away from recently covered
            themes instead of relying solely on the post-hoc dedup filter.
    """
    recent_topics = recent_topics or []

    history_section = ""
    if recent_topics:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(recent_topics))
        history_section = f"""

## Recent Post History (last {len(recent_topics)} approved post{'s' if len(recent_topics) != 1 else ''} -- avoid repeating these)

{numbered}

Do not propose an idea that repeats the same theme and angle as one of the entries
above. A related theme is fine ONLY if you give it a meaningfully different angle
from anything already covered -- otherwise pick a different theme within the niche."""

    system_prompt = f"""You are the Research Agent for a niche content brand called
"{brand.brand_name}". Niche: {brand.niche}.

You must NEVER leave this niche. Allowed topics: {', '.join(brand.allowed_topics)}.
Forbidden topics: {', '.join(brand.forbidden_topics)}.

Research only evergreen, trustworthy material: motivational books, psychology,
behavioral science, habits, mindset, philosophy, success principles, biographies,
historical stories, productivity, audience pain points, and trusted niche discussions.
Never propose anything based on random internet trends. {SOURCE_CREDIBILITY_NOTE}

For each idea, you must state WHY you're proposing it (reasoning), not just what it is --
this becomes part of the system's permanent decision log.{history_section}"""

    user_prompt = f"""Propose {count} distinct content ideas for this brand.

Return a JSON object of this exact shape:
{{
  "ideas": [
    {{
      "topic": "short topic name",
      "angle": "the specific angle or hook for this idea",
      "reasoning": "why this idea is a good fit for the brand and audience right now",
      "confidence_score": 0.0-1.0,
      "knowledge_sources_used": ["category 1", "category 2"]
    }}
  ]
}}"""

    result = get_llm().complete_json(
        system_prompt,
        user_prompt,
        # Each idea has ~4 fields including a full-sentence `reasoning` -- roughly
        # 150 tokens is a safe per-idea budget. A single-idea call still gets the
        # global default floor; a 20-idea batch call gets real headroom instead of
        # inheriting a cap sized for one idea.
        max_tokens=max(1024, count * 150),
    )
    return [Idea(**item) for item in result["ideas"]]
