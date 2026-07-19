"""
Research Agent.

Scope in this first slice: generates niche-locked ideas from the model's own knowledge,
weighted toward the trusted-source categories in the spec's Research Agent Rules. It does
NOT yet call a live search API or the ChromaDB dedup/knowledge-retrieval layer -- both are
direct follow-on additions (Content Batching / Knowledge Refresh Scheduler) once this
slice is proven, not new design decisions.
"""

from __future__ import annotations

from app.core.schemas import BrandProfile, Idea
from app.providers.llm import get_llm

SOURCE_CREDIBILITY_NOTE = (
    "When you cite knowledge_sources_used, prefer high-credibility categories in this "
    "order: books, peer-reviewed research, government publications, trusted podcasts, "
    "industry experts. Avoid citing 'general internet' or 'trending topics' as a source."
)


def run_research(brand: BrandProfile, count: int) -> list[Idea]:
    system_prompt = f"""You are the Research Agent for a niche content brand called
"{brand.brand_name}". Niche: {brand.niche}.

You must NEVER leave this niche. Allowed topics: {', '.join(brand.allowed_topics)}.
Forbidden topics: {', '.join(brand.forbidden_topics)}.

Research only evergreen, trustworthy material: motivational books, psychology,
behavioral science, habits, mindset, philosophy, success principles, biographies,
historical stories, productivity, audience pain points, and trusted niche discussions.
Never propose anything based on random internet trends. {SOURCE_CREDIBILITY_NOTE}

For each idea, you must state WHY you're proposing it (reasoning), not just what it is --
this becomes part of the system's permanent decision log."""

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

    result = get_llm().complete_json(system_prompt, user_prompt)
    return [Idea(**item) for item in result["ideas"]]
