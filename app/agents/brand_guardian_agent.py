"""
Brand Guardian Agent.

Scores generated content against the six-dimension rubric from the spec. The sixth
dimension, "strategic_fit", is where the review's "Editorial Director" concept was
folded in (v2.1 decision) rather than becoming a separate agent -- topic-diversity /
calendar-balance checks now feed into this score with real approved-post history
injected into the prompt (Content Diversity Check, v2). Prior to v2, the model
estimated strategic_fit qualitatively with no concrete evidence of what had already
been published, causing it to default to 4 almost every time.
"""

from __future__ import annotations

from app.core import config
from app.core.schemas import BrandGuardianResult, BrandProfile, GeneratedContent, RubricScores
from app.providers.llm import get_llm

PROMPT_VERSION = "v2"


def evaluate(
    brand: BrandProfile,
    content: GeneratedContent,
    recent_topics: list[str] | None = None,
) -> BrandGuardianResult:
    """
    Score `content` against the six-dimension rubric.

    Args:
        brand: The active brand profile.
        content: The post produced by the Content Producer Agent.
        recent_topics: Last N approved post topics+angles (newest-first), from
            get_recent_approved_topics(). Pass an empty list (or omit) when no
            history is available (first run, tests, etc.). When provided, injected
            into the system prompt so the LLM has concrete evidence for strategic_fit.
    """
    recent_topics = recent_topics or []

    history_section = ""
    if recent_topics:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(recent_topics))
        history_section = f"""

## Recent Post History (last {len(recent_topics)} approved post{'s' if len(recent_topics) != 1 else ''} — use this when scoring strategic_fit)

{numbered}

When scoring strategic_fit, be concrete:
- Score 1-2 if the new content closely matches one of the topics above (same theme, same angle).
- Score 3 if it is thematically adjacent but adds a meaningfully different perspective.
- Score 4-5 only if it introduces a fresh theme not covered in recent history."""

    system_prompt = f"""You are the Brand Guardian Agent for the brand "{brand.brand_name}".
Niche: {brand.niche}. Allowed topics: {', '.join(brand.allowed_topics)}.
Brand tone: {', '.join(brand.tone)}. Content philosophy: {', '.join(brand.content_philosophy)}.

Score the given content on these six dimensions, each 1-5:
- niche_fit: does this belong inside the configured niche?
- brand_alignment: does tone/style/voice match the brand?
- originality: does it feel fresh, not generic or cliche?
- value_to_audience: would this genuinely help the target reader?
- grammar_clarity: is it clean, readable, publish-ready?
- strategic_fit: does it add variety to the publishing calendar, or does it repeat a recently approved theme?

Be honest and critical -- do not default to high scores.{history_section}"""

    user_prompt = f"""Caption: {content.caption}
Image prompt: {content.image_prompt}
Hashtags: {', '.join(content.hashtags)}
CTA: {content.cta}

Return a JSON object of this exact shape:
{{
  "scores": {{
    "niche_fit": 1-5,
    "brand_alignment": 1-5,
    "originality": 1-5,
    "value_to_audience": 1-5,
    "grammar_clarity": 1-5,
    "strategic_fit": 1-5
  }},
  "reason": "one or two sentences explaining the scores, especially any low ones"
}}"""

    result = get_llm().complete_json(system_prompt, user_prompt)
    scores = RubricScores(**result["scores"])

    passed = (
        scores.min_dimension >= config.RUBRIC_MIN_DIMENSION
        and scores.average >= config.RUBRIC_PASS_AVERAGE
    )

    return BrandGuardianResult(
        scores=scores,
        passed=passed,
        reason=result.get("reason", ""),
        prompt_version=PROMPT_VERSION,
    )
