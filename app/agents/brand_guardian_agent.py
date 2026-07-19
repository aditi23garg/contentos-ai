"""
Brand Guardian Agent.

Scores generated content against the six-dimension rubric from the spec. The sixth
dimension, "strategic_fit", is where the review's "Editorial Director" concept was
folded in (v2.1 decision) rather than becoming a separate agent -- topic-diversity /
calendar-balance checks would feed into this score once the Content Diversity Check
(v2.2) is wired to real post history; for this first slice the model estimates it
qualitatively.
"""

from __future__ import annotations

from app.core import config
from app.core.schemas import BrandGuardianResult, BrandProfile, GeneratedContent, RubricScores
from app.providers.llm import get_llm

PROMPT_VERSION = "v1"


def evaluate(brand: BrandProfile, content: GeneratedContent) -> BrandGuardianResult:
    system_prompt = f"""You are the Brand Guardian Agent for the brand "{brand.brand_name}".
Niche: {brand.niche}. Allowed topics: {', '.join(brand.allowed_topics)}.
Brand tone: {', '.join(brand.tone)}. Content philosophy: {', '.join(brand.content_philosophy)}.

Score the given content on these six dimensions, each 1-5:
- niche_fit: does this belong inside the configured niche?
- brand_alignment: does tone/style/voice match the brand?
- originality: does it feel fresh, not generic or cliche?
- value_to_audience: would this genuinely help the target reader?
- grammar_clarity: is it clean, readable, publish-ready?
- strategic_fit: does it feel balanced/varied rather than repetitive for this brand?

Be honest and critical -- do not default to high scores."""

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
