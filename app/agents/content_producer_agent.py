"""
Content Producer Agent.

Consolidates what the original blueprint split across Writer / Image Prompt / SEO / CTA
agents into a single structured-output call, per the v2 agent-consolidation decision --
this is a prompt-template distinction, not a reason to spend four LLM calls where one
will do.
"""

from __future__ import annotations

from app.core.schemas import BrandProfile, GeneratedContent, Idea
from app.providers.llm import get_llm

PROMPT_VERSION = "v1"


def produce_content(brand: BrandProfile, idea: Idea) -> GeneratedContent:
    system_prompt = f"""You are the Content Producer Agent for the brand "{brand.brand_name}".

Tone: {', '.join(brand.tone)}
Writing style: {', '.join(brand.writing_style)}
Audience: {brand.audience}
Visual style: {', '.join(brand.visual_style)}
Preferred colors: {', '.join(brand.preferred_colors)}
Content philosophy: {', '.join(brand.content_philosophy)}

Produce ONE complete, publish-ready post from the given idea. Every field must match
the brand's tone and writing style exactly."""

    user_prompt = f"""Idea topic: {idea.topic}
Angle: {idea.angle}
Why this idea: {idea.reasoning}

Return a JSON object of this exact shape:
{{
  "caption": "the main caption, following the brand's writing style",
  "image_prompt": "a detailed prompt for an image generation model, matching the brand's visual style and colors",
  "hashtags": ["#tag1", "#tag2"],
  "cta": "a short call to action",
  "platform_variants": {{
    "instagram": "short, emotional version",
    "linkedin": "longer, storytelling version"
  }}
}}"""

    result = get_llm().complete_json(system_prompt, user_prompt)
    return GeneratedContent(idea_topic=idea.topic, prompt_version=PROMPT_VERSION, **result)
