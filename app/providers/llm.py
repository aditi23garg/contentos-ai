"""
LLM provider abstraction.

Groq and Ollama are both OpenAI-SDK-compatible, so a single client class covers both --
switching providers is purely a matter of which base_url/api_key/model get read from
config. Adding OpenAI/Claude/Gemini/DeepSeek later means adding a branch here, not
touching any agent code (Provider Abstraction requirement).
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from app.core import config


class LLMProvider:
    def __init__(self) -> None:
        if config.LLM_PROVIDER == "groq":
            self._client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
            self._model = config.GROQ_MODEL
        elif config.LLM_PROVIDER == "ollama":
            # Ollama's OpenAI-compatible endpoint accepts any non-empty api_key string.
            self._client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL)
            self._model = config.OLLAMA_MODEL
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Call the model and parse a JSON object from its response.

        We prompt for JSON-only output rather than relying on a provider-specific
        structured-output feature, since that keeps this working identically across
        Groq and Ollama models without per-provider branching.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                    + "\n\nRespond with ONLY a single valid JSON object. No prose, "
                    "no markdown code fences, no preamble.",
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        # Strip markdown fences if the model added them despite instructions.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON:\n{raw}") from exc


_provider: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """Lazy singleton so importing this module doesn't require API keys to be set."""
    global _provider
    if _provider is None:
        _provider = LLMProvider()
    return _provider
