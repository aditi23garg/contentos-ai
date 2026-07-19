"""
LLM provider abstraction.

Groq and Ollama are both OpenAI-SDK-compatible, so a single client class covers both --
switching providers is purely a matter of which base_url/api_key/model get read from
config. Adding OpenAI/Claude/Gemini/DeepSeek later means adding a branch here, not
touching any agent code (Provider Abstraction requirement).
"""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from app.core import config

logger = logging.getLogger(__name__)


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

        # Set on every complete_json() call so callers (graph nodes) can optionally
        # attach it to their AgentDecisionLog entry for full observability, without
        # complete_json()'s return contract having to change to a tuple.
        self.last_retry_count: int = 0

    def complete_json(
        self, system_prompt: str, user_prompt: str, max_retries: int = 2
    ) -> dict:
        """
        Call the model and parse a JSON object from its response.

        We prompt for JSON-only output rather than relying on a provider-specific
        structured-output feature, since that keeps this working identically across
        Groq and Ollama models without per-provider branching.

        The model occasionally truncates a long response mid-JSON (hits the token
        limit before finishing the object) or emits an unescaped/extra character
        around a string field. `max_tokens` gives long fields (e.g. Brand Guardian's
        `reason`) enough room to finish -- that's the preventative fix. The retry
        loop is the reactive fix for whatever still slips through: a fresh call
        almost always returns valid JSON.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
                + "\n\nRespond with ONLY a single valid JSON object. No prose, "
                "no markdown code fences, no preamble.",
            },
            {"role": "user", "content": user_prompt},
        ]

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(max_retries + 1):
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.7,
                max_tokens=config.LLM_MAX_TOKENS,
                messages=messages,
            )
            raw = response.choices[0].message.content.strip()
            try:
                parsed = self._extract_json(raw)
                self.last_retry_count = attempt
                return parsed
            except ValueError as exc:
                last_exc = exc
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
        self.last_retry_count = max_retries
        raise last_exc

    @staticmethod
    def _extract_json(raw: str) -> dict:
        # Strip markdown fences if the model added them despite instructions.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Second attempt: extract the first {...} block via regex in case the
        # model prefixed or suffixed extra text around the JSON object. Note this
        # does NOT recover truly truncated JSON (a missing closing brace still
        # fails here) -- that case is handled by the retry loop above, not this.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Model did not return valid JSON:\n{raw}")


_provider: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """Lazy singleton so importing this module doesn't require API keys to be set."""
    global _provider
    if _provider is None:
        _provider = LLMProvider()
    return _provider

