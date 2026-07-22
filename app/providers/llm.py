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
    # Matches the start of a plausible next JSON key, e.g. `"scores": ` --
    # used only to disambiguate a quote-then-comma inside _repair_json_string.
    _LOOKS_LIKE_NEXT_KEY = re.compile(r'^"[^"\\]*"\s*:')

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

        # Groq (and most recent Ollama builds) support OpenAI-style JSON mode --
        # constrained decoding that makes the model syntactically incapable of
        # emitting invalid/incomplete JSON, rather than just being *asked* to via
        # the system prompt. This is a strictly stronger fix than prompting harder
        # or raising max_tokens for the "model stops mid-object" failure mode.
        # Not every Ollama model supports it, so we probe once and remember the
        # result rather than assuming -- see _request() below.
        self._json_mode_supported: bool | None = None  # None = not probed yet

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
        max_tokens: int | None = None,
    ) -> dict:
        """
        Call the model and parse a JSON object from its response.

        Primary defense: constrained JSON mode via _request() when the provider
        supports it (Groq does) -- this makes the model syntactically incapable of
        emitting invalid or incomplete JSON, rather than just being *asked* to via
        the system prompt. Some models still stop generation mid-object even when
        the JSON so far is well within max_tokens (observed on Groq's
        llama-3.3-70b-versatile) -- that failure mode is what JSON mode targets;
        raising max_tokens alone does not fix it, since the model isn't running out
        of room, it's just stopping early.

        Two more layers below that: _extract_json()'s state-machine repair fixes
        the model still emitting an unescaped inner quote (a formatting mistake
        JSON mode can't always prevent either), and the retry loop is the last
        resort for whatever slips through both -- a fresh call almost always
        returns valid JSON.

        `max_tokens` still matters for genuinely large output (e.g. a batch of
        ~20 Research ideas) and defaults to config.LLM_MAX_TOKENS; callers
        producing more content than a single Guardian `reason` field should pass
        a higher value explicitly.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
                + "\n\nRespond with ONLY a single valid JSON object. No prose, "
                "no markdown code fences, no preamble. Output compact JSON on a "
                "single line -- no indentation, no pretty-printing, no extra "
                "whitespace between tokens.",
            },
            {"role": "user", "content": user_prompt},
        ]

        token_limit = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(max_retries + 1):
            response = self._request(messages, token_limit)
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

    def _request(self, messages: list[dict], token_limit: int):
        """
        Issue the chat completion, preferring constrained JSON mode when the
        provider supports it (see _json_mode_supported comment in __init__).

        Probes capability at most once per LLMProvider instance: on the first
        call we try with response_format set; if the provider rejects the
        parameter outright (some Ollama models do), we remember that and never
        try it again for this instance, instead of failing or re-probing every call.
        """
        kwargs = dict(model=self._model, temperature=0.7, max_tokens=token_limit, messages=messages)

        if self._json_mode_supported is not False:
            try:
                response = self._client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
                self._json_mode_supported = True
                return response
            except Exception as exc:  # noqa: BLE001 -- provider capability probe, not a parse error
                if self._json_mode_supported is None:
                    logger.info(
                        "Provider does not support response_format=json_object "
                        "(%s); falling back to prompt-only JSON enforcement for "
                        "the rest of this run.",
                        exc,
                    )
                self._json_mode_supported = False

        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _repair_json_string(raw: str) -> str:
        """
        Apply two lightweight repairs to recover from the most common model-side
        formatting mistakes:

        1. Strip bare control characters (\\x00-\\x1f except \\t, \\n, \\r) that are
           illegal inside JSON strings even when escaped.

        2. Escape unescaped double-quotes that appear *inside* a JSON string
           value — the canonical Guardian failure mode where the model writes:
               "reason": "The concept of "growth mindset" is well-established."
           which breaks strict JSON at the inner quote.

        Repair 2 uses a character-by-character state machine because a regex
        approach only matches *valid* string tokens and therefore can never see
        the malformed outer boundary needed to locate the bare inner quotes.
        The state machine tracks whether each `"` is a structural delimiter
        (opening/closing a JSON string) or a bare emphasis quote inside a value,
        and escapes the latter in-place.

        The `,` lookahead case is genuinely ambiguous on its own: a comma right
        after a quote could be a real field separator (`"reason": "text", "x": 1`)
        or just prose punctuation that happens to follow a bare inner quote
        (`"The word "discipline", used loosely, matters."`). We disambiguate by
        peeking *past* the comma: if what follows looks like a new JSON key
        (`"key":`) or end-of-input, it's a real closing delimiter; otherwise
        it's still inside the value and the quote gets escaped.
        """
        # --- repair 1: illegal control characters ---
        raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)

        # --- repair 2: state-machine quote escaper ---
        out: list[str] = []
        in_string = False     # True while inside a JSON string literal
        escape_next = False   # True when previous char was a backslash

        i = 0
        while i < len(raw):
            ch = raw[i]

            if escape_next:
                out.append(ch)
                escape_next = False
                i += 1
                continue

            if ch == "\\":
                out.append(ch)
                if in_string:
                    escape_next = True
                i += 1
                continue

            if ch == '"':
                if not in_string:
                    # Opening delimiter — start of a new JSON string.
                    in_string = True
                    out.append(ch)
                else:
                    # We are inside a string. Determine whether this `"` is the
                    # closing delimiter or a bare embedded emphasis quote.
                    # Heuristic: peek ahead (skip whitespace) — if the next
                    # non-whitespace char is a JSON structural token (`:`, `,`,
                    # `}`, `]`) or end-of-input, treat this as the closer.
                    # Otherwise it is a bare inner quote and we escape it.
                    rest = raw[i + 1:].lstrip(" \t\r\n")
                    if not rest or rest[0] in ":}]":
                        in_string = False
                        out.append(ch)
                    elif rest[0] == ",":
                        # Ambiguous on its own -- peek past the comma. A real
                        # closing comma is followed by a new key ("key": ...);
                        # otherwise this is prose punctuation after a bare
                        # inner quote and we're still inside the value.
                        after_comma = rest[1:].lstrip(" \t\r\n")
                        if not after_comma or LLMProvider._LOOKS_LIKE_NEXT_KEY.match(after_comma):
                            in_string = False
                            out.append(ch)
                        else:
                            out.append('\\"')  # escape the bare inner quote
                    else:
                        out.append('\\"')  # escape the bare inner quote
                i += 1
                continue

            out.append(ch)
            i += 1

        return "".join(out)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        # Strip markdown fences if the model added them despite instructions.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        # Attempt 1: parse as-is (the happy path — no repairs needed).
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 2: apply lightweight text repairs for the two most common
        # model-side mistakes (illegal control chars + unescaped inner quotes)
        # before trying to parse again.  This recovers the Guardian's "reason"
        # embedded-quote failure without spending a full retry LLM call.
        try:
            return json.loads(LLMProvider._repair_json_string(cleaned))
        except json.JSONDecodeError:
            pass

        # Attempt 3: extract the first {...} block via regex in case the model
        # prefixed or suffixed extra text, then apply repairs + parse.
        # Note: truly truncated JSON (missing closing brace) still fails here
        # and falls through to the retry loop in complete_json() — intentional.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
            try:
                return json.loads(LLMProvider._repair_json_string(match.group()))
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

