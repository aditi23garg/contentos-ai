"""
Smoke tests for LLMProvider._request()'s JSON-mode probing/fallback logic.
Mocks the OpenAI client entirely -- no live API needed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.path.insert(0, ".")

PASS, FAIL = "PASS", "FAIL"
errors: list[str] = []
total = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global total
    total += 1
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" -- {detail}" if detail else ""))
        errors.append(name)


import os  # noqa: E402
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

from app.providers.llm import LLMProvider  # noqa: E402

print("\n=== _request: JSON mode supported ===\n")

provider = LLMProvider.__new__(LLMProvider)  # bypass __init__'s real client setup
provider._model = "test-model"
provider._json_mode_supported = None
provider._client = MagicMock()
provider._client.chat.completions.create.return_value = "OK_RESPONSE"

result = provider._request([{"role": "user", "content": "hi"}], 100)
call_kwargs = provider._client.chat.completions.create.call_args.kwargs
check("T1 first call includes response_format", call_kwargs.get("response_format") == {"type": "json_object"})
check("T2 capability cached as supported", provider._json_mode_supported is True)
check("T3 returns the response", result == "OK_RESPONSE")

provider._client.chat.completions.create.reset_mock()
provider._request([{"role": "user", "content": "hi again"}], 100)
call_kwargs = provider._client.chat.completions.create.call_args.kwargs
check("T4 subsequent call still uses response_format (no re-probing needed)",
      call_kwargs.get("response_format") == {"type": "json_object"})


print("\n=== _request: JSON mode NOT supported (provider rejects param) ===\n")

provider2 = LLMProvider.__new__(LLMProvider)
provider2._model = "test-model"
provider2._json_mode_supported = None
provider2._client = MagicMock()


def create_side_effect(**kwargs):
    if "response_format" in kwargs:
        raise TypeError("unexpected keyword argument 'response_format'")
    return "FALLBACK_RESPONSE"


provider2._client.chat.completions.create.side_effect = create_side_effect

result2 = provider2._request([{"role": "user", "content": "hi"}], 100)
check("T5 falls back to no response_format on first call", result2 == "FALLBACK_RESPONSE")
check("T6 capability cached as unsupported", provider2._json_mode_supported is False)

provider2._client.chat.completions.create.reset_mock()
provider2._client.chat.completions.create.side_effect = None
provider2._client.chat.completions.create.return_value = "SECOND_CALL_RESPONSE"
provider2._request([{"role": "user", "content": "hi again"}], 100)
call_args_list = provider2._client.chat.completions.create.call_args_list
check("T7 second call skips the probe entirely (only one create() call, not two)",
      len(call_args_list) == 1, f"got {len(call_args_list)} calls")
check("T8 second call has no response_format param",
      "response_format" not in call_args_list[0].kwargs)

print()
if errors:
    print(f"  {len(errors)}/{total} test(s) FAILED: {', '.join(errors)}")
    sys.exit(1)
else:
    print(f"  All {total} tests passed.\n")
