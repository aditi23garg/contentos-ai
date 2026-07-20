"""
Smoke tests for the hardened _extract_json / _repair_json_string in llm.py.
Run with:  python test_json_repair.py
"""

import sys
sys.path.insert(0, ".")

from app.providers.llm import LLMProvider

PASS = "PASS"
FAIL = "FAIL"
errors = []

def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"  {FAIL}  {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        errors.append(name)


print("\n=== _extract_json smoke tests ===\n")

# T1: clean JSON -- happy path, no repair needed
clean = '{"scores": {"niche_fit": 5}, "reason": "Clean reason with no issues."}'
r = LLMProvider._extract_json(clean)
check("T1 clean JSON parses first attempt", r["reason"] == "Clean reason with no issues.")

# T2: the canonical Guardian bug -- unescaped double-quotes inside reason
broken = '{"scores": {"niche_fit": 5}, "reason": "The concept of "growth mindset" is well-established."}'
r = LLMProvider._extract_json(broken)
check("T2 embedded unescaped quotes repaired without retry", "growth mindset" in r["reason"])

# T3: multiple quoted phrases in one field
multi = '{"reason": "Both "grit" and "resilience" are key.", "scores": {"niche_fit": 4}}'
r = LLMProvider._extract_json(multi)
check("T3 multiple embedded quoted phrases repaired", "grit" in r["reason"] and "resilience" in r["reason"])

# T4: illegal control character (vertical tab) inside a string
with_ctrl = "{\"reason\": \"Some \x0b vertical tab reason.\", \"scores\": {\"niche_fit\": 4}}"
r = LLMProvider._extract_json(with_ctrl)
check("T4 illegal control character stripped", r["scores"]["niche_fit"] == 4)

# T5: markdown fences still stripped before repairs applied
fenced = "```json\n{\"scores\": {}, \"reason\": \"Good content.\"}\n```"
r = LLMProvider._extract_json(fenced)
check("T5 markdown fence stripped then parsed cleanly", r["reason"] == "Good content.")

# T6: markdown fence + inner quotes combo
fenced_broken = "```json\n{\"scores\": {}, \"reason\": \"The idea of \"personal growth\" is central.\"}\n```"
r = LLMProvider._extract_json(fenced_broken)
check("T6 markdown fence + embedded quotes both repaired", "personal growth" in r["reason"])

# T7: extra preamble text before the JSON block
preamble = 'Here is your JSON:\n{"scores": {"niche_fit": 5}, "reason": "All good."}'
r = LLMProvider._extract_json(preamble)
check("T7 preamble text before JSON block handled", r["reason"] == "All good.")

# T8: truly truncated JSON must still raise ValueError (falls back to retry loop)
truncated = '{"reason": "Incomplete JSON that never closes...'
try:
    LLMProvider._extract_json(truncated)
    check("T8 truncated JSON raises ValueError", False, "no exception raised")
except ValueError:
    check("T8 truncated JSON raises ValueError (retry loop handles it)", True)

print()
if errors:
    print(f"  {len(errors)} test(s) FAILED: {', '.join(errors)}")
    sys.exit(1)
else:
    print(f"  All 8 tests passed.\n")
