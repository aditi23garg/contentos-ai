"""
Guardian Discrimination Probe
==============================
Sends 4 hand-crafted content samples to the real BrandGuardianAgent (live LLM call)
and checks whether scores meaningfully differentiate quality levels.

Cases:
  A — genuinely strong, on-brand content                  (expect: high across the board)
  B — on-topic but generic/cliché, surface-level          (expect: niche_fit ok, originality low)
  C — wrong niche entirely (cryptocurrency)               (expect: niche_fit=1, low overall)
  D — on-topic but poor grammar, clickbait, off-tone      (expect: grammar_clarity low, brand_alignment low)

Run with:
    python probe_guardian.py
"""

from __future__ import annotations

from app.core.config import load_brand_profile
from app.core.schemas import GeneratedContent, Idea
from app.agents.brand_guardian_agent import evaluate

brand = load_brand_profile()

CASES: list[tuple[str, dict]] = [
    # -----------------------------------------------------------------------
    # A: Strong, specific, genuinely useful on-brand content
    # -----------------------------------------------------------------------
    ("A — Strong on-brand", {
        "caption": (
            "Fear doesn't disappear when you become brave — it just stops running your decisions. "
            "Every time you act despite the discomfort, you're rewiring the pattern. "
            "That's not motivation. That's neuroscience."
        ),
        "image_prompt": (
            "A single candle flame in a dark, minimal room. Warm soft gold light against "
            "deep green walls, beige linen texture in the foreground. Serene, composed."
        ),
        "hashtags": ["#Courage", "#MindsetShift", "#PersonalGrowth"],
        "cta": "What's one thing you'd do this week if fear wasn't a factor?",
    }),

    # -----------------------------------------------------------------------
    # B: On-topic but hollow / generic / cliché — motivational filler
    # -----------------------------------------------------------------------
    ("B — Generic & cliché", {
        "caption": (
            "Believe in yourself! You can do it! Every day is a new beginning. "
            "Keep pushing and never give up! Your dreams are waiting for you. "
            "Just keep going and you will succeed! 💪🔥"
        ),
        "image_prompt": "Sunrise over mountains with the word BELIEVE overlaid in bold text.",
        "hashtags": ["#Motivation", "#NeverGiveUp", "#BelieveInYourself", "#Success", "#Goals"],
        "cta": "Like and share if this inspired you!",
    }),

    # -----------------------------------------------------------------------
    # C: Completely wrong niche — cryptocurrency trading tips
    # -----------------------------------------------------------------------
    ("C — Wrong niche (crypto)", {
        "caption": (
            "Bitcoin just broke resistance at $68k — here's why altcoins are about to pump. "
            "The smart money is rotating into ETH and SOL right now. "
            "Don't get left behind this bull run. DYOR but this is the signal you've been waiting for."
        ),
        "image_prompt": "Green candlestick chart on a dark background with BTC price ticker.",
        "hashtags": ["#Bitcoin", "#Crypto", "#BullRun", "#DeFi", "#Trading"],
        "cta": "Drop a 🚀 if you're holding through the dip",
    }),

    # -----------------------------------------------------------------------
    # D: On-topic but terrible grammar, clickbait tone, off-brand voice
    # -----------------------------------------------------------------------
    ("D — Poor grammar & clickbait tone", {
        "caption": (
            "u wont beleive how this 1 WEIRD trick change my hole life!!!! "
            "doctors HATE this method but it work 100% guarantee. "
            "i was broke and sad now im happy success person. click link in bio NOW!!!"
        ),
        "image_prompt": "Bright red background with huge yellow text: SECRET REVEALED!!!",
        "hashtags": ["#LifeHack", "#SECRET", "#YouWontBelieve", "#ViralContent"],
        "cta": "CLICK LINK IN BIO RIGHT NOW BEFORE ITS GONE!!!",
    }),
]


def score_row(label: str, scores: dict, avg: float, passed: bool, reason: str) -> None:
    bar = lambda v: "#" * v + "." * (5 - v)
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    for dim, val in scores.items():
        print(f"  {dim:<22} {bar(val)} {val}/5")
    print(f"  {'AVERAGE':<22} {'':6} {avg:.2f}  |  passed={passed}")
    print(f"  reason: {reason[:120]}{'...' if len(reason)>120 else ''}")


print("\n" + "=" * 60)
print("  GUARDIAN DISCRIMINATION PROBE — live LLM calls")
print("=" * 60)

results = []
for label, fields in CASES:
    content = GeneratedContent(
        idea_topic=label,
        caption=fields["caption"],
        image_prompt=fields["image_prompt"],
        hashtags=fields["hashtags"],
        cta=fields["cta"],
    )
    print(f"\n  Evaluating: {label} ...", flush=True)
    result = evaluate(brand, content)
    results.append((label, result))
    score_row(label, result.scores.model_dump(), result.scores.average, result.passed, result.reason)

# ─── Analysis ───────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  ANALYSIS")
print("=" * 60)

averages = [(label, r.scores.average) for label, r in results]
spread = max(a for _, a in averages) - min(a for _, a in averages)
print(f"\n  Score spread (max - min avg): {spread:.2f}")
print(f"  {'Label':<30} {'Avg':>5}  {'niche':>5}  {'orig':>5}  {'gram':>5}")
print(f"  {'-'*55}")
for label, r in results:
    s = r.scores
    print(f"  {label:<30} {s.average:>5.2f}  {s.niche_fit:>5}  {s.originality:>5}  {s.grammar_clarity:>5}")

print()
if spread < 1.0:
    print("  [!] VERDICT: Spread < 1.0 -- Guardian is NOT discriminating.")
    print("      The rubric prompt needs tightening.")
elif spread < 1.5:
    print("  [!] VERDICT: Spread < 1.5 -- Marginal discrimination.")
    print("      Rubric prompt improvements recommended.")
else:
    print("  [OK] VERDICT: Spread >= 1.5 -- Guardian IS discriminating meaningfully.")

# Specific checks
_, crypto_result = results[2]
_, grammar_result = results[3]

print()
if crypto_result.scores.niche_fit >= 4:
    print("  [!] niche_fit FAILED: crypto post scored >= 4 on niche_fit (should be 1-2)")
else:
    print(f"  [OK] niche_fit: crypto post niche_fit = {crypto_result.scores.niche_fit} (correctly low)")

if grammar_result.scores.grammar_clarity >= 4:
    print("  [!] grammar_clarity FAILED: broken-grammar post scored >= 4 (should be 1-2)")
else:
    print(f"  [OK] grammar_clarity: broken-grammar post grammar_clarity = {grammar_result.scores.grammar_clarity} (correctly low)")
