"""
ContentOS AI — Phase 1, first runnable slice.

Runs: Research Agent -> Content Producer Agent -> Brand Guardian Agent, for one idea,
with a bounded regenerate loop, and prints the result plus the full decision log.

Usage:
    cp .env.example .env      # fill in GROQ_API_KEY (or set LLM_PROVIDER=ollama)
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import logging

from app.core.config import load_brand_profile
from app.graph.pipeline import build_pipeline

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")


def _print_decision_log(decisions: list) -> None:
    print("\n--- agent_decisions log ---")
    for entry in decisions:
        line = f"[{entry.timestamp:%H:%M:%S}] {entry.agent_name}: {entry.output_summary}"
        if entry.passed is not None:
            line += f" (passed={entry.passed}, avg_score={entry.scores.average:.2f})"
        print(line)


def main() -> None:
    brand = load_brand_profile()
    pipeline = build_pipeline()

    print(f"Running ContentOS AI Phase 1 pipeline for brand: {brand.brand_name}\n")
    final_state = pipeline.invoke({"brand": brand, "retries": 0})

    print("=" * 60)
    print(f"STATUS: {final_state['status'].upper()}")
    print("=" * 60)

    idea = final_state["idea"]
    content = final_state["content"]
    scores = final_state["guardian_result"].scores

    print(f"\nTopic: {idea.topic}")
    print(f"Angle: {idea.angle}")
    print(f"\nCaption:\n{content.caption}")
    print(f"\nImage prompt:\n{content.image_prompt}")
    print(f"\nHashtags: {' '.join(content.hashtags)}")
    print(f"CTA: {content.cta}")
    print("\nRubric scores:")
    for dim, val in scores.model_dump().items():
        print(f"  {dim}: {val}")
    print(f"  average: {scores.average:.2f}")

    _print_decision_log(final_state["decisions"])


if __name__ == "__main__":
    main()
