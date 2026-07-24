"""
ContentOS AI — Phase 1, Content Batching + Human Approval Gate.

Runs one full weekly-style batch cycle: Research generates a pool of candidate ideas,
dedup-filters them against previously approved history (ChromaDB) and against each
other, takes the top BATCH_SIZE by confidence, and runs each through Content Producer
-> Brand Guardian (with a bounded per-item retry) before persisting the whole batch to
SQLite. A Guardian pass lands in "pending_review", not "approved" -- run dashboard.py
to actually review, approve, reject, or edit each item.

Usage:
    cp .env.example .env      # fill in GROQ_API_KEY (or set LLM_PROVIDER=ollama)
    pip install -r requirements.txt
    python main.py
    streamlit run dashboard.py   # review the batch this just produced
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


def _print_batch_item(index: int, result: dict) -> None:
    idea = result["idea"]
    content = result["content"]
    scores = result["guardian_result"].scores

    label = "PENDING REVIEW" if result["status"] == "pending_review" else result["status"].upper()
    print("\n" + "-" * 60)
    print(f"[{index}] {label} — {idea.topic}")
    print("-" * 60)
    print(f"Angle: {idea.angle}")
    print(f"\nCaption:\n{content.caption}")
    print(f"\nImage prompt:\n{content.image_prompt}")
    print(f"\nHashtags: {' '.join(content.hashtags)}")
    print(f"CTA: {content.cta}")
    print("\nRubric scores:", {k: v for k, v in scores.model_dump().items()}, f"| average: {scores.average:.2f}")


def main() -> None:
    brand = load_brand_profile()
    pipeline = build_pipeline()

    print(f"Running ContentOS AI batch pipeline for brand: {brand.brand_name}\n")
    final_state = pipeline.invoke({"brand": brand})

    print("=" * 60)
    print(f"BATCH SUMMARY: {final_state['batch_summary']}")
    print("=" * 60)

    for i, result in enumerate(final_state["batch_results"], start=1):
        _print_batch_item(i, result)

    _print_decision_log(final_state["decisions"])

    pending = sum(1 for r in final_state["batch_results"] if r["status"] == "pending_review")
    if pending:
        print(f"\n{pending} item(s) awaiting your review — run: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
