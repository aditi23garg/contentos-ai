"""
Calibrate DEDUP_SIMILARITY_THRESHOLD against real approved-idea history, instead of
guessing a single number from one example.

Computes the full pairwise similarity matrix across every approved idea currently in
ChromaDB, using the same distance-to-similarity conversion vector_store.find_similar
uses. Splits pairs into two groups:

  - SAME-TOPIC pairs: idea.topic strings match (case-insensitive) -- a strong prior
    signal that the pair is likely a real duplicate, like the "Mindfulness" x2 case
    this script was built to investigate.
  - DIFF-TOPIC pairs: everything else -- these are more likely to be genuinely
    distinct ideas that just happen to live in the same narrow niche.

If SAME-TOPIC similarities cluster meaingfully higher than DIFF-TOPIC similarities,
there's a threshold in between that would separate them reasonably well. If the two
distributions overlap heavily, a single embedding-similarity threshold can't cleanly
tell "reworded duplicate" from "fresh angle, same theme" for this content -- and the
right fix is a different signal (e.g. an exact-topic-label check), not a smaller
number.

Usage:
    python calibrate_dedup_threshold.py                # full report
    python calibrate_dedup_threshold.py --top 20        # show 20 closest pairs (default 15)
"""

from __future__ import annotations

import argparse
import itertools

from app.repositories.db import IdeaRecord, get_session
from app.repositories.vector_store import get_vector_store


def _similarity(distance: float) -> float:
    # Mirrors VectorStore.find_similar's conversion exactly -- see that docstring
    # for why this is an approximation, not precise cosine similarity.
    return max(0.0, 1 - (distance / 2))


def main(top_n: int) -> None:
    session = get_session()
    approved = (
        session.query(IdeaRecord)
        .filter(IdeaRecord.status == "approved")
        .order_by(IdeaRecord.created_at)
        .all()
    )
    session.close()

    if len(approved) < 2:
        print(f"Only {len(approved)} approved idea(s) -- need at least 2 to compare. Run main.py a few more times first.")
        return

    store = get_vector_store()
    collection = store._collection

    # Pull every embedding once (id-aligned with `approved`) rather than doing one
    # query per idea -- collection.get() with embeddings=True gives us raw vectors
    # to compare pairwise ourselves, since we want ALL pairs, not just each idea's
    # top-5 nearest neighbors (which is all find_similar's query API exposes).
    ids = [str(row.id) for row in approved]
    got = collection.get(ids=ids, include=["embeddings"])
    id_to_vec = dict(zip(got["ids"], got["embeddings"]))

    import math

    def l2_distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    same_topic_sims: list[tuple[float, IdeaRecord, IdeaRecord]] = []
    diff_topic_sims: list[tuple[float, IdeaRecord, IdeaRecord]] = []

    for ra, rb in itertools.combinations(approved, 2):
        va, vb = id_to_vec.get(str(ra.id)), id_to_vec.get(str(rb.id))
        if va is None or vb is None:
            continue
        sim = _similarity(l2_distance(va, vb))
        bucket = same_topic_sims if ra.topic.strip().lower() == rb.topic.strip().lower() else diff_topic_sims
        bucket.append((sim, ra, rb))

    def summarize(label: str, pairs: list[tuple[float, IdeaRecord, IdeaRecord]]) -> None:
        if not pairs:
            print(f"{label}: no pairs")
            return
        sims = sorted((p[0] for p in pairs), reverse=True)
        n = len(sims)
        mean = sum(sims) / n
        print(f"{label}: n={n}  max={sims[0]:.4f}  mean={mean:.4f}  "
              f"median={sims[n // 2]:.4f}  min={sims[-1]:.4f}")
        buckets = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        for b in buckets:
            pct = sum(1 for s in sims if s >= b) / n * 100
            print(f"    >= {b:.2f}: {pct:5.1f}%")

    print(f"Loaded {len(approved)} approved ideas, {len(same_topic_sims) + len(diff_topic_sims)} pairs total\n")
    print("=== SAME topic label (likely-duplicate candidates) ===")
    summarize("same-topic", same_topic_sims)
    print("\n=== DIFFERENT topic label (likely-distinct candidates) ===")
    summarize("diff-topic", diff_topic_sims)

    print(f"\n=== {top_n} closest pairs overall (regardless of topic label) ===")
    all_pairs = sorted(same_topic_sims + diff_topic_sims, key=lambda p: p[0], reverse=True)
    for sim, ra, rb in all_pairs[:top_n]:
        same = "SAME-TOPIC" if ra.topic.strip().lower() == rb.topic.strip().lower() else "diff-topic"
        print(f"{sim:.4f}  [{same}]")
        print(f"    #{ra.id} {ra.topic!r}: {ra.angle}")
        print(f"    #{rb.id} {rb.topic!r}: {rb.angle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    main(args.top)
