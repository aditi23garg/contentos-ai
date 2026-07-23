"""
Smoke test for VectorStore's two-tier dedup threshold: a stored idea with a matching
topic label should be caught by the lower same_topic_threshold even when its
embedding similarity falls below the general threshold; a stored idea with a
different topic label at the same similarity should NOT be caught by the general
threshold alone.

Uses a real ChromaDB PersistentClient (temp dir) with a small deterministic fake
embedding function, so this exercises the actual add_idea/find_similar code path
and metadata storage -- not a mock of VectorStore itself -- without needing network
access to download a real embedding model.

"""

from __future__ import annotations

import shutil
import tempfile

from chromadb import EmbeddingFunction

from app.repositories.vector_store import VectorStore

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


class FixedVectorEF(EmbeddingFunction):
    """Maps specific known input strings to hand-picked vectors so we get exact,
    predictable similarity scores instead of depending on a real embedding model
    (which needs network access this test environment may not have)."""

    def __init__(self):
        # Chosen so that:
        #  - "near_dupe" is close-ish to "original" (mid similarity, ~0.65-0.7 band)
        #  - "distinct" is far from "original" (low similarity)
        self._vectors = {
            "original": [1.0, 0.0, 0.0],
            "near_dupe": [0.85, 0.53, 0.0],   # moderate distance from "original"
            "distinct": [0.0, 0.0, 1.0],       # orthogonal -- far from "original"
        }

    def __call__(self, input):
        return [self._vectors[t] for t in input]


tmp_dir = tempfile.mkdtemp()
try:
    store = VectorStore(persist_dir=tmp_dir, embedding_function=FixedVectorEF())
    store.add_idea("1", "original", topic="Growth Mindset")

    # Compute what similarity "near_dupe" actually gets against "original" with this
    # embedding, so the test's thresholds are grounded in the real numbers rather
    # than guessed -- mirrors how DEDUP_SAME_TOPIC_THRESHOLD itself was set from
    # calibrate_dedup_threshold.py's real data, not a guess.
    #
    # IMPORTANT: ChromaDB's query() returns SQUARED L2 distance directly (confirmed
    # empirically -- sum((x-y)**2) matches the returned "distance" exactly, no sqrt
    # applied by Chroma). find_similar() uses that raw squared value as-is in its
    # `1 - distance/2` formula, so the test must do the same -- NOT take an extra
    # sqrt(), which was an earlier bug in this test that silently computed the wrong
    # target similarity and made T2/T3 fail for the wrong reason.
    a, b = FixedVectorEF()._vectors["original"], FixedVectorEF()._vectors["near_dupe"]
    squared_l2 = sum((x - y) ** 2 for x, y in zip(a, b))
    near_dupe_similarity = max(0.0, 1 - (squared_l2 / 2))

    general_threshold = min(0.99, near_dupe_similarity + 0.1)   # deliberately ABOVE near_dupe's score
    same_topic_threshold = max(0.01, near_dupe_similarity - 0.1)  # deliberately BELOW near_dupe's score

    # --- T1: same topic label, similarity below general threshold but above
    # same_topic_threshold -> should be caught when topic is passed.
    matches = store.find_similar(
        "near_dupe", general_threshold, topic="Growth Mindset", same_topic_threshold=same_topic_threshold
    )
    check(
        "T1 same-topic candidate caught by lower same_topic_threshold "
        f"(similarity={near_dupe_similarity:.3f}, general={general_threshold:.3f}, same_topic={same_topic_threshold:.3f})",
        len(matches) == 1 and matches[0][0] == "original",
    )

    # --- T2: same similarity, but candidate's topic does NOT match stored idea's
    # topic -> should NOT be caught by the general threshold alone (same_topic gate
    # never applies since topics differ).
    matches = store.find_similar(
        "near_dupe", general_threshold, topic="Totally Different Topic", same_topic_threshold=same_topic_threshold
    )
    check(
        "T2 different-topic candidate at same similarity NOT caught (general threshold not met, "
        "same_topic gate doesn't apply)",
        len(matches) == 0,
    )

    # --- T3: without passing topic at all (old call signature / backward compat),
    # behavior matches the pre-existing single-threshold logic.
    matches = store.find_similar("near_dupe", general_threshold)
    check("T3 backward-compatible call (no topic arg) uses general threshold only", len(matches) == 0)

    # --- T4: a genuinely distinct idea (orthogonal vector) should never match,
    # regardless of topic label, even at a very low threshold.
    matches = store.find_similar(
        "distinct", 0.3, topic="Growth Mindset", same_topic_threshold=0.3
    )
    check("T4 genuinely distinct content not caught even with topic match + low threshold", len(matches) == 0)

    # --- T5: metadata actually persisted -- confirms add_idea's topic kwarg is
    # wired all the way through to ChromaDB, not silently dropped.
    raw = store._collection.get(ids=["1"], include=["metadatas"])
    check("T5 topic metadata persisted on add_idea", raw["metadatas"][0].get("topic") == "Growth Mindset")

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"\n  {total - len(errors)}/{total} tests passed.\n" if errors else f"\n  All {total} tests passed.\n")
if errors:
    raise SystemExit(1)