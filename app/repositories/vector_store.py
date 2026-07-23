"""
Vector store for idea deduplication (ChromaDB), per the spec's Content Batching /
Planner Logic requirement: "before a new idea reaches the Content Producer, embed it
and run a similarity search against ChromaDB's store of previously published/approved
posts."

Only approved ideas get added here (see graph/pipeline.py's persist_node) -- a
rejected idea shouldn't block a genuinely different future attempt at a similar angle.

Note: ChromaDB's default embedding function downloads a small ONNX model from the
internet on first use. That's a one-time cost on a machine with normal internet
access; it is NOT available in network-restricted environments (e.g. this was
developed and unit-tested with an injected fake embedding function -- see the
`embedding_function` parameter below -- specifically because of that constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str | Path | None = None, embedding_function: Any = None):
        from app.core import config

        path = Path(persist_dir) if persist_dir else config.CHROMA_PERSIST_DIR
        path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(path))
        kwargs = {"name": "approved_ideas"}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(**kwargs)

    def find_similar(
        self,
        text: str,
        threshold: float,
        topic: str | None = None,
        same_topic_threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return [(matched_text, similarity)] for stored ideas above threshold, best first.

        Two-tier threshold: if `topic` and `same_topic_threshold` are given, a stored
        idea whose metadata topic matches `topic` (case-insensitive) is compared
        against `same_topic_threshold` instead of `threshold`. See
        config.DEDUP_SAME_TOPIC_THRESHOLD for why -- exact topic-label match turned
        out to be a much stronger, free signal than embedding similarity alone for
        this content, per calibrate_dedup_threshold.py's findings against real data.
        Callers that don't pass `topic` get the old single-threshold behavior.
        """
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[text], n_results=min(5, count), include=["documents", "distances", "metadatas"]
        )
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)

        matches = []
        for doc, distance, meta in zip(documents, distances, metadatas):
            # Chroma's default space is squared L2 on normalized embeddings, which for
            # normalized vectors maps roughly to a 0 (identical) - 2 (opposite) range.
            # This similarity conversion is intentionally approximate -- good enough to
            # rank and threshold on, not meant to be a precise cosine similarity value.
            similarity = max(0.0, 1 - (distance / 2))

            effective_threshold = threshold
            stored_topic = (meta or {}).get("topic", "")
            if topic and same_topic_threshold is not None and stored_topic.strip().lower() == topic.strip().lower():
                effective_threshold = same_topic_threshold

            if similarity >= effective_threshold:
                matches.append((doc, similarity))

        matches.sort(key=lambda m: m[1], reverse=True)
        return matches

    def add_idea(self, idea_id: str, text: str, topic: str | None = None) -> None:
        self._collection.add(ids=[idea_id], documents=[text], metadatas=[{"topic": topic or ""}])


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Lazy singleton, mirroring the LLMProvider pattern in app/providers/llm.py."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store