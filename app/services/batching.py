"""
Content Batching support: filter a pool of candidate ideas against previously
approved ideas (dedup), filter near-duplicates within the same batch, then rank and
select the best subset for this cycle.

Kept as plain functions, not agent/LLM calls -- ranking here is a straightforward
sort by the Research Agent's own confidence_score. A smarter ranking (weighting in
source credibility, topic-diversity balance, etc.) is a natural place to extend this
later without touching the graph or the persistence layer.
"""

from __future__ import annotations

import difflib

from app.core.schemas import Idea
from app.repositories.vector_store import VectorStore


def idea_text(idea: Idea) -> str:
    return f"{idea.topic}: {idea.angle}"


def dedup_filter(
    ideas: list[Idea], vector_store: VectorStore, threshold: float
) -> tuple[list[Idea], list[str]]:
    """
    Split a pool of ideas into (survivors, filtered_notes) by checking each against
    previously approved ideas in the vector store. Order is preserved among survivors.
    """
    survivors: list[Idea] = []
    filtered_notes: list[str] = []

    for idea in ideas:
        matches = vector_store.find_similar(idea_text(idea), threshold)
        if matches:
            filtered_notes.append(f"'{idea.topic}' (similarity={matches[0][1]:.2f} to '{matches[0][0]}')")
        else:
            survivors.append(idea)

    return survivors, filtered_notes


def near_duplicate_filter(ideas: list[Idea], threshold: float = 0.9) -> tuple[list[Idea], int]:
    """
    Drop ideas that are near-duplicates of an earlier idea in the same list (the
    first occurrence is kept). This catches the case where one Research call
    proposes two near-identical ideas in the same batch -- a different failure mode
    than dedup_filter above, which checks against persisted history, not siblings
    in the same candidate pool.
    """
    kept: list[Idea] = []
    kept_texts: list[str] = []
    filtered_count = 0

    for idea in ideas:
        text = idea_text(idea)
        if any(difflib.SequenceMatcher(None, text.lower(), t.lower()).ratio() >= threshold for t in kept_texts):
            filtered_count += 1
            continue
        kept.append(idea)
        kept_texts.append(text)

    return kept, filtered_count


def select_top(ideas: list[Idea], count: int) -> list[Idea]:
    """Rank by confidence_score and take the top `count`."""
    return sorted(ideas, key=lambda i: i.confidence_score, reverse=True)[:count]


def build_batch_queue(
    candidates: list[Idea],
    vector_store: VectorStore,
    dedup_threshold: float,
    batch_size: int,
    near_dup_threshold: float = 0.9,
) -> tuple[list[Idea], list[Idea], str]:
    """
    The full Research -> Dedup -> Rank -> in-batch-dedup -> take-top pipeline in one
    call, so graph nodes don't need to know the individual filtering steps. Falls
    back to the single best original candidate (with a note explaining why) if
    everything gets filtered out, rather than returning an empty queue.

    Returns a three-tuple: (queue, surplus, note).
    - queue:   top `batch_size` ideas to run through production this cycle.
    - surplus: ideas that survived all filters but didn't make the batch cutoff,
               ordered by confidence (highest first). Saved to the Idea Library
               as status='backlog' so they aren't silently discarded.
    - note:    human-readable summary of filtering decisions for the decision log.
    """
    survivors, history_notes = dedup_filter(candidates, vector_store, dedup_threshold)
    ranked = select_top(survivors, len(survivors))
    deduped, batch_filtered_count = near_duplicate_filter(ranked, near_dup_threshold)
    queue = deduped[:batch_size]
    surplus = deduped[batch_size:]  # survivors that didn't make this week's cut

    if not queue:
        queue = select_top(candidates, 1)
        surplus = []  # nothing clean to backlog when we had to fall back to emergency pick
        note = (
            f"{'; '.join(history_notes) if history_notes else 'no history duplicates'} — "
            "all candidates were filtered; proceeding with the single best original idea anyway"
        )
    else:
        backlog_note = f", {len(surplus)} held for backlog" if surplus else ""
        note = (
            f"generated {len(candidates)} candidates, {len(history_notes)} filtered as repeats of "
            f"approved history, {batch_filtered_count} filtered as in-batch near-duplicates, "
            f"queued {len(queue)} for production{backlog_note}"
        )

    return queue, surplus, note

