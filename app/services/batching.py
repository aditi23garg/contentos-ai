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
    ideas: list[Idea],
    vector_store: VectorStore,
    threshold: float,
    same_topic_threshold: float | None = None,
) -> tuple[list[Idea], list[Idea], list[str]]:
    """
    Split a pool of ideas into (survivors, filtered_ideas, filtered_notes) by
    checking each against previously approved ideas in the vector store. Order is
    preserved among survivors. filtered_ideas (not just notes) is returned so the
    caller can tell which *specific* ideas were dropped -- needed to archive
    backlog-sourced ideas that are now stale, see build_batch_queue.

    `same_topic_threshold`, when given, is passed through to find_similar for its
    two-tier check (see config.DEDUP_SAME_TOPIC_THRESHOLD) -- a much lower bar
    applied only when a stored idea's topic label exactly matches this idea's.

    Also sets idea.dedup_note on every idea checked (mutates in place) so the
    per-idea result of this check travels with the Idea all the way to
    persist_batch_node -> IdeaRecord.dedup_note, not just into the batch-level
    agent_decisions note. Previously that column existed but nothing ever wrote to
    it for individual ideas.
    """
    survivors: list[Idea] = []
    filtered_ideas: list[Idea] = []
    filtered_notes: list[str] = []

    for idea in ideas:
        matches = vector_store.find_similar(
            idea_text(idea), threshold, topic=idea.topic, same_topic_threshold=same_topic_threshold
        )
        if matches:
            note = f"'{idea.topic}' (similarity={matches[0][1]:.2f} to '{matches[0][0]}')"
            idea.dedup_note = f"filtered as repeat: {note}"
            filtered_ideas.append(idea)
            filtered_notes.append(note)
        else:
            idea.dedup_note = f"no match >= {threshold} threshold found in approved history"
            survivors.append(idea)

    return survivors, filtered_ideas, filtered_notes


def near_duplicate_filter(ideas: list[Idea], threshold: float = 0.9) -> tuple[list[Idea], list[Idea]]:
    """
    Drop ideas that are near-duplicates of an earlier idea in the same list (the
    first occurrence is kept). This catches the case where one Research call
    proposes two near-identical ideas in the same batch -- a different failure mode
    than dedup_filter above, which checks against persisted history, not siblings
    in the same candidate pool. Returns (kept, filtered_ideas).
    """
    kept: list[Idea] = []
    kept_texts: list[str] = []
    filtered_ideas: list[Idea] = []

    for idea in ideas:
        text = idea_text(idea)
        if any(difflib.SequenceMatcher(None, text.lower(), t.lower()).ratio() >= threshold for t in kept_texts):
            filtered_ideas.append(idea)
            continue
        kept.append(idea)
        kept_texts.append(text)

    return kept, filtered_ideas


def select_top(ideas: list[Idea], count: int) -> list[Idea]:
    """Rank by confidence_score and take the top `count`."""
    return sorted(ideas, key=lambda i: i.confidence_score, reverse=True)[:count]


def build_batch_queue(
    candidates: list[Idea],
    vector_store: VectorStore,
    dedup_threshold: float,
    batch_size: int,
    near_dup_threshold: float = 0.9,
    same_topic_threshold: float | None = None,
) -> tuple[list[Idea], list[Idea], list[Idea], str]:
    """
    The full Research -> Dedup -> Rank -> in-batch-dedup -> take-top pipeline in one
    call, so graph nodes don't need to know the individual filtering steps. Falls
    back to the single best original candidate (with a note explaining why) if
    everything gets filtered out, rather than returning an empty queue.

    `candidates` may be a mix of freshly-researched Ideas and ideas read back from
    the backlog (Idea.source_backlog_id set) -- see repository.get_backlog_ideas.

    `same_topic_threshold`, when given, is passed through to dedup_filter for its
    two-tier check (see config.DEDUP_SAME_TOPIC_THRESHOLD).

    Returns a four-tuple: (queue, surplus, stale, note).
    - queue:   top `batch_size` ideas to run through production this cycle.
    - surplus: ideas that survived all filters but didn't make the batch cutoff,
               ordered by confidence (highest first). Freshly-researched ones get
               saved to the Idea Library as status='backlog'; backlog-sourced ones
               that are still surplus need no DB change, they're already 'backlog'.
    - stale:   backlog-sourced ideas that got filtered out this cycle by either
               dedup check (a similar idea is now in approved history) or the
               near-duplicate check (duplicates another candidate in this pool).
               These should be archived rather than left as 'backlog' forever --
               otherwise they'd get pulled and re-filtered every cycle indefinitely.
    - note:    human-readable summary of filtering decisions for the decision log.
    """
    survivors, history_filtered, history_notes = dedup_filter(
        candidates, vector_store, dedup_threshold, same_topic_threshold=same_topic_threshold
    )
    ranked = select_top(survivors, len(survivors))
    deduped, batch_filtered = near_duplicate_filter(ranked, near_dup_threshold)
    queue = deduped[:batch_size]
    surplus = deduped[batch_size:]  # survivors that didn't make this week's cut

    stale = [
        idea for idea in (*history_filtered, *batch_filtered)
        if idea.source_backlog_id is not None
    ]
    batch_filtered_count = len(batch_filtered)

    if not queue:
        queue = select_top(candidates, 1)
        surplus = []  # nothing clean to backlog when we had to fall back to emergency pick
        stale = [idea for idea in stale if idea not in queue]
        note = (
            f"{'; '.join(history_notes) if history_notes else 'no history duplicates'} — "
            "all candidates were filtered; proceeding with the single best original idea anyway"
        )
        # Overwrite whatever dedup_filter set on this specific idea -- it may say
        # "no match" (near-dup-filtered, not history-filtered) or "filtered as
        # repeat" (history match, overridden anyway). Either way, make the override
        # itself explicit on the persisted row rather than leaving an ambiguous note.
        queue[0].dedup_note = f"EMERGENCY OVERRIDE (dedup exhausted the batch): {queue[0].dedup_note}"
    else:
        backlog_note = f", {len(surplus)} held for backlog" if surplus else ""
        stale_note = f", {len(stale)} stale backlog item(s) archived" if stale else ""
        note = (
            f"generated {len(candidates)} candidates, {len(history_notes)} filtered as repeats of "
            f"approved history, {batch_filtered_count} filtered as in-batch near-duplicates, "
            f"queued {len(queue)} for production{backlog_note}{stale_note}"
        )

    return queue, surplus, stale, note
