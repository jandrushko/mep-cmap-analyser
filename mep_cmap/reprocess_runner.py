"""
mep_cmap.reprocess_runner
~~~~~~~~~~~~~~~~~~~~~~~~~
Driving a planned batch re-run, one file at a time.

Separate from :mod:`mep_cmap.reprocess`, which decides WHETHER a file can be
re-run, and from app.py, which knows how to load one. This is only the
sequencing: what happens after each file, what stops the run, and what each
file's queue entry becomes.

Pure, like the planner. Loading a recording, running the analysis and
comparing the results are injected, so the order of operations can be tested
against a handful of fakes rather than against a study of 135 recordings that
takes hours and cannot be run twice the same way.

The rule that matters
---------------------
A file is only marked complete when the run succeeded AND its numbers did not
move. Anything else is needs_review. A re-run exists to bring old analyses up
to current outputs, and some of those outputs are corrections -- the cSP/MEP
ratio was inverted -- so numbers changing is expected. Numbers changing
SILENTLY is the failure being designed out: a published figure that quietly
becomes a different figure, with the queue still showing a green tick.
"""

from __future__ import annotations

#: The file re-ran and its trial-level numbers are unchanged.
UNCHANGED = "unchanged"
#: The file re-ran and something moved. Needs a person.
CHANGED = "changed"
#: The analysis raised. Outputs are in an unknown state.
FAILED = "failed"
#: The run was stopped before this file was reached.
NOT_REACHED = "not_reached"
#: The file could not be loaded at all.
LOAD_FAILED = "load_failed"

#: review_flags key explaining why a re-run file needs looking at.
REVIEW_FLAG = "reprocess_changed_values"


def run_batch(items, *, load, analyse, compare, statuses,
              log=None, should_stop=None):
    """Re-run each planned file in turn. Returns a list of result dicts.

    ``items``   -- ``(key, plan)`` pairs, already filtered to runnable plans.
    ``load``    -- ``(key, plan) -> bool``; puts the file and its saved
                   session into the application. False means it could not be
                   loaded, and the file is skipped rather than analysed
                   against whatever the previous one left behind.
    ``analyse`` -- ``() -> (outcome, detail)`` where outcome is "ok" or
                   anything else. Must not return until the results are on
                   disk, or the comparison below reads the previous run's.
    ``compare`` -- ``(key) -> changes`` describing what moved, or a falsey
                   value when nothing did.
    ``statuses``-- the application's status constants, injected so this module
                   stays free of that import.

    Each file is loaded immediately before it is analysed, never in advance.
    Loading resets the per-file state the previous analysis is still being
    saved from, so a runner that pre-loaded would overwrite the session it had
    just written.
    """
    log = log or (lambda _msg: None)
    results = []

    for key, plan in list(items or ()):
        if should_stop is not None and should_stop():
            results.append(_result(key, NOT_REACHED, status=None))
            continue

        if not load(key, plan):
            log(f"could not load {key}; skipped.")
            # Status untouched. Failing to LOAD says nothing about the
            # analysis already on disk, and marking a complete file as needing
            # review because its recording would not open is misleading.
            results.append(_result(key, LOAD_FAILED, status=None))
            continue

        outcome, detail = analyse()
        if outcome != "ok":
            log(f"{key} failed: {detail}")
            results.append(_result(key, FAILED, status=statuses["needs_review"],
                                   detail=detail))
            continue

        changes = compare(key)
        if changes:
            results.append(_result(key, CHANGED,
                                   status=statuses["needs_review"],
                                   changes=changes))
        else:
            results.append(_result(key, UNCHANGED,
                                   status=statuses["complete"]))

    return results


def _result(key, outcome, *, status, detail=None, changes=None):
    return {"key": key, "outcome": outcome, "status": status,
            "detail": detail, "changes": changes}


def review_flag_for(result):
    """The review_flags entry explaining a result, or None when it needs none."""
    if result.get("outcome") == CHANGED:
        return {REVIEW_FLAG: _describe(result.get("changes"))}
    if result.get("outcome") == FAILED:
        return {REVIEW_FLAG: f"the re-run did not finish: "
                             f"{result.get('detail') or 'unknown error'}."}
    return None


def _describe(changes):
    """One line naming what moved, for the queue entry.

    Prefers the columns that actually raised the flag. A stimulus timestamp
    drifting in its last digit is reported but does not flag, so naming it
    here would point the analyst at the one change that does not matter.
    """
    changes = changes or {}
    cols = sorted(changes.get("flagging_columns")
                  or changes.get("columns") or ())
    if not cols:
        return "the re-run produced different values."
    shown = ", ".join(cols[:4]) + ("…" if len(cols) > 4 else "")
    return (f"the re-run changed {len(cols)} column(s): {shown}. The previous "
            f"values are in the change report.")


def summarise(results):
    """Counts by outcome, for the line printed when the run ends."""
    out = {UNCHANGED: 0, CHANGED: 0, FAILED: 0, LOAD_FAILED: 0,
           NOT_REACHED: 0}
    for r in results or ():
        out[r["outcome"]] = out.get(r["outcome"], 0) + 1
    return out
