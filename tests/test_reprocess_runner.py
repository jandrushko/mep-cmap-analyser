"""
Sequencing a batch re-run.

The planner decides which files CAN be re-run; this decides what happens to
each one after it has been. The rule under test throughout is that a file is
marked complete only when it both succeeded and produced the same numbers.
"""

import pytest

from mep_cmap import reprocess_runner as RR

STATUSES = {"complete": "complete", "needs_review": "needs_review"}


def _harness(analyse=None, compare=None, load=None, **kw):
    calls = []

    def _load(key, plan):
        calls.append(("load", key))
        return True if load is None else load(key, plan)

    def _analyse():
        calls.append(("analyse", None))
        return ("ok", None) if analyse is None else analyse()

    def _compare(key):
        calls.append(("compare", key))
        return None if compare is None else compare(key)

    return calls, dict(load=_load, analyse=_analyse, compare=_compare,
                       statuses=STATUSES, **kw)


def _items(*keys):
    return [(k, {"status": "ok"}) for k in keys]


# ── the rule ─────────────────────────────────────────────────────────────────

def test_a_file_whose_numbers_moved_is_not_marked_complete():
    """
    THE rule. Re-running old sessions through current code changes some
    numbers on purpose -- the cSP/MEP ratio was inverted and is now corrected.
    A file that quietly goes green while a published figure changed underneath
    it is the failure this whole exercise exists to prevent.
    """
    _, kw = _harness(compare=lambda k: {"columns": ["cSP_MEP_Ratio(ms/mV)"]})
    [res] = RR.run_batch(_items("a"), **kw)
    assert res["outcome"] == RR.CHANGED
    assert res["status"] == "needs_review"


def test_a_file_whose_numbers_held_is_marked_complete():
    _, kw = _harness()
    [res] = RR.run_batch(_items("a"), **kw)
    assert res["outcome"] == RR.UNCHANGED
    assert res["status"] == "complete"


# ── order of operations ──────────────────────────────────────────────────────

def test_each_file_is_loaded_immediately_before_it_is_analysed():
    """
    Not pre-loaded. Loading resets the per-file state the PREVIOUS analysis is
    still being autosaved from, so loading ahead overwrites the session just
    written -- destroying the manual edits the re-run exists to preserve.
    """
    calls, kw = _harness()
    RR.run_batch(_items("a", "b"), **kw)
    assert calls == [("load", "a"), ("analyse", None), ("compare", "a"),
                     ("load", "b"), ("analyse", None), ("compare", "b")]


def test_a_file_that_will_not_load_is_not_analysed():
    """
    Analysing after a failed load would run against whatever the previous file
    left in place, and produce a complete, plausible analysis of it under this
    file's name.
    """
    calls, kw = _harness(load=lambda k, p: False)
    [res] = RR.run_batch(_items("a"), **kw)
    assert ("analyse", None) not in calls
    assert res["outcome"] == RR.LOAD_FAILED


def test_a_file_that_will_not_load_keeps_its_existing_status():
    """
    Failing to open a recording says nothing about the analysis already on
    disk. Marking it for review would be a claim about results that were never
    touched.
    """
    _, kw = _harness(load=lambda k, p: False)
    [res] = RR.run_batch(_items("a"), **kw)
    assert res["status"] is None


# ── failure ──────────────────────────────────────────────────────────────────

def test_a_crashed_file_is_never_marked_complete():
    _, kw = _harness(analyse=lambda: ("error", "ValueError: boom"))
    [res] = RR.run_batch(_items("a"), **kw)
    assert res["outcome"] == RR.FAILED
    assert res["status"] == "needs_review"
    assert "boom" in res["detail"]


def test_a_crash_does_not_stop_the_remaining_files():
    """One unreadable recording must not cost the other hundred and twenty."""
    state = {"n": 0}

    def _analyse():
        state["n"] += 1
        return ("error", "boom") if state["n"] == 1 else ("ok", None)

    _, kw = _harness(analyse=_analyse)
    results = RR.run_batch(_items("a", "b"), **kw)
    assert [r["outcome"] for r in results] == [RR.FAILED, RR.UNCHANGED]


def test_a_crashed_file_is_not_compared():
    """Its outputs are in an unknown state; a diff against them means nothing."""
    calls, kw = _harness(analyse=lambda: ("error", "boom"))
    RR.run_batch(_items("a"), **kw)
    assert ("compare", "a") not in calls


# ── stopping ─────────────────────────────────────────────────────────────────

def test_stopping_leaves_the_rest_untouched():
    stop = {"now": False}

    def _analyse():
        stop["now"] = True
        return ("ok", None)

    _, kw = _harness(analyse=_analyse, should_stop=lambda: stop["now"])
    results = RR.run_batch(_items("a", "b", "c"), **kw)
    assert [r["outcome"] for r in results] == [
        RR.UNCHANGED, RR.NOT_REACHED, RR.NOT_REACHED]
    assert results[1]["status"] is None


# ── what the queue is told ───────────────────────────────────────────────────

def test_the_review_flag_names_the_columns_that_moved():
    flag = RR.review_flag_for(
        {"outcome": RR.CHANGED, "changes": {"columns": ["AUC", "Latency"]}})
    text = flag[RR.REVIEW_FLAG]
    assert "AUC" in text and "Latency" in text and "2" in text


def test_the_review_flag_names_the_columns_that_actually_flagged():
    """
    A stimulus timestamp drifting in its last digit is reported but does not
    raise the flag, so naming it here would point the analyst at the one
    change that does not matter.
    """
    flag = RR.review_flag_for({
        "outcome": RR.CHANGED,
        "changes": {"columns": ["Stim_Time(s)", "PTP(mV)"],
                    "flagging_columns": ["PTP(mV)"]}})
    text = flag[RR.REVIEW_FLAG]
    assert "PTP(mV)" in text
    assert "Stim_Time(s)" not in text


def test_the_review_flag_explains_a_failure():
    flag = RR.review_flag_for({"outcome": RR.FAILED, "detail": "boom"})
    assert "boom" in flag[RR.REVIEW_FLAG]


def test_an_unchanged_file_needs_no_flag():
    assert RR.review_flag_for({"outcome": RR.UNCHANGED}) is None


def test_the_summary_counts_every_outcome():
    _, kw = _harness()
    results = RR.run_batch(_items("a", "b"), **kw)
    assert RR.summarise(results)[RR.UNCHANGED] == 2
