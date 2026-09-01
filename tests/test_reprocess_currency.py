"""
Which recordings are behind the current trial schema.

Coverage, not correctness. On the study this was built against the split was
exactly even -- 131 files carrying all 55 current columns, 131 missing ten of
them -- and no summary dialogue had said so.
"""

import pytest

from mep_cmap import reprocess_currency as C

SCHEMA = ["File", "StimType", "Segment", "PTP(mV)", "Latency(ms)",
          "Onset_MethodsMedian(ms)", "MEP_Offset(ms)", "Condition"]
OLD = ["File", "StimType", "Segment", "PTP(mV)", "Latency(ms)"]


def test_a_file_with_every_current_column_is_current():
    assert C.classify(SCHEMA, SCHEMA) == C.CURRENT


def test_an_older_file_is_behind():
    assert C.classify(OLD, SCHEMA) == C.BEHIND


def test_a_missing_trial_file_is_absent_not_behind():
    """
    Nothing produced at all is a different situation from output that exists
    and is old: one has never been analysed, the other has results a reader
    might be relying on. Collapsing them hides which.
    """
    assert C.classify(None, SCHEMA) == C.ABSENT


def test_extra_columns_do_not_make_a_file_behind():
    """
    A file may carry columns the current schema dropped. That is worth knowing
    elsewhere, but it does not mean this file needs re-running to gain
    anything.
    """
    assert C.classify(SCHEMA + ["Legacy_Col"], SCHEMA) == C.CURRENT


def test_missing_columns_come_back_in_schema_order():
    """
    Not sorted: a report should read the way the trial file does, and a run of
    adjacent additions is then recognisable as one schema change rather than
    several unrelated ones.
    """
    assert C.missing_columns(OLD, SCHEMA) == [
        "Onset_MethodsMedian(ms)", "MEP_Offset(ms)", "Condition"]


def test_whitespace_in_a_header_is_not_a_missing_column():
    assert C.classify([" File ", "StimType", "Segment", "PTP(mV)",
                       "Latency(ms)", "Onset_MethodsMedian(ms)",
                       "MEP_Offset(ms)", " Condition"], SCHEMA) == C.CURRENT


# ── what the report says ─────────────────────────────────────────────────────

def _records():
    return [
        ("a", C.CURRENT, []),
        ("b", C.BEHIND, ["Condition", "MEP_Offset(ms)"]),
        ("c", C.BEHIND, ["Condition"]),
        ("d", C.ABSENT, []),
    ]


def test_the_summary_counts_each_verdict():
    assert C.summarise(_records()) == {
        C.CURRENT: 1, C.BEHIND: 2, C.ABSENT: 1}


def test_the_commonest_missing_column_leads():
    """
    A column missing from many files is one schema addition; missing from one
    is more likely that file having been analysed differently.
    """
    assert C.most_common_missing(_records())[0] == ("Condition", 2)


def test_only_behind_files_contribute_missing_columns():
    """An absent file is missing everything, which would swamp the count."""
    cols = dict(C.most_common_missing(_records()))
    assert "Latency(ms)" not in cols


def test_both_behind_and_absent_are_worth_rerunning():
    assert sorted(C.keys_needing_rerun(_records())) == ["b", "c", "d"]


def test_a_current_file_is_not_rerun():
    assert "a" not in C.keys_needing_rerun(_records())
