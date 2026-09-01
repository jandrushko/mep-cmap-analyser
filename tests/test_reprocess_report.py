"""
Diffing a re-run against the trial file it replaced.

The re-run is meant to change some numbers -- the cSP/MEP ratio was inverted
before v1.4 -- so "something changed" cannot be treated as an error. What must
never happen is a number changing without anyone being told.
"""

import pytest

from mep_cmap import reprocess_report as RP


def _row(seg, **kw):
    base = {"File": "x.txt", "StimType": "A", "Segment": seg,
            "PTP(mV)": "1.2000", "cSP_MEP_Ratio(ms/mV)": "0.5000"}
    base.update(kw)
    return base


# ── the corrected ratio, which is why this exists ────────────────────────────

def test_a_corrected_value_is_reported_with_both_numbers():
    old = [_row(1, **{"cSP_MEP_Ratio(ms/mV)": "2.0000"})]
    new = [_row(1, **{"cSP_MEP_Ratio(ms/mV)": "0.5000"})]
    f = RP.compare_trials(old, new)
    assert f["columns"] == ["cSP_MEP_Ratio(ms/mV)"]
    ex = f["changes"][0]["examples"][0]
    assert ex["was"] == "2.0000" and ex["now"] == "0.5000"
    assert RP.needs_review(f)


def test_an_unchanged_file_needs_no_review():
    rows = [_row(1), _row(2)]
    f = RP.compare_trials(rows, [dict(r) for r in rows])
    assert f["columns"] == []
    assert not RP.needs_review(f)


# ── additions are the point, not a change ────────────────────────────────────

def test_a_new_column_alone_does_not_flag_the_file():
    """
    A re-run exists to ADD things: the reporting record, the current trial
    schema. Treating those as changes would flag all 123 files and make the
    flag mean nothing.
    """
    old = [_row(1)]
    new = [_row(1, TMS_RAT_Complete="yes")]
    f = RP.compare_trials(old, new)
    assert f["added_columns"] == ["TMS_RAT_Complete"]
    assert f["columns"] == []
    assert not RP.needs_review(f)


def test_a_column_that_vanished_does_flag_the_file():
    """Not expected, and worth a person: something produced before no longer is."""
    old = [_row(1, Legacy_Col="7")]
    new = [_row(1)]
    f = RP.compare_trials(old, new)
    assert f["removed_columns"] == ["Legacy_Col"]
    assert RP.needs_review(f)


# ── trial counts ─────────────────────────────────────────────────────────────

def test_a_lost_trial_is_reported():
    f = RP.compare_trials([_row(1), _row(2)], [_row(1)])
    assert len(f["rows_only_old"]) == 1
    assert RP.needs_review(f)


def test_a_gained_trial_is_reported():
    f = RP.compare_trials([_row(1)], [_row(1), _row(2)])
    assert len(f["rows_only_new"]) == 1
    assert RP.needs_review(f)


# ── how values are compared ──────────────────────────────────────────────────

def test_blank_and_nan_are_the_same_absence():
    """
    Different versions of the writer spell an empty cell differently. A file
    whose blanks merely changed spelling has not changed.
    """
    f = RP.compare_trials([_row(1, AUC="")], [_row(1, AUC="nan")])
    assert f["columns"] == []


def test_surrounding_whitespace_is_not_a_change():
    f = RP.compare_trials([_row(1, AUC="1.5")], [_row(1, AUC=" 1.5 ")])
    assert f["columns"] == []


def test_a_changed_precision_IS_a_change():
    """
    The writer rounds explicitly, so identical inputs give identical text.
    Text that differs means the number or the rounding did -- both worth
    knowing, and neither should be hidden by a tolerance nobody chose.
    """
    f = RP.compare_trials([_row(1, AUC="1.5000")], [_row(1, AUC="1.5001")])
    assert f["columns"] == ["AUC"]


# ── the join ─────────────────────────────────────────────────────────────────

def test_conditions_are_part_of_the_identity():
    """
    Applying conditions splits one stimulus type into rows sharing a StimType:
    A-pre and A-post both report StimType=A. Joining without Condition pairs
    them arbitrarily and reads every row as changed.
    """
    old = [_row(1, Condition="pre", **{"PTP(mV)": "1.0"}),
           _row(1, Condition="post", **{"PTP(mV)": "2.0"})]
    new = [dict(r) for r in old]
    f = RP.compare_trials(old, new)
    assert "Condition" in f["identity"]
    assert f["columns"] == []


def test_a_file_with_nothing_to_join_on_is_refused_not_guessed():
    """
    Comparing by row ORDER is only correct while nothing was added or removed,
    which is the case worth detecting. So it refuses instead.
    """
    f = RP.compare_trials([{"PTP(mV)": "1"}], [{"PTP(mV)": "2"}])
    assert f.get("unjoinable")
    assert RP.needs_review(f)


def test_nothing_at_all_needs_review():
    assert RP.needs_review(None)


# ── the timing exemption ─────────────────────────────────────────────────────

def test_a_last_digit_timestamp_drift_is_reported_but_does_not_flag():
    """
    Seen on real data: Stim_Time 205.1477 against 205.1478 on one trial in
    twenty, which is arithmetic noise in the samples-to-seconds conversion. A
    file flagged for that alone teaches the analyst that the flag means "look
    at everything", which is the same as meaning nothing.

    Reported either way, so nothing is hidden.
    """
    old = [_row(1, **{"Stim_Time(s)": "205.1477"})]
    new = [_row(1, **{"Stim_Time(s)": "205.1478"})]
    f = RP.compare_trials(old, new)
    assert f["columns"] == ["Stim_Time(s)"]
    assert f["flagging_columns"] == []
    assert not RP.needs_review(f)
    assert RP.report_rows("x", f)[0]["Column"] == "Stim_Time(s)"


def test_a_measured_value_moving_still_flags_even_beside_a_timestamp():
    """
    The exemption is per column, not per file. A timestamp drifting must never
    excuse an amplitude that moved in the same file.
    """
    old = [_row(1, **{"Stim_Time(s)": "205.1477", "PTP(mV)": "0.19"})]
    new = [_row(1, **{"Stim_Time(s)": "205.1478", "PTP(mV)": "0.29"})]
    f = RP.compare_trials(old, new)
    assert set(f["columns"]) == {"Stim_Time(s)", "PTP(mV)"}
    assert f["flagging_columns"] == ["PTP(mV)"]
    assert RP.needs_review(f)


def test_the_exemption_is_a_named_list_not_a_tolerance():
    """
    A tolerance would apply everywhere and would mean choosing a size of
    change allowed to pass unreported on amplitudes and latencies too. Only
    these named columns are exempt, and a large jump in one is still exempt --
    which is the honest consequence of naming columns rather than sizes.
    """
    assert "PTP(mV)" not in RP.TIMING_COLUMNS
    assert "Latency(ms)" not in RP.TIMING_COLUMNS
    assert "Stim_Time(s)" in RP.TIMING_COLUMNS


# ── the study-level report ───────────────────────────────────────────────────

def test_report_rows_name_the_file_column_and_both_values():
    old = [_row(1, **{"cSP_MEP_Ratio(ms/mV)": "2.0000"})]
    new = [_row(1, **{"cSP_MEP_Ratio(ms/mV)": "0.5000"})]
    rows = RP.report_rows("sub-01_trials.csv", RP.compare_trials(old, new))
    assert rows[0]["File"] == "sub-01_trials.csv"
    assert rows[0]["Column"] == "cSP_MEP_Ratio(ms/mV)"
    assert rows[0]["ExampleWas"] == "2.0000"
    assert rows[0]["ExampleNow"] == "0.5000"
    assert rows[0]["TrialsAffected"] == 1
