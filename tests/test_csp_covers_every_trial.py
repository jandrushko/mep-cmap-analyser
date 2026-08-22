"""
A silent period must be measured on every trial of a cSP condition, not only
on the trials someone happened to open in the Data Inspector.

cSP columns used to be populated exclusively from stored segment metadata, so
an unreviewed trial had no cSP at all and the column described the analyst's
click history rather than the condition. That was hidden while stale markers
from a superseded detector lingered in the session and made coverage look
complete; once those started being re-detected, one recording went from 17 of
20 B trials carrying a cSP to 9 -- the nine that had been opened since.
"""

import inspect

from mep_cmap import pipeline


def _trial_row_source():
    src = inspect.getsource(pipeline)
    i = src.index("silent_dur = None")
    return src[i:i + 4000]


def test_the_pipeline_detects_a_silent_period_when_none_is_stored():
    body = _trial_row_source()
    assert "detect_csp_for_trial(" in body, (
        "the pipeline no longer detects a cSP for unreviewed trials, so the "
        "column will only describe trials opened in the Inspector")
    assert "cfg.csp_types" in body, (
        "detection is not restricted to the stimulus types assigned to cSP")


def test_stored_markers_still_win_over_detection():
    """
    A marker the analyst placed or checked is a decision. Detection fills
    gaps; it must never overrule a stored one.
    """
    body = _trial_row_source()
    stored = body.index('if mk in segments_metadata and "silent_start_idx"')
    detect = body.index("detect_csp_for_trial(")
    assert stored < detect, (
        "detection is reached before the stored-metadata branch")
    assert "else:" in body[stored:detect], (
        "detection is not in the else branch of the stored-metadata check")


def test_detection_uses_the_shared_entry_point_and_settings():
    body = _trial_row_source()
    assert "CspSettings.from_source(cfg)" in body, (
        "the pipeline is not building settings from its own config, so a "
        "reviewed and an unreviewed trial could be measured differently")
    assert "second_peak_ms" in body, (
        "the search is not anchored on the trial's own 2nd PTP peak")


def test_a_detection_failure_cannot_fail_the_run():
    body = _trial_row_source()
    seg = body[body.index("if stim_type in cfg.csp_types"):]
    assert "try:" in seg and "except Exception" in seg, (
        "a raised detector leaves the whole analysis dead rather than the "
        "trial blank")


def test_the_duration_feeds_the_summary_statistics():
    """
    silent_durs backs the per-condition mean/SD. A detected cSP that never
    reached it would show per trial and vanish from the summary.
    """
    body = _trial_row_source()
    assert body.count("silent_durs.append(silent_dur)") >= 2, (
        "the detected duration is not appended to silent_durs")
