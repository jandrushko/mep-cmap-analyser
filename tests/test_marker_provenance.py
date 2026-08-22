"""
A stored landmark has to record WHO placed it.

Auto-detected and hand-placed markers used to be stored identically. That
became a correctness problem when the silent-period detector changed from v4
to v5: markers written by v4 sat in the session, the pipeline prefers stored
metadata over detecting, and the results file reported a median cSP of 73 ms
on a recording where the Inspector, re-detecting with v5, showed about 95 ms
for the same trials. Neither number was wrong for its detector; nothing
recorded which detector each came from.

Dropping every stored marker on a version bump would have destroyed genuine
manual edits alongside the stale automatic ones, so provenance is what makes
the invalidation safe.
"""

import inspect

import pytest

from mep_cmap.detection import DETECTION_VERSION
from mep_cmap.inspector import (MARKER_DETECTOR_KEY, MARKER_SOURCE_KEY,
                                DataInspectorWindow as DIW,
                                marker_is_manual, stale_auto_markers)

FIELDS = ("ptp_min_idx", "ptp_max_idx", "onset_idx",
          "silent_start_idx", "silent_end_idx")


def test_a_dragged_marker_is_recorded_as_manual():
    src = inspect.getsource(DIW._update_meta)
    assert f"{MARKER_SOURCE_KEY}" in src or "MARKER_SOURCE_KEY" in src, (
        "_update_meta no longer records that the marker was placed by hand")
    assert '"manual"' in src


def test_manual_markers_survive_a_detector_change():
    meta = {
        "onset_idx": 100, "silent_start_idx": 200, "silent_end_idx": 300,
        MARKER_SOURCE_KEY: {"onset_idx": "manual"},
        MARKER_DETECTOR_KEY: "2026-modular-v4",
    }
    stale = stale_auto_markers(meta, FIELDS, DETECTION_VERSION)
    assert "onset_idx" not in stale, (
        "a hand-placed marker was dropped by a version bump")
    assert "silent_start_idx" in stale and "silent_end_idx" in stale


def test_nothing_is_dropped_when_the_detector_is_unchanged():
    meta = {
        "onset_idx": 100, "silent_start_idx": 200,
        MARKER_SOURCE_KEY: {},
        MARKER_DETECTOR_KEY: DETECTION_VERSION,
    }
    assert stale_auto_markers(meta, FIELDS, DETECTION_VERSION) == []


def test_metadata_predating_provenance_is_treated_as_stale():
    """
    The real sessions in the wild. No marker_source, no detector version, so
    it cannot be known who placed them -- and reusing a possibly-stale
    measurement is worse than re-detecting one the analyst can place again.
    """
    meta = {"onset_idx": 100, "silent_start_idx": 200, "silent_end_idx": 300,
            "_geometry": "d=0.00,e=-100.0/399.8,a=24.8/50.0"}
    stale = stale_auto_markers(meta, FIELDS, DETECTION_VERSION)
    assert set(stale) == {"onset_idx", "silent_start_idx", "silent_end_idx"}


def test_marker_is_manual_is_false_for_unknown_provenance():
    assert marker_is_manual({}, "onset_idx") is False
    assert marker_is_manual({MARKER_SOURCE_KEY: {}}, "onset_idx") is False
    assert marker_is_manual(None, "onset_idx") is False
    assert marker_is_manual(
        {MARKER_SOURCE_KEY: {"onset_idx": "manual"}}, "onset_idx") is True


def test_odd_session_contents_do_not_raise():
    """Session files are edited by hand and by older versions of the tool."""
    for meta in ({MARKER_SOURCE_KEY: "not-a-dict"},
                 {MARKER_SOURCE_KEY: None},
                 {MARKER_SOURCE_KEY: ["onset_idx"]}):
        assert marker_is_manual(meta, "onset_idx") is False
        stale_auto_markers(meta, FIELDS, DETECTION_VERSION)


def test_the_inspector_stamps_and_checks_the_detector_version():
    src = inspect.getsource(DIW._plot)
    assert "stale_auto_markers(" in src, (
        "_plot no longer drops auto markers from an older detector")
    assert "MARKER_DETECTOR_KEY" in src, (
        "_plot no longer stamps the detector version onto the metadata")
    assert src.index("stale_auto_markers(") < src.index("setdefault('ptp_min_idx'"), (
        "stale markers must be dropped BEFORE defaults are seeded, or the "
        "stale value is what gets kept")


def test_csp_duration_is_not_written_as_a_sentinel_string():
    """
    "Not Marked" made cSP_Duration(ms) a text column while its three sibling
    cSP columns stayed numeric, so read.csv typed one of the four differently
    and mean() on it returned NA without complaining.

    Comment lines are skipped: the comment explaining the removal necessarily
    names the string it removed.
    """
    from mep_cmap import pipeline
    offenders = [
        line.strip()
        for line in inspect.getsource(pipeline).splitlines()
        if "Not Marked" in line and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "the cSP duration sentinel is back; it breaks the column's dtype:\n  "
        + "\n  ".join(offenders))
