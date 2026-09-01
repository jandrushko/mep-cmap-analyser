"""Which latency gets reported, and what says so.

The reported latency comes from one of three places and the file must say
which, or a study that mixes them cannot be pooled -- the failure tms_rat.py
calls fatal.
"""

import pytest

from mep_cmap import column_groups as cg
from mep_cmap.detection.defaults import ONSET_DEFAULTS
from mep_cmap.pipeline import (
    LAT_COLS,
    LATENCY_SOURCES,
    PipelineConfig,
    resolve_reported_latency,
)


# ── precedence ───────────────────────────────────────────────────────────────

def test_a_manual_marker_beats_the_consensus():
    """The analyst looked at the trace. No automatic median outranks that."""
    val, src = resolve_reported_latency(21.5, 18.0, True)
    assert (val, src) == (21.5, "manual")


def test_the_consensus_beats_a_single_detector():
    """It does not depend on which detector happened to be selected."""
    val, src = resolve_reported_latency(21.5, 18.0, False)
    assert (val, src) == (18.0, "consensus")


def test_the_single_detector_is_the_fallback_not_a_blank():
    """
    Agreement is a setting. A study analysed before it was on by default must
    still report a latency rather than an empty column.
    """
    val, src = resolve_reported_latency(21.5, None, False)
    assert (val, src) == (21.5, "detector")


def test_nothing_detected_is_reported_as_none():
    val, src = resolve_reported_latency(None, None, False)
    assert (val, src) == (None, "none")


def test_a_marker_flag_without_a_value_falls_through():
    """
    manual_marked says what single_lat IS, not that it exists. A marker the
    caller rejected (at or before the stimulus) must not claim the trial.
    """
    val, src = resolve_reported_latency(None, 18.0, True)
    assert (val, src) == (18.0, "consensus")


def test_consensus_is_used_even_when_the_detector_found_nothing():
    """Coverage is the reason for the change: 1036 trials against 438."""
    val, src = resolve_reported_latency(None, 18.0, False)
    assert (val, src) == (18.0, "consensus")


@pytest.mark.parametrize("args,expected", [
    ((21.5, 18.0, True),  "manual"),
    ((21.5, 18.0, False), "consensus"),
    ((21.5, None, False), "detector"),
    ((None, None, False), "none"),
])
def test_every_declared_source_is_reachable(args, expected):
    """A value in LATENCY_SOURCES that nothing produces is a lie in the docs."""
    assert resolve_reported_latency(*args)[1] == expected
    assert expected in LATENCY_SOURCES


# ── the column ───────────────────────────────────────────────────────────────

def test_the_source_column_is_written():
    assert "Latency_Source" in LAT_COLS


def test_the_source_travels_with_the_latency():
    """
    Selecting the onset group and losing the definition of what was selected
    would produce a file whose latencies cannot be pooled with anyone else's.
    """
    onset = dict((k, cols) for k, _label, cols in cg.GROUPS)["onset"]
    assert "Latency(ms)" in onset
    assert "Latency_Source" in onset


def test_the_source_column_is_assigned_to_a_group():
    """Unassigned columns vanish from _trials_selected.csv."""
    assert not cg.unassigned(list(LAT_COLS))


# ── defaults ─────────────────────────────────────────────────────────────────

def test_agreement_runs_by_default():
    """
    The consensus is the reported latency wherever it exists. Left off, the
    reported latency would depend on a setting most analysts never open.
    """
    assert ONSET_DEFAULTS["onset_agreement"] is True


def test_outlier_review_is_off_by_default():
    """
    Review halts an analysis to put flagged trials to the analyst, which an
    unattended batch cannot answer.
    """
    assert PipelineConfig().enable_outlier_review is False


# ── NIBS-DAS export ──────────────────────────────────────────────────────────

def _das_schema():
    """The schema as the exporter itself loads it.

    Through nibs_das.load_schema rather than by rebuilding the path here: the
    schema directory is not an importable package, and a test that finds the
    asset by its own route can pass while the exporter finds nothing.
    """
    from mep_cmap import nibs_das
    schema = nibs_das.load_schema()
    assert schema is not None, "NIBS-DAS schema asset did not load"
    return schema


def _extended():
    return {m["name"]: m
            for m in _das_schema()["measures"]["extended"]}


def test_the_export_carries_the_latency_method():
    """
    A collated latency column of mixed provenance cannot be pooled. The
    export has to say which method produced each value.
    """
    m = _extended()["mep_lat_source"]
    assert m["from"] == "Latency_Source"


def test_the_method_field_is_not_given_units():
    """It is categorical. A unit on it would invite arithmetic."""
    assert _extended()["mep_lat_source"]["units"] is None


def test_the_method_sits_beside_the_latency_it_qualifies():
    """
    Column order is the header order, so adjacency is what stops a reader
    taking mep_lat without noticing mep_lat_source exists.
    """
    names = [m["name"] for m in _das_schema()["measures"]["extended"]]
    assert names.index("mep_lat_source") == names.index("mep_lat") + 1


def test_the_method_field_is_still_flagged_as_unagreed():
    """
    It is this tool's proposal, not NIBS-DAS vocabulary, and the schema must
    keep saying so until the authors adopt it.
    """
    status = _das_schema()["measures"]["_extended_status"]
    assert status.startswith("NOT AGREED")
    assert "mep_lat_source" in status


def test_the_latency_method_reaches_the_export_columns():
    """
    Schema-only additions are worth nothing if the reshape drops them, and a
    categorical measure is the first of its kind here.
    """
    from mep_cmap import nibs_das
    cols = nibs_das.columns(["fdi"], _das_schema(), extended=True)
    assert "mep_lat_source_fdi" in cols
    assert "mep_lat_fdi" in cols


def test_the_latency_method_is_off_when_extended_is():
    """A default export must not pre-empt the vocabulary decision."""
    from mep_cmap import nibs_das
    cols = nibs_das.columns(["fdi"], _das_schema(), extended=False)
    assert "mep_lat_source_fdi" not in cols
