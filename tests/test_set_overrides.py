"""
Per-recording deviations from a stimulation parameter set.

A recording delivered at 1.3 rMT when the protocol says 1.2 must be
recordable as a DEVIATION. Forcing a second set instead makes the study read
as two protocols deliberately run, rather than one protocol and one drift --
and which of those is true is what a reporting checklist exists to establish.
"""

import json
import os

import pytest

from mep_cmap import stim_params as sp
from mep_cmap.bidsify_state import BidsifyState


def _state(tmp_path):
    st = BidsifyState.load_or_create(str(tmp_path))
    st.param_sets = [
        sp.StimParamSet("test", nibs_type="TMS",
                        values={"StimulationIntensity": "50",
                                "IntensityScaling": "1.2",
                                "IntensityReference": "rMT"}),
        sp.StimParamSet("cond", nibs_type="TMS",
                        values={"IntensityScaling": "0.8"}),
    ]
    path = os.path.join(str(tmp_path), "sub-015.smr")
    rec = st.record_for(path)
    rec.marker_names = ["A", "B"]
    rec.code_sets = {"A": "test", "B": "cond"}
    return st, path


# ── the override itself ──────────────────────────────────────────────────────

def test_a_recording_can_differ_without_inventing_a_set(tmp_path):
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {"test": {"IntensityScaling": "1.3"}}

    sets = {s.name: s for s in st.effective_sets(path)}
    assert sets["test"].values["IntensityScaling"] == "1.3"
    # The other fields of that set are untouched.
    assert sets["test"].values["StimulationIntensity"] == "50"
    # And the other set is untouched.
    assert sets["cond"].values["IntensityScaling"] == "0.8"
    # The session protocol itself is NOT rewritten.
    assert st.param_sets[0].values["IntensityScaling"] == "1.2"


def test_another_recording_is_unaffected(tmp_path):
    """The deviation belongs to one recording, not to the study."""
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {"test": {"IntensityScaling": "1.3"}}

    other = os.path.join(str(tmp_path), "sub-016.smr")
    rec = st.record_for(other)
    rec.code_sets = {"A": "test"}
    sets = {s.name: s for s in st.effective_sets(other)}
    assert sets["test"].values["IntensityScaling"] == "1.2"


def test_the_set_name_and_type_are_not_overridable(tmp_path):
    """The name is the protocol ROLE, unchanged by the intensity used."""
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {
        "test": {"IntensityScaling": "1.3", "name": "something",
                 "nibs_type": "tES"}}
    s = next(s for s in st.effective_sets(path) if s.name == "test")
    assert s.name == "test" and s.nibs_type == "TMS"


def test_a_blank_override_clears_the_field_rather_than_writing_empty(tmp_path):
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {"test": {"IntensityReference": ""}}
    s = next(s for s in st.effective_sets(path) if s.name == "test")
    assert "IntensityReference" not in s.values


# ── what differs, reported as differing ──────────────────────────────────────

def test_deviations_name_the_protocol_value_and_the_delivered_one(tmp_path):
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {"test": {"IntensityScaling": "1.3"}}
    dev = st.deviations(path)
    assert dev == {"test": {"IntensityScaling": ("1.2", "1.3")}}


def test_an_override_that_restates_the_protocol_is_not_a_deviation(tmp_path):
    """It is not one, and reporting it would train the analyst to ignore the
    list that also carries the real ones."""
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {"test": {"IntensityScaling": "1.2"}}
    assert st.deviations(path) == {}


def test_no_overrides_means_no_deviations(tmp_path):
    st, path = _state(tmp_path)
    assert st.deviations(path) == {}


# ── the record the reporting assessment believes ─────────────────────────────

def test_the_analysis_record_carries_the_delivered_value(tmp_path):
    """
    Reading the session sets here would put the protocol's number in the
    record while the published sidecar carried the real one, and the
    group-level assessment believes the record.
    """
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {"test": {"IntensityScaling": "1.3"}}
    nr = st.nibs_record(path)
    assert nr["parameter_sets"]["test"]["values"]["IntensityScaling"] == "1.3"


# ── persistence ──────────────────────────────────────────────────────────────

def test_an_override_survives_a_reload(tmp_path):
    st, path = _state(tmp_path)
    st.record_for(path).set_overrides = {"test": {"IntensityScaling": "1.3"}}
    st.save()

    back = BidsifyState.load_or_create(str(tmp_path))
    rec = back.record_for(path, create=False)
    assert rec.set_overrides == {"test": {"IntensityScaling": "1.3"}}


def test_state_written_before_overrides_existed_reads_as_no_deviation(tmp_path):
    st, path = _state(tmp_path)
    st.save()
    raw = json.loads((tmp_path / "bidsify_state.json").read_text(encoding="utf-8"))
    for f in raw["files"]:
        f.pop("set_overrides", None)
    (tmp_path / "bidsify_state.json").write_text(json.dumps(raw),
                                                 encoding="utf-8")
    back = BidsifyState.load_or_create(str(tmp_path))
    assert back.record_for(path, create=False).set_overrides == {}
    assert back.deviations(path) == {}


def test_a_malformed_override_is_ignored_not_crashed_on(tmp_path):
    """This runs at startup; an unreadable entry must not stop the tool
    opening on a study the analyst still needs."""
    st, path = _state(tmp_path)
    st.save()
    raw = json.loads((tmp_path / "bidsify_state.json").read_text(encoding="utf-8"))
    raw["files"][0]["set_overrides"] = {"test": "not-a-dict"}
    (tmp_path / "bidsify_state.json").write_text(json.dumps(raw),
                                                 encoding="utf-8")
    back = BidsifyState.load_or_create(str(tmp_path))
    assert back.record_for(path, create=False).set_overrides == {}
