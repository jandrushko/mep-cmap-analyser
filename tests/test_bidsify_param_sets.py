"""
Parameter sets must survive being saved, and old state must still load.

State written by 1.4.0 has no parameter sets and no code assignment. It must
open without error, and it must not claim to describe stimulation it never
recorded.
"""

import json
import os

from mep_cmap import stim_params as sp
from mep_cmap.bids_schema import load_schema
from mep_cmap.bidsify_state import (STATUS_INCOMPLETE, STATUS_READY,
                                    BidsifyState, FileBidsRecord)


def _v1_state(root, **file_kw):
    """A state file exactly as version 1 wrote it: no param_sets, no code_sets."""
    payload = {
        "schema_state_version": 1,
        "modality": "TMS",
        "defaults": {},
        "container": "EDF",
        "powerline_hz": 50,
        "marker_name": "A",
        "rawdata_root": "",
        "files": [dict({"rel_path": "sub-001/a.smr", "reviewed": True,
                        "marker_names": ["A", "G"],
                        "stim_channel": "DigMark"}, **file_kw)],
    }
    with open(os.path.join(root, "bidsify_state.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh)


def _v1_path(tmp_path):
    """The v1 record's path, absolute.

    key_for() resolves against the dataset root via abspath, so a bare relative
    path is keyed against the working directory and never matches.
    """
    return os.path.join(str(tmp_path), "sub-001", "a.smr")


# ── migration ────────────────────────────────────────────────────────────────

def test_state_written_before_parameter_sets_still_loads(tmp_path):
    _v1_state(str(tmp_path))
    st = BidsifyState.load_or_create(str(tmp_path))
    assert st.param_sets == []
    rec = st.record_for(_v1_path(tmp_path), create=False)
    assert rec is not None
    assert rec.marker_names == ["A", "G"]
    assert rec.code_sets == {}


def test_an_old_reviewed_file_is_incomplete_not_ready(tmp_path):
    """It was Ready under the old rules. Its stimulation is undescribed under
    the new ones, and saying so is the honest answer."""
    _v1_state(str(tmp_path))
    st = BidsifyState.load_or_create(str(tmp_path))
    assert st.unassigned_codes(_v1_path(tmp_path)) == ["A", "G"]
    assert st.status(_v1_path(tmp_path), load_schema()) == STATUS_INCOMPLETE


def test_an_old_converted_file_stays_converted(tmp_path):
    """Already written to disk. Re-flagging it would invite a reconversion that
    changes nothing about the recording."""
    _v1_state(str(tmp_path), converted=True)
    st = BidsifyState.load_or_create(str(tmp_path))
    assert st.status(_v1_path(tmp_path), load_schema()) == "converted"


def test_a_file_with_no_ticked_codes_is_unaffected(tmp_path):
    _v1_state(str(tmp_path), marker_names=[])
    st = BidsifyState.load_or_create(str(tmp_path))
    assert st.unassigned_codes(_v1_path(tmp_path)) == []
    assert st.status(_v1_path(tmp_path), load_schema()) == STATUS_READY


# ── round trip ───────────────────────────────────────────────────────────────

def test_parameter_sets_survive_save_and_reload(tmp_path):
    st = BidsifyState.load_or_create(str(tmp_path))
    st.param_sets = [
        sp.StimParamSet("Mmax", nibs_type="PNS",
                        values={"StimulationIntensity": 45},
                        position="tibial_nerve"),
        sp.StimParamSet("MEP_120", nibs_type="TMS",
                        values={"StimulationIntensity": 60}),
    ]
    rec = st.record_for(os.path.join(str(tmp_path), "sub-001", "a.smr"))
    rec.reviewed = True
    rec.marker_names = ["A", "G"]
    rec.code_sets = {"A": "Mmax", "G": "MEP_120"}
    st.save()

    back = BidsifyState.load_or_create(str(tmp_path))
    assert [s.name for s in back.param_sets] == ["Mmax", "MEP_120"]
    assert back.param_sets[0].nibs_type == "PNS"
    assert back.param_sets[0].position == "tibial_nerve"
    got = back.record_for(os.path.join(str(tmp_path), "sub-001", "a.smr"),
                          create=False)
    assert got.code_sets == {"A": "Mmax", "G": "MEP_120"}


def test_a_fully_assigned_file_is_ready(tmp_path):
    st = BidsifyState.load_or_create(str(tmp_path))
    st.param_sets = [sp.StimParamSet("Mmax", nibs_type="PNS")]
    rec = st.record_for(os.path.join(str(tmp_path), "a.smr"))
    rec.reviewed = True
    rec.marker_names = ["A"]
    rec.code_sets = {"A": "Mmax"}
    assert st.unassigned_codes(os.path.join(str(tmp_path), "a.smr")) == []
    assert st.status(os.path.join(str(tmp_path), "a.smr"),
                     load_schema()) == STATUS_READY


def test_an_untouched_code_is_the_one_reported(tmp_path):
    st = BidsifyState.load_or_create(str(tmp_path))
    st.param_sets = [sp.StimParamSet("Mmax")]
    rec = st.record_for(os.path.join(str(tmp_path), "a.smr"))
    rec.reviewed = True
    rec.marker_names = ["A", "C", "G"]
    rec.code_sets = {"A": "Mmax", "G": "Mmax"}
    assert st.unassigned_codes(os.path.join(str(tmp_path), "a.smr")) == ["C"]


def test_the_state_version_was_bumped(tmp_path):
    """The record grew a field. A reader checking the version must be able to
    tell v1 from v2."""
    st = BidsifyState.load_or_create(str(tmp_path))
    st.save()
    with open(os.path.join(str(tmp_path), "bidsify_state.json"),
              encoding="utf-8") as fh:
        assert json.load(fh)["schema_state_version"] >= 2


def test_a_record_round_trips_through_its_own_dict():
    rec = FileBidsRecord(rel_path="a.smr", marker_names=["A"],
                         code_sets={"A": "Mmax"})
    assert FileBidsRecord.from_dict(rec.to_dict()).code_sets == {"A": "Mmax"}


# ── the description handed to the analysis record ────────────────────────────

def _two_set_state(tmp_path):
    st = BidsifyState.load_or_create(str(tmp_path))
    st.defaults = {"CoilType": "figure-of-8"}
    st.param_sets = [
        sp.StimParamSet("Mmax", nibs_type="PNS",
                        values={"StimulationIntensity": 45},
                        position="tibial_nerve"),
        sp.StimParamSet("MEP_120", nibs_type="TMS",
                        values={"StimulationIntensity": 60}),
        sp.StimParamSet("other_file", nibs_type="TMS",
                        values={"StimulationIntensity": 99}),
    ]
    path = os.path.join(str(tmp_path), "a.smr")
    rec = st.record_for(path)
    rec.reviewed = True
    rec.marker_names = ["A", "G"]
    rec.code_sets = {"A": "Mmax", "G": "MEP_120"}
    return st, path


def test_the_description_carries_the_parameter_sets_not_just_the_fields(tmp_path):
    """
    v6.3 puts most stimulation parameters on a named set. A flat mapping has
    one slot for a fact this recording has two of, and which of the two
    survived would depend on iteration order.
    """
    st, path = _two_set_state(tmp_path)
    nr = st.nibs_record(path)
    assert nr["modality"] == "TMS"
    assert nr["fields"]["CoilType"] == "figure-of-8"
    assert nr["code_sets"] == {"A": "Mmax", "G": "MEP_120"}
    assert nr["parameter_sets"]["Mmax"]["values"]["StimulationIntensity"] == 45
    assert nr["parameter_sets"]["MEP_120"]["values"]["StimulationIntensity"] == 60
    assert nr["parameter_sets"]["Mmax"]["nibs_type"] == "PNS"
    assert nr["parameter_sets"]["Mmax"]["position"] == "tibial_nerve"


def test_a_set_no_code_here_uses_is_left_out(tmp_path):
    """It belongs to another recording. Carrying it would describe
    stimulation that was never delivered in this one."""
    st, path = _two_set_state(tmp_path)
    assert "other_file" not in st.nibs_record(path)["parameter_sets"]


def test_the_description_of_an_unreviewed_file_is_empty_not_invented(tmp_path):
    st, _ = _two_set_state(tmp_path)
    nr = st.nibs_record(os.path.join(str(tmp_path), "never_touched.smr"))
    assert nr["code_sets"] == {}
    assert nr["parameter_sets"] == {}


def test_the_description_is_json_serialisable(tmp_path):
    """It is written into the reporting sidecar verbatim."""
    st, path = _two_set_state(tmp_path)
    nr = st.nibs_record(path)
    assert json.loads(json.dumps(nr)) == nr
