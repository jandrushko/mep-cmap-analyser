"""
The NIBS-DAS collated reshape.

The format is what gets uploaded to bignibsdata.com, so the failures that
matter are the silent ones: a column that means something different from what
its name says, a per-stimulus-type value stamped across a whole file, or two
trials merged into one row. None of those raise.
"""

import pytest

from mep_cmap import nibs_das as N
from mep_cmap.events_model import pair_key

SCHEMA = N.load_schema()


def _nibs(**sets):
    """A nibs_bids block with one parameter set per stimulus code."""
    return {
        "modality": "TMS",
        "fields": {"ManufacturersModelName": "BiStim2",
                   "PulseShape": "Monophasic",
                   "TargetRegion": "Left_M1",
                   "RestingMotorThreshold": 56,
                   "TargetingMethod": "manual",
                   "CurrentDirection": "normal"},
        "parameter_sets": {n: {"nibs_type": "TMS", "position": "",
                               "values": v} for n, v in sets.items()},
        "code_sets": {n: n for n in sets},
    }


def _trial(stim="SICI", n=1, ptp=0.75, rms=0.027, **extra):
    row = {"StimType": stim, "Stim_Label": stim, "Segment_Overall": n,
           "Condition": "", "PTP(mV)": ptp, "PreStimRMS": rms}
    row.update(extra)
    return row


# ── the schema asset ─────────────────────────────────────────────────────────

def test_the_schema_loads():
    assert SCHEMA and SCHEMA.get("schema_name") == "nibs_das"


def test_a_missing_schema_returns_none_rather_than_raising(monkeypatch):
    """An optional export must not stop an analysis producing measurements."""
    monkeypatch.setattr(N.os.path, "join", lambda *_a: "/nope/missing.json")
    assert N.load_schema() is None


# ── muscle slugs ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("channel,expected", [
    ("FDI", "fdi"), ("APB", "apb"), ("FDI_left", "fdi_left"),
    ("First Dorsal Interosseous", "first_dorsal_interosseous"),
    ("  EDC  ", "edc"), ("", "unknown"), (None, "unknown"), ("!!!", "unknown"),
])
def test_a_channel_label_becomes_a_column_suffix(channel, expected):
    assert N.muscle_slug(channel) == expected


def test_a_long_muscle_name_is_not_guessed_into_an_abbreviation():
    """
    Renaming 'First Dorsal Interosseous' to fdi would be a guess, and a reader
    comparing two datasets could not tell a real fdi from an invented one.
    """
    assert N.muscle_slug("First Dorsal Interosseous") != "fdi"


# ── stimulus mode ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,expected", [
    ("CSE", "single"), ("cse", "single"), ("SICI", "dual"), ("sici", "dual"),
    ("SICI_80", "dual"), ("ICF120", "dual"), ("LICI", "dual"),
    ("SICF", "dual"), ("LICF", "dual"), ("RMT", "single"),
])
def test_a_stimulus_label_resolves_to_a_mode(label, expected):
    assert N.stim_mode(label, SCHEMA["stim_mode_vocabulary"]) == expected


@pytest.mark.parametrize("label", ["", None, "wibble", "Block3", "42"])
def test_an_unrecognised_label_is_blank_not_guessed(label):
    """
    A guessed 'single' on a paired-pulse trial is a wrong number nothing
    downstream can detect. A blank is at least visible.
    """
    assert N.stim_mode(label, SCHEMA["stim_mode_vocabulary"]) == ""


# ── parameter-set scope ──────────────────────────────────────────────────────

def test_the_intensities_come_from_the_trials_own_parameter_set():
    """
    Both intensities and the ISI are parameter-set scope, so they differ
    BETWEEN stimulus types within one recording. Reading them once per file
    would state a conditioning intensity for trials that never had one.
    """
    nibs = _nibs(
        SICI={"ConditioningStimulusIntensity": 80,
              "TestStimulusIntensity": 120, "InterPulseInterval": 2},
        TEST={"TestStimulusIntensity": 120})
    sici = N.set_values_for_trial(nibs, "SICI")
    test = N.set_values_for_trial(nibs, "TEST")
    assert sici["ConditioningStimulusIntensity"] == 80
    assert "ConditioningStimulusIntensity" not in test


def test_a_split_code_prefers_its_own_set_over_the_bare_code():
    """
    The Conditions tab can split one code across conditions, each half with
    its own set. Checking the bare code first would return the unsplit set for
    a trial that has a more specific one.
    """
    nibs = _nibs(A={"TestStimulusIntensity": 100},
                 A_post={"TestStimulusIntensity": 140})
    nibs["code_sets"] = {"A": "A", pair_key("A", "post"): "A_post"}
    assert N.set_values_for_trial(nibs, "A", "post")["TestStimulusIntensity"] == 140
    assert N.set_values_for_trial(nibs, "A", "")["TestStimulusIntensity"] == 100


def test_an_unassigned_code_gives_nothing_rather_than_someone_elses_set():
    nibs = _nibs(SICI={"TestStimulusIntensity": 120})
    assert N.set_values_for_trial(nibs, "NOT_ASSIGNED") == {}


# ── the collated table ───────────────────────────────────────────────────────

def _two_muscle_study():
    nibs = _nibs(SICI={"ConditioningStimulusIntensity": 80,
                       "TestStimulusIntensity": 120,
                       "InterPulseInterval": 2})
    return [
        {"participant_id": "sub-01", "channel": "FDI", "nibs_bids": nibs,
         "trials": [_trial(n=1, ptp=0.74, rms=0.027),
                    _trial(n=2, ptp=0.57, rms=0.019)]},
        {"participant_id": "sub-01", "channel": "APB", "nibs_bids": nibs,
         "trials": [_trial(n=1, ptp=0.38, rms=0.034),
                    _trial(n=2, ptp=1.20, rms=0.025)]},
    ]


PARTICIPANTS = {"sub-01": {"participant_id": "sub-01", "age": 45,
                           "sex": "Male", "handedness": "Right"}}


def test_the_header_matches_the_published_example_exactly():
    """
    Their mep_example collated file IS the format. Twenty columns, fixed ones
    first, then the measures grouped by muscle.
    """
    cols, _, _ = N.collate(_two_muscle_study(), PARTICIPANTS, SCHEMA)
    assert cols == [
        "participant_id", "age", "sex", "handedness", "ManufacturerModelName",
        "protocol_name", "waveform", "current_direction", "targeting_method",
        "pulse_intensity_rmt", "second_pulse_intensity_rmt",
        "inter_stimulus_interval", "tms_pos_centre", "rmt_intensity",
        "trial_number", "tms_stim_mode",
        "mep_ampl_fdi", "pre_rms_fdi", "mep_ampl_apb", "pre_rms_apb"]


def test_the_channels_of_one_participant_merge_into_one_row_per_trial():
    """
    The format is wide on muscle. Two channels must not produce two rows for
    the same trial, or every trial is counted twice.
    """
    _, rows, _ = N.collate(_two_muscle_study(), PARTICIPANTS, SCHEMA)
    assert len(rows) == 2
    assert rows[0]["mep_ampl_fdi"] == 0.74 and rows[0]["mep_ampl_apb"] == 0.38
    assert rows[1]["mep_ampl_fdi"] == 0.57 and rows[1]["mep_ampl_apb"] == 1.20


def test_trials_are_keyed_on_the_chronological_number_not_the_per_type_one():
    """
    Segment restarts per stimulus type; Segment_Overall does not. Keying on
    the wrong one collides two different trials into a single row and one
    muscle's amplitude silently overwrites another trial's.
    """
    nibs = _nibs(A={"TestStimulusIntensity": 120},
                 B={"TestStimulusIntensity": 120})
    sess = [{"participant_id": "sub-01", "channel": "FDI", "nibs_bids": nibs,
             "trials": [_trial(stim="A", n=1, ptp=1.0, Segment=1),
                        _trial(stim="B", n=2, ptp=2.0, Segment=1)]}]
    _, rows, _ = N.collate(sess, PARTICIPANTS, SCHEMA)
    assert [r["mep_ampl_fdi"] for r in rows] == [1.0, 2.0]


def test_participant_facts_are_carried_onto_every_row():
    _, rows, _ = N.collate(_two_muscle_study(), PARTICIPANTS, SCHEMA)
    assert all(r["age"] == 45 and r["sex"] == "Male" for r in rows)


def test_the_waveform_is_lowercased_for_their_vocabulary():
    """NIBS-BIDS writes Monophasic; their example file writes monophasic."""
    _, rows, _ = N.collate(_two_muscle_study(), PARTICIPANTS, SCHEMA)
    assert rows[0]["waveform"] == "monophasic"


def test_a_parameter_set_value_reaches_the_row():
    _, rows, _ = N.collate(_two_muscle_study(), PARTICIPANTS, SCHEMA)
    assert rows[0]["pulse_intensity_rmt"] == 80
    assert rows[0]["second_pulse_intensity_rmt"] == 120
    assert rows[0]["inter_stimulus_interval"] == 2


def test_two_stimulus_types_do_not_share_one_sets_intensities():
    """The whole reason the resolve is per trial rather than per file."""
    nibs = _nibs(SICI={"ConditioningStimulusIntensity": 80,
                       "TestStimulusIntensity": 120},
                 TEST={"TestStimulusIntensity": 120})
    sess = [{"participant_id": "sub-01", "channel": "FDI", "nibs_bids": nibs,
             "trials": [_trial(stim="SICI", n=1), _trial(stim="TEST", n=2)]}]
    _, rows, _ = N.collate(sess, PARTICIPANTS, SCHEMA)
    assert rows[0]["pulse_intensity_rmt"] == 80
    assert rows[1]["pulse_intensity_rmt"] == ""


# ── extended measures ────────────────────────────────────────────────────────

def test_the_extended_measures_are_absent_by_default():
    """
    NIBS-DAS v1.0-beta defines mep_ampl and pre_rms only. Emitting our own
    names by default would pre-empt a vocabulary decision that belongs to the
    standard's authors.
    """
    cols, _, _ = N.collate(_two_muscle_study(), PARTICIPANTS, SCHEMA)
    assert not [c for c in cols if c.startswith(("csp_", "mep_lat", "mep_auc"))]


def test_the_extended_measures_appear_when_asked_for():
    sess = _two_muscle_study()
    sess[0]["trials"][0]["cSP_Duration(ms)"] = 112.5
    cols, rows, report = N.collate(sess, PARTICIPANTS, SCHEMA, extended=True)
    assert "csp_dur_fdi" in cols and "csp_dur_apb" in cols
    assert rows[0]["csp_dur_fdi"] == 112.5
    assert report["extended_measures"] is True


# ── what the caller has to tell the user ─────────────────────────────────────

def test_unmatched_stimulus_labels_are_reported():
    """Blank columns the analyst never asked for need naming, not hiding."""
    nibs = _nibs(WIBBLE={"TestStimulusIntensity": 120})
    sess = [{"participant_id": "sub-01", "channel": "FDI", "nibs_bids": nibs,
             "trials": [_trial(stim="WIBBLE", n=1)]}]
    _, rows, report = N.collate(sess, PARTICIPANTS, SCHEMA)
    assert rows[0]["tms_stim_mode"] == ""
    assert report["unmatched_stim_labels"] == ["WIBBLE"]


def test_the_slug_records_the_channel_it_came_from():
    """So a reader of the sidecar can tell what 'fdi' actually was."""
    _, _, report = N.collate(_two_muscle_study(), PARTICIPANTS, SCHEMA)
    assert report["muscles"] == {"fdi": ["FDI"], "apb": ["APB"]}
