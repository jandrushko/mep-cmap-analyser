"""
Assessing a STUDY against TMS-RAT.

The assessment is level 2 because the checklist is: it rates one protocol
within one study, and a quantitative item is Full only with a dispersion that
a single recording cannot supply.
"""

import json
import os

import pytest

from mep_cmap import tms_rat as T
from mep_cmap.reporting import load_tms_rat_mapping


def _rec(detection_version="d1", tool_version="1.4.5", nibs=None,
         conditions=None, filtering=None):
    return {
        "schema": "mep-cmap-reporting", "schema_version": "2",
        "tool_version": tool_version, "detection_version": detection_version,
        "filtering": dict({"sampling_rate_hz": 5000.0,
                           "bandpass_applied": True,
                           "bandpass_low_hz": 10.0,
                           "bandpass_high_hz": 1000.0,
                           "acquisition_filtering": None}, **(filtering or {})),
        "detection": {"onset_method": "peak_fraction",
                      "amplitude_window_ms": [20.0, 40.0]},
        "conditions": conditions if conditions is not None else {
            "A": {"n_trials": 20,
                  "trial_to_trial_interval_s": {"mean": 5.0, "sd": 0.3, "n": 20},
                  "ptp_mV": {"mean": 1.1, "sd": 0.2, "n": 20}}},
        "nibs_bids": nibs if nibs is not None else {
            "modality": "TMS",
            "fields": {"StimulationModality": "TMS", "Manufacturer": "Magstim",
                       "ManufacturersModelName": "200^2",
                       "CoilType": "figure-of-8", "CoilModel": "D70"},
            "parameter_sets": {}, "code_sets": {}},
    }


# ── refusing to pool ─────────────────────────────────────────────────────────

def test_records_from_two_detector_versions_are_refused():
    """
    The same recording yields a different onset under a different detector, so
    a pooled mean describes no analysis that was ever run.
    """
    recs = [_rec(detection_version="d1"), _rec(detection_version="d2")]
    fatal, _notes = T.check_poolable(recs)
    assert fatal and "different detector versions" in fatal[0]
    with pytest.raises(T.AnalysisFault):
        T.assess(recs, load_tms_rat_mapping())


def test_a_differing_tool_version_is_noted_but_not_fatal():
    """A tool version changes for reasons that need not touch a measurement."""
    recs = [_rec(tool_version="1.4.4"), _rec(tool_version="1.4.5")]
    fatal, notes = T.check_poolable(recs)
    assert not fatal
    assert notes and "more than one tool version" in notes[0]


# ── the aggregate rules ──────────────────────────────────────────────────────

def test_unique_reports_divergence_rather_than_choosing():
    val, divergent = T.agg_unique([10.0, 10.0, 30.0])
    assert val is None
    assert divergent == [10.0, 30.0]


def test_unique_agrees_when_every_file_agrees():
    assert T.agg_unique([10.0, 10.0, 10.0]) == (10.0, [])


def test_blanks_do_not_count_as_a_second_value():
    assert T.agg_unique([10.0, None, "", 10.0]) == (10.0, [])


def test_a_single_value_has_no_dispersion():
    """A zero SD would read as a measurement of no variability."""
    assert T.agg_mean_sd([4.0])["sd"] is None
    assert T.agg_mean_sd([4.0])["n"] == 1


def test_sum_and_list_ignore_what_is_not_there():
    assert T.agg_sum([10, None, 5]) == 15
    assert T.agg_list(["a", None, "a", ["b", "a"]]) == ["a", "b"]


# ── rating ───────────────────────────────────────────────────────────────────

def test_a_mean_without_a_dispersion_is_part_not_full():
    assert T.rate_quantitative({"mean": 1.0, "sd": None, "n": 1})[0] == T.PART
    assert T.rate_quantitative({"mean": 1.0, "sd": 0.2, "n": 9})[0] == T.FULL
    assert T.rate_quantitative(None)[0] == T.MISSING


def test_full_requires_names_the_fact_that_is_absent():
    item = {"full_requires": ["sampling rate", "filter type"]}
    rating, absent = T.rate_full_requires(item, {"sampling rate": 5000})
    assert rating == T.PART and absent == ["filter type"]


def test_alternatives_do_not_both_have_to_be_present():
    """A study reports a resting OR an active motor threshold. Demanding both
    would rate a correctly reported study Part."""
    item = {"provider": "RestingMotorThreshold|ActiveMotorThreshold"}
    nibs = [{"fields": {"RestingMotorThreshold": 48}, "parameter_sets": {}}]
    assert T.rate_nibs_item(item, nibs)[0] == T.FULL


def test_a_divergent_field_is_reported_as_an_analysis_fault():
    """Two coil models across a study is either two coils or a typo. Picking
    the majority would hide both."""
    recs = [_rec(), _rec()]
    recs[1]["nibs_bids"]["fields"]["CoilModel"] = "D50"
    results, faults = T.assess(recs, load_tms_rat_mapping())
    d2 = next(r for r in results if r["id"] == "D2")
    assert d2["rating"] == T.PART
    assert "DIVERGENT" in d2["comment"]
    assert any("differs across the study" in f for f in faults)


# ── the whole sheet ──────────────────────────────────────────────────────────

def test_every_mapping_item_gets_a_row():
    results, _ = T.assess([_rec()], load_tms_rat_mapping())
    assert len(results) == 88
    assert len({r["id"] for r in results}) == 88


def test_out_of_scope_items_are_missing_and_say_why():
    results, _ = T.assess([_rec()], load_tms_rat_mapping())
    b1 = next(r for r in results if r["id"] == "B1")
    assert b1["rating"] == T.MISSING
    assert "never sees" in b1["comment"]


def test_a_gap_item_names_itself_so_the_methods_text_can_carry_the_marker():
    results, _ = T.assess([_rec()], load_tms_rat_mapping())
    gap = next(r for r in results if r["source"] == "nibs_gap")
    assert f"[TMS-RAT {gap['id']}" in gap["comment"]


def test_participant_items_are_missing_without_a_participants_file():
    """The tool does not collect participant data. Inventing it from a
    filename would be inventing a measurement."""
    results, _ = T.assess([_rec()], load_tms_rat_mapping(), participants=None)
    a1 = next(r for r in results if r["id"] == "A1")
    assert a1["rating"] == T.MISSING and "participants.tsv" in a1["comment"]


def test_participant_items_are_rated_when_the_file_is_there():
    people = [{"age": 24, "sex": "F"}, {"age": 31, "sex": "M"},
              {"age": 28, "sex": "F"}]
    results, _ = T.assess([_rec()], load_tms_rat_mapping(), participants=people)
    a1 = next(r for r in results if r["id"] == "A1")
    assert a1["rating"] == T.FULL and "SD" in a1["comment"]
    a2 = next(r for r in results if r["id"] == "A2")
    assert "F n = 2" in a2["comment"]


def test_a_gate_item_is_left_for_the_rater():
    """Whether a section applies is a statement about the experiment. Deciding
    it from the presence of a column would let an incidental measurement
    switch on a whole section."""
    results, _ = T.assess([_rec()], load_tms_rat_mapping())
    g0 = next(r for r in results if r["id"] == "G0")
    assert g0["rating"] == T.BLANK and "rater" in g0["comment"]


def test_the_output_rows_match_the_sheet_column_order():
    results, _ = T.assess([_rec()], load_tms_rat_mapping(),
                          study="MyStudy", rater="JA", study_id="10.5281/x")
    rows = T.output_rows(results)
    assert list(rows[0]) == T.OUTPUT_COLUMNS
    assert rows[0]["Study"] == "MyStudy" and rows[0]["Rater"] == "JA"
    assert rows[0]["Version"] == "1.0"


def test_every_item_carries_the_input_row_the_workbook_expects():
    """The rating is written into INPUT!C{input_row}. A missing row number
    means a rating silently lands nowhere."""
    results, _ = T.assess([_rec()], load_tms_rat_mapping())
    missing = [r["id"] for r in results if not r["input_row"]]
    assert not missing, f"no input_row for {missing}"


# ── a rated item must say what it found ──────────────────────────────────────

def test_an_item_rated_full_states_its_values(tmp_path=None):
    """
    The methods draft is written FROM these comments. An item rated Full whose
    comment is blank comes out of the draft as an unknown, so the tool would
    print "[TMS-RAT D1: Stimulator]" for a stimulator whose model it knows.
    """
    results, _ = T.assess([_rec()], load_tms_rat_mapping())
    d1 = next(r for r in results if r["id"] == "D1")
    assert d1["rating"] == T.FULL
    assert "Magstim" in d1["comment"] and "200^2" in d1["comment"]


def test_the_methods_draft_uses_the_value_not_a_marker_where_it_knows_it():
    results, faults = T.assess([_rec()], load_tms_rat_mapping())
    md = T.methods_markdown(results, faults, study="S")
    assert "Stimulator: Manufacturer: Magstim" in md
    assert "[TMS-RAT D1" not in md


def test_a_gap_marker_is_not_printed_twice():
    results, faults = T.assess([_rec()], load_tms_rat_mapping())
    md = T.methods_markdown(results, faults, study="S")
    for line in md.splitlines():
        assert line.count("[TMS-RAT A4") <= 1, line


# ── the inter-trial interval ─────────────────────────────────────────────────

def _rec_interval(mean, sd, stim="A"):
    return _rec(conditions={stim: {
        "n_trials": 20,
        "trial_to_trial_interval_s": {"mean": mean, "sd": sd, "n": 20},
        "ptp_mV": {"mean": 1.1, "sd": 0.2, "n": 20}}})


def test_the_interval_dispersion_is_within_file_not_between_files():
    """
    G1 asks for the dispersion BETWEEN TRIALS. Every file averaging 5 s would
    report SD 0 from their means, which reads as a perfectly regular train and
    says nothing about the intervals actually delivered.
    """
    recs = [_rec_interval(5.0, 0.4) for _ in range(3)]
    stats, suspect = T.prov_trial_to_trial_interval(recs)
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["sd"] == pytest.approx(0.4)
    assert suspect == []


def test_an_sd_larger_than_the_mean_is_reported_as_an_analysis_fault():
    """
    A blocked design whose between-block gap is counted as an inter-trial
    interval gives mean 99.5 s with SD 375.9. That is not a train described by
    a mean, and reporting it as one would put a meaningless number in a
    methods section.
    """
    recs = [_rec_interval(99.5, 375.9, stim="B")]
    stats, suspect = T.prov_trial_to_trial_interval(recs)
    assert suspect and suspect[0][0] == "B"

    results, faults = T.assess(recs, load_tms_rat_mapping())
    g1 = next(r for r in results if r["id"] == "G1")
    assert g1["rating"] == T.PART, "a meaningless mean must not rate Full"
    assert any("not summarised by a mean" in f for f in faults)


def test_a_count_of_pulses_is_a_count():
    """'240.0' invites the reader to wonder what the fractional pulse was."""
    results, _ = T.assess([_rec(), _rec()], load_tms_rat_mapping())
    g2 = next(r for r in results if r["id"] == "G2")
    assert g2["comment"] == "40"


# ── the evoked response ──────────────────────────────────────────────────────

def _rec_mmax_and_mep():
    """One recording with an M-wave on A and MEPs on B, as a real one has."""
    return _rec(conditions={
        "A": {"n_trials": 3,
              "trial_to_trial_interval_s": {"mean": 9.4, "sd": 1.4, "n": 2},
              "ptp_mV": {"mean": 9.5767, "sd": 0.0569, "n": 3}},
        "B": {"n_trials": 20,
              "trial_to_trial_interval_s": {"mean": 8.0, "sd": 0.9, "n": 20},
              "ptp_mV": {"mean": 1.0015, "sd": 0.2032, "n": 20}}})


def test_the_evoked_response_is_never_pooled_across_conditions():
    """
    An M-wave at 9.58 mV averaged with an MEP at 1.00 mV gives 2.61 mV, an
    amplitude nothing produced, and a reader takes it for the study's MEP.
    The checklist's Full definition asks only for a central tendency and a
    dispersion, so the pooled number passes on paper -- which is why the
    pooling has to be refused rather than left to the rating.
    """
    results, _ = T.assess([_rec_mmax_and_mep()], load_tms_rat_mapping())
    g3 = next(r for r in results if r["id"] == "G3")
    assert "2.6" not in g3["comment"], "conditions were pooled"
    assert "A: mean 9.577" in g3["comment"]
    assert "B: mean 1.002" in g3["comment"]


def test_a_condition_without_a_dispersion_keeps_the_item_off_full():
    """One condition described and another not is a partially reported item."""
    rec = _rec_mmax_and_mep()
    rec["conditions"]["B"]["ptp_mV"]["sd"] = None
    results, _ = T.assess([rec], load_tms_rat_mapping())
    g3 = next(r for r in results if r["id"] == "G3")
    assert g3["rating"] == T.PART


def test_a_dispersion_says_what_it_is_a_dispersion_of():
    """A SD across FILES and a SD across TRIALS are different quantities that
    look identical as numbers."""
    one = T.assess([_rec_mmax_and_mep()], load_tms_rat_mapping())[0]
    g3 = next(r for r in one if r["id"] == "G3")
    assert "within one recording" in g3["comment"]

    many = T.assess([_rec_mmax_and_mep() for _ in range(3)],
                    load_tms_rat_mapping())[0]
    g3 = next(r for r in many if r["id"] == "G3")
    assert "between recordings" in g3["comment"]

    g1 = next(r for r in many if r["id"] == "G1")
    assert "typical within-file SD" in g1["comment"]


# ── divergence in a DERIVED item ─────────────────────────────────────────────

def test_two_sampling_rates_are_a_fault_not_a_missing_field():
    """
    2000 Hz in one recording and 4000 Hz in another is not "sampling rate not
    reported". Letting agg_unique's None fall through says the analyst should
    go and fill a field in, when the real problem is that the files were
    analysed differently and cannot share one methods sentence.
    """
    recs = [_rec(filtering={"sampling_rate_hz": 2000.0}),
            _rec(filtering={"sampling_rate_hz": 4000.0})]
    results, faults = T.assess(recs, load_tms_rat_mapping())
    j7 = next(r for r in results if r["id"] == "J7")
    assert "DIVERGENT sampling rate" in j7["comment"]
    assert "Not reported: sampling rate" not in j7["comment"]
    assert any("sampling rate differs across the study" in f for f in faults)
    assert j7["rating"] != T.FULL
    # Nor Missing: the fact WAS reported, just inconsistently, and a Missing
    # rating would contradict the comment beside it naming the values found.
    assert j7["rating"] == T.PART


def test_two_filter_settings_are_a_fault():
    """The case in the spec: two filter settings across a study."""
    recs = [_rec(filtering={"bandpass_high_hz": 450.0}),
            _rec(filtering={"bandpass_high_hz": 1000.0})]
    _results, faults = T.assess(recs, load_tms_rat_mapping())
    assert any("band-pass high cut-off differs" in f for f in faults)


def test_one_sampling_rate_across_the_study_raises_nothing():
    recs = [_rec(filtering={"sampling_rate_hz": 2000.0}) for _ in range(3)]
    results, faults = T.assess(recs, load_tms_rat_mapping())
    j7 = next(r for r in results if r["id"] == "J7")
    assert "DIVERGENT" not in j7["comment"]
    assert not any("differs across the study" in f for f in faults)
    assert "sampling rate: 2000.0" in j7["comment"]
