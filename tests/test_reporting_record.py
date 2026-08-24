"""
The per-recording reporting record.

A methods section, a reporting checklist and a data-sharing submission all ask
the same question -- what was actually done -- and the answer is spread across
a config, a NIBS-BIDS sidecar, the recording header and the results. This
record writes it down once, at the moment it is true, so none of the three has
to reconstruct it later from four sources that may since have diverged.
"""

import json

import pytest

from mep_cmap import reporting as R
from mep_cmap.bids import TOOL_VERSION
from mep_cmap.detection import CspSettings, DETECTION_VERSION
from mep_cmap.pipeline import PipelineConfig


def _rows():
    return [
        {"StimType": "B", "Time_Since_Last_Stim(s)": 5.1, "PTP(mV)": 1.10,
         "Latency(ms)": 22.8, "cSP_Duration(ms)": 81.4, "PreStimRMS": 0.012},
        # Latency carries the "Not Detected" sentinel on trials with no onset.
        {"StimType": "B", "Time_Since_Last_Stim(s)": 4.9, "PTP(mV)": 0.98,
         "Latency(ms)": "Not Detected", "cSP_Duration(ms)": 62.0,
         "PreStimRMS": 0.011},
        {"StimType": "C", "Time_Since_Last_Stim(s)": 5.0, "PTP(mV)": 0.55,
         "Latency(ms)": 25.0, "cSP_Duration(ms)": None, "PreStimRMS": 0.010},
    ]


def _nibs(fields=None, sets=None, code_sets=None, modality="TMS"):
    """A NIBS-BIDS block in the shape BidsifyState.nibs_record() builds.

    StimulationModality appears in ``fields`` as well as at the top level
    because effective_values() puts it there, and a helper that omitted it
    would test a shape the tool never produces.
    """
    return {
        "modality": modality,
        "fields": dict({"StimulationModality": modality}, **(fields or {})),
        "parameter_sets": dict(sets or {}),
        "code_sets": dict(code_sets or {}),
    }


def _record(**kw):
    cfg = kw.pop("cfg", None) or PipelineConfig(csp_types={"B"})
    return R.build_record(
        cfg, name="sub-001_emg", channel="EMG1", fs=5000.0, emg_unit="mV",
        source_format="EDF", rows=_rows(),
        nibs_record=kw.pop("nibs", _nibs()),
        tool_version=TOOL_VERSION, detection_version=DETECTION_VERSION, **kw)


def test_a_sentinel_string_is_not_averaged_as_a_number():
    """
    Trial tables carry sentinel strings in otherwise numeric columns.
    float() on such a column raises, and coercing it to 0 would be worse:
    an undetected onset would enter the mean as a 0 ms latency.
    """
    lat = _record()["conditions"]["B"]["latency_ms"]
    assert lat["n"] == 1 and lat["mean"] == pytest.approx(22.8)


def test_a_quantitative_fact_carries_dispersion_with_its_mean():
    """Full reporting of a quantitative item needs both, so they travel
    together and are never available singly."""
    ptp = _record()["conditions"]["B"]["ptp_mV"]
    assert set(ptp) >= {"mean", "sd", "n", "min", "max"}
    assert ptp["n"] == 2 and ptp["sd"] > 0


def test_a_single_value_reports_n_rather_than_a_zero_sd():
    """A zero SD reads as a measurement of no variability. n=1 says what
    actually happened."""
    one = R._mean_sd([4.2])
    assert one["n"] == 1 and one["sd"] is None


def test_a_column_of_nothing_is_absent_rather_than_zero():
    assert R._mean_sd([None, "", "n/a"]) is None
    assert _record()["conditions"]["C"]["csp_duration_ms"] is None


def test_the_silent_period_block_comes_from_the_shared_settings():
    """
    Read through CspSettings, so the record cannot describe a different
    configuration from the one the detector ran under.
    """
    cfg = PipelineConfig(csp_types={"B"}, csp_min_return_ms=55.0,
                         csp_max_mep_offset_ms=123.0)
    sp = _record(cfg=cfg)["silent_period"]
    s = CspSettings.from_source(cfg)
    assert sp["min_return_ms"] == s.min_return_ms == 55.0
    assert sp["max_offset_from_2nd_peak_ms"] == s.max_mep_offset_ms == 123.0
    assert "2nd peak" in sp["search_start"]


def test_both_version_stamps_are_recorded():
    """
    A study assembled from files analysed under different detectors is not
    one study. Level 2 can only refuse if level 1 wrote down which was used.
    """
    rec = _record()
    assert rec["tool_version"] == TOOL_VERSION
    assert rec["detection_version"] == DETECTION_VERSION


def test_acquisition_filtering_is_recorded_as_unknown_not_omitted():
    """
    TMS-RAT J7 treats acquisition and analysis filtering as two facts and
    needs both. Only the second is knowable here. A null says so; an absent
    key would read as an oversight.
    """
    f = _record()["filtering"]
    assert "acquisition_filtering" in f and f["acquisition_filtering"] is None
    assert f["sampling_rate_hz"] == 5000.0


def test_the_record_is_json_serialisable():
    rec = _record()
    assert json.loads(json.dumps(rec)) == rec


def test_missing_nibs_fields_names_what_is_blank():
    missing = R.missing_nibs_fields(_nibs())
    assert "StimulationModality" not in missing
    assert "CoilType" in missing and "ManufacturersModelName" in missing


def test_an_empty_value_counts_as_missing():
    assert "CoilType" in R.missing_nibs_fields(_nibs({"CoilType": ""}))
    assert "CoilType" not in R.missing_nibs_fields(
        _nibs({"CoilType": "figure-of-8"}))


def test_fields_belonging_to_another_modality_are_not_reported_as_gaps():
    """
    A TMS study has no electrodes and no transducer. Eight of the twenty-five
    required/recommended fields belong to tES, TUS or PNS, and naming them
    teaches the analyst that this list is noise.
    """
    missing = R.missing_nibs_fields(_nibs(modality="TMS"))
    for key in ("ElectrodeType", "TransducerModel", "MontageDescription",
                "StimulationType", "Current", "FundamentalFrequency",
                "SonicationDuration"):
        assert key not in missing, f"{key} does not apply to TMS"
    # And the reverse: a tES study is not asked for a coil.
    assert "CoilType" not in R.missing_nibs_fields(_nibs(modality="tES"))
    assert "ElectrodeType" in R.missing_nibs_fields(_nibs(modality="tES"))


def test_a_parameter_set_value_is_not_reported_as_an_unfilled_field():
    """
    v6.3 puts most stimulation parameters on a named set, not on the file.
    Reading only the file-level fields reports thirteen of the twenty-five as
    unset in the same breath as telling the analyst to go and set them --
    which they already have.
    """
    filled = _nibs(
        sets={"spTMS": {"nibs_type": "TMS",
                        "values": {"StimulationIntensity": 120,
                                   "IntensityReference": "rMT"}}},
        code_sets={"A": "spTMS"})
    missing = R.missing_nibs_fields(filled)
    assert "StimulationIntensity" not in missing
    assert "IntensityReference" not in missing
    # Still absent where it genuinely is.
    assert "PulseShape" in missing


def test_a_parameter_described_for_only_some_sets_is_still_a_gap():
    """
    Half a protocol described is a gap. A paired-pulse recording whose test
    pulse carries an intensity and whose conditioning pulse does not cannot
    report its intensities, and averaging the one that exists would state a
    protocol nobody ran.
    """
    half = _nibs(
        sets={"test": {"values": {"StimulationIntensity": 120}},
              "cond": {"values": {}}},
        code_sets={"A": "test", "B": "cond"})
    assert "StimulationIntensity" in R.missing_nibs_fields(half)


def test_a_set_this_recording_never_references_is_not_read():
    """
    Parameter sets are session-level. One that no code in THIS file uses
    describes stimulation that was never delivered here, so filling it must
    not close a gap in a recording that does not use it.
    """
    other = _nibs(
        sets={"other_file": {"values": {"StimulationIntensity": 120}}},
        code_sets={})
    # No code_sets at all: nothing says which sets this file used, so every
    # known set is read rather than none.
    assert "StimulationIntensity" not in R.missing_nibs_fields(other)
    # But once the file states its assignment, only the assigned set counts.
    named = dict(other)
    named["parameter_sets"] = {
        "other_file": {"values": {"StimulationIntensity": 120}},
        "mine": {"values": {}}}
    named["code_sets"] = {"A": "mine"}
    assert "StimulationIntensity" in R.missing_nibs_fields(named)


def test_the_sidecar_is_filed_with_the_report_family():
    from mep_cmap.results_layout import family_for
    assert family_for(f"sub-001_emg_{R.SIDECAR_SUFFIX}") == "report"


def test_the_sidecar_is_never_read_as_analysis_input():
    """A stacked JSON beside the trial files is exactly what a loose glob
    picks up."""
    from mep_cmap.results_layout import EXCLUDED_FROM_SEARCH
    assert "report" in EXCLUDED_FROM_SEARCH


def test_writing_the_sidecar_never_raises(tmp_path):
    p = R.write_sidecar(_record(), str(tmp_path), "sub-001_emg")
    assert p and p.endswith(R.SIDECAR_SUFFIX)
    assert json.load(open(p, encoding="utf-8"))["schema"] == "mep-cmap-reporting"
    # An unwritable destination blanks the path rather than failing the run.
    assert R.write_sidecar(_record(), "\0bad", "x") is None


def test_every_config_field_the_record_reads_actually_exists():
    """
    The record reads cfg by NAME. A name that is not a field returns the
    getattr default and the fact is silently absent -- which is how a real
    record came to state that a bandpass was applied without saying at what
    frequencies, the fields being highpass/lowpass and not bp_low/bp_high.

    Asserted against the dataclass rather than against a run, because a run
    with default settings cannot tell a correct None from a misspelled one.
    """
    import inspect
    import re
    from mep_cmap import reporting
    from mep_cmap.pipeline import PipelineConfig

    fields = set(PipelineConfig.__dataclass_fields__)
    src = inspect.getsource(reporting.analysis_facts)
    read = set(re.findall(r'\bg\("([a-z_0-9]+)"', src))
    unknown = sorted(read - fields)
    assert not unknown, (
        f"analysis_facts reads config fields that do not exist: {unknown}. "
        f"Each returns the default and is written as a null.")


def test_the_bandpass_cutoffs_are_reported_when_a_bandpass_is_applied():
    """TMS-RAT J7 needs the cut-off frequencies, not just that filtering
    happened. A null pair beside bandpass_applied=True is a Partial rating
    manufactured by the tool."""
    from mep_cmap.pipeline import PipelineConfig
    cfg = PipelineConfig(apply_bandpass=True, highpass=10.0, lowpass=1000.0)
    f = R.analysis_facts(cfg, fs=5000.0)["filtering"]
    assert f["bandpass_applied"] is True
    assert f["bandpass_low_hz"] == 10.0
    assert f["bandpass_high_hz"] == 1000.0


def test_the_recording_facts_are_taken_from_the_file_not_the_config_default():
    """
    Sampling rate, amplitude unit and source format are DISCOVERED when the
    file is opened. They reached the record as nulls while the pipeline was
    reading them off a config that had no such fields.
    """
    from mep_cmap.pipeline import PipelineConfig
    cfg = PipelineConfig()
    for field in ("emg_unit", "source_format", "nibs_record"):
        assert field in PipelineConfig.__dataclass_fields__, (
            f"PipelineConfig has no {field}, so the pipeline cannot record it")
    cfg.emg_unit, cfg.source_format = "mV", "edf"
    rec = R.build_record(cfg, name="x", fs=5000.0, emg_unit=cfg.emg_unit,
                         source_format=cfg.source_format, rows=[])
    assert rec["filtering"]["amplitude_units"] == "mV"
    assert rec["recording"]["source_format"] == "edf"


def test_a_pooled_folder_run_does_not_claim_one_stimulation_protocol(tmp_path):
    """
    A folder run pools every file into ONE analysis record. Attaching a
    NIBS-BIDS description there would state one protocol for recordings never
    checked to share one. "Not stated" is true; the wrong protocol is not.
    """
    import mep_cmap.pipeline as P

    src = tmp_path / "raw"
    src.mkdir()
    for n in ("a.txt", "b.txt"):
        (src / n).write_text("0\t0\n")

    seen = {}

    def _fake_write_outputs(*a, **kw):
        seen["nibs_record"] = kw.get("nibs_record")
        raise RuntimeError("stop here — the record decision has been made")

    logged = []
    described = {"modality": "TMS", "fields": {"CoilType": "figure-of-8"},
                 "parameter_sets": {}, "code_sets": {}}

    real = P.pipeline_write_outputs
    P.pipeline_write_outputs = _fake_write_outputs
    try:
        with pytest.raises(Exception):
            P.run_pipeline(
                str(src), 100.0, 100.0, 20.0, 40.0,
                prestim_ms=50.0, apply_humbug=False, apply_filter=False,
                apply_bandpass=False, apply_notch=False,
                highpass=10.0, lowpass=1000.0, notch_freq=50.0, notch_q=30.0,
                filter_order=4, nibs_record=described,
                log_callback=logged.append)
    finally:
        P.pipeline_write_outputs = real

    assert any("pooled into one analysis record" in m for m in logged), (
        f"the analyst was not told the description was dropped: {logged}")
