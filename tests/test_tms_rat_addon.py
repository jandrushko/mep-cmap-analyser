"""
The TMS-RAT report add-on, end to end on a study on disk.

Level 1 writes a record beside every recording's results; level 2 reads them
all and assesses the study. This covers the seam between them, including the
one that would silently break: an add-on is loaded from a FILE PATH with no
parent package, so a relative import inside it is skipped as a broken add-on
rather than raised.
"""

import csv
import json
import os

import pytest

from mep_cmap import addons as A
from mep_cmap import tms_rat as TR
from mep_cmap.reporting import SIDECAR_SUFFIX

ADDON_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "mep_cmap", "add_ons", "group_level",
    "tms_rat_report.py")


def _record(sub, detection_version="d1", coil="D70"):
    return {
        "schema": "mep-cmap-reporting", "schema_version": "2",
        "tool_version": "1.4.5", "detection_version": detection_version,
        "recording": {"file": sub, "channel": "EMG1", "source_format": "edf"},
        "filtering": {"sampling_rate_hz": 5000.0, "bandpass_applied": True,
                      "bandpass_low_hz": 10.0, "bandpass_high_hz": 1000.0,
                      "acquisition_filtering": None},
        "detection": {"onset_method": "peak_fraction",
                      "amplitude_window_ms": [20.0, 40.0]},
        "conditions": {"A": {"n_trials": 20,
                             "trial_to_trial_interval_s":
                                 {"mean": 5.0, "sd": 0.3, "n": 20},
                             "ptp_mV": {"mean": 1.1, "sd": 0.2, "n": 20}}},
        "nibs_bids": {"modality": "TMS",
                      "fields": {"StimulationModality": "TMS",
                                 "Manufacturer": "Magstim",
                                 "ManufacturersModelName": "200^2",
                                 "CoilType": "figure-of-8", "CoilModel": coil},
                      "parameter_sets": {}, "code_sets": {}},
    }


def _study(tmp_path, n=3, participants=True, **rec_kw):
    """A derivatives tree with one record per subject, as level 1 writes it."""
    deriv = tmp_path / "derivatives"
    for i in range(n):
        sub = f"sub-{i + 1:03d}"
        d = deriv / sub / "ses-01" / "results" / "report"
        d.mkdir(parents=True)
        rec = _record(sub, **rec_kw)
        (d / f"{sub}_ses-01_{SIDECAR_SUFFIX}").write_text(
            json.dumps(rec), encoding="utf-8")
    if participants:
        (tmp_path / "participants.tsv").write_text(
            "participant_id\tage\tsex\thandedness\n"
            "sub-001\t24\tF\tright\n"
            "sub-002\t31\tM\tright\n"
            "sub-003\t28\tF\tleft\n", encoding="utf-8")
    return deriv


class _Ctx:
    """The parts of GroupAddonContext this add-on uses."""

    def __init__(self, results_dir, config, log):
        self.results_dir = results_dir
        self.config = config
        self.log = log
        self.group_table = None
        self.bids_prefix = "group_level_LME_ready"


def _run(deriv, **cfg):
    logs = []
    entry = {"module": A._load_module_from_path(ADDON_PATH)}
    ctx = _Ctx(str(deriv), dict(cfg), logs.append)
    paths = entry["module"].run(ctx)
    return paths, logs


# ── the loader seam ──────────────────────────────────────────────────────────

def test_the_addon_imports_when_loaded_the_way_the_loader_loads_it():
    """
    discover_addons uses spec_from_file_location, so the module has no parent
    package. A relative import inside it raises at import time and the add-on
    is skipped with a log note rather than an error anyone would notice.
    """
    found = A.discover_addons([os.path.dirname(ADDON_PATH)],
                              log=lambda _m: None)
    names = {f["name"] for f in found}
    assert "tms_rat_report" in names, (
        "the add-on did not survive being loaded by path")
    entry = next(f for f in found if f["name"] == "tms_rat_report")
    assert entry["scope"] == "group_level"
    assert entry["module"].TR is not None, "mep_cmap.tms_rat did not import"


# ── the outputs ──────────────────────────────────────────────────────────────

def test_it_writes_the_three_outputs(tmp_path):
    deriv = _study(tmp_path)
    paths, _logs = _run(deriv, tms_rat_study="MyStudy", tms_rat_rater="JA",
                        tms_rat_protocol="Motor-evoked potentials")
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["MyStudy_methods.md", "MyStudy_tms-rat.csv",
                     "MyStudy_tms-rat.xlsx"]
    for p in paths:
        assert os.path.getsize(p) > 0


def test_the_csv_is_in_the_output_sheet_format(tmp_path):
    deriv = _study(tmp_path)
    paths, _ = _run(deriv, tms_rat_study="S1", tms_rat_rater="JA",
                    tms_rat_protocol="Motor-evoked potentials")
    csv_path = next(p for p in paths if p.endswith(".csv"))
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == TR.OUTPUT_COLUMNS
    assert len(rows) == 88
    assert {r["Study"] for r in rows} == {"S1"}
    assert {r["Rater"] for r in rows} == {"JA"}
    # Required comes from the checklist's own relevance table.
    assert {r["Required"] for r in rows} <= {"required", "sometimes",
                                             "not required", "-"}


def test_the_records_are_found_wherever_they_sit(tmp_path):
    deriv = _study(tmp_path, n=4)
    recs = _run(deriv)[1]
    assert any("4 recording(s) assessed" in m for m in recs)


def test_participant_items_are_rated_from_participants_tsv(tmp_path):
    deriv = _study(tmp_path)
    paths, _ = _run(deriv, tms_rat_study="S1")
    csv_path = next(p for p in paths if p.endswith(".csv"))
    with open(csv_path, encoding="utf-8") as fh:
        rows = {r["Section"] + r["Item"]: r for r in csv.DictReader(fh)}
    assert rows["A1"]["Rating"] == TR.FULL
    assert "SD" in rows["A1"]["Comments"]
    assert "F n = 2" in rows["A2"]["Comments"]


def test_without_participants_those_items_are_missing_and_say_so(tmp_path):
    deriv = _study(tmp_path, participants=False)
    paths, logs = _run(deriv, tms_rat_study="S1")
    csv_path = next(p for p in paths if p.endswith(".csv"))
    with open(csv_path, encoding="utf-8") as fh:
        rows = {r["Section"] + r["Item"]: r for r in csv.DictReader(fh)}
    assert rows["A1"]["Rating"] == TR.MISSING
    assert any("does not collect participant data" in m for m in logs)


# ── refusal and faults ───────────────────────────────────────────────────────

def test_mismatched_detector_versions_are_refused_not_pooled(tmp_path):
    deriv = _study(tmp_path, n=2)
    # Re-write one record under a different detector.
    p = sorted(deriv.rglob(f"*_{SIDECAR_SUFFIX}"))[0]
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["detection_version"] = "d2"
    p.write_text(json.dumps(rec), encoding="utf-8")

    paths, logs = _run(deriv, tms_rat_study="S1")
    assert paths == [], "an assessment was written across two detectors"
    assert any("REFUSED" in m for m in logs)


def test_a_divergent_field_is_reported_as_an_analysis_fault(tmp_path):
    deriv = _study(tmp_path, n=2)
    p = sorted(deriv.rglob(f"*_{SIDECAR_SUFFIX}"))[0]
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["nibs_bids"]["fields"]["CoilModel"] = "D50"
    p.write_text(json.dumps(rec), encoding="utf-8")

    paths, logs = _run(deriv, tms_rat_study="S1")
    assert paths, "the assessment should still be written"
    assert any("differs across the study" in m for m in logs)
    md = next(p for p in paths if p.endswith(".md"))
    assert "Analysis faults" in open(md, encoding="utf-8").read()


# ── the methods draft ────────────────────────────────────────────────────────

def test_the_methods_draft_marks_every_gap(tmp_path):
    """A paragraph that reads smoothly with the unknowable parts omitted looks
    finished, and those parts are exactly what the checklist exists to catch."""
    deriv = _study(tmp_path)
    paths, _ = _run(deriv, tms_rat_study="S1")
    md = open(next(p for p in paths if p.endswith(".md")), encoding="utf-8").read()
    assert "[TMS-RAT" in md
    assert "Search for `[TMS-RAT`" in md
    # The amplifier model is not in any recording, so it must be marked.
    assert "J1" in md


def test_the_workbook_is_filled_on_input_not_output(tmp_path):
    """OUTPUT is formulas that read INPUT and recalculate on open. Writing
    values there would replace the checklist's own logic with ours."""
    openpyxl = pytest.importorskip("openpyxl")
    deriv = _study(tmp_path)
    paths, _ = _run(deriv, tms_rat_study="S1", tms_rat_rater="JA",
                    tms_rat_id="10.5281/zenodo.1")
    xlsx = next(p for p in paths if p.endswith(".xlsx"))
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["INPUT"]
    assert ws["C3"].value == "S1"
    assert ws["C4"].value == "JA"
    assert ws["C2"].value == "10.5281/zenodo.1"
    # D1 (stimulator) is derivable and its row must carry a rating.
    mapping = json.load(open(os.path.join(
        os.path.dirname(ADDON_PATH), "..", "..", "schema",
        "tms_rat_v1_0.json"), encoding="utf-8"))
    d1 = next(i for i in mapping["items"] if i["id"] == "D1")
    assert ws.cell(row=d1["input_row"], column=3).value == TR.FULL
    # OUTPUT still holds its formulas.
    assert str(wb["OUTPUT"]["A2"].value).startswith("=")


def test_no_records_writes_nothing_and_says_why(tmp_path):
    empty = tmp_path / "derivatives"
    empty.mkdir()
    paths, logs = _run(empty, tms_rat_study="S1")
    assert paths == []
    assert any("no analysis records found" in m for m in logs)


def test_recordings_with_results_but_no_record_are_named(tmp_path):
    """
    An assessment covering one of three recordings looks exactly like one
    covering all three. The whole justification for doing this at group level
    is that a dispersion needs several recordings, so a silent sample of the
    study is the worst failure this add-on can have.
    """
    deriv = _study(tmp_path, n=1)
    # A second recording that was analysed before the record existed: it has
    # results and no reporting.json, which is what an older run leaves behind.
    old = deriv / "sub-099" / "ses-01" / "results" / "trial-level"
    old.mkdir(parents=True)
    (old / "sub-099_ses-01_emg_trials.csv").write_text("Trial\n1\n",
                                                      encoding="utf-8")

    paths, logs = _run(deriv, tms_rat_study="S1")
    assert any("sub-099_ses-01_emg" in m for m in logs), logs
    assert any("1 of 2 recordings" in m for m in logs), logs
    # And it survives into the written draft, not just the log.
    md = next(p for p in paths if p.endswith(".md"))
    assert "sub-099_ses-01_emg" in open(md, encoding="utf-8").read()


def test_a_complete_study_is_not_warned_about(tmp_path):
    deriv = _study(tmp_path, n=3)
    for i in range(3):
        sub = f"sub-{i + 1:03d}"
        d = deriv / sub / "ses-01" / "results" / "trial-level"
        d.mkdir(parents=True)
        (d / f"{sub}_ses-01_trials.csv").write_text("Trial\n1\n",
                                                   encoding="utf-8")
    _paths, logs = _run(deriv, tms_rat_study="S1")
    assert not any("recordings in this tree" in m for m in logs), logs


def test_participants_tsv_is_found_in_rawdata(tmp_path):
    """
    The tool BIDS-ifies into rawdata/ beside derivatives/, so that is where
    participants.tsv lands. Missing it reports "no participants.tsv in the
    dataset" for a study that has one -- a false statement about the study
    rather than a gap in it.
    """
    deriv = _study(tmp_path, participants=False)
    raw = tmp_path / "rawdata"
    raw.mkdir()
    (raw / "participants.tsv").write_text(
        "participant_id\tage\tsex\n"
        "sub-001\t24\tF\n"
        "sub-002\t31\tM\n", encoding="utf-8")

    paths, logs = _run(deriv, tms_rat_study="S1")
    assert any("read 2 participant row(s)" in m for m in logs), logs
    csv_path = next(p for p in paths if p.endswith(".csv"))
    with open(csv_path, encoding="utf-8") as fh:
        rows = {r["Section"] + r["Item"]: r for r in csv.DictReader(fh)}
    assert rows["A1"]["Rating"] == TR.FULL


def test_a_participants_file_without_the_column_says_which_column(tmp_path):
    """A file that exists but has only participant_id is a different problem
    from no file at all, and has a different fix."""
    deriv = _study(tmp_path, participants=False)
    raw = tmp_path / "rawdata"
    raw.mkdir()
    (raw / "participants.tsv").write_text(
        "participant_id\nsub-001\nsub-002\n", encoding="utf-8")

    paths, _ = _run(deriv, tms_rat_study="S1")
    csv_path = next(p for p in paths if p.endswith(".csv"))
    with open(csv_path, encoding="utf-8") as fh:
        rows = {r["Section"] + r["Item"]: r for r in csv.DictReader(fh)}
    assert rows["A1"]["Rating"] == TR.MISSING
    assert "no 'age' column" in rows["A1"]["Comments"]
    assert "no participants.tsv" not in rows["A1"]["Comments"]
