"""
The NIBS-DAS export add-on, end to end on a study on disk.

The reshape is tested in test_nibs_das.py. This covers what only shows up
against a filesystem: finding each recording's trials table, attributing it to
a participant, and building the folder NIBS-DAS specifies — including the
old/ rotation, which exists so a re-export cannot silently replace a collated
file someone has already analysed against.
"""

import csv
import datetime
import json
import os

from mep_cmap import addons as A
from mep_cmap.reporting import SIDECAR_SUFFIX

ADDON_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "mep_cmap", "add_ons", "group_level",
    "nibs_das_export.py")

TODAY = datetime.date.today().isoformat()


def _record(sub, channel="FDI"):
    return {
        "schema": "mep-cmap-reporting", "schema_version": "2",
        "tool_version": "1.4.6", "detection_version": "d1",
        "recording": {"file": f"{sub}_ses-01", "channel": channel,
                      "source_format": "edf"},
        "filtering": {"sampling_rate_hz": 5000.0},
        "detection": {"onset_method": "peak_fraction"},
        "conditions": {},
        "nibs_bids": {
            "modality": "TMS",
            "fields": {"ManufacturersModelName": "BiStim2",
                       "PulseShape": "Monophasic",
                       "TargetRegion": "Left_M1",
                       "TargetingMethod": "manual",
                       "RestingMotorThreshold": 56},
            "parameter_sets": {
                "SICI": {"nibs_type": "TMS", "position": "",
                         "values": {"ConditioningStimulusIntensity": 80,
                                    "TestStimulusIntensity": 120,
                                    "InterPulseInterval": 2}}},
            "code_sets": {"SICI": "SICI"}},
    }


def _trials(n=4):
    head = ["File", "Channel", "StimType", "Stim_Label", "Segment",
            "Segment_Overall", "Condition", "PTP(mV)", "PreStimRMS",
            "cSP_Duration(ms)", "Outlier_Decision", "Clipped"]
    rows = [dict(zip(head, ["f.edf", "", "SICI", "SICI", i + 1, i + 1, "",
                            round(0.5 + i / 10, 3), 0.02, 110.0, "Keep", ""]))
            for i in range(n)]
    return head, rows


def _study(tmp_path, subs=("sub-001", "sub-002"), channels=("FDI", "APB"),
           participants=True):
    """A derivatives tree as level 1 writes it: one record per channel."""
    deriv = tmp_path / "derivatives"
    for sub in subs:
        for chan in channels:
            base = deriv / sub / "ses-01" / "results"
            (base / "report").mkdir(parents=True, exist_ok=True)
            tag = f"{sub}_ses-01" + (f"_channel-{chan}" if len(channels) > 1
                                     else "")
            (base / "report" / f"{tag}_{SIDECAR_SUFFIX}").write_text(
                json.dumps(_record(sub, chan)), encoding="utf-8")
            head, rows = _trials()
            for r in rows:
                r["Channel"] = chan
            with open(base / f"{tag}_trials.csv", "w", encoding="utf-8",
                      newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=head)
                w.writeheader()
                w.writerows(rows)
    if participants:
        (tmp_path / "rawdata").mkdir(exist_ok=True)
        (tmp_path / "rawdata" / "participants.tsv").write_text(
            "participant_id\tage\tsex\thandedness\n"
            "sub-001\t45\tMale\tRight\n"
            "sub-002\t52\tMale\tRight\n", encoding="utf-8")
    return deriv


class _Ctx:
    def __init__(self, results_dir, config, log):
        self.results_dir = results_dir
        self.config = config
        self.log = log
        self.group_table = None
        self.bids_prefix = "group_level_LME_ready"


def _run(deriv, **cfg):
    logs = []
    module = A._load_module_from_path(ADDON_PATH)
    paths = module.run(_Ctx(str(deriv), dict(cfg), logs.append))
    return paths, logs


# ── the loader seam ──────────────────────────────────────────────────────────

def test_the_addon_imports_the_way_the_loader_loads_it():
    """
    Add-ons load from a file path with no parent package, so a relative import
    inside one is skipped as broken rather than raised.
    """
    found = A.discover_addons([os.path.dirname(ADDON_PATH)],
                              log=lambda _m: None)
    entry = next((f for f in found if f["name"] == "nibs_das_export"), None)
    assert entry, "the add-on did not survive being loaded by path"
    assert entry["scope"] == "group_level"
    assert entry["module"].ND is not None, "mep_cmap.nibs_das did not import"


# ── the folder ───────────────────────────────────────────────────────────────

def test_the_collated_folder_sits_beside_derivatives_not_inside_it(tmp_path):
    """
    NIBS-DAS is not a BIDS derivative. Its template uploads the raw and
    collated data as siblings, so filing this under derivatives/ would break
    the structure the platform expects.
    """
    deriv = _study(tmp_path)
    paths, _ = _run(deriv, nibs_das_study="mep_example")
    base = tmp_path / "collated_participant_data"
    assert base.is_dir()
    assert not (deriv / "collated_participant_data").exists()
    assert all(str(base) in p for p in paths)


def test_the_template_subfolders_are_created(tmp_path):
    """Created empty so the folder is upload-ready as it stands."""
    _run(_study(tmp_path), nibs_das_study="s")
    base = tmp_path / "collated_participant_data"
    for sub in ("usedata", "code", "outputfiles"):
        assert (base / sub / "old").is_dir()
    assert (base / "old").is_dir()


def test_the_filename_carries_the_date(tmp_path):
    """The date IS their version control."""
    paths, _ = _run(_study(tmp_path), nibs_das_study="mep_example")
    assert os.path.basename(paths[0]) == f"mep_exampledata_{TODAY}.csv"


def test_the_usedata_copy_is_written(tmp_path):
    """
    Their template reads the copy, never the original, so an analysis that
    mangles its input has not mangled the dataset.
    """
    paths, _ = _run(_study(tmp_path), nibs_das_study="s")
    use = tmp_path / "collated_participant_data" / "usedata" / f"sdata_{TODAY}_copy.csv"
    assert use.is_file()
    assert use.read_text() == open(paths[0], encoding="utf-8").read()


def test_a_re_export_retires_the_previous_file_rather_than_overwriting(tmp_path):
    """
    Someone may already have run an analysis against the existing table.
    Overwriting it invalidates their results with no record that it changed.
    """
    deriv = _study(tmp_path)
    paths, _ = _run(deriv, nibs_das_study="s")
    open(paths[0], "a", encoding="utf-8").write("sentinel\n")
    before = open(paths[0], encoding="utf-8").read()

    _run(deriv, nibs_das_study="s")
    old = tmp_path / "collated_participant_data" / "old" / f"sdata_{TODAY}.csv"
    assert old.is_file() and old.read_text() == before
    assert "sentinel" not in open(paths[0], encoding="utf-8").read()


# ── the table ────────────────────────────────────────────────────────────────

def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_two_channels_of_a_participant_become_one_row_per_trial(tmp_path):
    paths, _ = _run(_study(tmp_path), nibs_das_study="s")
    rows = _read(paths[0])
    assert len(rows) == 8                    # 2 participants x 4 trials
    assert rows[0]["mep_ampl_fdi"] and rows[0]["mep_ampl_apb"]


def test_participants_tsv_is_found_beside_derivatives(tmp_path):
    """BIDS puts it at the root of the RAW dataset, not the derivatives tree."""
    paths, _ = _run(_study(tmp_path), nibs_das_study="s")
    assert _read(paths[0])[0]["age"] == "45"


def test_a_recording_with_no_sub_entity_is_left_out_and_named(tmp_path):
    """
    participant_id is the join key to participants.tsv and to every other
    dataset on the platform. A recording stem in that column is not a
    participant label, so substituting one would be worse than the gap.
    """
    deriv = _study(tmp_path, subs=("nosub",))
    paths, logs = _run(deriv, nibs_das_study="s")
    assert paths == []
    assert any("no sub- entity" in m for m in logs)


def test_the_sidecar_records_what_the_format_cannot_carry(tmp_path):
    """
    Outlier_Decision and Clipped have no home in NIBS-DAS, so a clipped trial
    sits in the table looking like an ordinary lower number. The sidecar is
    the only place that can be said.
    """
    paths, _ = _run(_study(tmp_path), nibs_das_study="s")
    meta = json.load(open(paths[1], encoding="utf-8"))
    dropped = {d["column"] for d in meta["NotRepresented"]}
    assert {"Outlier_Decision", "Clipped", "Units_Assumed"} <= dropped
    assert meta["Muscles"] == {"fdi": ["FDI"], "apb": ["APB"]}
    assert meta["GeneratedBy"][0]["Version"] == "1.4.6"


def test_the_sidecar_states_the_units(tmp_path):
    """The collated format states mV by convention and never writes it down."""
    paths, _ = _run(_study(tmp_path), nibs_das_study="s")
    meta = json.load(open(paths[1], encoding="utf-8"))
    assert meta["Measures"]["mep_ampl_<muscle>"]["Units"] == "mV"


# ── extended measures ────────────────────────────────────────────────────────

def test_extended_measures_are_off_by_default(tmp_path):
    """A default export must not pre-empt the naming decision."""
    paths, _ = _run(_study(tmp_path), nibs_das_study="s")
    assert "csp_dur_fdi" not in _read(paths[0])[0]


def test_extended_measures_are_flagged_as_proposals_when_on(tmp_path):
    paths, logs = _run(_study(tmp_path), nibs_das_study="s",
                       nibs_das_extended=True)
    assert "csp_dur_fdi" in _read(paths[0])[0]
    meta = json.load(open(paths[1], encoding="utf-8"))
    assert meta["ExtendedMeasuresAreProposals"] is True
    assert any("not yet standard" in m for m in logs)


def test_an_unrecognised_stimulus_label_is_named_in_the_log(tmp_path):
    deriv = _study(tmp_path)
    for p in deriv.rglob("*_trials.csv"):
        text = p.read_text().replace("SICI", "WIBBLE")
        p.write_text(text)
    paths, logs = _run(deriv, nibs_das_study="s")
    assert any("WIBBLE" in m for m in logs)
    assert _read(paths[0])[0]["tms_stim_mode"] == ""


# ── where the folder goes ────────────────────────────────────────────────────

def test_the_project_root_is_found_by_walking_up_to_derivatives(tmp_path):
    """
    results_dir for a group-level add-on is the folder holding whichever
    group CSV was selected, so its parent is wherever the analyst saved that
    file rather than anything about the study.
    """
    from mep_cmap import addons as A
    module = A._load_module_from_path(ADDON_PATH)
    deep = tmp_path / "derivatives" / "sub-001" / "ses-01" / "results"
    deep.mkdir(parents=True)
    assert module.project_root_for(str(deep)) == str(tmp_path)


def test_an_explicit_project_folder_wins(tmp_path):
    from mep_cmap import addons as A
    module = A._load_module_from_path(ADDON_PATH)
    assert module.project_root_for(str(tmp_path), "/somewhere/else") == \
        "/somewhere/else"


def test_the_export_lands_beside_derivatives_from_a_nested_results_dir(tmp_path):
    """The failure this fixes: exporting next to a group CSV buried in a tree."""
    deriv = _study(tmp_path)
    nested = deriv / "sub-001" / "ses-01" / "results"
    paths, logs = _run(nested, nibs_das_study="s")
    assert paths, "nothing was written"
    assert paths[0].startswith(str(tmp_path / "collated_participant_data"))
    assert any(str(tmp_path / "collated_participant_data") in m for m in logs)


def test_the_log_gives_the_absolute_path(tmp_path):
    """
    A relative path here always printed 'collated_participant_data', which
    names the folder and says nothing about where it is.

    Asserted with os.path.isabs rather than a leading "/": on Windows an
    absolute path starts with a drive letter, and the stricter check passed
    on the machine it was written on and failed everywhere else.
    """
    _, logs = _run(_study(tmp_path), nibs_das_study="s")
    written = [m for m in logs if m.startswith("nibs_das_export: written to ")]
    assert written, "the destination was never logged"
    assert os.path.isabs(written[0].split("written to ", 1)[1].strip())


def test_a_relative_project_folder_is_resolved_against_the_study(tmp_path):
    """
    Typing 'test' into a folder setting means beside my study, not wherever
    the application happened to be launched from. Resolving it against the
    working directory puts the export somewhere the analyst cannot guess.
    """
    from mep_cmap import addons as A
    module = A._load_module_from_path(ADDON_PATH)
    deep = tmp_path / "derivatives" / "sub-001" / "ses-01" / "results"
    deep.mkdir(parents=True)
    got = module.project_root_for(str(deep), "test")
    assert got == str(tmp_path / "test")
    assert os.path.isabs(got)


def test_a_relative_project_folder_is_named_in_the_log(tmp_path):
    deriv = _study(tmp_path)
    _, logs = _run(deriv, nibs_das_study="s", nibs_das_root="test")
    assert any("relative path, taken as" in m for m in logs)


def test_the_returned_root_is_always_absolute(tmp_path):
    from mep_cmap import addons as A
    module = A._load_module_from_path(ADDON_PATH)
    for override in ("", "test", str(tmp_path / "abs")):
        assert os.path.isabs(module.project_root_for(str(tmp_path), override))
