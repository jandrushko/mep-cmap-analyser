"""
The BIDS derivatives layout (v1.4.8).

    <study>/derivatives/mep-cmap-analyser/dataset_description.json
    <study>/derivatives/mep-cmap-analyser/sub-XX/ses-YY/emg/...

and what happens to studies written in the legacy layout
(<study>/derivatives/sub-XX/ses-YY/...): read where they are, archived for a
recording once it is re-run, or moved wholesale by the migration command.
"""

import json
import os

from mep_cmap import bids as B
from mep_cmap import layout_migration as LM
from mep_cmap.bids import StudyMetadata


META = StudyMetadata(participant_id="sub-001", session="ses-01")


def _touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _session(path, recording):
    return _touch(path, json.dumps({"file_path": recording}))


# ── where things go ──────────────────────────────────────────────────────────

def test_session_path_is_in_the_emg_folder(tmp_path):
    src = str(tmp_path / "rawdata" / "test000.smr")
    p = B.session_path_for(src, META, str(tmp_path))
    rel = os.path.relpath(p, tmp_path).split(os.sep)
    assert rel[:5] == ["derivatives", "mep-cmap-analyser", "sub-001",
                       "ses-01", "emg"]
    assert rel[-1].endswith("_session.json")


def test_legacy_session_path(tmp_path):
    src = str(tmp_path / "rawdata" / "test000.smr")
    p = B.session_path_for(src, META, str(tmp_path), legacy=True)
    assert os.path.relpath(p, tmp_path).split(os.sep)[:3] == \
        ["derivatives", "sub-001", "ses-01"]


def test_any_of_the_three_roots_means_the_same_place(tmp_path):
    want = B.session_output_dir(str(tmp_path), META)
    for root in (tmp_path, tmp_path / "derivatives",
                 tmp_path / "derivatives" / "mep-cmap-analyser"):
        assert B.session_output_dir(str(root), META) == want


def test_study_root_found_from_either_layout(tmp_path):
    new = B.session_path_for(str(tmp_path / "r.smr"), META, str(tmp_path))
    old = B.session_path_for(str(tmp_path / "r.smr"), META, str(tmp_path),
                             legacy=True)
    assert B.study_root_for(new) == str(tmp_path)
    assert B.study_root_for(old) == str(tmp_path)
    # A session folder that does not exist yet (the save case).
    assert B.study_root_for(os.path.dirname(new)) == str(tmp_path)


def test_derivatives_description(tmp_path):
    p = B.write_derivatives_description(str(tmp_path), version="9.9.9")
    d = json.load(open(p, encoding="utf-8"))
    assert d["DatasetType"] == "derivative"
    assert d["GeneratedBy"][0]["Name"] == "MEP-CMAP Analyser"
    assert d["GeneratedBy"][0]["Version"] == "9.9.9"
    assert os.path.dirname(p).endswith("mep-cmap-analyser")
    # Left alone once it exists: the analyst may have edited it.
    _touch(p, '{"Name": "mine"}')
    B.write_derivatives_description(str(tmp_path))
    assert json.load(open(p, encoding="utf-8")) == {"Name": "mine"}


def test_pipeline_writes_through_the_shared_rule():
    src = open(os.path.join(os.path.dirname(B.__file__), "pipeline.py"),
               encoding="utf-8").read()
    assert "session_output_dir(_study_root, meta)" in src
    assert "write_derivatives_description(_study_root)" in src
    assert "_make_deriv_base" not in src


def test_app_alias_forwards_legacy(tmp_path):
    from mep_cmap import app as A
    src = str(tmp_path / "r.smr")
    assert A.session_path_for(src, META, str(tmp_path), legacy=True) == \
        B.session_path_for(src, META, str(tmp_path), legacy=True)


def test_find_session_for_reads_both_layouts(tmp_path):
    from mep_cmap.bidsify import find_session_for
    src = str(tmp_path / "rawdata" / "test000.smr")
    old = _session(B.session_path_for(src, META, str(tmp_path), legacy=True),
                   "rawdata/test000.smr")
    assert find_session_for(src, META, str(tmp_path)) == old
    new = _session(B.session_path_for(src, META, str(tmp_path)),
                   "rawdata/test000.smr")
    assert find_session_for(src, META, str(tmp_path)) == new


# ── a legacy study to work on ────────────────────────────────────────────────

def _legacy_study(root):
    """Two recordings in one legacy ses- folder, one with a longer prefix."""
    ses = os.path.join(root, "derivatives", "sub-001", "ses-01")
    P, P2 = "sub-001_ses-01_test000", "sub-001_ses-01_test000_run2"
    _session(os.path.join(ses, P + "_session.json"), "rawdata/test000.smr")
    _session(os.path.join(ses, P2 + "_session.json"), "rawdata/test000_run2.smr")
    files = {
        "P": [os.path.join(ses, "trials", P + "_A.csv"),
              os.path.join(ses, "results", "trial-level", P + "_trials.csv"),
              os.path.join(ses, "figures", P + "_onset_methods_figures", "a.png")],
        "P2": [os.path.join(ses, "trials", P2 + "_A.csv")],
    }
    for lst in files.values():
        for f in lst:
            _touch(f)
    return ses, P, P2, files


def test_legacy_items_respect_a_longer_prefix(tmp_path):
    ses, P, P2, files = _legacy_study(str(tmp_path))
    got = set(LM.legacy_items(ses, P))
    assert os.path.join(ses, P + "_session.json") in got
    assert os.path.join(ses, "trials", P + "_A.csv") in got
    # The figures folder goes whole, not file by file.
    assert os.path.join(ses, "figures", P + "_onset_methods_figures") in got
    assert not any(P2 in os.path.basename(g) for g in got)


# ── (a) a re-run archives that recording's legacy copy ───────────────────────

def test_rerun_archives_only_that_recordings_legacy_outputs(tmp_path):
    root = str(tmp_path)
    ses, P, P2, files = _legacy_study(root)
    src = os.path.join(root, "rawdata", "test000.smr")
    new_sess = _session(B.session_path_for(src, META, root),
                        "rawdata/test000.smr")
    out = LM.archive_legacy_for_recording(src, META, root, new_sess)
    assert out and out["moved"] == 4 and not out["failed"]
    for f in files["P"]:
        assert not os.path.exists(f)
    for f in files["P2"]:
        assert os.path.exists(f)                    # the other recording
    assert os.path.exists(os.path.join(ses, P2 + "_session.json"))
    arch = out["archive_root"]
    assert os.path.dirname(os.path.dirname(arch)) == root   # beside derivatives
    assert os.path.exists(os.path.join(arch, "sub-001", "ses-01",
                                       "trials", P + "_A.csv"))


def test_nothing_moves_until_the_new_session_exists(tmp_path):
    root = str(tmp_path)
    ses, P, P2, files = _legacy_study(root)
    src = os.path.join(root, "rawdata", "test000.smr")
    missing = B.session_path_for(src, META, root)
    assert LM.archive_legacy_for_recording(src, META, root, missing) is None
    assert all(os.path.exists(f) for f in files["P"])


# ── (b) migrating a whole study ──────────────────────────────────────────────

def test_migration_moves_everything_into_the_emg_folder(tmp_path):
    root = str(tmp_path)
    ses, P, P2, files = _legacy_study(root)
    plan = LM.plan_migration(root)
    assert plan.n_sessions == 2 and not plan.archive and not plan.conflicts
    out = LM.apply_migration(plan)
    assert out["moved"] == len(plan.moves) and not out["failed"]

    emg = os.path.join(root, "derivatives", "mep-cmap-analyser",
                       "sub-001", "ses-01", "emg")
    for f in files["P"] + files["P2"]:
        assert not os.path.exists(f)
        assert os.path.exists(os.path.join(emg, os.path.relpath(f, ses)))
    assert not os.path.exists(ses)                  # emptied and removed
    assert os.path.isfile(out["manifest"])
    assert os.path.dirname(out["manifest"]) == root
    assert os.path.isfile(os.path.join(root, "derivatives", "mep-cmap-analyser",
                                       "dataset_description.json"))
    # The moved session still resolves its recording against the study folder.
    sess = os.path.join(emg, P + "_session.json")
    assert B.study_root_for(sess) == root


def test_migration_archives_a_recording_already_rerun(tmp_path):
    root = str(tmp_path)
    ses, P, P2, files = _legacy_study(root)
    src = os.path.join(root, "rawdata", "test000.smr")
    _session(B.session_path_for(src, META, root), "rawdata/test000.smr")
    plan = LM.plan_migration(root)
    assert {i.prefix for i in plan.archive} == {P}
    assert {p for _s, _d, p in plan.moves} == {P2}
    LM.apply_migration(plan)
    assert not any(os.path.exists(f) for f in files["P"])
    assert os.path.isdir(os.path.join(root, "superseded_derivatives"))


def test_migration_from_the_derivatives_folder_itself(tmp_path):
    root = str(tmp_path)
    _legacy_study(root)
    plan = LM.plan_migration(os.path.join(root, "derivatives"))
    assert plan.study_root == root and plan.moves


def test_queue_entries_follow_their_session(tmp_path):
    root = str(tmp_path)
    ses, P, P2, files = _legacy_study(root)
    old = os.path.join(ses, P + "_session.json")

    class FE:
        derivatives_json = old
    fe = FE()
    plan = LM.plan_migration(root)
    LM.apply_migration(plan)
    assert LM.remap_session_paths([fe], plan) == 1
    assert os.path.isfile(fe.derivatives_json)
    assert B.is_in_pipeline_dir(fe.derivatives_json)


def test_a_migrated_study_has_nothing_left_to_migrate(tmp_path):
    root = str(tmp_path)
    _legacy_study(root)
    LM.apply_migration(LM.plan_migration(root))
    again = LM.plan_migration(root)
    assert not again.moves and not again.archive


# ── every module agrees on where derivatives/ is ─────────────────────────────

def _three_roots(tmp_path):
    return (str(tmp_path), str(tmp_path / "derivatives"),
            str(tmp_path / "derivatives" / "mep-cmap-analyser"))


def test_queue_file_stays_in_derivatives_whatever_the_root(tmp_path):
    from mep_cmap.dataset_session import DatasetSession, _study_root_from_deriv
    want = os.path.join(str(tmp_path), "derivatives")
    for root in _three_roots(tmp_path):
        assert DatasetSession(root)._deriv_dir == want
        assert _study_root_from_deriv(root) == str(tmp_path)


def test_channel_config_sidecars_never_nest_a_second_derivatives(tmp_path):
    from mep_cmap import sidecars as S
    rec = str(tmp_path / "rawdata" / "test000.smr")
    saved = S._deriv_root[0]
    try:
        paths = set()
        for root in _three_roots(tmp_path):
            S._deriv_root[0] = root
            paths.add(str(S.sidecar_path(rec, ".smr_config.json")))
    finally:
        S._deriv_root[0] = saved
    assert len(paths) == 1
    p = paths.pop()
    assert os.path.relpath(p, tmp_path).split(os.sep)[0] == "derivatives"
    assert "mep-cmap-analyser" not in p
