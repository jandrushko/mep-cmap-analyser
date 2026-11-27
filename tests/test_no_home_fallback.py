"""
No study open must never mean "use the home folder" (v1.4.12).

A queue file once created in the home folder, with no study set, sat there
for months. Its recordings moved; then opening the BIDS-ify tab with nothing
loaded resolved them by searching the whole home folder, OneDrive included,
and the tool appeared frozen for minutes.
"""

import os
import pathlib

from mep_cmap.dataset_session import (DatasetSession, _searchable_study_root,
                                      _to_absolute)
from mep_cmap.bidsify_state import BidsifyState

PKG = pathlib.Path(__file__).resolve().parents[1] / "mep_cmap"
HOME = os.path.expanduser("~")


# ── nothing is written without a study folder ────────────────────────────────

def test_a_queue_with_no_study_is_never_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds = DatasetSession(derivatives_root="")
    ds.add_file(str(tmp_path / "rec.smr"))
    assert ds.save() is False
    assert not (tmp_path / "derivatives").exists()


def test_bidsify_state_with_no_study_reads_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bidsify_state.json").write_text('{"modality": "PNS"}')
    st = BidsifyState.load_or_create("")
    assert st.modality != "PNS"             # the working folder's file ignored
    st.save()                               # and nothing written, no error
    assert (tmp_path / "bidsify_state.json").read_text() == '{"modality": "PNS"}'


def test_no_home_folder_fallback_remains():
    app = (PKG / "app.py").read_text(encoding="utf-8")
    body = app[app.index("def _get_or_create_dataset"):]
    body = body[:body.index("\n    def ", 10)]
    assert "expanduser" not in body
    tab = (PKG / "bidsify_tab.py").read_text(encoding="utf-8")
    root = tab[tab.index("def _bidsify_state_root"):]
    root = root[:root.index("\n    def ", 10)]
    assert "expanduser" not in root
    refresh = tab[tab.index("def _bidsify_tab_refresh"):]
    assert "No study open" in refresh[:1200]


# ── the queue is adopted once a study is chosen ──────────────────────────────

class _Var:
    def __init__(self, v=""):
        self.v = v

    def get(self):
        return self.v


def test_in_memory_queue_joins_the_study_queue(tmp_path):
    from mep_cmap.app import TMSAnalysisApp
    import types
    stub = types.SimpleNamespace(derivatives_path=_Var(""), _dataset=None)
    ds = TMSAnalysisApp._get_or_create_dataset(stub)
    assert ds.derivatives_root == ""
    ds.add_file(str(tmp_path / "rawdata" / "rec.smr"))
    stub.derivatives_path.v = str(tmp_path)
    ds2 = TMSAnalysisApp._get_or_create_dataset(stub)
    assert ds2.derivatives_root == str(tmp_path)
    assert [os.path.basename(f.path) for f in ds2.files] == ["rec.smr"]
    assert ds2.save() is True


# ── a moved file is only searched for inside a study ─────────────────────────

def test_search_scope_refuses_wide_folders(tmp_path):
    assert not _searchable_study_root("")
    assert not _searchable_study_root(HOME)
    assert not _searchable_study_root(os.path.dirname(HOME))      # above home
    assert not _searchable_study_root(os.path.splitdrive(HOME)[0] + os.sep
                                      if os.path.splitdrive(HOME)[0] else os.sep)
    assert not _searchable_study_root(
        os.path.join(HOME, "OneDrive - Northumbria University - Production Azure AD"))
    assert _searchable_study_root(str(tmp_path / "Pilot data"))


def test_moved_file_still_found_inside_a_study(tmp_path):
    study = tmp_path / "Pilot data"
    moved = study / "rawdata" / "sub-01" / "rec.smr"
    moved.parent.mkdir(parents=True)
    moved.write_text("x")
    assert _to_absolute("old/place/rec.smr", str(study)) == str(moved)


def test_no_search_from_the_home_folder(monkeypatch, tmp_path):
    walked = []
    monkeypatch.setattr(os, "walk", lambda *a, **k: walked.append(a) or iter(()))
    _to_absolute("gone/rec.smr", HOME)
    assert walked == []
