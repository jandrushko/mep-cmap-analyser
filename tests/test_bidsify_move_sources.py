"""
Moving originals into sourcedata instead of copying them.

This is the only step in BIDS-ify that destroys anything, and a native
recording is the one artefact the tool cannot regenerate. Every rule here
answers "could this delete the only copy?".
"""

import os

import pytest

from mep_cmap.bidsify import (DatasetLayout, Plan, _remove_verified_source)


class _Item:
    def __init__(self, source_path):
        self.source_path = source_path


class _PF:
    """The parts of PlannedFile that the move step reads."""
    def __init__(self, source_path, sourcedata_path):
        self.item = _Item(source_path)
        self.sourcedata_path = sourcedata_path


def _pair(tmp_path, data=b"recording-bytes", copy=True, copy_data=None):
    src = tmp_path / "raw" / "a.smr"
    dst = tmp_path / "sourcedata" / "a.smr"
    src.parent.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(data)
    if copy:
        dst.write_bytes(copy_data if copy_data is not None else data)
    return _PF(str(src), str(dst)), src, dst


# ── the happy path ───────────────────────────────────────────────────────────

def test_a_verified_original_is_removed_and_the_archive_reported(tmp_path):
    pf, src, dst = _pair(tmp_path)
    got = _remove_verified_source(pf, log=lambda _m: None)
    assert got == str(dst)
    assert not src.exists()
    assert dst.exists()


# ── the refusals ─────────────────────────────────────────────────────────────

def test_nothing_is_deleted_when_the_copy_is_absent(tmp_path):
    pf, src, _dst = _pair(tmp_path, copy=False)
    logs = []
    assert _remove_verified_source(pf, log=logs.append) == ""
    assert src.exists(), "the only copy was deleted"
    assert any("not there" in m for m in logs)


def test_a_short_copy_is_not_treated_as_a_copy(tmp_path):
    """
    A copy interrupted by a full disk or a sync client leaves a file that
    exists and is not the recording. Existence is not the test; size is.
    """
    pf, src, _dst = _pair(tmp_path, data=b"x" * 5000, copy_data=b"x" * 12)
    logs = []
    assert _remove_verified_source(pf, log=logs.append) == ""
    assert src.exists()
    assert any("bytes against" in m for m in logs)


def test_a_source_that_is_already_the_archive_is_left_alone(tmp_path):
    """Deleting it would remove the copy the move exists to preserve."""
    p = tmp_path / "sourcedata" / "a.smr"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"data")
    pf = _PF(str(p), str(p))
    assert _remove_verified_source(pf, log=lambda _m: None) == ""
    assert p.exists()


def test_a_failure_to_delete_is_reported_not_raised(tmp_path):
    """The conversion succeeded. Failing the file because a delete was
    refused would report a good recording as bad."""
    pf, _src, _dst = _pair(tmp_path)
    pf.item.source_path = str(tmp_path / "raw")     # a directory, not a file
    logs = []
    assert _remove_verified_source(pf, log=logs.append) == ""
    assert any("original kept" in m for m in logs)


# ── the plan carries the choice ──────────────────────────────────────────────

def test_copying_is_the_default(tmp_path):
    plan = Plan(layout=DatasetLayout(rawdata_root=str(tmp_path / "rawdata")))
    assert plan.move_sources is False


def test_the_preview_says_when_originals_will_be_deleted(tmp_path):
    """The preview is the last thing shown before anything is written, and
    this is the only irreversible part of the run."""
    lay = DatasetLayout(rawdata_root=str(tmp_path / "rawdata"))
    assert "DELETED" not in Plan(layout=lay).preview_text()
    assert "DELETED" in Plan(layout=lay, move_sources=True).preview_text()

# ── the state remembers it, per study ────────────────────────────────────────

def test_the_setting_survives_a_reload(tmp_path):
    from mep_cmap.bidsify_state import BidsifyState
    st = BidsifyState.load_or_create(str(tmp_path))
    st.move_sources = True
    st.save()
    assert BidsifyState.load_or_create(str(tmp_path)).move_sources is True


def test_a_state_file_written_before_the_option_existed_defaults_to_copy(tmp_path):
    """A destructive setting must never be inherited by a study that was
    never asked the question."""
    import json
    from mep_cmap.bidsify_state import BidsifyState
    (tmp_path / "bidsify_state.json").write_text(
        json.dumps({"modality": "TMS", "container": "EDF"}), encoding="utf-8")
    assert BidsifyState.load_or_create(str(tmp_path)).move_sources is False


# ── a moved file keeps everything recorded about it ──────────────────────────

def test_repoint_carries_the_record_to_the_new_path(tmp_path):
    """
    Records are keyed by path. Without this the reviewed flag, marker names
    and stimulation set assignment silently revert to unset, and the recording
    presents itself as never having been touched.
    """
    from mep_cmap.bidsify_state import BidsifyState
    st = BidsifyState.load_or_create(str(tmp_path))
    old = str(tmp_path / "raw" / "a.smr")
    new = str(tmp_path / "sourcedata" / "a.smr")
    rec = st.record_for(old)
    rec.reviewed = True
    rec.marker_names = ["A"]
    rec.code_sets = {"A": "spTMS"}

    st.repoint(old, new)
    moved = st.record_for(new, create=False)
    assert moved is not None and moved.reviewed is True
    assert moved.code_sets == {"A": "spTMS"}
    # And the old key no longer resolves to a record.
    assert st.record_for(old, create=False) is None


def test_repointing_an_unknown_file_is_harmless(tmp_path):
    from mep_cmap.bidsify_state import BidsifyState
    st = BidsifyState.load_or_create(str(tmp_path))
    st.repoint(str(tmp_path / "nope.smr"), str(tmp_path / "also-nope.smr"))
