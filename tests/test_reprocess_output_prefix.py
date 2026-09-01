"""
Where a batch re-run writes its results.

The prefix a run writes under is derived from the study metadata and the
source file stem, and both rules have drifted between versions. Of the 135
sessions in the study the re-runner was built against, 84 would land on a
DIFFERENT prefix today than the one they were saved under -- some gaining the
stem's own BIDS tokens, some losing a token to deduplication.

Recomputing therefore does not overwrite the old results. It forks them: the
originals stay exactly where the group merge and the analysis scripts already
point, beside a second set nothing reads. The numbers do not change, they
disagree, and the stale branch is the cited one.

app.py and pipeline.py cannot be imported here (matplotlib's Tk backend), so
these read the source.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
APP = (ROOT / "app.py").read_text(encoding="utf-8")
PIPE = (ROOT / "pipeline.py").read_text(encoding="utf-8")


def _between(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


def test_the_pipeline_accepts_an_override():
    assert "bids_prefix_override=None," in PIPE


def test_the_override_replaces_the_computed_prefix():
    body = _between(PIPE, "_bids_prefix = _make_bids_prefix",
                    "def _bids_path")
    assert "if bids_prefix_override:" in body
    assert "_bids_prefix = str(bids_prefix_override)" in body


def test_the_channel_tag_is_still_applied_on_top():
    """
    A multi-channel run tags each pass so the passes do not overwrite each
    other. Pinning the prefix must not cost that, or two channels of a
    re-run would land on one filename.
    """
    body = _between(PIPE, "if bids_prefix_override:", "def _bids_path")
    assert "channel-" in body


def test_the_snapshot_carries_it():
    body = _between(APP, "def _snapshot_analysis_params", "return params")
    assert "bids_prefix_override" in body


def test_the_worker_passes_it_to_the_pipeline():
    body = _between(APP, "run_pipeline(", "self.msg_q.put((\"log\", \"✅")
    assert "bids_prefix_override" in body


def test_the_autosave_is_pinned_too():
    """
    Session JSONs are named by the same rule, so without this the session
    forks exactly as the results do -- and the queue's derivatives_json still
    points at the original, so a second pass restores stale state and undoes
    the first.

    Pinned in session_path(), NOT in _autosave_session. _autosave_session
    computes a bids_prefix of its own and then discards it, calling
    session_path() for the path actually written -- so the pin sat in dead
    code and a real run took the study from 135 sessions to 161 before anyone
    noticed.
    """
    body = _between(APP, "def session_path", "def _autosave_session")
    assert "prefix_override" in body
    assert '_bids_prefix_override' in body


def test_session_path_for_honours_the_override():
    """The rule itself, tested directly rather than by reading app.py."""
    from mep_cmap.bids import session_path_for
    pinned = session_path_for("/study/rawdata/sub-01/x_emg.txt",
                              None, "/study/derivatives",
                              prefix_override="sub-01_ses-01_task-30")
    assert pinned.endswith("sub-01_ses-01_task-30_session.json")


def test_session_path_for_recomputes_without_one():
    from mep_cmap.bids import session_path_for
    plain = session_path_for("/study/rawdata/sub-01/x_emg.txt",
                             None, "/study/derivatives")
    assert plain.endswith("x_emg_session.json")


def test_an_ordinary_run_is_unaffected():
    """
    Read with getattr and a None default, and applied only when truthy, so an
    app that never sets it recomputes exactly as before.
    """
    assert 'getattr(self, "_bids_prefix_override", None)' in APP
    for chunk in APP.split('getattr(self, "_bids_prefix_override"')[1:]:
        assert chunk.lstrip().startswith(", None)")
