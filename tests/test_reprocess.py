"""
Planning a re-run of an already-analysed recording.

The rules here were derived from a real study of 135 sessions, and two of them
are counter-intuitive enough that they will be "corrected" back into bugs by
anyone who reasons about the problem from first principles instead of looking
at data. Both are pinned below.
"""

import os

import pytest

from mep_cmap import reprocess as R

CHANS = ["Ch 1 - Handgrip 1", "Ch 2 - Handgrip 2", "Ch 3 - MEP",
         "Ch 4 - Stim", "Ch 5 -Pre EMG", "Ch 6 - EMG 2"]


def _sess(**kw):
    # settings.outlier_review is False in all 135 sessions of the study this
    # was built against, which is what makes them re-runnable at all; a
    # session without it is refused, and has its own test below.
    base = {"channel_choice": "Ch 3 - MEP", "channel_idx": 2,
            "marker_choice": "A", "analyse_channels": None,
            "settings": {"outlier_review": False}}
    base.update(kw)
    return base


# ── the two counter-intuitive rules ──────────────────────────────────────────

def test_a_name_index_disagreement_is_reported_not_resolved():
    """
    THE rule. In the study this was built against, 77 sessions named
    'Ch 1 - Handgrip 1' while their index pointed at 'Ch 3 - MEP', and the
    saved figures show the analyses ran on MEP. Trusting the NAME -- the
    obvious rule, since a position means nothing after a re-export -- would
    have re-run 77 files against a handgrip force trace and produced a
    complete, plausible analysis of the wrong signal.

    Trusting the index instead is the same mistake facing the other way. So
    neither is chosen automatically.
    """
    status, resolved, note = R.resolve_channels(
        _sess(channel_choice="Ch 1 - Handgrip 1", channel_idx=2), CHANS)
    assert status == R.CONFLICT
    assert resolved is None
    assert "Ch 1 - Handgrip 1" in note and "Ch 3 - MEP" in note


def test_an_empty_analyse_channels_is_single_channel_not_unknown():
    """
    130 of 135 sessions have nothing here, because the set is only populated
    when the assignment dialogue picks more than one. Reading it as "unknown"
    and skipping would refuse almost every file for the wrong reason.
    """
    status, resolved, _ = R.resolve_channels(_sess(analyse_channels=None),
                                             CHANS)
    assert status == R.OK
    assert resolved["analyse_channels"] == [2]


# ── channel resolution ───────────────────────────────────────────────────────

def test_agreement_resolves_cleanly():
    status, resolved, note = R.resolve_channels(_sess(), CHANS)
    assert status == R.OK and note == ""
    assert resolved == {"channel_choice": "Ch 3 - MEP", "channel_idx": 2,
                        "analyse_channels": [2]}


def test_a_multi_channel_selection_is_kept_and_includes_the_analysed_one():
    _, resolved, _ = R.resolve_channels(
        _sess(analyse_channels=[2, 5]), CHANS)
    assert resolved["analyse_channels"] == [2, 5]


def test_an_out_of_range_saved_channel_is_dropped_not_carried():
    _, resolved, _ = R.resolve_channels(
        _sess(analyse_channels=[2, 99]), CHANS)
    assert resolved["analyse_channels"] == [2]


def test_a_name_absent_from_the_file_with_a_usable_index_is_a_conflict():
    status, _, note = R.resolve_channels(
        _sess(channel_choice="Ch 9 - Gone", channel_idx=2), CHANS)
    assert status == R.CONFLICT and "Ch 3 - MEP" in note


def test_a_name_absent_from_the_file_with_no_usable_index_is_a_skip():
    status, _, _ = R.resolve_channels(
        _sess(channel_choice="Ch 9 - Gone", channel_idx=99), CHANS)
    assert status == R.SKIP


def test_a_recording_with_no_channels_is_a_skip():
    assert R.resolve_channels(_sess(), [])[0] == R.SKIP


def test_a_session_with_no_channel_at_all_is_a_skip():
    assert R.resolve_channels(
        _sess(channel_choice="", channel_idx=None), CHANS)[0] == R.SKIP


# ── what plan_file refuses ───────────────────────────────────────────────────

def test_a_missing_marker_choice_is_a_skip():
    plan = R.plan_file(_sess(marker_choice=""), CHANS, path="x.txt")
    assert plan["status"] == R.SKIP
    assert "marker_choice" in plan["notes"][-1]


def test_a_format_needing_a_unit_without_one_saved_is_a_skip():
    """
    Every amplitude in the file scales by this. The suggestion the dialogue
    would offer is good enough for a person looking at the evidence and not
    good enough to apply silently across a study -- which is why the trial
    files carry Units_Assumed at all.
    """
    plan = R.plan_file(_sess(), CHANS, needs_unit=True, path="x.mat")
    assert plan["status"] == R.SKIP
    assert "amplitude unit" in plan["notes"][-1]


def test_a_saved_unit_is_accepted():
    plan = R.plan_file(_sess(emg_unit="mV"), CHANS, needs_unit=True,
                       path="x.mat")
    assert plan["status"] == R.OK


def test_a_session_whose_outlier_review_was_on_is_a_skip():
    """
    Which flagged trials the analyst kept is not saved anywhere the pipeline
    reads back: pipeline_review_outliers returns no rejections at all when
    review is off. Re-running such a file would put every removed trial back
    into the mean and label it "Not flagged" -- a changed published number
    with nothing recording that it changed.
    """
    plan = R.plan_file(_sess(settings={"outlier_review": True}), CHANS,
                       path="x.txt")
    assert plan["status"] == R.SKIP
    assert "outlier review" in plan["notes"][-1]


def test_an_unrecorded_outlier_review_setting_is_a_skip():
    """
    Absent means ON. The variable defaults to True where it is constructed and
    True again where a session restores it, so a session that never recorded
    it opens the dialogue on the re-run.
    """
    _s = _sess()
    del _s["settings"]
    assert R.plan_file(_s, CHANS, path="x.txt")["status"] == R.SKIP


def test_inspector_exclusions_do_not_trigger_that_refusal():
    """
    A different mechanism, and one that DOES survive: exclusions live in
    segments_metadata, which the session carries. 13 of the 135 sessions in
    the study hold 41 of them between them, and every one is re-runnable.
    """
    plan = R.plan_file(
        _sess(segments_metadata={"A:1": {"exclude": True}}), CHANS,
        path="x.txt")
    assert plan["status"] == R.OK


def test_a_windows_path_resolves_on_posix_without_being_called_moved():
    """
    Sessions store a path relative to the study root, written with the
    separator of the machine that ran the analysis. os.path.join does not
    translate, so on POSIX "rawdata\\sub-01\\x.txt" becomes one absurd
    filename that never exists.

    The basename fallback then finds the file anyway, which is what hid this:
    on the real study it reported 104 of 106 recordings as having MOVED when
    almost none had.
    """
    stored = r"rawdata\sub-01\second-session\tms\x.txt"
    want = os.path.join("/study", "rawdata", "sub-01", "second-session",
                        "tms", "x.txt")
    path, note = R.locate_recording(stored, "/study", {},
                                    isfile=lambda p: p == want)
    assert path == want
    assert note == ""          # found where the session said, so not "moved"


def test_the_recorded_path_beats_an_ambiguous_basename():
    """
    sub-y001 has a first-session and a second-session holding identically
    named files. Its sessions name the right one; only the separators stopped
    it resolving, and the file was refused as ambiguous instead.
    """
    stored = r"rawdata\sub-y001\second-session\tms\x.txt"
    want = os.path.join("/study", "rawdata", "sub-y001", "second-session",
                        "tms", "x.txt")
    recordings = {"x.txt": ["/study/rawdata/sub-y001/first-session/tms/x.txt",
                            want]}
    path, _ = R.locate_recording(stored, "/study", recordings,
                                 isfile=lambda p: p == want)
    assert path == want


def test_the_literal_path_is_tried_first():
    """
    A POSIX filename may legitimately contain a backslash, and where the
    recorded path resolves as written it is the better evidence.
    """
    stored = r"rawdata\odd\name.txt"
    literal = os.path.join("/study", stored)
    path, note = R.locate_recording(stored, "/study", {},
                                    isfile=lambda p: p == literal)
    assert path == literal
    assert note == ""


def test_a_genuinely_missing_recording_is_still_refused():
    """The separator repair must not turn a missing file into a found one."""
    path, note = R.locate_recording(r"young\sub-y002\x.txt", "/study", {},
                                    isfile=lambda p: False)
    assert path is None
    assert "not in this study" in note


def test_an_unlocatable_recording_is_a_skip_before_anything_else():
    plan = R.plan_file(_sess(), CHANS, path=None, path_note="gone.")
    assert plan["status"] == R.SKIP and plan["notes"] == ["gone."]


def test_a_moved_recording_is_repaired_not_merely_ok():
    """Where a file was found is worth reporting even when nothing else changed."""
    plan = R.plan_file(_sess(), CHANS, path="new/x.txt",
                       path_note="the recording has moved.")
    assert plan["status"] == R.REPAIRED


# ── sessions that disagree about which recording they describe ───────────────

def test_a_recording_claimed_by_several_sessions_is_found():
    """
    135 sessions in the study name only 122 distinct recordings. One rest
    recording is claimed by SIX sessions carrying four different task labels,
    so at most one of them is right. Run unattended, that recording would be
    analysed six times and filed under names it is not.
    """
    shared = R.sessions_sharing_a_recording([
        ("task-30", r"C:\s\rawdata\sub-01\rest.txt"),
        ("task-60", r"C:\s\rawdata\sub-01\rest.txt"),
        ("task-REST", r"C:\s\rawdata\sub-01\rest.txt"),
        ("mwave", r"C:\s\rawdata\sub-01\mwave.txt"),
    ])
    assert list(shared) == ["rest.txt"]
    assert sorted(shared["rest.txt"]) == ["task-30", "task-60", "task-REST"]


def test_a_recording_claimed_once_is_not_reported():
    shared = R.sessions_sharing_a_recording([
        ("a", r"C:\s\one.txt"), ("b", r"C:\s\two.txt")])
    assert shared == {}


def test_the_claim_is_matched_across_platforms():
    """
    Sessions are written on whichever machine ran the analysis, so a Windows
    path can be read on macOS and the reverse.
    """
    shared = R.sessions_sharing_a_recording([
        ("a", r"C:\study\rawdata\x.txt"),
        ("b", "/Users/j/study/rawdata/x.txt"),
    ])
    assert sorted(shared["x.txt"]) == ["a", "b"]


def test_the_refusal_note_says_how_many_and_which():
    note = R.shared_recording_note(r"C:\s\rest.txt", ["a", "b", "c"])
    assert "3" in note and "rest.txt" in note


# ── locating a recording that has moved ──────────────────────────────────────

def test_a_moved_recording_is_found_by_name():
    """
    74 of 135 sessions in the study pointed at a path that no longer existed,
    because the study was reorganised. Those files are not missing and their
    sessions are not stale; only the folder above them changed.
    """
    idx = R.index_recordings([r"C:\s\rawdata\sub-01\a.txt"])
    path, note = R.locate_recording(r"old\young\a.txt", r"C:\s", idx,
                                    isfile=lambda p: False)
    assert path == r"C:\s\rawdata\sub-01\a.txt"
    assert "moved" in note


def test_two_recordings_of_the_same_name_are_refused():
    """
    Attaching one participant's manual marker edits to another participant's
    recording produces a complete, plausible analysis of the wrong data.
    """
    idx = R.index_recordings([r"C:\s\rawdata\sub-01\a.txt",
                              r"C:\s\rawdata\sub-02\a.txt"])
    path, note = R.locate_recording("a.txt", r"C:\s", idx,
                                    isfile=lambda p: False)
    assert path is None and "cannot be told apart" in note


def test_the_stored_path_wins_when_it_still_exists():
    idx = R.index_recordings([r"C:\s\rawdata\elsewhere\a.txt"])
    path, note = R.locate_recording(r"rawdata\sub-01\a.txt", r"C:\s", idx,
                                    isfile=lambda p: True)
    assert path.endswith(r"sub-01\a.txt") and note == ""


def test_a_windows_path_is_split_on_a_mac():
    """
    Sessions are written on whichever machine ran the analysis. os.path.
    basename only splits on the current platform's separator, so a Windows
    path read on macOS comes back whole and matches nothing.
    """
    assert R._basename(r"rawdata\sub-01\a.txt") == "a.txt"
    assert R._basename("rawdata/sub-01/a.txt") == "a.txt"


# ── grouping conflicts ───────────────────────────────────────────────────────

def test_identical_conflicts_become_one_question():
    """
    77 identical questions get answered by clicking through them. The study
    reduced to two groups: one for MEP files, one for M-wave.
    """
    mwave = [c.replace("Ch 3 - MEP", "Ch 3 - Mwave") for c in CHANS]
    s = _sess(channel_choice="Ch 1 - Handgrip 1")
    items = ([(f"mep{i}", s, CHANS) for i in range(62)]
             + [(f"mw{i}", s, mwave) for i in range(15)])
    groups = R.group_conflicts(items)
    assert len(groups) == 2
    assert sorted(len(v) for v in groups.values()) == [15, 62]


def test_a_conflict_group_offers_both_readings():
    sig = R.conflict_signature(_sess(channel_choice="Ch 1 - Handgrip 1"),
                               CHANS)
    d = R.describe_conflict(sig)
    assert d["name_points_at"] == ("Ch 1 - Handgrip 1", 0)
    assert d["index_points_at"] == ("Ch 3 - MEP", 2)


def test_answering_by_index_takes_the_channel_the_index_points_at():
    out = R.apply_conflict_answer(
        _sess(channel_choice="Ch 1 - Handgrip 1"), CHANS, R.BY_INDEX)
    assert out == {"channel_choice": "Ch 3 - MEP", "channel_idx": 2,
                   "analyse_channels": [2]}


def test_answering_by_name_takes_the_named_channel():
    out = R.apply_conflict_answer(
        _sess(channel_choice="Ch 1 - Handgrip 1"), CHANS, R.BY_NAME)
    assert out == {"channel_choice": "Ch 1 - Handgrip 1", "channel_idx": 0,
                   "analyse_channels": [0]}


def test_declining_to_answer_leaves_the_file_alone():
    """An analyst who does not know must be able to say so."""
    assert R.apply_conflict_answer(_sess(), CHANS, R.UNRESOLVED) is None


# ── what the queue is told ───────────────────────────────────────────────────

_STATUSES = {"needs_review": "needs_review"}


def test_a_conflict_marks_the_file_for_review_in_the_queue():
    """
    A run report is read once and closed. Six months later a file that was
    quietly not reprocessed is indistinguishable from one that was.
    """
    plan = {"status": R.CONFLICT, "notes": ["names X but index Y."]}
    assert R.queue_status_for(plan, _STATUSES) == "needs_review"
    assert R.REVIEW_FLAG in R.review_flag_for(plan)


def test_a_skip_does_not_change_the_queue_status():
    """
    It is a statement about the session, not about the analysis on disk.
    Marking a complete file as needing review because its recording moved
    would be misleading.
    """
    assert R.queue_status_for({"status": R.SKIP, "notes": []},
                              _STATUSES) is None
    assert R.review_flag_for({"status": R.SKIP}) is None


def test_a_clean_file_is_left_alone():
    assert R.queue_status_for({"status": R.OK, "notes": []},
                              _STATUSES) is None


def test_summarise_counts_every_status():
    plans = [{"status": R.OK}, {"status": R.OK}, {"status": R.CONFLICT},
             {"status": R.SKIP}]
    assert R.summarise(plans) == {R.OK: 2, R.REPAIRED: 0, R.CONFLICT: 1,
                                  R.SKIP: 1}
