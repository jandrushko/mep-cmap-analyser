"""Tests for mep_cmap.archive_superseded."""

import csv
import os

import pytest

from mep_cmap import archive_superseded as AS


def _touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _session(root, sub="sub-01", ses="ses-01"):
    d = os.path.join(root, sub, ses)
    os.makedirs(d, exist_ok=True)
    return d


def _study(root, prefix="sub-01_ses-01_task-rest_measure-CSE",
           flat=True, foldered=True, sub="sub-01"):
    """A session with a claimed prefix, in either or both layouts."""
    d = _session(root, sub=sub)
    _touch(os.path.join(d, f"{prefix}_session.json"), "{}")
    if flat:
        _touch(os.path.join(d, "results", f"{prefix}_trials.csv"))
        _touch(os.path.join(d, "results", f"{prefix}_trials.json"))
    if foldered:
        _touch(os.path.join(d, "results", "trial-level",
                            f"{prefix}_trials.csv"))
        _touch(os.path.join(d, "results", "trial-level",
                            f"{prefix}_trials.json"))
    return d, prefix


# ── prefix parsing ───────────────────────────────────────────────────────────

def test_prefix_of_uses_longest_suffix():
    # _summary_with_outliers.csv also ends with _summary.csv; the longer
    # family must win or the prefix keeps a fragment of the suffix.
    assert AS._prefix_of("sub-01_summary_with_outliers.csv") == "sub-01"
    assert AS._prefix_of("sub-01_summary.csv") == "sub-01"
    assert AS._prefix_of("sub-01_trials_selected.csv") == "sub-01"
    assert AS._prefix_of("sub-01_trials.csv") == "sub-01"


def test_prefix_of_rejects_unknown_names():
    assert AS._prefix_of("notes.txt") == ""
    assert AS._prefix_of("figure.png") == ""


# ── selection ────────────────────────────────────────────────────────────────

def test_flat_copy_archived_when_foldered_exists(tmp_path):
    root = str(tmp_path / "derivatives")
    _study(root)
    plan = AS.plan_archive(root)

    assert plan.n_items == 2
    assert {i.reason for i in plan.items} == {AS.REASON_FLAT}
    assert all(os.sep + "trial-level" + os.sep not in i.path
               for i in plan.items)
    assert len(plan.current) == 2
    assert plan.stale == []


def test_flat_only_session_is_never_archived(tmp_path):
    # The whole point: a session with no current copy must keep its flat one,
    # or the merge drops it instead of double-counting it.
    root = str(tmp_path / "derivatives")
    _study(root, flat=True, foldered=False)
    plan = AS.plan_archive(root)

    assert plan.n_items == 0
    assert len(plan.stale) == 1
    assert len(plan.stale[0].flat_files) == 2
    for p in plan.stale[0].flat_files:
        assert os.path.isfile(p)


def test_unclaimed_prefix_archived_in_either_layout(tmp_path):
    root = str(tmp_path / "derivatives")
    d, _ = _study(root)
    _touch(os.path.join(d, "results", "orphan_task-x_trials.csv"))
    _touch(os.path.join(d, "results", "trial-level",
                        "orphan_task-x_trials.json"))
    plan = AS.plan_archive(root)

    unclaimed = [i for i in plan.items if i.reason == AS.REASON_UNCLAIMED]
    assert len(unclaimed) == 2
    assert {os.path.basename(i.path) for i in unclaimed} == {
        "orphan_task-x_trials.csv", "orphan_task-x_trials.json"}


def test_prefix_is_claimed_only_within_its_own_session(tmp_path):
    # Two participants can carry the same task names. A prefix claimed in one
    # folder must not vouch for a file in another.
    root = str(tmp_path / "derivatives")
    _study(root, prefix="sub-01_ses-01_task-rest_measure-CSE", sub="sub-01")
    other = _session(root, sub="sub-02")
    _touch(os.path.join(other, "sub-02_ses-01_task-rest_measure-CSE_session.json"),
           "{}")
    _touch(os.path.join(other, "results", "trial-level",
                        "sub-01_ses-01_task-rest_measure-CSE_trials.csv"))

    plan = AS.plan_archive(root)
    strays = [i for i in plan.items
              if i.session_dir == other and i.reason == AS.REASON_UNCLAIMED]
    assert len(strays) == 1


def test_folder_without_session_json_is_skipped(tmp_path):
    # Nothing to judge its outputs against, so nothing is archived from it.
    root = str(tmp_path / "derivatives")
    d = _session(root)
    _touch(os.path.join(d, "results", "sub-01_ses-01_task-x_trials.csv"))
    plan = AS.plan_archive(root)
    assert plan.n_items == 0
    assert plan.current == []


def test_missing_root_returns_empty_plan(tmp_path):
    plan = AS.plan_archive(str(tmp_path / "nope"))
    assert plan.n_items == 0 and plan.stale == []


def test_all_result_families_are_considered(tmp_path):
    # Archiving only trials would leave a half-migrated tree.
    root = str(tmp_path / "derivatives")
    d, prefix = _study(root)
    for suffix in ("summary.csv", "onset_methods.csv", "segments.npz"):
        _touch(os.path.join(d, "results", f"{prefix}_{suffix}"))
        _touch(os.path.join(d, "results", AS.RL.family_for(suffix),
                            f"{prefix}_{suffix}"))
    plan = AS.plan_archive(root)
    flat = [os.path.basename(i.path) for i in plan.items
            if i.reason == AS.REASON_FLAT]
    assert f"{prefix}_summary.csv" in flat
    assert f"{prefix}_onset_methods.csv" in flat
    assert f"{prefix}_segments.npz" in flat


# ── moving ───────────────────────────────────────────────────────────────────

def test_apply_moves_and_leaves_current_alone(tmp_path):
    root = str(tmp_path / "derivatives")
    d, prefix = _study(root)
    plan = AS.plan_archive(root)
    arc = str(tmp_path / "superseded" / "run1")

    res = AS.apply_archive(plan, arc)

    assert res["moved"] == 2
    assert res["failed"] == []
    assert not os.path.exists(os.path.join(d, "results",
                                           f"{prefix}_trials.csv"))
    assert os.path.isfile(os.path.join(d, "results", "trial-level",
                                       f"{prefix}_trials.csv"))


def test_archive_keeps_path_relative_to_derivatives(tmp_path):
    root = str(tmp_path / "derivatives")
    d, prefix = _study(root)
    plan = AS.plan_archive(root)
    arc = str(tmp_path / "superseded" / "run1")
    AS.apply_archive(plan, arc)

    landed = os.path.join(arc, "sub-01", "ses-01", "results",
                          f"{prefix}_trials.csv")
    assert os.path.isfile(landed)


def test_archive_never_overwrites_an_earlier_copy(tmp_path):
    root = str(tmp_path / "derivatives")
    d, prefix = _study(root)
    arc = str(tmp_path / "superseded" / "run1")
    AS.apply_archive(AS.plan_archive(root), arc)

    # Same path archived again by a later run.
    _touch(os.path.join(d, "results", f"{prefix}_trials.csv"), "second")
    AS.apply_archive(AS.plan_archive(root), arc)

    base = os.path.join(arc, "sub-01", "ses-01", "results")
    assert os.path.isfile(os.path.join(base, f"{prefix}_trials.csv"))
    assert os.path.isfile(os.path.join(base, f"{prefix}_trials__2.csv"))


def test_apply_is_a_move_not_a_copy(tmp_path):
    root = str(tmp_path / "derivatives")
    d, prefix = _study(root)
    plan = AS.plan_archive(root)
    AS.apply_archive(plan, str(tmp_path / "arc"))
    remaining = os.listdir(os.path.join(d, "results"))
    assert remaining == ["trial-level"]


# ── manifest ─────────────────────────────────────────────────────────────────

def test_manifest_lists_every_move_with_its_reason(tmp_path):
    root = str(tmp_path / "derivatives")
    d, prefix = _study(root)
    _touch(os.path.join(d, "results", "orphan_trials.csv"))
    plan = AS.plan_archive(root)
    arc = str(tmp_path / "arc")
    res = AS.apply_archive(plan, arc)

    with open(res["manifest"], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == plan.n_items
    assert set(rows[0]) == set(AS.MANIFEST_HEADER)
    reasons = {r["Reason"] for r in rows}
    assert reasons == {AS.REASON_FLAT, AS.REASON_UNCLAIMED}
    for r in rows:
        assert not os.path.isabs(r["OriginalPath"])
        assert not os.path.isabs(r["ArchivePath"])


def test_manifest_records_the_prefix_a_file_lost_to(tmp_path):
    root = str(tmp_path / "derivatives")
    _study(root)
    plan = AS.plan_archive(root)
    arc = str(tmp_path / "arc")
    res = AS.apply_archive(plan, arc)
    with open(res["manifest"], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["Prefix"] for r in rows)


def test_manifest_written_before_any_move(tmp_path):
    # An interrupted archive must still say what it intended.
    root = str(tmp_path / "derivatives")
    _study(root)
    plan = AS.plan_archive(root)
    arc = str(tmp_path / "arc")
    path = AS.write_manifest(plan, arc)
    assert os.path.isfile(path)
    for item in plan.items:
        assert os.path.isfile(item.path)


# ── archive root placement ───────────────────────────────────────────────────

def test_default_archive_root_is_outside_derivatives(tmp_path):
    root = str(tmp_path / "study" / "derivatives")
    arc = AS.default_archive_root(root)
    assert os.path.abspath(root) not in os.path.abspath(arc)


def test_session_walk_ignores_an_archive_left_inside(tmp_path):
    root = str(tmp_path / "derivatives")
    _study(root)
    inner = os.path.join(root, "_archive", "20260101T000000",
                         "sub-01", "ses-01")
    _touch(os.path.join(inner, "sub-01_ses-01_task-rest_measure-CSE_session.json"),
           "{}")
    _touch(os.path.join(inner, "results", "stray_trials.csv"))
    plan = AS.plan_archive(root)
    assert all("_archive" not in i.path for i in plan.items)


# ── reporting ────────────────────────────────────────────────────────────────

def test_describe_names_counts_and_stale(tmp_path):
    root = str(tmp_path / "derivatives")
    _study(root)
    _study(root, prefix="sub-01_ses-01_task-hold_measure-CSE",
           flat=True, foldered=False)
    text = AS.describe(AS.plan_archive(root))
    assert "Superseded" in text
    assert "sub-01_ses-01_task-hold_measure-CSE" in text
