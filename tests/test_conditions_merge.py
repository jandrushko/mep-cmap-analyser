"""
Combining conditions back into one.

The inverse of a split, and the reason it exists is that undo only reaches the
last action: once the analyst has done anything else, or reopened the file,
the halves of a mis-split condition were permanent. The only route back was
Reload from file, which discards every other edit to the table.
"""

import pytest

from mep_cmap import conditions as C
from mep_cmap.conditions import ConditionRow, ConditionError


def _rows():
    return [ConditionRow(stim_type="A", condition="pre", trials=(0, 1, 2)),
            ConditionRow(stim_type="A", condition="post", trials=(3, 4)),
            ConditionRow(stim_type="B", condition="", trials=(0, 1))]


def test_a_split_can_be_undone_after_the_fact():
    """The property that matters: split then merge returns the original."""
    start = [ConditionRow(stim_type="A", condition="A", trials=(0, 1, 2, 3))]
    split = C.split_row(start, 0, (2, 3), new_condition="post")
    assert len(split) == 2
    merged = C.merge_rows(split, [0, 1])
    assert len(merged) == 1
    assert merged[0].trials == (0, 1, 2, 3)
    assert merged[0].stim_type == "A"


def test_every_trial_survives_the_merge():
    out = C.merge_rows(_rows(), [0, 1])
    assert out[0].trials == (0, 1, 2, 3, 4)


def test_the_other_rows_are_left_alone():
    out = C.merge_rows(_rows(), [0, 1])
    assert len(out) == 2
    assert out[1].stim_type == "B" and out[1].trials == (0, 1)


def test_the_merged_row_takes_the_first_name():
    """What someone undoing a split expects: the original row's name."""
    assert C.merge_rows(_rows(), [0, 1])[0].condition == "pre"


def test_an_explicit_name_wins():
    assert C.merge_rows(_rows(), [0, 1], condition="all")[0].condition == "all"


def test_an_unnamed_first_row_falls_back_to_a_name_that_is_set():
    """Combining an unnamed row with a named one must not lose the only label."""
    rows = [ConditionRow(stim_type="A", condition="", trials=(0,)),
            ConditionRow(stim_type="A", condition="post", trials=(1,))]
    assert C.merge_rows(rows, [0, 1])[0].condition == "post"


def test_the_merged_row_sits_where_the_first_one_was():
    rows = [ConditionRow(stim_type="B", condition="x", trials=(0,)),
            ConditionRow(stim_type="A", condition="pre", trials=(1,)),
            ConditionRow(stim_type="A", condition="post", trials=(2,))]
    out = C.merge_rows(rows, [1, 2])
    assert [r.condition for r in out] == ["x", "pre"]


# ── what it refuses ──────────────────────────────────────────────────────────

def test_two_stimulus_types_cannot_be_combined():
    """
    Trial indices are numbered per stimulus type, so a row holding both would
    collide indices that mean different trials, and would claim responses to
    different stimuli are one condition.
    """
    with pytest.raises(ConditionError) as exc:
        C.merge_rows(_rows(), [0, 2])
    assert "different stimulus types" in str(exc.value)


def test_one_row_is_not_a_merge():
    with pytest.raises(ConditionError):
        C.merge_rows(_rows(), [0])


def test_a_row_that_does_not_exist_is_refused():
    with pytest.raises(ConditionError):
        C.merge_rows(_rows(), [0, 99])


def test_a_trial_in_two_rows_appears_once():
    """
    Splitting cannot produce this, but a hand-set trial list can, and a
    duplicated trial would be double-counted in every summary downstream.
    """
    rows = [ConditionRow(stim_type="A", condition="a", trials=(0, 1, 2)),
            ConditionRow(stim_type="A", condition="b", trials=(2, 3))]
    assert C.merge_rows(rows, [0, 1])[0].trials == (0, 1, 2, 3)


def test_the_input_rows_are_not_mutated():
    """The undo stack holds the previous list; mutating it would corrupt it."""
    rows = _rows()
    before = [(r.stim_type, r.condition, r.trials) for r in rows]
    C.merge_rows(rows, [0, 1])
    assert [(r.stim_type, r.condition, r.trials) for r in rows] == before
