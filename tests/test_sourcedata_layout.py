"""
Where native copies go, for each of the two BIDS study layouts.

Both layouts are valid; mixing them is not. sourcedata/ inside rawdata/ while
derivatives/ is a sibling leaves native files inside the dataset a validator
walks and inside every recursive glob of rawdata/**.
"""

import os

from mep_cmap.bidsify import DatasetLayout, default_sourcedata_root


def test_the_sibling_layout_puts_sourcedata_beside_rawdata(tmp_path):
    """rawdata/, sourcedata/ and derivatives/ end up at the same level."""
    raw = tmp_path / "rawdata"
    raw.mkdir()
    got = default_sourcedata_root(str(raw))
    assert os.path.normpath(got) == os.path.normpath(str(tmp_path / "sourcedata"))
    assert os.path.dirname(os.path.normpath(got)) == os.path.normpath(str(tmp_path))


def test_the_canonical_layout_keeps_sourcedata_inside_the_dataset(tmp_path):
    """When the raw dataset IS the study root, sourcedata/ belongs within it."""
    root = tmp_path / "my_study"
    root.mkdir()
    got = default_sourcedata_root(str(root))
    assert os.path.normpath(got) == os.path.normpath(str(root / "sourcedata"))


def test_an_existing_sourcedata_folder_always_wins(tmp_path):
    """
    A study half-converted under the old default and half under the new one is
    worse than one consistently in the old place: the native copy of a
    recording would sit in whichever folder existed the day it was BIDS-ified,
    and nothing would say which.
    """
    raw = tmp_path / "rawdata"
    (raw / "sourcedata").mkdir(parents=True)
    got = default_sourcedata_root(str(raw))
    assert os.path.normpath(got) == os.path.normpath(str(raw / "sourcedata"))


def test_a_trailing_separator_does_not_change_the_answer(tmp_path):
    raw = tmp_path / "rawdata"
    raw.mkdir()
    assert (os.path.normpath(default_sourcedata_root(str(raw) + os.sep))
            == os.path.normpath(default_sourcedata_root(str(raw))))


def test_the_folder_name_is_matched_case_insensitively(tmp_path):
    """Windows hands back RawData as readily as rawdata."""
    raw = tmp_path / "RawData"
    raw.mkdir()
    got = default_sourcedata_root(str(raw))
    assert os.path.normpath(got) == os.path.normpath(str(tmp_path / "sourcedata"))


# ── through the layout ───────────────────────────────────────────────────────

def test_the_layout_resolves_it_when_not_given(tmp_path):
    raw = tmp_path / "rawdata"
    raw.mkdir()
    lay = DatasetLayout(rawdata_root=str(raw))
    assert os.path.normpath(lay.sourcedata_root) \
        == os.path.normpath(str(tmp_path / "sourcedata"))


def test_an_explicit_sourcedata_root_is_never_second_guessed(tmp_path):
    raw = tmp_path / "rawdata"
    raw.mkdir()
    chosen = str(tmp_path / "originals")
    lay = DatasetLayout(rawdata_root=str(raw), sourcedata_root=chosen)
    assert lay.sourcedata_root == chosen
