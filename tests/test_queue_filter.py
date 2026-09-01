"""
The file-queue filter.

Filtering HIDES rows rather than greying them out, which means a hidden file
cannot be selected, which means "Run selected" can never act on a file the
analyst cannot see. That is the property worth protecting: a long batch run
started against invisible files is expensive to discover afterwards.
"""

import pytest

from mep_cmap.app import (QUEUE_FILTER_ALL, queue_row_matches,
                          _queue_type_label, queue_type_choice_label,
                          _QUEUE_TYPE_LABELS)
from mep_cmap.dataset_session import (FileEntry, STATUS_NOT_STARTED,
                                      STATUS_COMPLETE, STATUS_LABELS)


def _fe(path, status=STATUS_NOT_STARTED, label=""):
    fe = FileEntry(path=path, label=label or path.rsplit("/", 1)[-1])
    fe.status = status
    return fe


SMR = _fe("C:/study/rawdata/sub-002/ses-2/Spike/sub-002_limb-left_250414.smr")
TXT = _fe("C:/study/rawdata/sub-009/ses-2/sub-009_limb-right_251031.txt",
          status=STATUS_COMPLETE)


# ── the type label ───────────────────────────────────────────────────────────

def test_the_type_label_is_the_one_the_column_shows():
    assert _queue_type_label("/x/a.smr") == "SMR"
    assert _queue_type_label("/x/a.vhdr") == "BrainVision"
    assert _queue_type_label("/x/a.SMR") == "SMR", "Windows yields .SMR too"


def test_an_unknown_extension_still_gets_a_label():
    assert _queue_type_label("/x/a.xyz") == "XYZ"
    assert _queue_type_label("/x/README") == "—"


# ── no filter ────────────────────────────────────────────────────────────────

def test_the_default_shows_everything():
    for fe in (SMR, TXT):
        assert queue_row_matches(fe)


# ── status ───────────────────────────────────────────────────────────────────

def test_filtering_by_status_keeps_only_that_status():
    not_started = STATUS_LABELS[STATUS_NOT_STARTED]
    assert queue_row_matches(SMR, status=not_started)
    assert not queue_row_matches(TXT, status=not_started)


def test_the_filter_matches_the_label_the_analyst_sees():
    """The dropdown offers display labels, and the row shows display labels.
    Comparing against the raw status code would match nothing."""
    assert queue_row_matches(TXT, status=STATUS_LABELS[STATUS_COMPLETE])


# ── type ─────────────────────────────────────────────────────────────────────

def test_filtering_by_type_keeps_only_that_type():
    assert queue_row_matches(SMR, ftype="SMR")
    assert not queue_row_matches(SMR, ftype="TXT")
    assert queue_row_matches(TXT, ftype="TXT")


# ── free text ────────────────────────────────────────────────────────────────

def test_the_text_term_finds_a_subject():
    assert queue_row_matches(SMR, term="sub-002")
    assert not queue_row_matches(TXT, term="sub-002")


def test_the_text_term_finds_a_limb_or_a_date():
    assert queue_row_matches(SMR, term="limb-left")
    assert not queue_row_matches(SMR, term="limb-right")
    assert queue_row_matches(SMR, term="250414")


def test_the_text_term_matches_any_part_of_the_path():
    """So the analyst need not know which column holds the thing."""
    assert queue_row_matches(SMR, term="Spike")


def test_the_text_term_ignores_case_and_surrounding_space():
    assert queue_row_matches(SMR, term="  SUB-002 ")


def test_an_empty_term_is_not_a_filter():
    assert queue_row_matches(SMR, term="   ")


# ── combining ────────────────────────────────────────────────────────────────

def test_the_filters_are_combined_with_and():
    not_started = STATUS_LABELS[STATUS_NOT_STARTED]
    assert queue_row_matches(SMR, status=not_started, ftype="SMR",
                             term="sub-002")
    # Right status and type, wrong subject.
    assert not queue_row_matches(SMR, status=not_started, ftype="SMR",
                                 term="sub-009")
    # Right subject and type, wrong status.
    assert not queue_row_matches(SMR, status=STATUS_LABELS[STATUS_COMPLETE],
                                 ftype="SMR", term="sub-002")


def test_all_on_one_axis_does_not_disable_the_others():
    assert not queue_row_matches(TXT, status=QUEUE_FILTER_ALL, ftype="SMR")


# ── the type dropdown carries a count ────────────────────────────────────────

@pytest.mark.parametrize("choice,expected", [
    ("MAT (24)", "MAT"), ("BDF (0)", "BDF"), ("BrainVision (2)", "BrainVision"),
    ("EDF", "EDF"), ("All", "All"), ("", ""),
])
def test_the_count_is_stripped_before_matching(choice, expected):
    assert queue_type_choice_label(choice) == expected


def test_a_label_that_merely_ends_in_a_bracket_is_left_alone():
    """Only a trailing count is display decoration. Anything else is the name."""
    assert queue_type_choice_label("ODD (x)") == "ODD (x)"
    assert queue_type_choice_label("ODD ()") == "ODD ()"


def test_filtering_works_through_the_displayed_choice():
    """
    The dropdown shows 'SMR (1)' but the row shows 'SMR'. Comparing the two
    directly would match nothing, and the queue would empty on every type
    filter.
    """
    assert queue_row_matches(SMR, ftype="SMR (1)")
    assert not queue_row_matches(TXT, ftype="SMR (1)")


def test_every_readable_format_can_be_named_in_the_filter():
    """
    A supported format absent from the dropdown reads as one the tool cannot
    open, which is the wrong conclusion and one that has been drawn. BDF is
    the case that prompted this.
    """
    labels = set(_QUEUE_TYPE_LABELS.values())
    assert {"BDF", "EDF", "ACQ", "ADIBIN", "SMR", "TXT", "MAT", "CSV",
            "BrainVision"} <= labels
    bdf = _fe("C:/study/rawdata/sub-01/sub-01_emg.bdf")
    assert _queue_type_label(bdf.path) == "BDF"
    assert queue_row_matches(bdf, ftype="BDF (1)")


def test_an_empty_type_matches_nothing_rather_than_everything():
    """An entry showing (0) must filter to an empty queue, not to no filter."""
    assert not queue_row_matches(SMR, ftype="BDF (0)")
    assert not queue_row_matches(TXT, ftype="BDF (0)")
