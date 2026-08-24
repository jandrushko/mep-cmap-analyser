"""
Tests for describing what a crop selection contains.

The dialogue showed the ranges as times -- "[12.40 - 88.10]" -- which says
nothing about the events inside them. A recruitment curve, a block of 120% aMT
trials and an iSP block can sit in one recording with no visible boundary
between them, and a collaborator reported counting tick marks by eye to work
out which events were the first 90.
"""

import pathlib

from mep_cmap.selection_summary import (format_selection, summarise_selection)

# 120 events: A 1-45, C 46-75, A 76-120 -- so A's file numbering is
# discontinuous in time, which is the case that catches naive index arithmetic.
STIM = {"A": [float(i) for i in list(range(1, 46)) + list(range(76, 121))],
        "C": [float(i) for i in range(46, 76)]}


def test_counts_and_indices_are_per_type():
    sel = summarise_selection(STIM, [(0.5, 45.5)])
    assert len(sel) == 1
    a = sel[0]
    assert a.stim_type == "A"
    assert (a.n_selected, a.n_total) == (45, 90)
    assert a.spans == [(1, 45)]


def test_indices_are_positions_within_the_type_not_the_file():
    """
    Matches the Data Inspector's "C - segment 3/15". Two different numbers for
    the same trial would be worse than none.
    """
    sel = summarise_selection(STIM, [(45.5, 75.5)])
    c = [s for s in sel if s.stim_type == "C"][0]
    assert c.spans == [(1, 30)], "C's first event must be #1, not #46"


def test_a_discontinuous_selection_is_not_collapsed():
    """
    Collapsing #1-20 and #55-70 to #1-70 would claim 70 trials where 36 were
    chosen. The spans are reported as they are.
    """
    sel = summarise_selection({"A": [float(i) for i in range(1, 121)]},
                              [(0.5, 20.5), (54.5, 70.5)])
    assert sel[0].spans == [(1, 20), (55, 70)]
    assert sel[0].n_selected == 36


def test_overlapping_ranges_do_not_double_count():
    sel = summarise_selection({"A": [float(i) for i in range(1, 21)]},
                              [(0.5, 10.5), (5.0, 15.5)])
    assert sel[0].n_selected == 15


def test_types_with_nothing_selected_are_omitted():
    sel = summarise_selection(STIM, [(45.5, 75.5)])
    assert [s.stim_type for s in sel] == ["C"]


# ── The displayed line ────────────────────────────────────────────────────────

def test_multiple_types_are_listed_separately():
    line = format_selection(STIM, [(0.5, 45.5), (45.5, 75.5)])
    assert "2 ranges" in line
    assert "A: 45 events" in line
    assert "C: 30 events" in line


def test_single_type_gives_count_and_position_in_the_file():
    line = format_selection({"A": [float(i) for i in range(1, 121)]},
                            [(0.5, 90.5)])
    assert "1 range" in line
    assert "90 events" in line
    assert "of 120" in line


def test_a_fragmented_selection_is_capped_rather_than_wrapping():
    """Five spans on one line would be harder to read than no summary."""
    line = format_selection(
        {"A": [float(i) for i in range(1, 121)]},
        [(0.5, 20.5), (54.5, 70.5), (80.5, 85.5), (100.5, 105.5),
         (110.5, 112.5)], max_spans=3)
    assert "(+2 more)" in line
    assert line.count("#") == 3


def test_singular_and_plural_read_correctly():
    assert "1 range " in format_selection({"F": [10.0]}, [(9.0, 11.0)])
    assert "1 event " in format_selection({"F": [10.0]}, [(9.0, 11.0)])


def test_an_empty_selection_says_so():
    assert "no stimulus events inside" in format_selection(STIM, [(500.0, 600.0)])


def test_no_ranges_keeps_the_original_prompt():
    assert "drag on the plot" in format_selection(STIM, [])


def test_no_events_at_all_does_not_raise():
    assert format_selection({}, [(0.0, 10.0)])
    assert format_selection(None, [(0.0, 10.0)])


# ── Wiring ────────────────────────────────────────────────────────────────────

APP = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def test_the_crop_dialogue_shows_the_summary():
    a = APP.index("def _update_list_label")
    b = APP.index("_update_list_label()", a + 10)
    body = APP[a:b]
    # Summarised against the events of the channel ON SCREEN. Ranges are
    # chosen per channel, and each channel has its own event sources, so a
    # summary built from a variable captured when the window opened would
    # count the first channel's trials under the second channel's ranges.
    assert 'format_selection(state["stim"]' in body


def test_the_summary_counts_the_ranges_of_the_channel_on_screen():
    a = APP.index("def _update_list_label")
    b = APP.index("_update_list_label()", a + 10)
    body = APP[a:b]
    assert "spans = _spans()" in body, (
        "the summary must read the on-screen channel's ranges, not a "
        "list captured when the window opened")


def test_the_dialogue_states_where_every_channel_stands():
    """
    A channel never opened here has no ranges and is analysed whole. That is a
    legitimate choice and an easy accident, and they are indistinguishable
    unless the window says so without being asked.
    """
    a = APP.index("def _update_list_label")
    b = APP.index("_update_list_label()", a + 10)
    body = APP[a:b]
    assert "whole recording" in body
    assert "per_chan" in body


def test_a_summary_failure_cannot_break_the_dialogue():
    """It is an aid; the crop tool must work without it."""
    a = APP.index("def _update_list_label")
    b = APP.index("_update_list_label()", a + 10)
    assert "except Exception" in APP[a:b]


def test_the_label_left_aligns_both_lines():
    a = APP.index("info = tk.Label(footer, textvariable=list_lbl")
    assert 'justify="left"' in APP[a:a + 200]
