"""
The Conditions tab reads the range chosen for the channel it is showing.

It read the file-wide snapshot value, so one channel's selection was applied
to every channel's trial list. Conditions are assigned by trial INDEX, so that
is not a display problem: the table numbers its trials from one channel's
selection while the analysis numbers them from another's, and every assignment
lands on the wrong trial.
"""

import pathlib

import pytest


SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "conditions_tab.py").read_text(encoding="utf-8")


class _Tab:
    """Just enough of the tab to exercise the resolver."""

    from mep_cmap.conditions_tab import ConditionsTabMixin as _M
    _cond_crop_for = _M._cond_crop_for
    _cond_ranges_differ_by_channel = _M._cond_ranges_differ_by_channel

    def __init__(self, chan_settings, current, names=None):
        self._chan_settings = chan_settings
        self._current = current
        self._names = names or [(i, f"ch{i}") for i in chan_settings]

    def _cond_selected_channel_idx(self):
        return self._current

    def _cond_channel_names(self):
        return self._names


_PARAMS = {"crop_ranges": [(0.0, 10.0)], "crop_start": None, "crop_end": None}


def test_a_channel_uses_its_own_ranges():
    tab = _Tab({0: {"crop_ranges": [(1.0, 2.0)]},
                1: {"crop_ranges": [(5.0, 6.0)]}}, current=1)
    assert tab._cond_crop_for(_PARAMS) == ([(5.0, 6.0)], None, None)


def test_the_other_channels_ranges_are_not_used():
    tab = _Tab({0: {"crop_ranges": [(1.0, 2.0)]},
                1: {"crop_ranges": [(5.0, 6.0)]}}, current=0)
    got, _s, _e = tab._cond_crop_for(_PARAMS)
    assert got == [(1.0, 2.0)]
    assert (5.0, 6.0) not in got


def test_a_channel_with_no_ranges_falls_back_to_the_file_wide_pair():
    """What a single-channel study and every older session carry."""
    tab = _Tab({0: {}}, current=0)
    assert tab._cond_crop_for(_PARAMS) == ([(0.0, 10.0)], None, None)


def test_a_channel_left_whole_is_not_given_another_channels_range():
    """An explicit empty list is a choice, not an absence."""
    tab = _Tab({0: {"crop_ranges": [(1.0, 2.0)]},
                1: {"crop_ranges": []}}, current=1)
    assert tab._cond_crop_for(_PARAMS) == (None, None, None)


def test_an_explicit_channel_overrides_the_dropdown():
    tab = _Tab({0: {"crop_ranges": [(1.0, 2.0)]},
                1: {"crop_ranges": [(5.0, 6.0)]}}, current=1)
    assert tab._cond_crop_for(_PARAMS, 0) == ([(1.0, 2.0)], None, None)


# ── whether a rebuild is needed ──────────────────────────────────────────────

def test_identical_ranges_are_not_reported_as_differing():
    """Rebuilding then would discard hand-made assignments to reach an
    identical list."""
    tab = _Tab({0: {"crop_ranges": [(1.0, 2.0)]},
                1: {"crop_ranges": [(1.0, 2.0)]}}, current=0)
    assert tab._cond_ranges_differ_by_channel() is False


def test_differing_ranges_are_reported():
    tab = _Tab({0: {"crop_ranges": [(1.0, 2.0)]},
                1: {"crop_ranges": [(5.0, 6.0)]}}, current=0)
    assert tab._cond_ranges_differ_by_channel() is True


def test_one_channel_whole_and_one_cropped_differ():
    tab = _Tab({0: {}, 1: {"crop_ranges": [(5.0, 6.0)]}}, current=0)
    assert tab._cond_ranges_differ_by_channel() is True


# ── wiring ───────────────────────────────────────────────────────────────────

def test_no_crop_site_still_reads_the_file_wide_value_directly():
    """Three sites cropped from the snapshot; all must go through the
    resolver, or they drift apart again."""
    assert 'crop_stim_times(events, _p.get("crop_ranges")' not in SRC
    assert 'crop_ranges=params.get("crop_ranges")' not in SRC


def test_switching_channel_rebuilds_the_trial_list():
    a = SRC.index("def _cond_channel_changed")
    b = SRC.index("def _cond_ranges_differ_by_channel")
    body = SRC[a:b]
    assert "_cond_reload_events_for_channel" in body


def test_the_reload_reads_that_channels_own_events():
    """Event sources are per channel too, so re-cutting with another
    channel's events would replace one mismatch with a subtler one."""
    a = SRC.index("def _cond_reload_events_for_channel")
    body = SRC[a:a + 2000]
    assert "channel_idx=self._cond_selected_channel_idx()" in body


def test_the_status_line_names_the_channel_it_describes():
    """Two channels of one recording can be cut differently, and the table
    looks simply wrong on the second unless it says which it describes."""
    assert "within the range selected for" in SRC
