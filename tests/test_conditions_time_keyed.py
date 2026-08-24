"""
Conditions are keyed by stimulus TIME, and cover the whole recording.

The Conditions table shows one channel's selected data, which is what the
analyst asked for. The events FILE cannot be that narrow: pipeline_load_file
uses these rows exclusively when they exist, so an event with no row is not an
unconditioned trial, it is not a trial at all.

With per-channel ranges that silently emptied every channel whose range
differed from the one last displayed -- conditions assigned while viewing
Sol-CL handed TA six onsets from a range TA does not cover, and TA measured
nothing.
"""

import pytest

from mep_cmap import conditions as C
from mep_cmap.conditions_tab import ConditionsTabMixin


class _Tab:
    """Just enough of the tab to exercise the row builder."""

    _cond_event_rows_for_recording = \
        ConditionsTabMixin._cond_event_rows_for_recording

    def __init__(self, all_events, existing=None, channel=0):
        self._all_events = all_events
        self.condition_event_rows = existing or []
        self._channel = channel

        class _P:
            @staticmethod
            def get():
                return "recording.smr"
        self.file_path = _P()

    def _configured_events(self, _path, channel_idx=None):
        return dict(self._all_events), []

    def _cond_selected_channel_idx(self):
        return self._channel


ALL = {"A": [0.100, 0.600, 1.100, 5.000, 9.610]}


def test_every_event_in_the_recording_gets_a_row():
    """An event with no row is dropped from the analysis entirely."""
    tab = _Tab(ALL)
    rows = tab._cond_event_rows_for_recording([])
    assert [r["onset"] for r in rows] == [0.100, 0.600, 1.100, 5.000, 9.610]


def test_an_unconditioned_event_keeps_its_stimulus_type():
    """So it is still analysed, just without a condition -- which is what an
    unassigned recording has always looked like."""
    rows = _Tab(ALL)._cond_event_rows_for_recording([])
    assert all(r["trial_type"] == "A" for r in rows)
    groups, _decoded = C.group_events(rows)
    assert groups == {"A": [0.100, 0.600, 1.100, 5.000, 9.610]}


def test_the_shown_channels_assignments_win():
    shown = [{"onset": 0.600, "duration": 0.0,
              "trial_type": "A", "condition": "pre"}]
    rows = _Tab(ALL)._cond_event_rows_for_recording(shown)
    by_onset = {r["onset"]: r for r in rows}
    assert by_onset[0.600]["condition"] == "pre"
    assert by_onset[0.100]["condition"] == C.NA


def test_assignments_from_another_channel_survive():
    """
    A condition is a property of the trial, not of the channel that happened
    to be displayed when it was assigned. Switching channel and applying again
    must accumulate rather than replace.
    """
    earlier = [{"onset": 5.000, "duration": 0.0,
                "trial_type": "A", "condition": "post"}]
    shown = [{"onset": 0.100, "duration": 0.0,
              "trial_type": "A", "condition": "pre"}]
    rows = _Tab(ALL, existing=earlier)._cond_event_rows_for_recording(shown)
    by_onset = {r["onset"]: r for r in rows}
    assert by_onset[5.000]["condition"] == "post"
    assert by_onset[0.100]["condition"] == "pre"


def test_a_reassignment_replaces_the_earlier_one():
    earlier = [{"onset": 0.100, "duration": 0.0,
                "trial_type": "A", "condition": "pre"}]
    shown = [{"onset": 0.100, "duration": 0.0,
              "trial_type": "A", "condition": "post"}]
    rows = _Tab(ALL, existing=earlier)._cond_event_rows_for_recording(shown)
    hits = [r for r in rows if r["onset"] == 0.100]
    assert len(hits) == 1 and hits[0]["condition"] == "post"


def test_onsets_are_matched_despite_float_error():
    """The same event re-read from the file must not become a second row."""
    shown = [{"onset": 0.10000000000000003, "duration": 0.0,
              "trial_type": "A", "condition": "pre"}]
    rows = _Tab(ALL)._cond_event_rows_for_recording(shown)
    assert len([r for r in rows if abs(r["onset"] - 0.100) < 1e-3]) == 1


def test_the_rows_are_sorted_by_onset():
    shown = [{"onset": 9.610, "duration": 0.0,
              "trial_type": "A", "condition": "last"}]
    rows = _Tab(ALL)._cond_event_rows_for_recording(shown)
    assert [r["onset"] for r in rows] == sorted(r["onset"] for r in rows)


def test_an_unreadable_recording_does_not_lose_what_was_assigned():
    """Reading the full event list is best effort; the assignments are not."""
    class _Broken(_Tab):
        def _configured_events(self, *_a, **_k):
            raise OSError("gone")

    shown = [{"onset": 0.600, "duration": 0.0,
              "trial_type": "A", "condition": "pre"}]
    rows = _Broken(ALL)._cond_event_rows_for_recording(shown)
    assert [r["condition"] for r in rows] == ["pre"]


def test_two_channels_with_different_ranges_both_find_their_trials():
    """
    The regression, end to end: TA's range holds event 1, Sol-CL's holds
    events 4-5. Both must survive one events file.
    """
    from mep_cmap.pipeline import crop_stim_times

    shown = [{"onset": 5.000, "duration": 0.0,
              "trial_type": "A", "condition": "post"}]
    rows = _Tab(ALL)._cond_event_rows_for_recording(shown)
    groups, _d = C.group_events(rows)

    ta = crop_stim_times(groups, [(0.0, 0.3)], None, None)
    assert sum(len(v) for v in ta.values()) == 1, "TA lost its only trial"

    solcl = crop_stim_times(groups, [(4.5, 10.0)], None, None)
    assert sum(len(v) for v in solcl.values()) == 2
