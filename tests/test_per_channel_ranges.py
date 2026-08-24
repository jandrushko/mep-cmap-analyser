"""
Time ranges are chosen per channel.

A range is a choice about one channel's trace: a block clean on the
contralateral recording can be unusable on the ipsilateral one. One list
imposed on every channel made that impossible to say, and the imposition was
silent.
"""

import json

import pytest

from mep_cmap import reporting as R
from mep_cmap.pipeline import PipelineConfig


def _cfg(**kw):
    return PipelineConfig(csp_types=set(), **kw)


# ── what the record says was analysed ────────────────────────────────────────

def test_the_whole_recording_is_recorded_as_such():
    """
    None and [] are different answers. The first says the recording was
    analysed entire; the second would say a selection was made and was empty,
    which no run can produce.
    """
    assert R._analysed_ranges(_cfg()) is None


def test_the_ranges_analysed_are_recorded():
    got = R._analysed_ranges(_cfg(crop_ranges=[(1.0, 2.5), (10.0, 12.0)]))
    assert got == [[1.0, 2.5], [10.0, 12.0]]


def test_a_legacy_single_span_is_recorded_too():
    """Sessions written before multiple ranges existed carry only the pair."""
    got = R._analysed_ranges(_cfg(crop_start=3.0, crop_end=9.0))
    assert got == [[3.0, 9.0]]


def test_the_ranges_reach_the_record_and_are_json_safe():
    rec = R.build_record(
        _cfg(crop_ranges=[(1.0, 2.0)]), name="sub-001", channel="TA",
        fs=5000.0, emg_unit="mV", source_format="smr", rows=[],
        tool_version="1.4.6", detection_version="d1")
    assert rec["recording"]["analysed_ranges_s"] == [[1.0, 2.0]]
    assert json.loads(json.dumps(rec)) == rec


def test_two_channels_of_one_file_can_differ_in_the_record():
    """
    This is the fact the record exists to carry: without it, two channels of
    one recording report different trial counts and nothing explains why.
    """
    ta = R.build_record(_cfg(crop_ranges=[(0.0, 60.0)]), name="f", channel="TA",
                        rows=[], tool_version="1.4.6", detection_version="d1")
    cl = R.build_record(_cfg(), name="f", channel="TA-CL",
                        rows=[], tool_version="1.4.6", detection_version="d1")
    assert ta["recording"]["analysed_ranges_s"] == [[0.0, 60.0]]
    assert cl["recording"]["analysed_ranges_s"] is None


def test_a_malformed_range_is_dropped_rather_than_crashing():
    """This runs at the end of a completed analysis; a bad value must not
    lose the results that have already been measured."""
    got = R._analysed_ranges(_cfg(crop_ranges=[(1.0, 2.0), ("x",), None]))
    assert got == [[1.0, 2.0]]


# ── the config still carries them per run ────────────────────────────────────

def test_the_config_field_exists_and_defaults_to_none():
    """The worker resolves it per channel and passes one list per run; the
    pipeline itself knows nothing about there being other channels."""
    assert "crop_ranges" in PipelineConfig.__dataclass_fields__
    assert _cfg().crop_ranges is None

# ── the shared resolver ──────────────────────────────────────────────────────
#
# Four readers now: the analysis worker, the Conditions tab, the preview, and
# the range picker. A channel cut differently by any two of them numbers its
# trials differently, and conditions are assigned by trial INDEX.

from mep_cmap.pipeline import crop_for_channel

_P = {"crop_ranges": [(0.0, 10.0)], "crop_start": None, "crop_end": None}


def test_a_channel_uses_its_own_ranges_everywhere():
    cs = {0: {"crop_ranges": [(1.0, 2.0)]}, 1: {"crop_ranges": [(5.0, 6.0)]}}
    assert crop_for_channel(cs, 1, _P) == ([(5.0, 6.0)], None, None)
    assert crop_for_channel(cs, 0, _P) == ([(1.0, 2.0)], None, None)


def test_a_channel_without_ranges_falls_back_to_the_file_wide_pair():
    assert crop_for_channel({0: {}}, 0, _P) == ([(0.0, 10.0)], None, None)


def test_a_stale_file_wide_range_does_not_reach_a_channel_left_whole():
    """
    The failure that reached the preview: a file-wide range of 8.31-9.89 s
    survived in the session while both channels carried none, so a channel
    the analyst had left whole was cut to a window with no stimuli in it.
    An explicit empty list is a choice and must not inherit.
    """
    cs = {0: {"crop_ranges": [(1.0, 2.0)]}, 1: {"crop_ranges": []}}
    assert crop_for_channel(cs, 1, _P) == (None, None, None)


def test_string_keys_from_a_reloaded_session_still_resolve():
    """chan_settings comes back from JSON with string keys."""
    cs = {"1": {"crop_ranges": [(5.0, 6.0)]}}
    assert crop_for_channel(cs, 1, _P) == ([(5.0, 6.0)], None, None)


def test_the_legacy_single_span_is_carried_through():
    p = {"crop_ranges": None, "crop_start": 3.0, "crop_end": 9.0}
    assert crop_for_channel({}, 0, p) == (None, 3.0, 9.0)


def test_missing_settings_and_params_do_not_raise():
    assert crop_for_channel(None, 0, None) == (None, None, None)


# ── the readers all go through it ────────────────────────────────────────────

def test_the_preview_resolves_per_channel():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
    assert 'crop_ranges=params.get("crop_ranges")' not in src, (
        "the preview cropped one channel's trace to another channel's "
        "selection")
    assert "crop_for_channel" in src


def test_the_preview_names_the_range_when_nothing_falls_inside_it():
    """'Check the selected marker' sends the analyst to the one thing that is
    probably right."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
    assert "cropped_to" in src
    assert "data range selected" in src

# ── choosing which channel to preview ────────────────────────────────────────

class _Prev:
    """Just enough of the mixin to exercise the channel choice."""

    from mep_cmap.preview import PreviewDetectionMixin as _M
    _preview_pick_channel = _M._preview_pick_channel

    def __init__(self, idxs, channel_idx=0):
        self._idxs = idxs
        self.channel_idx = channel_idx

    def _analysis_channel_indices(self):
        return self._idxs


def test_one_analysed_channel_is_not_asked_about():
    """There is no question to put, and a dialogue every time would be
    friction on the commonest case."""
    p = _Prev([2], channel_idx=2)
    assert p._preview_pick_channel({"channel_idx": 2}) == 2


def test_no_analysed_channels_falls_back_to_the_header_channel():
    p = _Prev([], channel_idx=3)
    assert p._preview_pick_channel({"channel_idx": 3}) == 3


def test_the_snapshot_is_not_rewritten_by_previewing_another_channel():
    """
    The snapshot describes the run the analyst has set up. Previewing a
    different channel must not edit it, or the preview would silently change
    what Run Analysis is about to do.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
    a = src.index("_ch = self._preview_pick_channel(params)")
    body = src[a:a + 600]
    assert "params = dict(params)" in body, (
        "the channel must be applied to a COPY of the snapshot")


def test_cancelling_the_channel_choice_cancels_the_preview():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
    a = src.index("_ch = self._preview_pick_channel(params)")
    body = src[a:a + 300]
    assert "if _ch is None:" in body and "Preview cancelled" in body


def test_the_channel_is_chosen_before_the_file_is_read():
    """Reading first and then offering to change channel would read twice."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
    assert (src.index("_ch = self._preview_pick_channel(params)")
            < src.index("self._preview_running = True"))


def test_the_log_names_the_channel_being_previewed():
    """Four previews of one file are otherwise indistinguishable in the log."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "preview.py").read_text(encoding="utf-8")
    assert "on {_nm}" in src

# ── settings tab 1a does not own must survive a channel confirm ──────────────

class _Snap:
    """Just enough of the app to exercise _snapshot_chan_settings."""

    from mep_cmap.app import TMSAnalysisApp as _A
    _snapshot_chan_settings = _A._snapshot_chan_settings

    _chan_settings_keys = ("label_map", "window_map")

    def __init__(self, existing=None):
        self._chan_settings = dict(existing or {})
        self.channel_idx = 0
        self.label_map = {"A": "MEP"}
        self.window_map = {"A": (20.0, 400.0)}


def test_confirming_a_channel_keeps_its_data_ranges():
    """
    The entry is REPLACED and rebuilt only from the keys tab 1a writes, so a
    per-channel setting owned by anything else was discarded on every Confirm
    Setup. The ranges were chosen, wiped, and the run fell back to the
    file-wide list -- one channel's selection applied to all of them.
    """
    app = _Snap({0: {"crop_ranges": [(1.0, 2.0)], "label_map": {"A": "old"}}})
    app._snapshot_chan_settings(0)
    assert app._chan_settings[0]["crop_ranges"] == [(1.0, 2.0)]
    # ...while what tab 1a DOES own is refreshed from the live table.
    assert app._chan_settings[0]["label_map"] == {"A": "MEP"}


def test_a_channel_with_nothing_stored_is_unaffected():
    app = _Snap()
    app._snapshot_chan_settings(0)
    assert "crop_ranges" not in app._chan_settings[0]
    assert app._chan_settings[0]["window_map"] == {"A": (20.0, 400.0)}


def test_confirming_one_channel_does_not_touch_another():
    app = _Snap({0: {"crop_ranges": [(1.0, 2.0)]},
                 1: {"crop_ranges": [(5.0, 6.0)]}})
    app._snapshot_chan_settings(0)
    assert app._chan_settings[1]["crop_ranges"] == [(5.0, 6.0)]


def test_an_empty_range_list_is_preserved_not_dropped():
    """It means the channel was left whole, which is a choice."""
    app = _Snap({0: {"crop_ranges": []}})
    app._snapshot_chan_settings(0)
    assert app._chan_settings[0]["crop_ranges"] == []


def test_crop_ranges_is_not_a_tab_1a_key():
    """If it were, a channel confirm would overwrite it from an attribute
    that does not exist, and the preservation above would never fire."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    a = src.index("self._chan_settings_keys = (")
    b = src.index(")", a)
    assert "crop_ranges" not in src[a:b]


def test_a_channel_confirm_preserves_what_tab_1a_does_not_own():
    """The general rule, not a crop_ranges special case: any per-channel
    setting owned elsewhere was discarded on every Confirm Setup."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    a = src.index("def _snapshot_chan_settings")
    b = src.index("def _reset_chan_settings_to_defaults", a)
    body = src[a:b]
    assert "if k not in self._chan_settings_keys" in body
    assert "snap.update(_keep)" in body

# ── an edge trial must survive the range it was selected in ──────────────────
#
# Selecting a range means "analyse these trials", not "throw the rest of the
# recording away". An epoch needs signal before the stimulus and after it, so a
# range dragged to end just after the last pulse -- the natural way to draw it
# -- left that pulse without enough data and the trial was silently dropped.

from mep_cmap.pipeline import epoch_pad_seconds


def test_the_pad_covers_the_whole_epoch():
    assert epoch_pad_seconds(pre_ms=100.0, post_ms=400.0) == pytest.approx(0.5)


def test_the_widest_per_type_window_wins():
    """One pad serves every stimulus type in the file, so it must be big
    enough for the greediest of them."""
    wm = {"A": (20.0, 50.0), "B": (100.0, 800.0), "C": (None, None)}
    assert epoch_pad_seconds(pre_ms=20.0, post_ms=400.0,
                             window_map=wm) == pytest.approx(0.9)


def test_the_review_lead_in_is_included():
    """The Inspector deliberately cuts a wider lead-in than the analysis
    window, and it cuts from the same padded signal."""
    assert epoch_pad_seconds(pre_ms=20.0, post_ms=400.0,
                             prestim_ms=100.0) == pytest.approx(0.5)


def test_a_malformed_window_entry_does_not_break_the_pad():
    wm = {"A": None, "B": (10.0,), "C": (20.0, 50.0)}
    assert epoch_pad_seconds(pre_ms=0.0, post_ms=0.0,
                             window_map=wm) == pytest.approx(0.07)


def test_nothing_configured_gives_no_pad():
    assert epoch_pad_seconds() == 0.0


# ── the crop itself ──────────────────────────────────────────────────────────

def test_the_padded_signal_reaches_past_the_range(tmp_path, monkeypatch):
    """
    The regression, in the shape it was found: 20 events in a range ending
    9.989 s, the last at 9.610 s needing data to 10.010 s. The picker counted
    twenty and the run measured nineteen.
    """
    import numpy as np
    import mep_cmap.pipeline as P

    fs = 1000.0
    emg = np.zeros(int(12.0 * fs))
    monkeypatch.setattr(P, "extract_emg_waveform_and_fs",
                        lambda *a, **k: (emg, fs, "mV"))
    monkeypatch.setattr(P, "extract_stim_times",
                        lambda *a, **k: {"A": [0.100, 9.610]})

    _e, t, _fs, _u, st = P.pipeline_load_file(
        "x", 0, "A", crop_ranges=[(0.000, 9.989)], epoch_pad_s=0.5)
    # The stimuli are those inside the range AS DRAWN...
    assert st["A"] == [0.100, 9.610]
    # ...while the signal reaches far enough for the last one to be cut whole.
    assert t.max() >= 9.610 + 0.4
    assert t.min() <= 0.100 - 0.1


def test_the_padding_never_admits_a_stimulus_outside_the_range(tmp_path,
                                                               monkeypatch):
    """The pad is signal, not selection. A stimulus just outside the range
    the analyst drew must not become a trial."""
    import numpy as np
    import mep_cmap.pipeline as P

    fs = 1000.0
    emg = np.zeros(int(12.0 * fs))
    monkeypatch.setattr(P, "extract_emg_waveform_and_fs",
                        lambda *a, **k: (emg, fs, "mV"))
    monkeypatch.setattr(P, "extract_stim_times",
                        lambda *a, **k: {"A": [1.000, 5.000, 9.000]})

    _e, _t, _fs, _u, st = P.pipeline_load_file(
        "x", 0, "A", crop_ranges=[(4.000, 6.000)], epoch_pad_s=2.0)
    assert st["A"] == [5.000], "a padded stimulus was analysed as a trial"


def test_no_pad_keeps_the_previous_behaviour(monkeypatch):
    import numpy as np
    import mep_cmap.pipeline as P

    fs = 1000.0
    emg = np.zeros(int(12.0 * fs))
    monkeypatch.setattr(P, "extract_emg_waveform_and_fs",
                        lambda *a, **k: (emg, fs, "mV"))
    monkeypatch.setattr(P, "extract_stim_times", lambda *a, **k: {"A": [5.0]})

    _e, t, _fs, _u, _st = P.pipeline_load_file(
        "x", 0, "A", crop_ranges=[(4.0, 6.0)])
    assert t.min() >= 4.0 and t.max() <= 6.0


def test_the_legacy_single_span_is_padded_too(monkeypatch):
    import numpy as np
    import mep_cmap.pipeline as P

    fs = 1000.0
    emg = np.zeros(int(12.0 * fs))
    monkeypatch.setattr(P, "extract_emg_waveform_and_fs",
                        lambda *a, **k: (emg, fs, "mV"))
    monkeypatch.setattr(P, "extract_stim_times", lambda *a, **k: {"A": [5.9]})

    _e, t, _fs, _u, st = P.pipeline_load_file(
        "x", 0, "A", crop_start=4.0, crop_end=6.0, epoch_pad_s=0.5)
    assert st["A"] == [5.9]
    assert t.max() >= 6.3
