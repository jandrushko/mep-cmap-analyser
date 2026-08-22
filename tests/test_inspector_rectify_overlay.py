"""
The rectified trace and the RMS envelope are DISPLAY overlays.

They exist so an analyst can judge a cortical silent period offset against the
signal the detector actually thresholded, rather than against the raw trace,
which the detector never looks at. That makes them useful and also makes them
dangerous: if the visible trace also decided where a marker landed, ticking a
display box would silently change a measurement.

The Inspector needs a live Tk root and a matplotlib backend, so the rules are
asserted against the class and its source rather than by driving the window.
"""

import inspect

import numpy as np
import pytest

from mep_cmap.inspector import DataInspectorWindow as DIW


MARKER_FIELDS = ("ptp_min_idx", "ptp_max_idx", "onset_idx",
                 "silent_start_idx", "silent_end_idx", "mep_offset_idx")


def test_every_marker_is_measured_on_the_raw_trace():
    for field in MARKER_FIELDS:
        assert DIW._SNAP_SOURCE[field] == "raw", (
            f"{field} is snapped to something other than the raw trace. A "
            f"peak-to-peak landmark on |EMG| finds the larger peak twice and "
            f"reports a positive amplitude for a biphasic response.")


def test_every_drawn_marker_declares_a_snap_source():
    """A marker _add draws but _SNAP_SOURCE omits would default silently."""
    for field in set(DIW.DOT_COLOURS):
        assert field in DIW._SNAP_SOURCE, (
            f"{field} is drawn but has no _SNAP_SOURCE entry")


def test_a_non_raw_snap_source_is_refused_rather_than_honoured():
    """
    The guard is the point of the table. If someone adds "rectified" here in
    future, it must fail loudly rather than quietly change what is measured.
    """
    insp = DIW.__new__(DIW)
    emg = np.array([1.0, -2.0, 3.0])
    assert insp._snap_array("ptp_min_idx", emg) is emg

    insp._SNAP_SOURCE = dict(DIW._SNAP_SOURCE, ptp_min_idx="rectified")
    with pytest.raises(ValueError):
        insp._snap_array("ptp_min_idx", emg)


def test_markers_are_handed_the_snap_array_not_the_visible_one():
    """
    _add must route through _snap_array. Capturing `emg` directly would work
    today and break the moment an overlay is drawn into the same variable.
    """
    src = inspect.getsource(DIW._plot)
    start = src.index("def _add(")
    body = src[start:src.index("_add(m['ptp_min_idx']", start)]
    assert "_snap_array(" in body, (
        "_add no longer routes through _snap_array")
    assert "DraggablePoint(" in body
    seg = body[body.index("DraggablePoint("):]
    assert "_snap" in seg.split(")")[0], (
        "DraggablePoint is being handed an array other than the snap array")


def test_the_overlays_are_drawn_behind_the_raw_trace():
    """
    Order matters: an overlay drawn after the trace covers the thing the
    analyst is reading amplitudes off.
    """
    src = inspect.getsource(DIW._plot)
    assert src.index("_draw_overlays(") < src.index(
        "self.ax_raw.plot(self.t, emg"), (
        "overlays are drawn on top of the raw trace")


def test_the_envelope_is_mirrored():
    """
    Drawn one-sided the envelope sits in the positive half of the axis and
    collides with the MEP's positive peak, which is where the peak-to-peak
    markers have to stay legible.
    """
    src = inspect.getsource(DIW._draw_overlays)
    assert "fill_between(self.t, -env, env" in src, (
        "the envelope band is no longer mirrored about zero")
    assert "-env" in src


def test_the_overlay_never_writes_to_segment_metadata():
    """
    Display state is not a measurement and must not reach the saved edits.
    """
    for fn in (DIW._draw_overlays, DIW._rect_and_envelope):
        src = inspect.getsource(fn)
        assert "_update_meta" not in src
        assert "self.meta" not in src


def test_the_overlay_uses_the_detectors_own_envelope():
    """
    The value of showing the envelope is that it is the one the detector
    thresholded. Recomputing it a second way here would put a different
    picture in front of the analyst from the one that produced the marker.
    """
    src = inspect.getsource(DIW._rect_and_envelope)
    assert "compute_rms_envelope" in src, (
        "the overlay no longer uses the detector's envelope function")
    assert "rms_window_ms" in src, (
        "the overlay is not using the configured RMS window")
    assert "_csp_settings()" in src, (
        "the overlay is not reading the settings the analysis runs under")


def test_the_threshold_and_its_baseline_use_one_definition():
    """
    The threshold, the band shading it and the x-limit must agree about which
    samples the baseline is. They did not at first: the threshold was built
    over the analysis baseline while the view started at visible_pre_ms, so
    the EMG it was normalised to was off the left edge of the plot and the
    line appeared to float below a corridor it was a fraction of.
    """
    for fn in (DIW._rect_and_envelope, DIW._draw_overlays, DIW._plot):
        assert "_baseline_window_ms()" in inspect.getsource(fn), (
            f"{fn.__name__} does not use the shared baseline window")


def test_the_baseline_window_is_the_analysis_baseline():
    insp = DIW.__new__(DIW)
    insp.t = np.linspace(-200.0, 600.0, 4000, endpoint=False)

    insp._analysis_pre_ms = 100.0
    assert insp._baseline_window_ms() == (-100.0, 0.0)

    # Falls back to the segment's own pre-stimulus extent, never to a constant.
    insp._analysis_pre_ms = None
    assert insp._baseline_window_ms() == (-200.0, 0.0)


def test_the_view_is_never_cropped_inside_the_baseline():
    """
    With the envelope on, the x-limit must reach at least the start of the
    baseline, so the evidence behind the threshold is on screen.
    """
    src = inspect.getsource(DIW._plot)
    seg = src[src.index("_xlim_left ="):src.index("set_xlim(")]
    assert "show_env_var" in seg and "_baseline_window_ms()" in seg, (
        "the x-limit no longer widens to include the baseline window")
    assert "min(_xlim_left" in seg, (
        "the x-limit is being replaced rather than widened")
