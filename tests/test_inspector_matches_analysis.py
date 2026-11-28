"""
The Data Inspector (and the Preview's lower plot, which is the Inspector)
detects onsets exactly as the analysis does (v1.4.16).

It searched from the PTP window start (10 ms by default) instead of the
analysis's window, so on an FDI M-wave with Min lat 3 ms every trial it drew
showed 10.0 ms -- part-way through the response -- while the analysis and the
Preview's upper plot found ~5 ms. With the window fixed it still used the
100 ms display lead-in as its baseline where the analysis uses the type's
20 ms Pre, which moved 10 of 31 onsets by 0.4-1 ms. With both the same, it
matched the analysis on every trial of that recording.
"""

import pathlib
import types

import numpy as np

from mep_cmap.pipeline import onset_search_window, ARTEFACT_FLOOR_MS
from mep_cmap.detection.dispatch import dispatch_onset

SRC = (pathlib.Path(__file__).resolve().parents[1] / "mep_cmap" / "inspector.py"
       ).read_text(encoding="utf-8")


def test_the_inspector_uses_the_analysis_search_window():
    assert "search_start_ms=self.ptp_start_ms or 10" not in SRC
    assert "from .pipeline import onset_search_window" in SRC
    i = SRC.index("_s_lo, _s_hi = onset_search_window(")
    j = SRC.index("onset_ms = dispatch_onset(", i)
    call = SRC[j:j + 400]
    assert "search_start_ms=_s_lo" in call and "search_end_ms=_s_hi" in call


def test_the_inspector_detects_on_the_types_own_pre():
    j = SRC.index("onset_ms = dispatch_onset(")
    call = SRC[j:j + 400]
    assert "_emg_det" in call and "pre_ms=_det_pre" in call


def test_window_starts_at_the_profile_not_the_ptp_start():
    cfg = types.SimpleNamespace(ptp_start=10.0, ptp_end=50.0)
    # Upper-limb M-wave profile as on tab 1a: the search must start at 3 ms.
    assert onset_search_window(cfg, 3.0, 12.0)[0] == 3.0
    # Never inside the artefact floor, and TMS unchanged at the PTP start.
    assert onset_search_window(cfg, 1.0, 12.0)[0] == ARTEFACT_FLOOR_MS
    assert onset_search_window(cfg, 18.0, 28.0)[0] == 10.0


def test_trimming_the_lead_in_reproduces_the_analysis_segment():
    # The Inspector's segment (100 ms lead-in) trimmed to the type's 20 ms Pre
    # is the analysis segment, so the detector sees the same trace.
    fs, rng = 5000.0, np.random.default_rng(1)
    t = (np.arange(int(0.16 * fs)) - int(0.1 * fs)) / fs * 1000.0
    x = rng.normal(0, 0.01, t.size)
    u = np.clip(t - 5.0, 0, None)
    x += np.where(t >= 5.0, 3.0 * np.sin(2 * np.pi * u / 8.0) * np.exp(-u / 6.0), 0)
    cut = int(round((100.0 - 20.0) * fs / 1000.0))
    kw = dict(search_start_ms=3.0, search_end_ms=50.0, min_latency_ms=3.0, max_latency_ms=12.0)
    inspector = dispatch_onset(x[cut:], fs, {"onset_method": "methods_median"}, pre_ms=20.0, **kw)
    analysis = dispatch_onset(x[cut:].copy(), fs, {"onset_method": "methods_median"}, pre_ms=20.0, **kw)
    assert inspector == analysis and inspector is not None and 4.0 <= inspector <= 6.5


# ── no onset, no offset / duration / AUC; PTP kept ───────────────────────────

PIPE = (pathlib.Path(__file__).resolve().parents[1] / "mep_cmap" / "pipeline.py"
        ).read_text(encoding="utf-8")


def test_pipeline_blanks_offset_duration_and_auc_without_an_onset():
    i = PIPE.index("No onset, no response: an offset is the END of a response")
    block = PIPE[i:i + 900]
    assert "if man_lat is None and auto_lat is None:" in block
    assert "mep_offset_ms = mep_duration_ms = None" in block
    j = PIPE.index("auc_val = None\n        if man_lat is None and auto_lat is None:")
    assert "pass" in PIPE[j:j + 500]


def test_inspector_says_so_and_keeps_the_ptp():
    assert 'ptp_txt += " (onset not detected)"' in SRC
    assert "no response detected" not in SRC   # a missed onset is not proof of none
    i = SRC.index("def _redraw_auc_fill")
    assert 'if m.get("onset_auto_failed"):' in SRC[i:i + 900]
    assert '"    |AUC|: not measured (no onset)"' in SRC
    assert 'field in ("onset_idx", "mep_offset_idx") and _onset_missing' in SRC
    assert "MEP offset (no onset, not reported)" in SRC
