"""
Onset detection kept out of the stimulus artefact (v1.4.15).

Real-data evidence (not in the repository): on an FDI M-wave recruitment
curve 1.4.14 put an onset on the artefact (2.0-2.2 ms) on every trial below
and near motor threshold; on lower-limb Mmax trials it reported the flat
post-artefact plateau (4.2 ms). With these rules the sub-threshold trials
have no onset, the small M-waves are found at ~5.6 ms, the Mmax onsets land
on the M-wave (5.4-7.2 ms), and 186 of 191 trials in a TMS study -- every
TMS MEP -- are identical to 1.4.14.

These tests rebuild the shapes synthetically.
"""

import numpy as np
import pytest

from mep_cmap.detection import artefact as A
from mep_cmap.detection import onset_methods_median as MM
from mep_cmap.detection import dispatch as DSP
from mep_cmap.pipeline import LAT_COLS, low_agreement_warnings

FS, PRE = 5000.0, 100.0
RNG = np.random.default_rng(3)


def _trace(post_ms=60.0, noise=0.01):
    n_pre, n_post = int(PRE * FS / 1000), int(post_ms * FS / 1000)
    x = RNG.normal(0, noise, n_pre + n_post)
    t = (np.arange(n_pre + n_post) - n_pre) / FS * 1000.0
    return x, t


def _artefact(x, t, tail=-0.12, tau=2.5):
    """A sharp pulse, then a dip that drifts back to baseline (FDI-like)."""
    x = x.copy()
    x += np.where((t >= 0) & (t < 1.0), 3.0 * np.sin(np.pi * np.clip(t, 0, 1)), 0)
    x += np.where(t >= 1.0, tail * np.exp(-(t - 1.0) / tau), 0)
    return x


def _mwave(x, t, onset=5.0, amp=3.0):
    """A biphasic M-wave starting at ``onset`` ms."""
    x = x.copy()
    u = np.clip(t - onset, 0, None)
    x += np.where(t >= onset, amp * np.sin(2 * np.pi * u / 8.0) * np.exp(-u / 6.0), 0)
    return x


def _agree(x, start=2.0, end=50.0, lo=2.0, hi=12.0):
    return MM.compute_onset_agreement(
        x, FS, pre_ms=PRE, search_start_ms=start, search_end_ms=end,
        min_latency_ms=lo, max_latency_ms=hi, min_peak_amplitude=0.05)


# ── the gate ─────────────────────────────────────────────────────────────────

def test_artefact_is_detected_at_an_early_search_start():
    x, t = _trace()
    assert A.artefact_at_start(_artefact(x, t), FS, PRE, 2.0)


def test_never_at_or_after_the_artefact_window():
    # Every TMS profile starts at 8 ms or later; nothing there is artefact.
    x, t = _trace()
    y = _artefact(x, t, tail=-2.0, tau=40.0)          # an absurdly long tail
    assert not A.artefact_at_start(y, FS, PRE, A.ARTEFACT_WINDOW_MS)
    assert not A.artefact_at_start(y, FS, PRE, 10.0)


def test_a_quiet_start_is_not_an_artefact():
    x, t = _trace()
    assert not A.artefact_at_start(x, FS, PRE, 2.0)


# ── the rules ────────────────────────────────────────────────────────────────

def test_settled_start_never_moves_earlier():
    x, t = _trace()
    y = _mwave(_artefact(x, t), t)
    assert A.settled_search_start(y, FS, PRE, 2.0, 50.0) >= 2.0
    assert A.settled_search_start(x, FS, PRE, 2.0, 50.0) == 2.0   # nothing to settle


def test_plateau_is_moved_to_baseline_on_a_copy_only():
    x, t = _trace()
    y = x.copy()
    y[t >= 1.0] -= 1.4                                 # flat plateau, -1.4 mV
    y = _mwave(y, t, onset=6.0, amp=4.0)
    z = A.plateau_corrected(y, FS, PRE, 4.0)
    assert z is not y
    assert abs(np.median(z[(t > 4) & (t < 5.5)])) < 0.1
    assert np.allclose(y[t < 0], z[t < 0])             # baseline untouched
    assert A.plateau_corrected(x, FS, PRE, 4.0) is x   # no plateau, no copy


def test_window_edge():
    assert A.on_window_edge(2.0, 2.0, FS)
    assert A.on_window_edge(2.2, 2.0, FS)
    assert not A.on_window_edge(2.6, 2.0, FS)
    assert not A.on_window_edge(None, 2.0, FS)
    assert not A.on_window_edge(2.0, None, FS)


def test_quorum_needs_a_majority():
    pm = {"bigoni": 5.0, "bigoni_walkback": 5.2, "rms_envelope": None,
          "cusum": None, "peak_fraction": 5.6}
    assert MM.consensus_onset(pm) == 5.2
    pm["peak_fraction"] = None
    assert MM.consensus_onset(pm) is None


# ── end to end on synthetic trials ───────────────────────────────────────────

def test_no_m_wave_gives_no_onset():
    x, t = _trace()
    assert _agree(_artefact(x, t)).consensus_ms is None


def test_a_clear_m_wave_is_found_after_the_artefact():
    x, t = _trace()
    on = _agree(_mwave(_artefact(x, t), t, onset=5.0)).consensus_ms
    assert on is not None and 4.0 <= on <= 6.5


def test_tms_trace_is_exactly_as_before(monkeypatch):
    # Quiet at a 10 ms start, MEP at 22 ms: the gate stays shut and the
    # result equals the one with every new rule switched off.
    x, t = _trace()
    y = _mwave(x, t, onset=22.0, amp=1.0)
    kw = dict(start=10.0, end=50.0, lo=18.0, hi=28.0)
    now = _agree(y, **kw).consensus_ms
    monkeypatch.setattr(MM, "artefact_at_start", lambda *a, **k: False)
    assert _agree(y, **kw).consensus_ms == now
    assert now is not None and 20.0 <= now <= 24.0


def test_dispatch_uses_the_same_gate(monkeypatch):
    x, t = _trace()
    y = _artefact(x, t)
    lat = DSP.dispatch_onset(y, FS, {"onset_method": "methods_median"}, pre_ms=PRE,
                             search_start_ms=2.0, search_end_ms=50.0,
                             min_latency_ms=2.0, max_latency_ms=12.0)
    assert lat is None


# ── the review flag ──────────────────────────────────────────────────────────

def _row(st, lat, n, cond=""):
    r = [None] * len(LAT_COLS)
    r[LAT_COLS.index("StimType")] = st
    r[LAT_COLS.index("Condition")] = cond
    r[LAT_COLS.index("Latency(ms)")] = lat
    r[LAT_COLS.index("Onset_Methods_N")] = n
    return r


def test_low_agreement_trials_are_named_not_blanked():
    rows = [_row("B", 22.0, 5), _row("B", 23.0, 2), _row("B", None, 0), _row("B", 21.0, 1)]
    (msg,) = low_agreement_warnings(rows)
    assert msg.startswith("B:") and "#1, #3" in msg and "2 trial(s)" in msg
    assert low_agreement_warnings([_row("B", 22.0, 4)]) == []
