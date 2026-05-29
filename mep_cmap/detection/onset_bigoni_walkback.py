"""
mep_cmap.detection.onset_bigoni_walkback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modified Bigoni derivative-based onset detector with baseline walkback.

This method extends the original Bigoni et al. (2022) algorithm with a
post-detection walkback step that moves the onset back to the true point
of signal departure from baseline, correcting the systematic late-placement
that occurs when the longest positive-derivative run starts mid-rise.

Algorithm
---------
Steps 1–11 are identical to onset_bigoni.py (faithful Bigoni implementation).

Step 12 — Baseline walkback (modification):
    After the longest-run onset is found, scan backward sample-by-sample
    from that point toward the physiological floor. At each step, check
    whether the signal has returned to within `walkback_sd_mult` SDs of
    the pre-stimulus baseline mean. The walkback stops at the last sample
    that is still clearly above baseline — this is the true departure point.

    Walkback threshold = baseline_mean + walkback_sd_mult * baseline_sd

    A conservative multiplier (default 1.0) ensures we stop at a genuine
    departure rather than landing in noise. Increase to be more conservative
    (onset placed later), decrease to be more aggressive (onset placed earlier).

Reference for base algorithm:
    Bigoni C, Cadic-Melchior A, Vassiliadis P, et al.
    J Neural Eng. 19 (2022) 024002.
    https://doi.org/10.1088/1741-2552/ac636c
    https://github.com/clovbig/MEP_latency

  • detect_mep_onset_bigoni_walkback
"""

import numpy as np
from scipy.signal import savgol_filter


def detect_mep_onset_bigoni_walkback(
        signal, fs, *,
        pre_ms=100,
        search_start_ms=5,
        search_end_ms=60,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        smooth_window_ms=0.5,
        min_run_ms=0.5,
        artefact_blank_ms=2.0,
        walkback_sd_mult=1.0):
    """
    Bigoni derivative method + baseline walkback refinement.

    Parameters
    ----------
    signal              : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs                  : float           sampling frequency in Hz
    pre_ms              : float           ms of pre-stim data in the segment
    search_start_ms     : float           ms post-stim to begin search
    search_end_ms       : float           ms post-stim to end search
    min_latency_ms      : float or None   physiological floor (ms post-stim)
    max_latency_ms      : float or None   physiological ceiling (ms post-stim)
    min_peak_amplitude  : float           amplitude gate in mV (default 0.05)
    smooth_window_ms    : float           Savitzky-Golay window in ms (default 0.5)
    min_run_ms          : float           minimum chunk length in ms (default 0.5)
    artefact_blank_ms   : float           hard floor ms post-stim
    walkback_sd_mult    : float           SD multiplier for walkback threshold
                          (default 1.0). Lower = earlier onset, higher = later.

    Returns
    -------
    latency_ms : float, or None if no MEP detected or onset is ambiguous
    """
    ms_per_samp = 1000.0 / fs

    # ── Index arithmetic ──────────────────────────────────────────────────────
    stim_idx  = int(pre_ms * fs / 1000)
    win_start = int((pre_ms + search_start_ms) * fs / 1000)
    win_end   = int((pre_ms + search_end_ms)   * fs / 1000)
    win_end   = min(win_end, len(signal))

    if win_start >= win_end or win_start >= len(signal):
        return None

    # Physiological bounds
    _min_lat = min_latency_ms if min_latency_ms is not None else artefact_blank_ms
    _max_lat = max_latency_ms if max_latency_ms is not None else search_end_ms
    _min_lat = max(_min_lat, artefact_blank_ms)

    # ── Amplitude gate ────────────────────────────────────────────────────────
    window = signal[win_start:win_end].copy()
    if len(window) == 0:
        return None

    ptp = float(np.max(window) - np.min(window))
    if ptp < min_peak_amplitude:
        return None

    # ── Pre-stimulus baseline stats (used in walkback step) ───────────────────
    pre_signal   = signal[:stim_idx]
    baseline_mean = float(np.mean(np.abs(pre_signal))) if len(pre_signal) > 0 else 0.0
    baseline_sd   = float(np.std(pre_signal, ddof=1))  if len(pre_signal) > 1 else 0.0
    walkback_thr  = baseline_mean + walkback_sd_mult * baseline_sd

    # ── Optional smoothing ────────────────────────────────────────────────────
    if smooth_window_ms > 0:
        sg_win = max(3, int(smooth_window_ms * fs / 1000))
        if sg_win % 2 == 0:
            sg_win += 1
        if sg_win < len(window):
            window = savgol_filter(window, window_length=sg_win, polyorder=2)

    # ── Polarity correction ───────────────────────────────────────────────────
    p_peak = int(np.argmax(window))
    n_peak = int(np.argmin(window))

    if p_peak > n_peak:
        window = -window
        p_peak = int(np.argmax(window))

    if p_peak < 2:
        return None

    # ── First derivative over rising portion ─────────────────────────────────
    first_derv = np.diff(window[:p_peak])

    # ── Double-diff run finding (faithful Bigoni) ─────────────────────────────
    idx_positive = np.argwhere(first_derv > 0).flatten()
    if len(idx_positive) < 2:
        return None

    idx_positive_diff = np.where(np.diff(idx_positive) == 1)[0]
    if len(idx_positive_diff) < 2:
        return None

    idx_positive_diff_diff = np.where(np.diff(idx_positive_diff) == 1)[0]
    if len(idx_positive_diff_diff) == 0:
        return None

    split_points = (np.where(np.diff(idx_positive_diff_diff) > 1)[0] + 1).tolist()
    chunks = np.split(idx_positive_diff_diff, split_points)

    min_run_samp = max(1, int(min_run_ms * fs / 1000))
    c_longest    = None
    max_len      = min_run_samp

    for c in chunks:
        if len(c) > max_len:
            max_len   = len(c)
            c_longest = c

    if c_longest is None:
        return None

    # Bigoni onset — start of longest positive-derivative run
    onset_in_window = int(idx_positive[idx_positive_diff[c_longest[0]]])
    onset_global    = win_start + onset_in_window

    # ── Walkback step ─────────────────────────────────────────────────────────
    # Scan backward from the Bigoni onset toward the physiological floor.
    # Stop at the last sample where |signal| is still above the walkback
    # threshold — this is the true departure from baseline.
    floor_idx = int((pre_ms + _min_lat) * fs / 1000)
    wb_onset  = onset_global  # default: keep Bigoni onset if walkback finds nothing

    for i in range(onset_global, floor_idx - 1, -1):
        if abs(signal[i]) <= walkback_thr:
            # Signal has returned to baseline — onset is one sample forward
            wb_onset = i + 1
            break

    onset_global = max(wb_onset, floor_idx)
    latency_ms   = (onset_global - stim_idx) * ms_per_samp

    # ── Physiological bounds check ────────────────────────────────────────────
    if latency_ms < _min_lat or latency_ms > _max_lat:
        return None

    return round(float(latency_ms), 2)
