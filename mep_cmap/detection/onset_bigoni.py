"""
mep_cmap.detection.onset_bigoni
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Derivative-based MEP onset detector based on the method described in:

    Bigoni C, Cadic-Melchior A, Vassiliadis P, et al.
    "An automatized method to determine latencies of motor-evoked potentials
    under physiological and pathophysiological conditions."
    J Neural Eng. 19 (2022) 024002.
    https://doi.org/10.1088/1741-2552/ac636c

Original implementation: https://github.com/clovbig/MEP_latency

The algorithm does not rely on pre-stimulus baseline statistics and therefore
does not require threshold tuning. It identifies the onset as the start of
the longest consecutive run of positive-derivative samples in the rising edge
of the MEP waveform.

Adaptations for MEP-CMAP Analyser:
- Variable sampling rate (original fixed to 3 kHz)
- Variable search window (original fixed to 10–50 ms)
- Physiological latency bounds (min_latency_ms / max_latency_ms)
- Optional Savitzky-Golay smoothing before differentiation
- min_run_ms expressed in time rather than samples (sampling-rate agnostic)
- Returns None rather than 'nan' when detection fails

  • detect_mep_onset_bigoni
"""

import numpy as np
from scipy.signal import savgol_filter


def detect_mep_onset_bigoni(
        signal, fs, *,
        pre_ms=100,
        search_start_ms=5,
        search_end_ms=60,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        smooth_window_ms=0.5,
        min_run_ms=0.5,
        artefact_blank_ms=2.0):
    """
    Derivative-based MEP onset detector (Bigoni et al., J Neural Eng 2022).

    Faithful implementation of the double-diff run-finding algorithm from
    https://github.com/clovbig/MEP_latency (epoch_c.py, latency_bigoni_method)
    with adaptations for variable sampling rate and search windows.

    Algorithm (matching original)
    ------------------------------
    1.  Extract the post-stimulus search window.
    2.  Amplitude gate — return None if no MEP present.
    3.  Optional Savitzky-Golay smoothing.
    4.  Find positive peak (argmax) and negative peak (argmin).
    5.  If negative peak comes before positive peak, negate the signal
        so the dominant deflection always rises positively.
    6.  Compute np.diff over signal[:p_peak] — the rising portion only.
    7.  Find indices where derivative > 0 (idx_positive).
    8.  Find consecutive pairs within idx_positive (idx_positive_diff).
    9.  Find consecutive pairs within idx_positive_diff (idx_positive_diff_diff).
   10.  Split into chunks; find the longest chunk >= min_run_samp.
   11.  Onset = idx_positive[idx_positive_diff[c_longest[0]]] mapped back
        to the full signal index space.
   12.  Apply physiological bounds — return None if outside.

    Parameters
    ----------
    signal              : 1-D np.ndarray  EMG segment (pre-stim + post-stim)
    fs                  : float           sampling frequency in Hz
    pre_ms              : float           ms of pre-stim data in the segment
    search_start_ms     : float           ms post-stim to begin search (default 5)
    search_end_ms       : float           ms post-stim to end search (default 60)
    min_latency_ms      : float or None   physiological floor (ms post-stim).
                          If None, defaults to artefact_blank_ms.
    max_latency_ms      : float or None   physiological ceiling (ms post-stim).
                          If None, defaults to search_end_ms.
    min_peak_amplitude  : float           amplitude gate in mV (default 0.05)
    smooth_window_ms    : float           Savitzky-Golay window in ms (default 0.5).
                          Set to 0 to disable. Small values preserve edge sharpness.
    min_run_ms          : float           minimum chunk length in ms (default 0.5).
                          Equivalent to min_len=4 samples at 3 kHz in the original.
    artefact_blank_ms   : float           hard floor — onset never placed before
                          this time (ms post-stim).

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

    # ── Optional smoothing ────────────────────────────────────────────────────
    if smooth_window_ms > 0:
        sg_win = max(3, int(smooth_window_ms * fs / 1000))
        if sg_win % 2 == 0:
            sg_win += 1
        if sg_win < len(window):
            window = savgol_filter(window, window_length=sg_win, polyorder=2)

    # ── Steps 4–5: polarity correction (faithful to original) ────────────────
    p_peak = int(np.argmax(window))
    n_peak = int(np.argmin(window))

    if p_peak > n_peak:
        # Negative deflection comes first — negate so dominant rise is positive
        window = -window
        p_peak = int(np.argmax(window))

    if p_peak < 2:
        return None

    # ── Step 6: first derivative over rising portion only ────────────────────
    first_derv = np.diff(window[:p_peak])

    # ── Steps 7–9: double-diff run finding (faithful to original) ────────────
    idx_positive = np.argwhere(first_derv > 0).flatten()
    if len(idx_positive) < 2:
        return None

    idx_positive_diff = np.where(np.diff(idx_positive) == 1)[0]
    if len(idx_positive_diff) < 2:
        return None

    idx_positive_diff_diff = np.where(np.diff(idx_positive_diff) == 1)[0]
    if len(idx_positive_diff_diff) == 0:
        return None

    # Split into chunks of consecutive indices
    split_points = (np.where(np.diff(idx_positive_diff_diff) > 1)[0] + 1).tolist()
    chunks = np.split(idx_positive_diff_diff, split_points)

    # ── Step 10: find the longest chunk >= min_run_samp ───────────────────────
    min_run_samp = max(1, int(min_run_ms * fs / 1000))
    c_longest    = None
    max_len      = min_run_samp

    for c in chunks:
        if len(c) > max_len:
            max_len   = len(c)
            c_longest = c

    if c_longest is None:
        return None

    # ── Step 11: map onset back to full signal index space ────────────────────
    # Matches original: idx_positive[idx_positive_diff[c_longest[0]]]
    onset_in_window = int(idx_positive[idx_positive_diff[c_longest[0]]])
    onset_global    = win_start + onset_in_window
    latency_ms      = (onset_global - stim_idx) * ms_per_samp

    # ── Step 12: physiological bounds check ──────────────────────────────────
    if latency_ms < _min_lat or latency_ms > _max_lat:
        return None

    return round(float(latency_ms), 2)
