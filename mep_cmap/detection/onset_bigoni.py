"""
mep_cmap.detection.onset_bigoni
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Derivative-based MEP onset detector based on the method described in:

    Bigoni C, Cadic-Melchior A, Vassiliadis P, et al.
    "An automatized method to determine latencies of motor-evoked potentials
    under physiological and pathophysiological conditions."
    J Neural Eng. 19 (2022) 024002.
    https://doi.org/10.1088/1741-2552/ac636c

The algorithm does not rely on pre-stimulus baseline statistics and therefore
does not require 'magic numbers' for threshold tuning. It identifies the onset
as the start of the steepest sustained rising edge in the MEP waveform.

Key adaptation for MEP-CMAP Analyser:
- Variable sampling rate (not fixed to 3 kHz)
- Variable search window (not fixed to 10–50 ms)
- Physiological latency bounds (min_latency_ms / max_latency_ms)
- Active-contraction compatibility: works on the raw signal rather than
  requiring near-silent pre-stimulus baseline
- Returns None rather than a floor value when detection is uncertain

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
        smooth_window_ms=2.0,
        min_run_ms=1.0,
        artefact_blank_ms=2.0):
    """
    Derivative-based MEP onset detector (Bigoni et al., J Neural Eng 2022).

    Algorithm
    ---------
    1.  Extract the post-stimulus search window [search_start_ms, search_end_ms].
    2.  Apply amplitude gate — return None if no MEP is present.
    3.  Optionally smooth with a Savitzky-Golay filter to reduce derivative noise.
    4.  Find the dominant peak (max absolute value) and its polarity.
    5.  If the MEP is negative-first (trough before peak), negate the window so
        the dominant deflection always rises positively — this ensures the
        derivative approach works correctly for biphasic MEPs of either polarity.
    6.  Compute the approximate first derivative (np.diff) of the rectified
        signal from the start of the window up to and including the dominant peak.
    7.  Find all consecutive runs of positive derivative samples.
    8.  Select the longest such run — this corresponds to the steepest sustained
        rising edge of the MEP, which is the most reliable indicator of onset.
    9.  The onset is the first sample of that longest run.
   10.  Apply physiological bounds: clamp to [min_latency_ms, max_latency_ms].
        Return None if the result falls outside the window after clamping.

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
    smooth_window_ms    : float           Savitzky-Golay window in ms (default 2.0 ms).
                          Set to 0 to disable smoothing.
    min_run_ms          : float           minimum length of positive-derivative run
                          to be considered as a candidate onset (default 1.0 ms).
                          Prevents single-sample noise spikes from being selected.
    artefact_blank_ms   : float           hard floor — onset never placed before
                          this time (ms post-stim), regardless of other settings.

    Returns
    -------
    latency_ms : float, or None if no MEP detected or onset is ambiguous
    """
    ms_per_samp = 1000.0 / fs

    # ── Index arithmetic ──────────────────────────────────────────────────────
    stim_idx     = int(pre_ms * fs / 1000)
    win_start    = int((pre_ms + search_start_ms) * fs / 1000)
    win_end      = int((pre_ms + search_end_ms)   * fs / 1000)
    win_end      = min(win_end, len(signal))

    if win_start >= win_end or win_start >= len(signal):
        return None

    # Physiological bounds
    _min_lat = min_latency_ms if min_latency_ms is not None else artefact_blank_ms
    _max_lat = max_latency_ms if max_latency_ms is not None else search_end_ms
    _min_lat = max(_min_lat, artefact_blank_ms)  # hard floor

    # ── Amplitude gate ────────────────────────────────────────────────────────
    window = signal[win_start:win_end].copy()
    if len(window) == 0:
        return None

    ptp = float(np.max(window) - np.min(window))
    if ptp < min_peak_amplitude:
        return None

    # ── Optional smoothing ────────────────────────────────────────────────────
    # Savitzky-Golay smoothing reduces derivative noise without shifting edges.
    # Window length must be odd and >= 3.
    if smooth_window_ms > 0:
        sg_win = max(3, int(smooth_window_ms * fs / 1000))
        if sg_win % 2 == 0:
            sg_win += 1
        if sg_win < len(window):
            window = savgol_filter(window, window_length=sg_win, polyorder=2)

    # ── Step 4: find dominant peak and polarity ───────────────────────────────
    # Use the max absolute value as the dominant peak.
    abs_window   = np.abs(window)
    peak_local   = int(np.argmax(abs_window))   # index within window

    # Determine polarity: is the dominant peak positive or negative?
    # If negative (trough before peak in original), negate so the rising
    # edge always goes upward — required for derivative logic.
    if window[peak_local] < 0:
        window = -window

    # ── Step 5: slice from window start to (inclusive of) dominant peak ───────
    # Only examine the rising portion of the MEP.
    rise_segment = window[:peak_local + 1]
    if len(rise_segment) < 2:
        return None

    # ── Step 6: approximate first derivative ─────────────────────────────────
    deriv = np.diff(rise_segment)       # length = len(rise_segment) - 1

    # ── Step 7–8: find all positive-derivative runs, pick the longest ─────────
    min_run_samp = max(1, int(min_run_ms * fs / 1000))
    positive     = deriv > 0

    # Run-length encode the positive mask
    best_start  = None
    best_len    = 0
    run_start   = None

    for i, p in enumerate(positive):
        if p:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len > best_len:
                    best_len  = run_len
                    best_start = run_start
                run_start = None

    # Handle run extending to the end
    if run_start is not None:
        run_len = len(positive) - run_start
        if run_len > best_len:
            best_len   = run_len
            best_start = run_start

    # No qualifying positive run found
    if best_start is None or best_len < min_run_samp:
        return None

    # ── Step 9: onset = first sample of the longest positive-derivative run ───
    # best_start is an index into `deriv`, which is offset by 1 from `window`.
    # Adding win_start converts back to the full-signal index space.
    onset_in_window = best_start          # index in rise_segment / window
    onset_global    = win_start + onset_in_window
    latency_ms      = (onset_global - stim_idx) * ms_per_samp

    # ── Step 10: physiological bounds check ──────────────────────────────────
    if latency_ms < _min_lat or latency_ms > _max_lat:
        return None

    return round(float(latency_ms), 2)
