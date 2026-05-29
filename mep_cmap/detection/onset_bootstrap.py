"""
mep_cmap.detection.onset_bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bootstrap-calibrated MEP onset detector using a peak-anchored backward scan
within a physiologically bounded window.

  • detect_mep_onset_bootstrap
"""

import numpy as np
from .bootstrap_baseline import compute_bootstrap_threshold


def detect_mep_onset_bootstrap(
        signal, fs, *,
        pre_ms=100,
        peak_search_start_ms=10,
        peak_search_end_ms=60,
        min_latency_ms=None,
        max_latency_ms=None,
        min_peak_amplitude=0.05,
        criterion=1.96,
        n_boot=500,
        min_duration_ms=5,
        artefact_blank_ms=2,
        seed=42):
    """
    Bootstrap-calibrated MEP onset detector — peak-anchored backward scan
    within a physiologically bounded window.

    Strategy
    --------
    1.  Confirm MEP exists via amplitude gate in PTP window.
    2.  Establish pre-stim bootstrap threshold.
    3.  Within [min_latency_ms, max_latency_ms], find the largest absolute
        peak — this anchors detection to the actual MEP rather than the
        first noise crossing.
    4.  Scan BACKWARD from that peak to find where the sliding-window mean
        first drops below threshold — that crossing is the onset.
    5.  Onset is bounded on the left by min_latency_ms (physiological floor).

    This approach is more robust than forward scanning because:
    - It ignores early artefacts and noise spikes before the MEP
    - Biphasic MEPs are handled correctly: the backward scan from the
      dominant peak crosses the true onset regardless of polarity
    - The physiological bounds prevent grossly implausible placements

    Parameters
    ----------
    signal              : 1-D EMG segment (pre-stim + post-stim)
    fs                  : sampling frequency in Hz
    pre_ms              : ms of pre-stim data in the segment
    peak_search_start_ms: start of PTP search window (ms post-stim)
    peak_search_end_ms  : end of PTP search window (ms post-stim)
    min_latency_ms      : earliest plausible onset (ms post-stim).
                          If None, defaults to artefact_blank_ms.
    max_latency_ms      : latest plausible onset (ms post-stim).
                          If None, defaults to peak_search_end_ms.
    min_peak_amplitude  : minimum MEP amplitude gate (mV)
    criterion           : z-score multiplier for onset threshold (default 1.96)
    n_boot              : bootstrap iterations (default 500)
    min_duration_ms     : sliding window width in ms (default 5 ms)
    artefact_blank_ms   : hard minimum — search never starts before this
    seed                : RNG seed for reproducibility

    Returns
    -------
    latency_ms : float, or None if no MEP detected
    """
    samples_before    = int(pre_ms                          * fs / 1000)
    peak_search_start = int((pre_ms + peak_search_start_ms) * fs / 1000)
    peak_search_end   = int((pre_ms + peak_search_end_ms)   * fs / 1000)
    peak_search_end   = min(peak_search_end, len(signal))

    # Physiological window for onset search
    _min_lat = min_latency_ms if min_latency_ms is not None else artefact_blank_ms
    _max_lat = max_latency_ms if max_latency_ms is not None else peak_search_end_ms
    _min_lat = max(_min_lat, artefact_blank_ms)  # hard floor

    onset_search_start = int((pre_ms + _min_lat) * fs / 1000)
    onset_search_end   = int((pre_ms + _max_lat) * fs / 1000)
    onset_search_end   = min(onset_search_end, len(signal))
    win_samp           = max(2, int(min_duration_ms * fs / 1000))

    if peak_search_start >= peak_search_end:
        return None

    # ── Amplitude gate ────────────────────────────────────────────────────────
    post_abs = np.abs(signal[peak_search_start:peak_search_end])
    if len(post_abs) == 0 or post_abs.max() < min_peak_amplitude:
        return None

    # ── Bootstrap threshold from pre-stim baseline ───────────────────────────
    pre_abs = np.abs(signal[:samples_before])
    threshold, mu_abs, _ = compute_bootstrap_threshold(
        pre_abs, criterion=criterion, n_boot=n_boot, seed=seed)
    if threshold is None:
        return None

    # ── Peak-anchored backward scan ───────────────────────────────────────────
    if onset_search_start >= onset_search_end:
        return None

    search_win  = np.abs(signal[onset_search_start:onset_search_end])
    if len(search_win) == 0:
        return None
    peak_local  = int(np.argmax(search_win))
    peak_global = onset_search_start + peak_local

    # scan_start placed 2×win_samp before the peak to ensure enough room for
    # the backward scan on steep short-duration responses (M-waves, fast MEPs).
    scan_start  = max(onset_search_start, peak_global - 2 * win_samp)
    scan_range  = max(1, peak_global - scan_start)
    sustain     = max(1, min(win_samp // 2, scan_range // 3))
    onset_idx   = onset_search_start   # default: physiological floor
    below_count = 0

    for i in range(scan_start, onset_search_start - 1, -1):
        i1       = min(len(signal), i + win_samp)
        win_mean = float(np.mean(np.abs(signal[i:i1]))) if i1 > i else 0.0
        if win_mean < threshold:
            below_count += 1
            if below_count >= sustain:
                onset_idx = min(i + sustain, peak_global)
                break
        else:
            below_count = 0

    latency_ms = (onset_idx - samples_before) * 1000.0 / fs
    return round(float(latency_ms), 2)
