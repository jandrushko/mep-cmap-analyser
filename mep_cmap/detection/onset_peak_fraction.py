"""
mep_cmap.detection.onset_peak_fraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MEP onset detection via peak-fraction amplitude threshold + slope backtracking.

  • detect_mep_onset_peak_fraction
"""

import numpy as np


def detect_mep_onset_peak_fraction(signal, fs,
                                   pre_ms=100,
                                   poststim_start_ms=5,
                                   poststim_end_ms=60,
                                   peak_frac=0.15,
                                   min_consecutive=5,
                                   min_peak_amplitude=0.05,
                                   slope_threshold=0.05):
    """
    MEP onset detection using peak-fraction amplitude threshold + slope backtracking.

    Finds the largest peak in the post-stimulus window, sets a threshold at
    peak_frac of that peak, then backtracks to find where the signal first
    rises through that threshold with a minimum slope.

    Parameters
    ----------
    signal              : 1-D EMG segment (pre-stim + post-stim)
    fs                  : sampling frequency in Hz
    pre_ms              : ms of pre-stim data in the segment
    poststim_start_ms   : ms after stim to begin search
    poststim_end_ms     : ms after stim to end search
    peak_frac           : fraction of peak amplitude for threshold crossing
    min_consecutive     : min samples continuously above threshold
    min_peak_amplitude  : minimum MEP amplitude to qualify (mV)
    slope_threshold     : minimum slope (mV/ms) for onset confirmation

    Returns
    -------
    latency_ms : float, or None if no MEP detected
    """
    samples_before = int(pre_ms * fs / 1000)
    start_idx      = int((pre_ms + poststim_start_ms) * fs / 1000)
    end_idx        = int((pre_ms + poststim_end_ms)   * fs / 1000)

    rectified = np.abs(signal)
    post_stim = rectified[start_idx:end_idx]
    peak_val  = np.max(post_stim)

    if peak_val < min_peak_amplitude:
        return None

    threshold = peak_frac * peak_val
    above     = rectified[start_idx:end_idx] > threshold

    for i in range(len(above) - min_consecutive + 1):
        if np.all(above[i:i + min_consecutive]):
            for j in range(i - 1, 0, -1):
                idx          = start_idx + j
                slope_per_ms = (rectified[idx] - rectified[idx - 1]) / (1000 / fs)
                if slope_per_ms < slope_threshold:
                    onset_sample = idx + 1
                    break
            else:
                onset_sample = start_idx
            return round((onset_sample - samples_before) * 1000 / fs, 2)

    return None
