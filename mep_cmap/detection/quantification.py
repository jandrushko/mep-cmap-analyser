"""
mep_cmap.detection.quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared signal quantification functions used by both the pipeline and
the Data Inspector, ensuring a single source of truth for all scalar
trial metrics.

  • compute_ptp          — peak-to-peak amplitude within a window
  • compute_auc          — area under the rectified signal between two indices
  • compute_prestim_rms  — pre-stimulus RMS
  • compute_prestim_ptp  — pre-stimulus peak-to-peak
"""

import numpy as np

try:
    from ..compat import _np_trapz, _np_ptp
except ImportError:
    # Fallback for standalone use / testing
    _np_trapz = np.trapz
    _np_ptp   = np.ptp


def compute_ptp(segment, start_idx, end_idx):
    """
    Peak-to-peak amplitude of *segment* within [start_idx, end_idx).

    Parameters
    ----------
    segment   : 1-D np.ndarray  EMG trial segment
    start_idx : int             window start (samples)
    end_idx   : int             window end (samples, exclusive)

    Returns
    -------
    ptp : float  peak-to-peak amplitude (same units as segment)
    """
    window = segment[start_idx:end_idx]
    if len(window) == 0:
        return 0.0
    return float(_np_ptp(window))


def compute_auc(segment, onset_idx, end_idx, fs):
    """
    Area under the rectified EMG signal from onset_idx to end_idx.

    Uses the trapezoidal rule on |segment|. The result is in mV·s when
    the segment is in mV and fs is in Hz.

    Parameters
    ----------
    segment   : 1-D np.ndarray  EMG trial segment
    onset_idx : int             onset sample index (inclusive)
    end_idx   : int             end sample index (exclusive)
    fs        : float           sampling frequency in Hz

    Returns
    -------
    auc : float, or None if window is empty or invalid
    """
    if end_idx <= onset_idx:
        return None
    window = np.abs(segment[onset_idx:end_idx])
    if len(window) == 0:
        return None
    return float(_np_trapz(window, dx=1.0 / fs))


def compute_prestim_rms(prestim_segment):
    """
    Root-mean-square of the pre-stimulus segment.

    Parameters
    ----------
    prestim_segment : 1-D np.ndarray  pre-stimulus EMG samples

    Returns
    -------
    rms : float
    """
    if len(prestim_segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(prestim_segment ** 2)))


def compute_prestim_ptp(prestim_segment):
    """
    Peak-to-peak amplitude of the pre-stimulus segment.

    Parameters
    ----------
    prestim_segment : 1-D np.ndarray  pre-stimulus EMG samples

    Returns
    -------
    ptp : float
    """
    if len(prestim_segment) == 0:
        return 0.0
    return float(_np_ptp(prestim_segment))
