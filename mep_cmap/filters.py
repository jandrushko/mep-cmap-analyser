"""
mep_cmap.filters
~~~~~~~~~~~~~~~~
EMG signal filtering functions.

  • adaptive_mains_cancel  — least-squares mains noise subtraction
  • design_notch_sos       — IIR notch filter design (with harmonics)
"""

import numpy as np
from scipy.signal import iirnotch


def adaptive_mains_cancel(data: np.ndarray,
                          fs: float,
                          mains_freq: float = 50.0,
                          n_harmonics: int = 6,
                          win_s: float = 1.0,
                          overlap: float = 0.5) -> np.ndarray:
    """
    Subtract a sine-series model of the mains (and its harmonics) from *data*.

    Parameters
    ----------
    data        : 1-D EMG trace (numpy array)
    fs          : sampling frequency (Hz)
    mains_freq  : 50.0 Hz (Europe) or 60.0 Hz (US/Japan)
    n_harmonics : how many integer harmonics to model (>=1)
    win_s       : window length for each adaptive fit (seconds)
    overlap     : fraction overlap between consecutive windows (0-0.9)

    Returns
    -------
    cleaned : numpy array, same length as *data*
    """
    if n_harmonics < 1:
        return data.copy()

    n    = len(data)
    step = int(win_s * fs * (1 - overlap))
    win  = int(win_s * fs)
    if win < 4:
        raise ValueError("Window too short for adaptive mains cancel")

    t       = np.arange(n) / fs
    cleaned = data.copy()

    basis = []
    for h in range(1, n_harmonics + 1):
        ang = 2 * np.pi * mains_freq * h * t
        basis.append(np.sin(ang))
        basis.append(np.cos(ang))
    B = np.column_stack(basis)

    for start in range(0, n, step):
        stop      = min(start + win, n)
        Bw        = B[start:stop]
        yw        = cleaned[start:stop]
        c, *_     = np.linalg.lstsq(Bw, yw, rcond=None)
        cleaned[start:stop] = yw - Bw @ c

    return cleaned


def design_notch_filters(fs: float, f0: float, q: float,
                         include_harmonics: bool = False) -> list:
    """
    Return a list of (b, a) pairs implementing a notch at *f0* Hz and,
    if requested, every integer multiple (harmonic) up to Nyquist.

    Parameters
    ----------
    fs                : sampling frequency in Hz
    f0                : fundamental notch frequency in Hz
    q                 : Q-factor
    include_harmonics : if True, also notch at 2*f0, 3*f0, ...

    Returns
    -------
    list of (b, a) coefficient pairs, one per notch frequency

    Notes
    -----
    These are TRANSFER-FUNCTION coefficients from ``iirnotch``, not
    second-order sections. Apply them with ``filtfilt(b, a, x)`` in a loop;
    passing them to ``sosfilt``/``sosfiltfilt`` will not work. The function
    was previously named ``design_notch_sos``, which said otherwise; that
    name is kept as an alias below so existing scripts keep running.
    """
    if not np.isfinite(f0) or f0 <= 0:
        raise ValueError(f"Notch frequency must be finite and positive, got {f0!r}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling frequency must be finite and positive, got {fs!r}")

    nyq = 0.5 * fs
    # Largest harmonic index that could still fall below Nyquist. Bounding the
    # range up front replaces a `while True` whose exit depended on adding
    # float('inf') to an int counter -- that worked, but it turned `n` into a
    # float and would have looped forever had f0 ever reached this function
    # as 0 (f0 * inf -> nan, and nan >= nyq is False).
    n_max = int(nyq // f0) if include_harmonics else 1

    return [iirnotch(f0 * n / nyq, q)
            for n in range(1, n_max + 1)
            if f0 * n < nyq]


#: Former name. ``iirnotch`` returns (b, a), never second-order sections, so
#: the old name misdescribed the return value; kept so external analysis
#: scripts written against v1.4.x and earlier continue to import cleanly.
design_notch_sos = design_notch_filters

