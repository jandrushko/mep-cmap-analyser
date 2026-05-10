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


def design_notch_sos(fs: float, f0: float, q: float,
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
    """
    nyq      = 0.5 * fs
    sos_list = []
    n        = 1
    while True:
        f = f0 * n
        if f >= nyq:
            break
        sos_list.append(iirnotch(f / nyq, q))
        n += 1 if include_harmonics else float('inf')
    return sos_list
