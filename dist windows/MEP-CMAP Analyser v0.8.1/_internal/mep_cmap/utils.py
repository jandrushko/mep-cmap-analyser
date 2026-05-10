"""
mep_cmap.utils
~~~~~~~~~~~~~~
Small shared helper functions used across multiple modules.
"""

import numpy as np
import pandas as pd


def _add_time_and_digmark(df, stim_idx, fs):
    """
    Add Time_ms and DigMark columns to a segment DataFrame.

    Parameters
    ----------
    df        : DataFrame whose rows are samples and columns are trials
    stim_idx  : row index corresponding to DigMark (t = 0 ms)
    fs        : sampling rate in Hz

    Returns
    -------
    Copy of *df* with two new leading columns:
        • 'Time_ms'  — negative before pulse, positive after
        • 'DigMark'  — 1 at stim_idx, 0 elsewhere
    """
    dt_ms   = 1000 / fs
    n_rows  = len(df)
    time_ms = (np.arange(n_rows) - stim_idx) * dt_ms
    dig     = np.zeros(n_rows, dtype=int)
    dig[stim_idx] = 1

    df = df.copy()
    df.insert(0, "Time_ms", time_ms)
    df.insert(1, "DigMark", dig)
    return df
