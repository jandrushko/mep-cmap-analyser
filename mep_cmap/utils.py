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


# Characters no filename may contain on Windows (and "/" anywhere), plus the
# escape character itself so that the encoding stays reversible.
_FILENAME_UNSAFE = frozenset('<>:"/\\|?*%')


def filename_token(label) -> str:
    """A stimulus type or label made safe to put inside a filename.

    Stimulus types come from the recording, not from the analyst: Spike2
    DigMark codes between 32 and 126 decode to their ASCII character, so a
    perfectly ordinary marker code 62 arrives as ">" and code 63 as "?".
    Written straight into a path, those raise ``OSError: [Errno 22]`` on
    Windows part way through a run, and "/" would quietly create a
    subfolder on every platform.

    Only the characters that cannot appear in a filename are encoded, as
    ``%XX`` (their hex code), together with "%" itself and control
    characters. Everything else is left exactly as it was, so every name
    that already worked is unchanged, and no two different labels can map
    to the same token -- which a replace-with-underscore rule cannot
    promise: "=", ">" and "?" all became "_" and overwrote one another.
    """
    out = []
    for ch in str(label):
        if ch in _FILENAME_UNSAFE or ord(ch) < 32 or ord(ch) == 127:
            out.append(f"%{ord(ch):02X}")
        else:
            out.append(ch)
    return "".join(out)
