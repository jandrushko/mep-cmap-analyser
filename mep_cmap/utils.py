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


# ── Windows long paths ───────────────────────────────────────────────────────
#
# Windows refuses any path over 260 characters (MAX_PATH) unless long paths
# are switched on machine-wide, which university PCs rarely allow -- and it
# says so as "[Errno 2] No such file or directory", not as a length problem.
# A study kept a few folders deep in OneDrive reaches that easily: the BIDS
# layout (derivatives/mep-cmap-analyser/sub/ses/emg), a full BIDS prefix and a
# condition label in each figure's name came to 268 characters on a lab PC,
# and the run failed at its first figure after writing the trial CSVs.
#
# The extended-length form, \\?\C:\..., is exempt from the limit with no
# setting at all. Paths for WRITING go through long_path(); anything a person
# reads or a file stores goes through display_path(), so the prefix never
# reaches a log line, a session JSON or a manifest.

_LONG_PREFIX = "\\\\?\\"
_LONG_UNC_PREFIX = "\\\\?\\UNC\\"

# Longest path Windows accepts without the extended-length form.
WINDOWS_MAX_PATH = 259


def long_path(path):
    """``path`` in a form Windows will write however long it is.

    Absolute and normalised, then prefixed with ``\\\\?\\`` (or
    ``\\\\?\\UNC\\`` for a network share). Unchanged on other systems, for an
    empty path, or if it is already in that form. Everything joined beneath
    the result inherits the exemption, so prefixing an output folder once is
    enough for every file written into it.
    """
    import os
    if os.name != "nt" or not path:
        return path
    p = str(path)
    if p.startswith(_LONG_PREFIX):
        return p
    p = os.path.abspath(p)
    if p.startswith("\\\\"):                 # \\server\share\...
        return _LONG_UNC_PREFIX + p[2:]
    return _LONG_PREFIX + p


def display_path(path):
    """``path`` without the extended-length prefix, for logs and storage."""
    if not path:
        return path
    p = str(path)
    if p.startswith(_LONG_UNC_PREFIX):
        return "\\\\" + p[len(_LONG_UNC_PREFIX):]
    if p.startswith(_LONG_PREFIX):
        return p[len(_LONG_PREFIX):]
    return p


def path_too_long_hint(exc):
    """A plain explanation if ``exc`` is Windows refusing a long path; else "".

    For the places the extended-length form cannot reach (a program that
    strips it, a share that rejects it), so the analyst is told what to do
    rather than shown "No such file or directory" for a folder that exists.
    """
    import os
    if os.name != "nt" or not isinstance(exc, OSError):
        return ""
    names = [getattr(exc, "filename", None), getattr(exc, "filename2", None)]
    longest = max((len(display_path(n)) for n in names if n), default=0)
    if longest <= WINDOWS_MAX_PATH:
        return ""
    return (f"That path is {longest} characters long, and Windows refuses "
            f"paths over {WINDOWS_MAX_PATH + 1} unless long paths are enabled. "
            f"Move the study to a shorter folder (closer to the top of "
            f"OneDrive or the drive), or shorten the condition labels.")
