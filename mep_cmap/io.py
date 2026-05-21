"""
mep_cmap.io
~~~~~~~~~~~
Format-agnostic public API for reading EMG data files.

Supported formats (auto-detected from file header)
----------------------------------------------------
  Spike-2 text export  — header contains "SUMMARY" / "START" / "CHANNEL" blocks
  LabChart text export — header line 0 starts with "Interval="

Adding a new format
-------------------
  1. Create mep_cmap/formats/<format>.py with the three public functions.
  2. Add detection logic to detect_format().
  3. Add a dispatch branch to each of the three public functions below.
  4. Nothing else in the codebase needs to change.

Public API
----------
  detect_format(file_path)                     -> 'spike2' | 'labchart' | ...
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]
"""

import os as _os

from .formats import spike2   as _spike2
from .formats import labchart as _labchart


def _resolve_path(file_path: str) -> str:
    """
    Resolve a possibly-relative path to absolute.

    Paths stored in the dataset JSON may be relative (for cross-computer /
    OneDrive portability) and may use backslashes on Windows.  This function
    normalises the slashes and searches a cascade of candidate roots until the
    file is found.
    """
    # Normalise backslashes → OS separator
    file_path = _os.path.normpath(file_path.replace("\\", _os.sep))
    if _os.path.isabs(file_path) and _os.path.exists(file_path):
        return file_path
    if _os.path.isabs(file_path):
        return file_path  # absolute but missing — let open() raise clearly

    import sys as _sys
    candidates = [_os.getcwd()]
    try:
        candidates.append(_os.path.dirname(_os.path.abspath(__file__)))
        candidates.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    except Exception:
        pass
    try:
        candidates.append(_os.path.dirname(_os.path.abspath(_sys.argv[0])))
    except Exception:
        pass
    # Walk up from cwd looking for the study root (contains derivatives/)
    walk = _os.getcwd()
    for _ in range(8):
        if _os.path.isdir(_os.path.join(walk, "derivatives")):
            candidates.append(walk)
            break
        parent = _os.path.dirname(walk)
        if parent == walk:
            break
        walk = parent

    for root in candidates:
        resolved = _os.path.normpath(_os.path.join(root, file_path))
        if _os.path.isfile(resolved):
            return resolved

    # Nothing found — return joined to cwd so open() gives a clear error
    return _os.path.normpath(_os.path.join(_os.getcwd(), file_path))


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(file_path: str) -> str:
    """
    Inspect the file header and return a format identifier string.

    Returns
    -------
    'labchart' — LabChart text export (line 0 starts with 'Interval=')
    'spike2'   — Spike-2 text export (default / fallback)
    """
    file_path = _resolve_path(file_path)
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
    if first_line.startswith('Interval='):
        return 'labchart'
    return 'spike2'


# ─────────────────────────────────────────────────────────────────────────────
# Public API — dispatches to the correct format reader
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return channel names for display in the channel selector."""
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'labchart':
        return _labchart.list_waveform_channels(file_path)
    return _spike2.list_waveform_channels(file_path)


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Load EMG waveform, sampling rate, and voltage unit for the given channel.

    Parameters
    ----------
    file_path   : path to the data file
    channel_idx : 0-based channel index

    Returns
    -------
    emg  : np.ndarray  raw EMG samples
    fs   : int         sampling frequency in Hz
    unit : str | None  voltage unit (e.g. 'mV'), or None
    """
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'labchart':
        return _labchart.extract_emg_waveform_and_fs(file_path, channel_idx)
    return _spike2.extract_emg_waveform_and_fs(file_path, channel_idx)


def extract_stim_times(file_path: str, marker_name: str) -> dict:
    """
    Return stimulation timestamps.

    For Spike-2 : marker_name selects the DigMark channel
                  (e.g. 'Keyboard', 'TTL').
    For LabChart: marker_name is used as the stim-type label
                  (single uppercase letter, e.g. 'A').

    Returns
    -------
    dict mapping stim_type -> list[float]  (timestamps in seconds)
    """
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'labchart':
        return _labchart.extract_stim_times(file_path, marker_name)
    return _spike2.extract_stim_times(file_path, marker_name)
