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

from .formats import spike2   as _spike2
from .formats import labchart as _labchart


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
    fmt = detect_format(file_path)
    if fmt == 'labchart':
        return _labchart.extract_stim_times(file_path, marker_name)
    return _spike2.extract_stim_times(file_path, marker_name)
