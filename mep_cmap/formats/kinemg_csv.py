"""
mep_cmap.formats.kinemg_csv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reader for KinEMG software CSV exports.

File structure
--------------
  Line 0:  Author,KinEMG
  Line 1:  TimeStamp,<date/time string>
  Line 2:  Sample Clock Rate,<Hz as float>
  Line 3:  (blank)
  Line 4:  <chan0>,<chan1>,<chan2>,...       ← channel names (Dev1/ai0 style)
  Line 5+: <float>,<float>,...              ← one sample per row, one column per channel

Notes
-----
* No time column — a synthetic time axis is constructed from fs.
* No stim/trigger channel is expected; ``extract_stim_times`` returns an empty
  dict so the caller falls back to manual event marking or the Format Wizard
  pattern for generic_tsv.
* The file is comma-delimited with CRLF line endings.
* Channel names are returned as-is from line 4 (e.g. "Dev1/ai0").

Public API  (mirrors the io.py contract)
-----------------------------------------
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_ROWS = 5   # rows before numeric data begins (0-based row 5 is first data row)


def _parse_header(file_path: str) -> tuple[float, list[str]]:
    """
    Read the KinEMG header and return (fs, channel_names).

    Raises
    ------
    ValueError  if the header cannot be parsed correctly.
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as fh:
        lines = [fh.readline() for _ in range(_HEADER_ROWS)]

    # Line 2: "Sample Clock Rate,<Hz>"
    rate_line = lines[2].strip()
    try:
        fs = float(rate_line.split(',', 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"KinEMG CSV: cannot parse sampling rate from line 3: {rate_line!r}"
        ) from exc

    # Line 4: channel names
    chan_line = lines[4].strip()
    channel_names = [c.strip() for c in chan_line.split(',') if c.strip()]
    if not channel_names:
        raise ValueError(
            f"KinEMG CSV: cannot parse channel names from line 5: {chan_line!r}"
        )

    return fs, channel_names


def _load_data(file_path: str) -> tuple[np.ndarray, float, list[str]]:
    """
    Return (data_2d, fs, channel_names).
    data_2d has shape (n_samples, n_channels).
    """
    fs, channel_names = _parse_header(file_path)
    data = np.loadtxt(file_path, delimiter=',', skiprows=_HEADER_ROWS)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data, fs, channel_names


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list[str]:
    """Return all channel names found in the header (line 4)."""
    try:
        _, channel_names = _parse_header(file_path)
        return channel_names
    except Exception:
        return ['Channel 1']


def extract_emg_waveform_and_fs(
    file_path: str,
    channel_idx: int = 0,
) -> tuple[np.ndarray, int, Optional[str]]:
    """
    Extract a single channel as a 1-D waveform.

    Parameters
    ----------
    channel_idx : 0-based index into the list returned by list_waveform_channels().

    Returns
    -------
    emg  : np.ndarray  1-D array of raw EMG samples
    fs   : int         sampling frequency in Hz
    unit : str | None  always None (unit not encoded in KinEMG CSV header)
    """
    data, fs, channel_names = _load_data(file_path)

    n_channels = data.shape[1]
    if channel_idx >= n_channels:
        raise IndexError(
            f"KinEMG CSV has {n_channels} channel(s); "
            f"requested index {channel_idx} is out of range."
        )

    emg = data[:, channel_idx].astype(float)
    return emg, int(round(fs)), None


def extract_stim_times(file_path: str, marker_name: str) -> dict[str, list[float]]:
    """
    KinEMG CSV files do not contain a dedicated trigger/stim channel.

    Returns an empty dict so the caller (app.py) falls through to the generic
    stim-time detection path (threshold crossing on the raw signal).
    """
    return {}
