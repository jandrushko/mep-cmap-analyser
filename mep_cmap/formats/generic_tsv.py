"""
mep_cmap.formats.generic_tsv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reader for headerless / generic tab- (or space-) separated text files.

These files contain only numeric data — no format-identifying header line —
so the io layer detects them as 'generic_tsv' and, on the first encounter,
launches FormatWizard (format_wizard.py) to let the user define layout and
channel roles.  The wizard writes a sidecar config file alongside the data
file; subsequent opens read the sidecar directly without prompting.

Sidecar file
------------
  <data_file_stem>.tsv_config.json

  {
    "layout":         "column_wise" | "row_wise",
    "delimiter":      "tab" | "space" | "comma",
    "fs":             2148,              // Hz — inferred or user-supplied
    "time_col":       0,                 // column index of time axis (column-wise)
                                         // or row index of time row (row-wise)
                                         // or null if absent
    "skip_rows":      0,                 // lines to skip at the top of the file
                                         // (non-numeric metadata / header lines)
    "trials_stacked": true,              // column-wise only: time axis resets per trial
    "channels": [
      {
        "col":    0,                     // 0-based column index (column-wise)
                                         // or row index (row-wise)
        "name":   "Trigger",
        "role":   "emg" | "stim" | "ignore",
        "unit":   "V"
      },
      {
        "col":    1,
        "name":   "Biceps EMG",
        "role":   "emg",
        "unit":   "mV"
      }
    ]
  }

Supported file layouts
----------------------
Column-wise (each column = one channel, rows = time samples):
  Typical for LabChart exports, simple CSV/TSV recordings.

Row-wise (each row = one channel, columns = time samples):
  Used by Delsys Trigno and similar systems.  Common pattern:
    Row 0 = TTL / stim trigger channel (~5 V pulses, otherwise ~0 V)
    Row 1 = continuous EMG recording (mV scale)
  The trigger channel may have a large startup artifact at sample 0
  that is well outside the ~0 V baseline — this is handled robustly
  in both auto-detection and stim-time extraction.

Public API  (mirrors the io.py contract)
-----------------------------------------
  has_config(file_path)                        -> bool
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sidecar_path(file_path: str) -> Path:
    return Path(file_path).with_suffix('.tsv_config.json')


def has_config(file_path: str) -> bool:
    """Return True if a valid sidecar config exists for this file."""
    p = _sidecar_path(file_path)
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text(encoding='utf-8'))
        return bool(cfg.get('channels'))
    except Exception:
        return False


def load_config(file_path: str) -> dict:
    """Load and return the sidecar config dict, raising if missing."""
    p = _sidecar_path(file_path)
    if not p.exists():
        raise FileNotFoundError(
            f"No generic-TSV config found for {os.path.basename(file_path)}.\n"
            "Open the file through the GUI to run the Format Wizard."
        )
    return json.loads(p.read_text(encoding='utf-8'))


def save_config(file_path: str, cfg: dict) -> None:
    """Write the sidecar config dict next to the data file."""
    p = _sidecar_path(file_path)
    p.write_text(json.dumps(cfg, indent=2), encoding='utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# Raw data loader
# ─────────────────────────────────────────────────────────────────────────────

def _delimiter_char(cfg: dict) -> str:
    d = cfg.get('delimiter', 'tab')
    return {'tab': '\t', 'space': ' ', 'comma': ','}.get(d, '\t')


def _load_array(file_path: str, cfg: dict) -> np.ndarray:
    """
    Load the full file as a 2-D array (rows × cols).

    Respects the ``skip_rows`` config key: that many lines are skipped at the
    top of the file before parsing begins (useful for ignoring metadata /
    non-numeric header lines that appear before the data).

    Always returns a 2-D array, even for single-row files.
    """
    sep  = _delimiter_char(cfg)
    skip = int(cfg.get('skip_rows', 0))
    arr  = np.loadtxt(file_path, delimiter=sep, skiprows=skip)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list[str]:
    """
    Return the names of all non-ignored, non-stim channels.
    Falls back to ['Channel 1'] if config is absent.
    """
    if not has_config(file_path):
        return ['Channel 1']
    cfg = load_config(file_path)
    names = [
        ch['name']
        for ch in cfg['channels']
        if ch.get('role', 'emg') not in ('ignore', 'stim', 'time')
    ]
    return names if names else ['Channel 1']


def extract_emg_waveform_and_fs(
    file_path: str,
    channel_idx: int = 0,
) -> tuple[np.ndarray, int, Optional[str]]:
    """
    Extract a single EMG channel as a 1-D waveform.

    Column-wise
    -----------
    The waveform is the selected column across all rows.  If
    ``trials_stacked`` is True the time axis resets every N rows; the data
    is returned as one concatenated array in sample order.

    Row-wise
    --------
    The file is oriented so that each row is one channel and columns are
    time samples.  Common for Delsys Trigno exports:

        Row 0  →  TTL trigger channel (assigned role "stim")
        Row 1  →  Continuous EMG recording (assigned role "emg")

    The ``col`` field in the channel config holds the **row index** of that
    channel.  The full row is returned as a 1-D waveform; the pipeline then
    uses ``extract_stim_times`` to find TMS pulses and epochs the signal.

    Parameters
    ----------
    channel_idx : 0-based index into the list returned by
                  list_waveform_channels() — counting only non-ignored
                  non-stim channels.
    """
    cfg = load_config(file_path)
    fs  = int(cfg['fs'])

    emg_channels = [
        ch for ch in cfg['channels']
        if ch.get('role', 'emg') not in ('ignore', 'stim', 'time')
    ]
    if channel_idx >= len(emg_channels):
        channel_idx = 0
    ch_cfg = emg_channels[channel_idx]
    unit   = ch_cfg.get('unit') or None

    arr    = _load_array(file_path, cfg)
    layout = cfg.get('layout', 'column_wise')

    if layout == 'column_wise':
        col      = ch_cfg['col']
        waveform = arr[:, col].astype(float)

    else:  # row_wise
        # 'col' stores the row index for row-wise files (wizard convention)
        row = ch_cfg['col']
        if arr.ndim == 2 and row < arr.shape[0]:
            waveform = arr[row, :].astype(float)
        else:
            waveform = arr.flatten().astype(float)

    return waveform, fs, unit


def extract_stim_times(
    file_path: str,
    marker_name: str = 'A',
) -> dict[str, list[float]]:
    """
    Derive stimulation times from the designated stim/trigger channel.

    Returns stim times in **pipeline-absolute** coordinates — seconds from
    the first sample of the waveform returned by extract_emg_waveform_and_fs,
    matching the time axis that pipeline_load_file builds with
    ``np.arange(n_samples) / fs``.

    Column-wise stacked trials
    --------------------------
    The local time axis resets each trial; we detect trial boundaries from
    the time-column resets, find the rising-edge crossing within each block,
    and place it at ``abs_sample / fs``.

    Column-wise continuous
    ----------------------
    One global rising-edge pass; absolute times read from the time column
    directly (or computed as ``sample / fs`` if there is no time column).

    Row-wise — continuous recording with TTL trigger row
    -----------------------------------------------------
    Row-wise files (e.g. Delsys Trigno) typically have:
      • One row designated as "stim" containing ~0 V baseline and ~5 V
        TTL pulses at each TMS delivery.
      • One or more rows designated as "emg" containing the continuous
        EMG signal.

    The trigger row may have a large startup artifact at sample 0 (e.g.
    -0.75 V on a Delsys system).  Detection is therefore based on a
    percentile-robust threshold rather than the global minimum.

    Rising-edge positions (in samples from the start of the recording)
    are converted directly to seconds: ``edge_sample / fs``.  These align
    with the absolute time axis of the EMG waveform returned by
    extract_emg_waveform_and_fs, so the pipeline can epoch correctly.

    If no stim channel is configured, returns an empty dict.
    """
    cfg = load_config(file_path)

    stim_channels = [
        ch for ch in cfg['channels']
        if ch.get('role') == 'stim'
    ]
    if not stim_channels:
        return {}

    stim_cfg = stim_channels[0]
    fs       = int(cfg['fs'])
    label    = (marker_name[:1].upper() if marker_name else 'A')
    arr      = _load_array(file_path, cfg)
    layout   = cfg.get('layout', 'column_wise')
    time_col = cfg.get('time_col')

    stim_times: list[float] = []

    if layout == 'column_wise':
        stim_col    = stim_cfg['col']
        stim_signal = arr[:, stim_col].astype(float)
        trials_stacked = cfg.get('trials_stacked', False)

        if trials_stacked and time_col is not None:
            # Stacked trials: time axis resets each trial
            t_axis = arr[:, time_col].astype(float)
            resets = np.where(np.diff(t_axis) < -1e-6)[0]
            trial_boundaries = np.concatenate(
                [[0], resets + 1, [len(stim_signal)]]
            )
            for i in range(len(trial_boundaries) - 1):
                s = int(trial_boundaries[i])
                e = int(trial_boundaries[i + 1])
                sweep = stim_signal[s:e]
                thr   = sweep.max() * 0.5
                if thr <= 0:
                    continue
                above = (sweep >= thr).astype(int)
                edges = np.where(np.diff(above) == 1)[0]
                if len(edges):
                    stim_times.append((s + edges[0] + 1) / fs)

        else:
            # Continuous recording
            thr = stim_signal.max() * 0.5
            if thr <= 0:
                return {}
            above = (stim_signal >= thr).astype(int)
            edges = np.where(np.diff(above) == 1)[0]
            if time_col is not None:
                t_axis = arr[:, time_col].astype(float)
                t0     = t_axis[0]
                stim_times = (t_axis[edges + 1] - t0).tolist()
            else:
                stim_times = ((edges + 1) / fs).tolist()

    else:  # row_wise
        if arr.ndim != 2:
            return {}

        stim_row_idx = stim_cfg['col']
        if stim_row_idx >= arr.shape[0]:
            return {}

        stim_signal = arr[stim_row_idx, :].astype(float)

        # Use a percentile-robust lower bound when checking whether the signal
        # is usable, but threshold on the global maximum: trigger pulses are
        # so sparse (typically < 0.01 % of samples) that any percentile-based
        # peak estimate lands inside the EMG noise floor.
        #
        # This correctly handles the Delsys Trigno startup artifact: the very
        # first sample is often a large negative transient (e.g. -0.75 V) while
        # all subsequent baseline samples are within ±1 mV, and TTL highs are
        # at ~5 V.  Global-max * 0.5 ≈ 2.5 V catches all real trigger edges.
        thr = stim_signal.max() * 0.5

        if thr <= 0:
            return {}

        above = (stim_signal >= thr).astype(int)
        edges = np.where(np.diff(above) == 1)[0]

        # Convert sample indices to absolute seconds
        stim_times = ((edges + 1) / fs).tolist()

    return {label: stim_times} if stim_times else {}
