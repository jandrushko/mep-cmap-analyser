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
    "layout":       "column_wise" | "row_wise",
    "delimiter":    "tab" | "space" | "comma",
    "fs":           4000,               // Hz — inferred or user-supplied
    "time_col":     0,                  // column index of time axis, or null
    "trials_stacked": true,             // column-wise only: time axis resets per trial
    "channels": [
      {
        "col":    1,                    // 0-based column index (column-wise)
                                        // or row index (row-wise)
        "name":   "EMG – FDI",
        "role":   "emg" | "stim" | "ignore",
        "unit":   "mV"                  // optional
      },
      ...
    ]
  }

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
    """Load the full file as a 2-D array (rows × cols)."""
    sep = _delimiter_char(cfg)
    return np.loadtxt(file_path, delimiter=sep)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list[str]:
    """
    Return the names of all non-ignored, non-time, non-stim channels.
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

    For **column-wise** files the waveform is the selected column, with trial
    blocks concatenated in order.  If 'trials_stacked' is True the time axis
    resets every N rows; we concatenate all blocks into one long array.

    For **row-wise** files (each row = one trial) the waveform is the
    concatenation of all rows for the selected channel.

    Parameters
    ----------
    channel_idx : 0-based index into the list returned by
                  list_waveform_channels() — i.e. counting only non-ignored
                  non-stim channels.
    """
    cfg = load_config(file_path)
    fs  = int(cfg['fs'])

    # Resolve the actual column/row index in the raw array
    emg_channels = [
        ch for ch in cfg['channels']
        if ch.get('role', 'emg') not in ('ignore', 'stim', 'time')
    ]
    if channel_idx >= len(emg_channels):
        channel_idx = 0
    ch_cfg = emg_channels[channel_idx]
    unit   = ch_cfg.get('unit') or None

    arr = _load_array(file_path, cfg)

    layout = cfg.get('layout', 'column_wise')

    if layout == 'column_wise':
        col = ch_cfg['col']
        waveform = arr[:, col].astype(float)
        # Trials stacked: time axis resets → data is already one continuous
        # array in sample order; just return it as-is.

    else:  # row_wise
        row = ch_cfg['col']  # 'col' stores the row index for row-wise files
        # Each row is one trial; concatenate across trials
        # arr shape: (n_trials, n_samples_per_trial)
        waveform = arr[row, :].astype(float) if arr.ndim == 2 else arr.astype(float)

    return waveform, fs, unit


def extract_stim_times(
    file_path: str,
    marker_name: str = 'A',
) -> dict[str, list[float]]:
    """
    Derive stimulation times from the designated stim/trigger channel.

    Returns stim times in **pipeline-absolute** coordinates — i.e. seconds
    from the first sample of the concatenated waveform, matching the time
    axis that pipeline_load_file builds with ``np.arange(n_samples) / fs``.

    Column-wise stacked trials
    --------------------------
    The local time axis (col 0) resets each trial, so it cannot be used
    directly.  Instead we detect rising-edge threshold crossings within
    each trial's sample block and convert the sample index to an absolute
    time: ``abs_time = abs_sample_index / fs``.

    Column-wise continuous (no resets)
    ------------------------------------
    The time axis is monotonically increasing; we find crossings globally
    and read absolute time from the time column directly.

    Row-wise
    --------
    Each row is one trial; the stim crossing within that row is found and
    placed at ``trial_index * samples_per_trial / fs + t_within``.

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
    time_col = cfg.get('time_col')   # may be None

    stim_times: list[float] = []

    if layout == 'column_wise':
        stim_col    = stim_cfg['col']
        stim_signal = arr[:, stim_col].astype(float)

        trials_stacked = cfg.get('trials_stacked', False)

        if trials_stacked and time_col is not None:
            # ── Stacked trials: time axis resets each trial ───────────────────
            # Detect trial boundaries by finding where the time axis decreases.
            t_axis = arr[:, time_col].astype(float)
            resets = np.where(np.diff(t_axis) < -1e-6)[0]
            trial_boundaries = np.concatenate(
                [[0], resets + 1, [len(stim_signal)]]
            )
            for i in range(len(trial_boundaries) - 1):
                s = int(trial_boundaries[i])
                e = int(trial_boundaries[i + 1])
                sweep = stim_signal[s:e]
                thr = sweep.max() * 0.5
                if thr <= 0:
                    continue
                above = (sweep >= thr).astype(int)
                edges = np.where(np.diff(above) == 1)[0]
                if len(edges):
                    # Convert sample index to pipeline-absolute time
                    abs_sample = s + edges[0] + 1
                    stim_times.append(abs_sample / fs)

        else:
            # ── Continuous recording: one global pass ─────────────────────────
            thr = stim_signal.max() * 0.5
            if thr <= 0:
                return {}
            above = (stim_signal >= thr).astype(int)
            edges = np.where(np.diff(above) == 1)[0]
            if time_col is not None:
                # Time column is monotonically increasing — read absolute times
                # directly, but offset so they start from 0 (pipeline convention)
                t_axis = arr[:, time_col].astype(float)
                t0     = t_axis[0]
                stim_times = (t_axis[edges + 1] - t0).tolist()
            else:
                stim_times = ((edges + 1) / fs).tolist()

    else:  # row_wise — each row is one trial
        n_trials, n_samples = arr.shape
        stim_row_idx = stim_cfg['col']
        trial_dur    = n_samples / fs
        stim_sweep   = arr[stim_row_idx, :].astype(float)
        thr = stim_sweep.max() * 0.5
        if thr > 0:
            above = (stim_sweep >= thr).astype(int)
            edges = np.where(np.diff(above) == 1)[0]
            if len(edges):
                t_within = (edges[0] + 1) / fs
                for trial_i in range(n_trials):
                    stim_times.append(trial_i * trial_dur + t_within)

    return {label: stim_times} if stim_times else {}
