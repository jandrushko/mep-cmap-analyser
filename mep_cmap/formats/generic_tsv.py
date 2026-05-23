"""
mep_cmap.formats.generic_tsv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reader for headerless / generic tab- (or space-) separated text files.

These files contain only numeric data — no format-identifying header line —
so the io layer detects them as 'generic_tsv' and, on the first encounter,
launches FormatWizard (format_wizard.py) to let the user define layout and
channel roles.  The wizard writes a sidecar config file alongside the data
file; subsequent opens read the sidecar directly without prompting.

Performance
-----------
All file I/O is delegated to the Rust extension ``mep_cmap_io`` when it is
available.  The Rust reader (``load_2d``) uses an 8 MB BufReader and Rust's
native f64 parser, which is 10–20× faster than np.loadtxt for column-wise
files and avoids the memory-allocation hazard that caused "Not Responding"
on row-wise files with hundreds of thousands of columns.

If the Rust extension is not found (e.g. during development without a build)
the module falls back to the pure-Python / NumPy path transparently.

Sidecar file
------------
  <data_file_stem>.tsv_config.json

  {
    "layout":         "column_wise" | "row_wise",
    "delimiter":      "tab" | "space" | "comma",
    "fs":             2148,
    "time_col":       0,        // column or row index of time axis, null if absent
    "skip_rows":      0,        // non-numeric header lines to skip
    "trials_stacked": false,    // column-wise only: time axis resets per trial
    "channels": [
      { "col": 0, "name": "Trigger",    "role": "stim", "unit": "V"  },
      { "col": 1, "name": "Biceps EMG", "role": "emg",  "unit": "mV" }
    ]
  }

Public API  (mirrors the io.py contract)
-----------------------------------------
  has_config(file_path)                        -> bool
  sniff(file_path)                             -> (n_rows, n_cols, fs_detected)
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

# ── Try to load the Rust extension ───────────────────────────────────────────
try:
    import mep_cmap_io as _rust
    _RUST_AVAILABLE = True
except ImportError:
    _rust = None               # type: ignore[assignment]
    _RUST_AVAILABLE = False


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
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _delimiter_char(cfg: dict) -> str:
    d = cfg.get('delimiter', 'tab')
    return {'tab': '\t', 'space': ' ', 'comma': ','}.get(d, '\t')


def _delimiter_name(cfg: dict) -> str:
    return cfg.get('delimiter', 'tab')


def _load_array_python(file_path: str, cfg: dict) -> np.ndarray:
    """
    Pure-Python / NumPy fallback loader.  Used when the Rust extension is
    unavailable.  Returns a 2-D array (rows × cols).
    """
    sep  = _delimiter_char(cfg)
    skip = int(cfg.get('skip_rows', 0))
    arr  = np.loadtxt(file_path, delimiter=sep, skiprows=skip)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Public sniff helper (used by FormatWizard)
# ─────────────────────────────────────────────────────────────────────────────

def sniff(
    file_path: str,
    delimiter: str = 'tab',
    skip_rows: int = 0,
) -> tuple[int, int, Optional[float]]:
    """
    Quickly return (n_data_rows, n_cols, fs_detected) without loading the
    full array.

    Delegates to the Rust ``generic_tsv_sniff`` function when available;
    falls back to a pure-Python line-count scan otherwise.

    Used by FormatWizard._sniff_shape and _load_data to avoid the
    "max_rows=10000 on a 2-row × 423k-col file" memory hazard.
    """
    if _RUST_AVAILABLE:
        return _rust.generic_tsv_sniff(file_path, delimiter, skip_rows)

    # Python fallback
    sep = {'tab': '\t', 'space': ' ', 'comma': ','}.get(delimiter, '\t')
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as fh:
            for _ in range(skip_rows):
                fh.readline()
            first = fh.readline()
            if not first.strip():
                return 0, 0, None
            n_cols = len(first.split(sep))
            n_rows = 1 + sum(1 for _ in fh)
        return n_rows, n_cols, None
    except Exception:
        return 0, 0, None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list[str]:
    """Return the names of all non-ignored, non-stim channels."""
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

    Delegates to ``mep_cmap_io.generic_tsv_extract_waveform`` (Rust) when
    available, otherwise falls back to np.loadtxt.

    Column-wise
    -----------
    Waveform = target column across all rows.

    Row-wise
    --------
    Waveform = target row across all columns.  Common for Delsys Trigno
    exports where row 0 = TTL trigger and row 1 = continuous EMG.
    The ``col`` field in the channel config holds the row index.
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
    target = int(ch_cfg['col'])
    layout = cfg.get('layout', 'column_wise')

    if _RUST_AVAILABLE:
        samples, unit_out = _rust.generic_tsv_extract_waveform(
            file_path,
            _delimiter_name(cfg),
            int(cfg.get('skip_rows', 0)),
            target,
            layout,
            unit,
        )
        return np.asarray(samples, dtype=float), fs, unit_out

    # Python fallback
    arr = _load_array_python(file_path, cfg)
    if layout == 'column_wise':
        waveform = arr[:, target].astype(float)
    else:
        if arr.ndim == 2 and target < arr.shape[0]:
            waveform = arr[target, :].astype(float)
        else:
            waveform = arr.flatten().astype(float)

    return waveform, fs, unit


def extract_stim_times(
    file_path: str,
    marker_name: str = 'A',
) -> dict[str, list[float]]:
    """
    Derive stimulation times from the designated stim/trigger channel.

    Returns stim times in pipeline-absolute coordinates (seconds from the
    first sample of the waveform returned by extract_emg_waveform_and_fs).

    Delegates to ``mep_cmap_io.generic_tsv_extract_stim_times`` (Rust) when
    available, otherwise falls back to the NumPy rising-edge detector.

    Row-wise startup-artifact handling
    ------------------------------------
    The Delsys Trigno trigger row often has a -0.75 V transient at sample 0.
    Both the Rust and Python paths threshold on global_max * 0.5 (~2.5 V for
    a 5 V TTL rail) which is well above the noise floor and unaffected by the
    startup transient.
    """
    cfg = load_config(file_path)

    stim_channels = [ch for ch in cfg['channels'] if ch.get('role') == 'stim']
    if not stim_channels:
        return {}

    stim_cfg  = stim_channels[0]
    fs        = float(cfg['fs'])
    label     = (marker_name[:1].upper() if marker_name else 'A')
    layout    = cfg.get('layout', 'column_wise')
    time_col  = cfg.get('time_col')
    stim_col  = int(stim_cfg['col'])
    trials_stacked = cfg.get('trials_stacked', False)

    if _RUST_AVAILABLE:
        return _rust.generic_tsv_extract_stim_times(
            file_path,
            _delimiter_name(cfg),
            int(cfg.get('skip_rows', 0)),
            stim_col,
            layout,
            fs,
            int(time_col) if time_col is not None else -1,
            bool(trials_stacked),
            label,
        )

    # ── Python / NumPy fallback ───────────────────────────────────────────────
    arr = _load_array_python(file_path, cfg)
    stim_times: list[float] = []

    if layout == 'column_wise':
        stim_signal = arr[:, stim_col].astype(float)

        if trials_stacked and time_col is not None:
            t_axis = arr[:, time_col].astype(float)
            resets = np.where(np.diff(t_axis) < -1e-6)[0]
            trial_boundaries = np.concatenate([[0], resets + 1, [len(stim_signal)]])
            for i in range(len(trial_boundaries) - 1):
                s = int(trial_boundaries[i])
                e = int(trial_boundaries[i + 1])
                sweep = stim_signal[s:e]
                thr = sweep.max() * 0.5
                if thr <= 0:
                    continue
                edges = np.where(np.diff((sweep >= thr).astype(int)) == 1)[0]
                if len(edges):
                    stim_times.append((s + edges[0] + 1) / fs)
        else:
            thr = stim_signal.max() * 0.5
            if thr <= 0:
                return {}
            edges = np.where(np.diff((stim_signal >= thr).astype(int)) == 1)[0]
            if time_col is not None:
                t_axis = arr[:, time_col].astype(float)
                stim_times = (t_axis[edges + 1] - t_axis[0]).tolist()
            else:
                stim_times = ((edges + 1) / fs).tolist()

    else:  # row_wise
        if arr.ndim != 2 or stim_col >= arr.shape[0]:
            return {}
        stim_signal = arr[stim_col, :].astype(float)
        thr = stim_signal.max() * 0.5
        if thr <= 0:
            return {}
        edges = np.where(np.diff((stim_signal >= thr).astype(int)) == 1)[0]
        stim_times = ((edges + 1) / fs).tolist()

    return {label: stim_times} if stim_times else {}
