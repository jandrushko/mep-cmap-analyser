"""
mep_cmap.formats.spike2
~~~~~~~~~~~~~~~~~~~~~~~
Spike-2 text export reader.

Spike-2 exports are recognised by a "SUMMARY" block near the top of the file,
followed by a "START" block containing waveform samples, and optional "Marker"
blocks containing DigMark event timestamps.

Public API (mirrors the io.py contract)
----------------------------------------
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Performance
-----------
If the compiled Rust extension ``mep_cmap_io`` is importable, all three
functions delegate to it.  The Rust implementation is 4-7x faster on large
files because it avoids the per-line Python float() overhead and returns a
numpy array directly without an intermediate Python list.

If the extension is not available (e.g. first run before building, or an
unsupported platform) the module falls back to the pure-Python implementation
transparently — no user action required.
"""

import re
from collections import defaultdict
import numpy as np

# ── Try to import the compiled Rust extension ─────────────────────────────────
# Guard against a partially-installed or stub mep_cmap_io package (e.g. on a
# machine where the Rust crate exists but has never been compiled).  We check
# for the specific function we need rather than just a successful import.
try:
    import mep_cmap_io as _rust
    _RUST_AVAILABLE = callable(getattr(_rust, 'spike2_list_channels', None))
except ImportError:
    _RUST_AVAILABLE = False


def list_waveform_channels(file_path: str) -> list:
    """Return the channel names that appear as Waveform rows in the SUMMARY block."""
    if _RUST_AVAILABLE:
        return _rust.spike2_list_channels(file_path)
    return _list_waveform_channels_py(file_path)


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Return (waveform, fs, unit) for the requested channel (0-based index).

    Parameters
    ----------
    file_path   : path to a Spike-2 exported .txt file
    channel_idx : 0-based index of the waveform channel to load

    Returns
    -------
    emg  : np.ndarray  raw EMG samples
    fs   : int         sampling frequency in Hz
    unit : str | None  voltage unit string (e.g. 'mV'), or None
    """
    if _RUST_AVAILABLE:
        arr, fs, unit = _rust.spike2_extract_waveform(file_path, channel_idx)
        return np.asarray(arr, dtype=float), fs, unit
    return _extract_emg_waveform_and_fs_py(file_path, channel_idx)


def extract_stim_times(file_path: str, marker_name: str) -> dict:
    """
    Read a Spike-2 .txt file and return stimulation timestamps.

    Parameters
    ----------
    file_path   : path to Spike-2 exported .txt file
    marker_name : marker channel name (e.g. 'Keyboard', 'TTL')

    Returns
    -------
    dict mapping stim_type (single character) -> list of timestamps (seconds)
    """
    if _RUST_AVAILABLE:
        return dict(_rust.spike2_extract_stim_times(file_path, marker_name))
    return _extract_stim_times_py(file_path, marker_name)


# ── Pure-Python fallbacks ─────────────────────────────────────────────────────

def _list_waveform_channels_py(file_path: str) -> list:
    names = []
    with open(file_path, "rb") as f:
        raw = f.read()
    lines_iter = iter(raw.decode("latin-1", errors="replace")
                      .replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    for line in lines_iter:
        if line.startswith('"SUMMARY"'):
            break
    for _ in range(40):
        try:
            line = next(lines_iter)
        except StopIteration:
            break
        parts = [tok.strip().strip('"') for tok in line.split('\t')]
        if len(parts) >= 3 and parts[1] == "Waveform":
            names.append(parts[2] or f"Chan {len(names)+1}")
    return names or ["Waveform-1"]


def _extract_emg_waveform_and_fs_py(file_path: str, channel_idx: int = 0):
    numeric_re = re.compile(r"^-?\d+(\.\d+)?$")

    with open(file_path, "rb") as f:
        raw = f.read()
    text  = raw.decode("latin-1", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Collect every Waveform row in the SUMMARY block
    summary_rows = []
    for i, line in enumerate(lines):
        if line.startswith('"SUMMARY"'):
            for j in range(i + 1, min(i + 40, len(lines))):
                parts = [tok.strip().strip('"') for tok in lines[j].split('\t')]
                if len(parts) >= 2 and parts[1] == "Waveform":
                    row_fs   = next((int(float(tok))
                                     for tok in parts[2:]
                                     if numeric_re.match(tok) and float(tok) >= 100),
                                    None)
                    row_unit = next((tok for tok in parts
                                     if re.fullmatch(r"[a-zA-Zµμ]+[Vv]", tok)),
                                    None)
                    summary_rows.append((j, row_fs, row_unit))
            break

    if not summary_rows:
        raise ValueError("No Waveform channels found in SUMMARY.")

    try:
        _, fs, unit = summary_rows[channel_idx]
    except IndexError:
        raise ValueError(
            f"Channel #{channel_idx+1} requested but only {len(summary_rows)} found.")

    # Jump to START, then skip channel_idx waveform blocks
    line_no = next(i for i, l in enumerate(lines) if l.startswith('"START"')) + 1
    for _ in range(channel_idx):
        while line_no < len(lines) and not lines[line_no].startswith('"CHANNEL"'):
            line_no += 1
        line_no += 2

    # Read samples until next CHANNEL / EOF
    emg_vals = []
    for l in lines[line_no:]:
        if l.startswith('"CHANNEL"'):
            break
        try:
            emg_vals.append(float(l.strip()))
        except ValueError:
            continue

    return np.asarray(emg_vals, float), fs, unit


def _extract_stim_times_py(file_path: str, marker_name: str) -> dict:
    stim_dict = defaultdict(list)
    pattern   = re.compile(r'^([\d.]+)\s+"(.{1})\?\?\?"')

    with open(file_path, "rb") as f:
        raw = f.read()
    text  = raw.decode("latin-1", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    block_start = None
    for i in range(len(lines)):
        if lines[i].strip().startswith('"Marker"') and i + 2 < len(lines):
            current_marker = lines[i + 2].strip().strip('"')
            if current_marker == marker_name:
                block_start = i + 3
                break

    if block_start is None:
        return stim_dict

    for line in lines[block_start:]:
        if line.strip().startswith('"CHANNEL"'):
            break
        match = pattern.match(line.strip())
        if match:
            stim_dict[match.group(2)].append(float(match.group(1)))

    return stim_dict
