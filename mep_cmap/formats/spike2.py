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
"""

import re
from collections import defaultdict
import numpy as np


def list_waveform_channels(file_path: str) -> list:
    """Return the channel names that appear as Waveform rows in the SUMMARY block."""
    names = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith('"SUMMARY"'):
                break
        for _ in range(40):
            parts = [tok.strip().strip('"') for tok in f.readline().split('\t')]
            if len(parts) >= 3 and parts[1] == "Waveform":
                names.append(parts[2] or f"Chan {len(names)+1}")
    return names or ["Waveform-1"]


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
    numeric_re = re.compile(r"^-?\d+(\.\d+)?$")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

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
    stim_dict = defaultdict(list)
    pattern   = re.compile(r'^([\d.]+)\s+"(.{1})\?\?\?"')

    with open(file_path, 'r') as f:
        lines = f.readlines()

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
