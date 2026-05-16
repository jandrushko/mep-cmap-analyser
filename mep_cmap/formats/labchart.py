"""
mep_cmap.formats.labchart
~~~~~~~~~~~~~~~~~~~~~~~~~
LabChart text export reader.

LabChart exports are recognised by an "Interval=\\t" header on line 0.
Each recording block has its own header section followed by tab-separated
columns: time, ch1, ch2, ... chN. Multiple blocks (one per trial) are
concatenated into a single continuous waveform using ExcelDateTime for
absolute timing. Gaps between blocks are zero-filled.

LabChart exports pre-align each block so that t=0 corresponds to the
stimulation event — no DigMark/Marker channel required.

Public API (mirrors the io.py contract)
----------------------------------------
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Notes on extract_stim_times
----------------------------
  marker_name is repurposed as the stim-type label (single uppercase letter,
  e.g. 'A'). The stim channel is auto-detected by searching channel names for
  'stim', 'trig', or 'ttl' (case-insensitive); falls back to channel index 3
  (Ch 4) if none found. The stim pulse is detected either at t=0 (LabChart
  pre-centres blocks) or via threshold crossing on the analogue stim channel.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_blocks(lines: list) -> list:
    """
    Parse a LabChart text export into a list of block dicts.

    Each block dict contains:
      fs         : int   — sampling rate in Hz
      edt_sec    : float — absolute start time (Excel serial days → seconds)
      channels   : list  — channel name strings
      units      : list  — unit strings per channel
      data_start : int   — line index of first data row
      data_end   : int   — line index one past last data row
    """
    block_starts = [i for i, l in enumerate(lines) if l.startswith('Interval=')]
    blocks = []
    for b_idx, start in enumerate(block_starts):
        try:
            interval_s = float(lines[start].split('\t')[1].strip().split()[0])
            fs         = round(1.0 / interval_s)
            edt_val    = float(lines[start + 1].split('\t')[1].strip())
            edt_sec    = edt_val * 86400.0
            ch_line    = next((lines[start + k] for k in range(2, 9)
                               if lines[start + k].startswith('ChannelTitle')), '')
            channels   = ch_line.strip().split('\t')[1:] if ch_line else []
            unit_line  = next((lines[start + k] for k in range(2, 9)
                               if lines[start + k].startswith('UnitName')), '')
            units      = unit_line.strip().split('\t')[1:] if unit_line else []
            data_start = start + 9
            data_end   = (block_starts[b_idx + 1]
                          if b_idx + 1 < len(block_starts) else len(lines))
            blocks.append(dict(
                fs=fs, edt_sec=edt_sec, channels=channels, units=units,
                data_start=data_start, data_end=data_end,
            ))
        except Exception:
            continue
    return blocks


def _abs_start(blocks: list, lines: list) -> float:
    """Absolute time (seconds) of the very first sample in block 0."""
    try:
        t_local = float(lines[blocks[0]['data_start']].strip().split('\t')[0])
    except Exception:
        t_local = 0.0
    return blocks[0]['edt_sec'] + t_local


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return channel names from the first block header."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    blocks = _parse_blocks(lines)
    if not blocks:
        return ['Channel 1']
    return [c.strip() for c in blocks[0]['channels']] or ['Channel 1']


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Concatenate all LabChart blocks into one continuous waveform.

    Blocks are placed at their absolute times using ExcelDateTime.
    Gaps between blocks are zero-filled.

    Returns
    -------
    emg  : np.ndarray  concatenated waveform
    fs   : int         sampling rate in Hz
    unit : str | None  unit string, or None if not available
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    blocks = _parse_blocks(lines)
    if not blocks:
        raise ValueError("No LabChart data blocks found.")

    fs   = blocks[0]['fs']
    unit = None
    if channel_idx < len(blocks[0]['units']):
        u = blocks[0]['units'][channel_idx].strip().strip('*')
        if u:
            unit = u

    t0_abs = _abs_start(blocks, lines)

    # Estimate total output length from last block
    last = blocks[-1]
    try:
        t_last_local = float(
            lines[last['data_end'] - 1].strip().split('\t')[0])
    except Exception:
        t_last_local = 0.5
    total_samples = int(np.ceil(
        ((last['edt_sec'] + t_last_local) - t0_abs) * fs)) + 10

    output = np.zeros(total_samples, dtype=float)

    col = channel_idx + 1   # column 0 is time
    for block in blocks:
        try:
            t_local_start = float(
                lines[block['data_start']].strip().split('\t')[0])
        except Exception:
            t_local_start = 0.0
        sample_offset = int(round(
            ((block['edt_sec'] + t_local_start) - t0_abs) * fs))

        samples = []
        for row in lines[block['data_start']:block['data_end']]:
            parts = row.strip().split('\t')
            if len(parts) > col:
                try:
                    samples.append(float(parts[col]))
                except ValueError:
                    pass
        if not samples:
            continue
        arr     = np.array(samples, dtype=float)
        end_idx = sample_offset + len(arr)
        if end_idx > len(output):
            output = np.pad(output, (0, end_idx - len(output)))
        output[sample_offset:end_idx] = arr

    return output, fs, unit


def extract_stim_times(file_path: str, marker_name: str = 'A') -> dict:
    """
    Detect stimulation times from each LabChart block.

    Each block is pre-centred by LabChart so t=0 is the stimulation.
    We use t=0 directly; if t=0 is not present we fall back to threshold
    crossing on the analogue stim channel.

    Parameters
    ----------
    marker_name : used as the stim-type label (single uppercase letter).
                  The stim channel is auto-detected by channel name.

    Returns
    -------
    dict mapping label -> list of absolute timestamps (seconds)
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    blocks = _parse_blocks(lines)
    if not blocks:
        return {}

    fs     = blocks[0]['fs']
    t0_abs = _abs_start(blocks, lines)
    label  = (marker_name[:1].upper() if marker_name else 'A')

    # Auto-detect stim channel index
    channels    = [c.strip() for c in blocks[0]['channels']]
    stim_ch_idx = next(
        (i for i, c in enumerate(channels)
         if any(k in c.lower() for k in ('stim', 'trig', 'ttl'))),
        min(3, len(channels) - 1))
    stim_col = stim_ch_idx + 1   # +1 for the time column

    stim_times = []
    for block in blocks:
        time_v, stim_v = [], []
        for row in lines[block['data_start']:block['data_end']]:
            parts = row.strip().split('\t')
            if len(parts) > stim_col:
                try:
                    time_v.append(float(parts[0]))
                    stim_v.append(float(parts[stim_col]))
                except ValueError:
                    pass
        if not time_v:
            continue

        try:
            t_local_start = float(
                lines[block['data_start']].strip().split('\t')[0])
        except Exception:
            t_local_start = time_v[0]

        abs_block_start = (block['edt_sec'] + t_local_start) - t0_abs
        time_arr = np.array(time_v)
        stim_arr = np.array(stim_v)

        # Strategy 1: LabChart pre-centres blocks at t=0 = stim
        t0_idx = np.argmin(np.abs(time_arr))
        if abs(time_arr[t0_idx]) < 2.0 / fs:
            stim_times.append(abs_block_start + (time_arr[t0_idx] - time_v[0]))
            continue

        # Strategy 2: threshold crossing on stim channel
        if stim_arr.max() > 0.1:
            threshold = stim_arr.max() * 0.5
            edges = np.where(
                np.diff((stim_arr >= threshold).astype(int)) == 1)[0]
            if len(edges):
                stim_times.append(
                    abs_block_start + (time_arr[edges[0]] - time_v[0]))

    return {label: stim_times} if stim_times else {}
