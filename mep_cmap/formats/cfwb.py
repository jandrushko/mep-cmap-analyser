"""
mep_cmap.formats.cfwb
~~~~~~~~~~~~~~~~~~~~~
Reader for ADInstruments CFWB binary files (.adibin).

These are produced by LabChart via File → Export → ADInstruments Binary.
The format is fully documented in ADIBinaryFormat.h published by
ADInstruments (2001–2009):
  http://cdn.adinstruments.com/adi-web/manuals/translatebinary/ADIBinaryFormat.h

File layout (all values little-endian, 1-byte packed)
------------------------------------------------------
  [68 bytes]            File header (CFWBINARY struct)
  [96 bytes × NChannels] Channel headers (CFWBCHANNEL struct)
  [interleaved samples]  NChannels (or NChannels+1 if TimeChannel=1)
                         values per sample row.

DataFormat codes
  1 = float64 (double precision)
  2 = float32 (single precision)
  3 = int16   — physical = scale × (raw + offset)

For float formats scale = 1.0 and offset = 0.0.

Stim time detection
-------------------
The CFWB format has no explicit marker/comment channel.  Stimulation times
are derived from a trigger/TTL channel (auto-detected by title keyword
"stim", "trig", or "ttl"; falls back to the last channel) using the same
rising-edge threshold method as the LabChart text reader.

Performance
-----------
All three public functions delegate to the Rust extension ``mep_cmap_io``
when available.  The Rust reader works entirely on the raw bytes of the
pre-loaded file and never allocates intermediate Python lists, making it
~10× faster than the NumPy fallback for large files.

Public API  (mirrors the io.py contract)
-----------------------------------------
  is_cfwb(file_path)                           -> bool
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

# ── Try to load the Rust extension ───────────────────────────────────────────
try:
    import mep_cmap_io as _rust
    _RUST_AVAILABLE = (
        callable(getattr(_rust, 'cfwb_list_channels',      None))
        and callable(getattr(_rust, 'cfwb_extract_waveform',    None))
        and callable(getattr(_rust, 'cfwb_extract_stim_times',  None))
    )
except ImportError:
    _rust = None            # type: ignore[assignment]
    _RUST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MAGIC          = b'CFWB'
_HEADER_SIZE    = 68
_CHAN_HDR_SIZE  = 96
_FMT_F64        = 1
_FMT_F32        = 2
_FMT_I16        = 3

# Struct formats (little-endian, 1-byte aligned)
# File header fields we actually use:
#   off 8:  secsPerTick (d)
#   off 52: NChannels   (l)
#   off 56: SamplesPerChannel (l)
#   off 60: TimeChannel (l)
#   off 64: DataFormat  (l)
_FILE_HDR_FMT = '<4sl d 5l d d 4l'   # 68 bytes

# Channel header: 32s title + 32s units + 4d (scale, offset, high, low)
_CHAN_HDR_FMT = '<32s 32s 4d'         # 96 bytes


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cstr(b: bytes) -> str:
    """Decode a null-terminated C string from a fixed-length byte field."""
    end = b.find(b'\x00')
    return b[:end].decode('latin-1', errors='replace').strip() if end >= 0 \
           else b.decode('latin-1', errors='replace').strip()


def _read_header(file_path: str) -> dict:
    """
    Parse the CFWB file header and all channel headers.

    Returns a dict with keys:
        secs_per_tick, n_channels, samples_per_channel,
        time_channel (bool), data_format (int),
        channels: list of {'title', 'units', 'scale', 'offset'}
        data_offset (int): byte offset of first sample
    """
    with open(file_path, 'rb') as fh:
        raw_hdr = fh.read(_HEADER_SIZE)

    if len(raw_hdr) < _HEADER_SIZE:
        raise ValueError("File too small to be a valid CFWB binary.")
    if raw_hdr[:4] != _MAGIC:
        raise ValueError(
            f"Not a CFWB file — magic bytes are {raw_hdr[:4]!r}, expected b'CFWB'."
        )

    (magic, version, secs_per_tick,
     year, month, day, hour, minute,
     second, trigger,
     n_channels, samples_per_channel,
     time_channel_flag, data_format) = struct.unpack_from(_FILE_HDR_FMT, raw_hdr)

    n_channels          = max(0, n_channels)
    samples_per_channel = max(0, samples_per_channel)
    time_channel        = bool(time_channel_flag)
    fs                  = round(1.0 / secs_per_tick)

    # Channel headers
    channels = []
    with open(file_path, 'rb') as fh:
        fh.seek(_HEADER_SIZE)
        for _ in range(n_channels):
            raw_ch = fh.read(_CHAN_HDR_SIZE)
            if len(raw_ch) < _CHAN_HDR_SIZE:
                break
            title_b, units_b, scale, offset, _, _ = struct.unpack(_CHAN_HDR_FMT, raw_ch)
            channels.append({
                'title':  _parse_cstr(title_b),
                'units':  _parse_cstr(units_b),
                'scale':  scale,
                'offset': offset,
            })
        data_offset = fh.tell()

    return {
        'secs_per_tick':       secs_per_tick,
        'fs':                  fs,
        'n_channels':          n_channels,
        'samples_per_channel': samples_per_channel,
        'time_channel':        time_channel,
        'data_format':         data_format,
        'channels':            channels,
        'data_offset':         data_offset,
    }


def _load_channel_py(file_path: str, hdr: dict, channel_idx: int) -> np.ndarray:
    """
    Pure-Python / NumPy fallback: extract one channel from the binary data.
    """
    fmt      = hdr['data_format']
    n_ch     = hdr['n_channels']
    n_samp   = hdr['samples_per_channel']
    tc       = hdr['time_channel']
    data_cols = n_ch + (1 if tc else 0)
    col       = channel_idx + (1 if tc else 0)
    ch        = hdr['channels'][channel_idx]

    dtype_map = {_FMT_F64: np.float64, _FMT_F32: np.float32, _FMT_I16: np.int16}
    dt = dtype_map.get(fmt, np.float64)

    with open(file_path, 'rb') as fh:
        fh.seek(hdr['data_offset'])
        raw = np.frombuffer(fh.read(), dtype=dt)

    if raw.size < data_cols * n_samp:
        n_samp = raw.size // data_cols

    raw = raw[:n_samp * data_cols].reshape(n_samp, data_cols)
    col_data = raw[:, col].astype(np.float64)

    # Apply scale / offset (matters for i16; no-op for float formats)
    return ch['scale'] * (col_data + ch['offset'])


# ─────────────────────────────────────────────────────────────────────────────
# Detection helper
# ─────────────────────────────────────────────────────────────────────────────

def is_cfwb(file_path: str) -> bool:
    """Return True if the file starts with the CFWB magic bytes."""
    try:
        with open(file_path, 'rb') as fh:
            return fh.read(4) == _MAGIC
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list[str]:
    """Return channel names from the CFWB channel headers."""
    if _RUST_AVAILABLE:
        return _rust.cfwb_list_channels(file_path)
    try:
        hdr = _read_header(file_path)
        names = [ch['title'] for ch in hdr['channels']]
        return names if names else ['Channel 1']
    except Exception:
        return ['Channel 1']


def extract_emg_waveform_and_fs(
    file_path: str,
    channel_idx: int = 0,
) -> tuple[np.ndarray, int, Optional[str]]:
    """
    Extract a single channel as a 1-D float64 waveform.

    Parameters
    ----------
    channel_idx : 0-based index into the list from list_waveform_channels().

    Returns
    -------
    emg  : np.ndarray  1-D array of calibrated samples
    fs   : int         sampling frequency in Hz
    unit : str | None  unit string from the channel header, or None
    """
    if _RUST_AVAILABLE:
        samples, fs, unit = _rust.cfwb_extract_waveform(file_path, channel_idx)
        return np.asarray(samples, dtype=float), int(fs), unit

    hdr = _read_header(file_path)
    idx = min(channel_idx, hdr['n_channels'] - 1)
    ch  = hdr['channels'][idx]
    unit = ch['units'] if ch['units'] else None
    emg  = _load_channel_py(file_path, hdr, idx)
    return emg, int(hdr['fs']), unit


def extract_stim_times(
    file_path: str,
    marker_name: str = 'A',
) -> dict[str, list[float]]:
    """
    Detect stimulation times from the CFWB trigger/TTL channel.

    The trigger channel is identified by searching channel titles for
    "stim", "trig", or "ttl" (case-insensitive).  If none is found the
    last channel is used as a fallback.

    Rising edges on the trigger signal (threshold = 50 % of peak) are
    returned as absolute timestamps in seconds from the first sample.

    Returns
    -------
    dict mapping label -> list[float]  e.g. {'A': [0.050, 0.550, ...]}
    """
    if _RUST_AVAILABLE:
        return dict(_rust.cfwb_extract_stim_times(file_path, marker_name))

    label = (marker_name[:1].upper() if marker_name else 'A')
    hdr   = _read_header(file_path)
    titles = [ch['title'].lower() for ch in hdr['channels']]

    # Auto-detect trigger channel
    stim_idx = next(
        (i for i, t in enumerate(titles)
         if any(k in t for k in ('stim', 'trig', 'ttl'))),
        len(titles) - 1
    )

    stim_sig = _load_channel_py(file_path, hdr, stim_idx)
    if stim_sig.size == 0:
        return {}

    max_val = stim_sig.max()
    if max_val <= 0:
        return {}

    thr   = max_val * 0.5
    edges = np.where(np.diff((stim_sig >= thr).astype(np.int8)) == 1)[0]
    times = ((edges + 1) / hdr['fs']).tolist()

    return {label: times} if times else {}
