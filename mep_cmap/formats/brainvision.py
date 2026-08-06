"""
mep_cmap.formats.brainvision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BrainVision Core Data Format reader (.vhdr / .vmrk / .eeg triplet).

BrainVision splits a recording across three sibling files:
  .vhdr  INI-style text header  (channels, sampling interval, binary format)
  .vmrk  INI-style text markers (stimulus/response/segment events)
  .eeg   raw binary sample data

Any one of the three may be passed as ``file_path``; the header (.vhdr) is
resolved from it, and the data/marker files are located from the header's
``DataFile`` / ``MarkerFile`` entries (falling back to same-basename siblings).

This is a lightweight native reader — no MNE / FieldTrip dependency — so it
adds nothing to the binary footprint.  It supports the common variants:
BinaryFormat INT_16 / INT_32 / IEEE_FLOAT_32 and DataOrientation MULTIPLEXED /
VECTORIZED.  Samples are scaled by each channel's resolution into that
channel's native unit (typically µV); no further rescaling is applied.

Stimulation times come from the .vmrk markers, grouped by marker description
(e.g. 'S255'), excluding structural 'New Segment' boundaries.

Public API (mirrors the io.py contract)
----------------------------------------
  is_brainvision(file_path)                    -> bool   (detection helper)
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Dependency: none beyond NumPy.
"""

import os

import numpy as np

_HEADER_SIGNATURE = 'BrainVision Data Exchange Header File'

_DTYPE_MAP = {
    'INT_16': np.dtype('<i2'),
    'INT_32': np.dtype('<i4'),
    'IEEE_FLOAT_32': np.dtype('<f4'),
}

_BOUNDARY_MARKER_TYPES = {'new segment'}


# ── Detection helper ──────────────────────────────────────────────────────────

def _resolve_vhdr(file_path: str) -> str:
    """Return the .vhdr path for any member of the triplet."""
    base, ext = os.path.splitext(file_path)
    if ext.lower() == '.vhdr':
        return file_path
    cand = base + '.vhdr'
    return cand if os.path.exists(cand) else file_path


def is_brainvision(file_path: str) -> bool:
    """True if the file (or its sibling .vhdr) is a BrainVision header."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ('.vhdr', '.vmrk', '.eeg'):
        return False
    vhdr = _resolve_vhdr(file_path)
    if not os.path.exists(vhdr) or os.path.splitext(vhdr)[1].lower() != '.vhdr':
        return False
    try:
        with open(vhdr, 'r', encoding='utf-8', errors='replace') as f:
            return _HEADER_SIGNATURE in f.readline()
    except Exception:
        return False


# ── Header / marker parsing ───────────────────────────────────────────────────

def _parse_ini(path: str) -> dict:
    """
    Parse a BrainVision INI-style text file into {section: {key: value}} plus a
    special 'Channel Infos'/'Marker Infos' list of the raw entry lines (which
    are positional CSV, not key=value in the usual sense).
    """
    sections = {}
    order_lines = {}
    current = None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            s = line.strip()
            if not s or s.startswith(';'):
                continue
            if s.startswith('[') and s.endswith(']'):
                current = s[1:-1].strip()
                sections.setdefault(current, {})
                order_lines.setdefault(current, [])
                continue
            if current is None:
                continue
            order_lines[current].append(line)
            if '=' in line:
                k, v = line.split('=', 1)
                sections[current][k.strip()] = v.strip()
    return {'sections': sections, 'lines': order_lines}


def _header(file_path: str) -> dict:
    vhdr = _resolve_vhdr(file_path)
    if os.path.splitext(vhdr)[1].lower() != '.vhdr':
        raise ValueError(f"BrainVision: no .vhdr header found for {file_path!r}.")
    parsed = _parse_ini(vhdr)
    sec = parsed['sections']
    common = sec.get('Common Infos', {})
    bininfo = sec.get('Binary Infos', {})

    hdr_dir = os.path.dirname(vhdr)
    n_chan = int(common.get('NumberOfChannels', 0))
    sampling_interval_us = float(common.get('SamplingInterval', 0) or 0)
    if sampling_interval_us <= 0:
        raise ValueError("BrainVision: missing/invalid SamplingInterval.")
    fs = int(round(1e6 / sampling_interval_us))

    orientation = common.get('DataOrientation', 'MULTIPLEXED').upper()
    bin_fmt = bininfo.get('BinaryFormat', 'INT_16').upper()
    if bin_fmt not in _DTYPE_MAP:
        raise ValueError(f"BrainVision: unsupported BinaryFormat {bin_fmt!r}.")

    # Channel infos: Ch<n>=<name>,<ref>,<resolution>,<unit>
    names, resolutions, units = [], [], []
    for line in parsed['lines'].get('Channel Infos', []):
        if not line.startswith('Ch') or '=' not in line:
            continue
        _, val = line.split('=', 1)
        parts = [p.replace('\\1', ',') for p in val.split(',')]
        names.append(parts[0].strip() if parts else '')
        res = 0.0
        if len(parts) >= 3 and parts[2].strip():
            try:
                res = float(parts[2])
            except ValueError:
                res = 0.0
        resolutions.append(res if res else 1.0)
        units.append(parts[3].strip() if len(parts) >= 4 else '')

    if n_chan == 0:
        n_chan = len(names)
    if len(names) < n_chan:
        names += [f'Ch{i + 1}' for i in range(len(names), n_chan)]
        resolutions += [1.0] * (n_chan - len(resolutions))
        units += [''] * (n_chan - len(units))

    # Resolve data + marker files (header entries, then basename fallback)
    base = os.path.splitext(vhdr)[0]
    data_file = common.get('DataFile', '')
    data_path = os.path.join(hdr_dir, data_file) if data_file else ''
    if not (data_path and os.path.exists(data_path)):
        data_path = base + '.eeg'
    marker_file = common.get('MarkerFile', '')
    marker_path = os.path.join(hdr_dir, marker_file) if marker_file else ''
    if not (marker_path and os.path.exists(marker_path)):
        marker_path = base + '.vmrk'

    return dict(fs=fs, n_chan=n_chan, orientation=orientation,
                dtype=_DTYPE_MAP[bin_fmt], names=names[:n_chan],
                resolutions=resolutions[:n_chan], units=units[:n_chan],
                data_path=data_path, marker_path=marker_path)


def _norm_unit(u):
    if not u:
        return None
    s = str(u).strip()
    low = s.lower()
    if low in ('microvolts', 'microvolt', 'uv', '\u00b5v', '\u03bcv'):
        return '\u00b5V'
    if low in ('millivolts', 'millivolt', 'mv'):
        return 'mV'
    if low in ('volts', 'volt', 'v'):
        return 'V'
    # Normalise a stray Greek-mu micro sign to the MICRO SIGN used elsewhere
    return s.replace('\u03bc', '\u00b5')


# ── Public API ────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return channel names from the .vhdr [Channel Infos]."""
    return _header(file_path)['names'] or ['Ch1']


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Return one channel's waveform (scaled to its native unit), sample rate,
    and unit string.

    Returns
    -------
    emg  : np.ndarray  waveform (native unit, resolution-scaled only)
    fs   : int         sampling rate in Hz
    unit : str | None  normalised unit string (e.g. 'µV')
    """
    h = _header(file_path)
    n_chan = h['n_chan']
    if not (0 <= channel_idx < n_chan):
        raise IndexError(
            f"channel_idx {channel_idx} out of range (0..{n_chan - 1})")
    if not os.path.exists(h['data_path']):
        raise ValueError(f"BrainVision: data file not found ({h['data_path']!r}).")

    raw = np.fromfile(h['data_path'], dtype=h['dtype'])
    if raw.size % n_chan != 0:
        # Trim any trailing partial frame defensively
        raw = raw[:raw.size - (raw.size % n_chan)]

    if h['orientation'] == 'VECTORIZED':
        chan = raw.reshape(n_chan, -1)[channel_idx]
    else:  # MULTIPLEXED
        chan = raw.reshape(-1, n_chan)[:, channel_idx]

    emg = chan.astype(float) * h['resolutions'][channel_idx]
    return emg, h['fs'], _norm_unit(h['units'][channel_idx])


def extract_stim_times(file_path: str, marker_name: str = '') -> dict:
    """
    Return stim times from the .vmrk markers, grouped by marker description
    (e.g. 'S255'), excluding 'New Segment' boundaries.

    ``marker_name`` is accepted for API parity but deliberately ignored: the
    .vmrk descriptions define the stim types, and this tool supports many
    types per recording (paired-pulse, multi-intensity).  Filtering here would
    silently drop every type except the one the marker dropdown happens to
    hold.  Same contract as formats/edf.py.

    Returns
    -------
    dict mapping description -> list of timestamps (seconds).
    """
    h = _header(file_path)
    fs = h['fs']
    if not os.path.exists(h['marker_path']):
        return {}

    parsed = _parse_ini(h['marker_path'])
    out = {}
    for line in parsed['lines'].get('Marker Infos', []):
        if not line.startswith('Mk') or '=' not in line:
            continue
        _, val = line.split('=', 1)
        parts = [p.replace('\\1', ',') for p in val.split(',')]
        if len(parts) < 3:
            continue
        mtype = parts[0].strip()
        desc = parts[1].strip()
        if mtype.lower() in _BOUNDARY_MARKER_TYPES:
            continue
        try:
            pos = int(float(parts[2]))
        except ValueError:
            continue
        label = desc if desc else (mtype if mtype else 'A')
        # .vmrk positions are 1-based ("Position in data points"): the leading
        # 'New Segment' marker sits at position 1, i.e. the first sample, t=0.
        # Subtract 1 before converting, or every event lands one sample late.
        out.setdefault(label, []).append(max(pos - 1, 0) / fs)

    for label in out:
        out[label].sort()

    return out
