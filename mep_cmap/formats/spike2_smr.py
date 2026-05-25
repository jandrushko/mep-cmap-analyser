"""
mep_cmap.formats.spike2_smr
Native Spike2 .smr file reader using the Neo library.

On first open: dialog to choose EMG channel and stim/trigger channel.
Config saved to <file>.smr_config.json sidecar.

Stim time grouping mirrors smr2txt.py: per-event marker codes are decoded
from Neo event labels or waveforms so that DigMark events are split by
code letter (A, B, C...) exactly as the text-export reader does.

extract_stim_times(path, marker_name) behaviour:
  - If marker_name matches a known code in the stim channel,
    return only events with that code: {"A": [t1, t2, ...]}
  - If marker_name is the channel name (first scan), return all
    codes grouped: {"A": [...], "B": [...], ...}
  - Falls back to analogue threshold crossing when no event channels exist.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Graceful dependency check
# ---------------------------------------------------------------------------

def _neo_available() -> bool:
    try:
        import neo  # noqa: F401
        return True
    except ImportError:
        return False


def _require_neo():
    if not _neo_available():
        raise ImportError(
            "The 'neo' package is required to read native Spike2 .smr files.\n"
            "Install it with:  pip install neo"
        )


# ---------------------------------------------------------------------------
# Sidecar config helpers
# ---------------------------------------------------------------------------

def _sidecar_path(file_path: str) -> Path:
    return Path(file_path).with_suffix(".smr_config.json")


def has_config(file_path: str) -> bool:
    p = _sidecar_path(file_path)
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return bool(cfg.get("emg_channel") and cfg.get("stim_channel"))
    except Exception:
        return False


def load_config(file_path: str) -> dict:
    p = _sidecar_path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"No SMR config found for {Path(file_path).name}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_config(file_path: str, emg_channel: str, stim_channel: str) -> None:
    p = _sidecar_path(file_path)
    p.write_text(
        json.dumps({"emg_channel": emg_channel, "stim_channel": stim_channel},
                   indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Segment + rawio cache (LRU-1)
# ---------------------------------------------------------------------------

_cache_lock   = threading.Lock()
_cached_path:  list = [None]
_cached_seg:   list = [None]
_cached_names: list = [None]


def _b2s(x):
    return x.decode("latin-1", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)


def _load(file_path: str):
    """Load SMR via Neo, cache result. Returns (seg, analogue_names)."""
    with _cache_lock:
        if _cached_path[0] == file_path and _cached_seg[0] is not None:
            return _cached_seg[0], _cached_names[0]

    _require_neo()
    import neo
    import warnings

    try:
        reader = neo.io.Spike2IO(filename=file_path, try_signal_grouping=False)
    except TypeError:
        reader = neo.io.Spike2IO(filename=file_path)

    # Use rawio header for analogue names (more reliable than seg.analogsignals[i].name)
    analogue_names = None
    try:
        rawio = reader.rawio
        rawio.parse_header()
        analogue_names = [_b2s(m["name"]) for m in rawio.header["signal_channels"]]
    except Exception:
        try:
            from neo.rawio import Spike2RawIO
            rawio2 = Spike2RawIO(filename=file_path)
            rawio2.parse_header()
            analogue_names = [_b2s(m["name"]) for m in rawio2.header["signal_channels"]]
        except Exception:
            analogue_names = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*can not be converted to a quantity.*")
        warnings.filterwarnings("ignore", message=".*UnitWarning.*")
        block = reader.read_block(lazy=False)
    if not block.segments:
        raise ValueError(f"No segments found in {file_path}")
    seg = block.segments[0]

    if not analogue_names:
        analogue_names = [sig.name for sig in seg.analogsignals] or ["Channel 1"]

    with _cache_lock:
        _cached_path[0]  = file_path
        _cached_seg[0]   = seg
        _cached_names[0] = analogue_names

    return seg, analogue_names


def clear_cache():
    with _cache_lock:
        _cached_path[0]  = None
        _cached_seg[0]   = None
        _cached_names[0] = None


# ---------------------------------------------------------------------------
# Marker code decoding  (mirrors smr2txt.py _derive_labels)
# ---------------------------------------------------------------------------

def _decode_marker_code(raw) -> str:
    """
    Decode a single marker code value to a printable character.

    Handles bytes, numeric strings, and direct characters.
    Returns the single ASCII letter (e.g. 'A', 'B') or '?' if undecodable.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("latin-1", errors="replace").strip()
    s = str(raw).strip()
    # Numeric string (e.g. "66") -> chr(66) = 'B'
    if s.isdigit():
        v = int(s)
        if 32 <= v <= 126:
            return chr(v)
    # Already a single printable ASCII character
    if len(s) == 1 and 32 <= ord(s[0]) <= 126:
        return s
    # Multi-char: return as-is (e.g. already decoded)
    return s if s else "?"


def _get_event_codes(evt) -> list:
    """
    Return a list of per-event marker codes for a Neo Event/Epoch object.

    Priority order (same as smr2txt.py _derive_labels):
    1. evt.labels  (array of per-event label bytes/strings)
    2. evt.waveforms (Spike2 DigMark stores the code as first sample)
    3. Fall back to the channel name repeated for each event
    """
    n = len(evt.times)

    # 1. labels attribute
    if hasattr(evt, "labels") and evt.labels is not None:
        try:
            labs = [_decode_marker_code(_b2s(lb).strip()) for lb in evt.labels]
            if len(labs) == n and any(lb != "" for lb in labs):
                return labs
        except Exception:
            pass

    # 2. waveforms (single-sample; code stored as first value)
    if hasattr(evt, "waveforms") and evt.waveforms is not None:
        try:
            codes = []
            for w in evt.waveforms:
                v = int(float(str(w.flat[0])))
                codes.append(chr(v) if 32 <= v <= 126 else "?")
            if len(codes) == n:
                return codes
        except Exception:
            pass

    # 3. fallback: channel name for all events
    return [evt.name] * n


def get_event_codes_for_channel(file_path: str, channel_name: str) -> list:
    """
    Return the sorted list of unique marker codes present in the named
    event/epoch/spike channel.  Used by app.py to populate the code picker.
    """
    seg, _ = _load(file_path)
    candidates = list(seg.events) + list(seg.epochs) + list(seg.spiketrains)
    cl = channel_name.lower()
    target = next(
        (c for c in candidates if c.name.lower() == cl or cl in c.name.lower()),
        None
    )
    if target is None:
        return []
    codes = _get_event_codes(target)
    return sorted(set(codes))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STIM_KW = ("stim", "trig", "ttl", "digmark", "keyboard")


def _is_trigger(name: str) -> bool:
    return any(kw in name.lower() for kw in _STIM_KW)


def get_channel_info(file_path: str) -> dict:
    seg, analogue_names = _load(file_path)
    return {
        "analogue": analogue_names,
        "events":   [e.name for e in seg.events],
        "epochs":   [e.name for e in seg.epochs],
        "spikes":   [s.name for s in seg.spiketrains],
    }


def list_waveform_channels(file_path: str) -> list:
    try:
        _, names = _load(file_path)
        return names if names else ["Channel 1"]
    except ImportError:
        raise
    except Exception:
        return ["Channel 1"]


def list_event_channels(file_path: str) -> list:
    try:
        info = get_channel_info(file_path)
        names = info["events"] + info["epochs"] + info["spikes"]
        names += [n for n in info["analogue"] if _is_trigger(n)]
        return names
    except ImportError:
        raise
    except Exception:
        return []


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0) -> tuple:
    seg, analogue_names = _load(file_path)
    if not seg.analogsignals:
        raise ValueError(f"No analogue signals found in {file_path}")

    sig = None

    # Only resolve by sidecar name when channel_idx == 0 (the primary EMG).
    # For any other index the caller (inspector extra channel) is requesting
    # a specific channel by position — honour that directly.
    if channel_idx == 0 and has_config(file_path):
        target_name = load_config(file_path).get("emg_channel", "")
        tl = target_name.lower()
        for i, n in enumerate(analogue_names):
            if n.lower() == tl or tl in n.lower():
                if i < len(seg.analogsignals):
                    sig = seg.analogsignals[i]
                break

    if sig is None:
        idx = min(channel_idx, len(seg.analogsignals) - 1)
        sig = seg.analogsignals[idx]

    emg  = np.asarray(sig).flatten().astype(float)
    fs   = int(round(float(sig.sampling_rate.rescale("Hz").magnitude)))
    unit = None
    try:
        u = str(sig.units.dimensionality).strip().split()
        unit = u[-1] if u else None
    except Exception:
        pass
    return emg, fs, unit


def extract_stim_times(file_path: str, marker_name: str = "A") -> dict:
    """
    Return stim times grouped by marker code, normalised to t=0.

    If marker_name matches a specific code present in the stim channel
    (e.g. 'A'), only that code's events are returned.
    If marker_name matches the channel name itself (e.g. 'DigMark'),
    all codes are returned grouped: {"A": [...], "B": [...], ...}.
    """
    seg, analogue_names = _load(file_path)

    t0 = (float(seg.analogsignals[0].t_start.rescale("s").magnitude)
          if seg.analogsignals
          else float(seg.t_start.rescale("s").magnitude))

    stim_ch = (load_config(file_path).get("stim_channel", marker_name)
               if has_config(file_path) else marker_name)

    sl = stim_ch.lower()
    ml = marker_name.lower()

    # --- Find the stim event channel ---
    evt_all = list(seg.events) + list(seg.epochs) + list(seg.spiketrains)
    target  = None
    if evt_all:
        target = next((c for c in evt_all if c.name.lower() == sl), None)
        if target is None:
            target = next((c for c in evt_all if sl in c.name.lower()), None)
        if target is None:
            target = next((c for c in evt_all if _is_trigger(c.name)), None)
        if target is None:
            target = evt_all[0]

    if target is not None:
        codes     = _get_event_codes(target)
        times_abs = target.times.rescale("s").magnitude
        times_rel = times_abs - t0

        # Group by code
        by_code: dict = {}
        for t, code in zip(times_rel, codes):
            if t >= 0:
                by_code.setdefault(code, []).append(float(t))

        if not by_code:
            return {}

        # If marker_name is a specific code that exists, return only that
        if marker_name in by_code:
            return {marker_name: by_code[marker_name]}

        # If marker_name is the channel name or a fallback, return all codes
        return by_code

    # --- Analogue threshold crossing fallback ---
    trig_sig = None
    for i, n in enumerate(analogue_names):
        if n.lower() == sl or sl in n.lower() or _is_trigger(n):
            if i < len(seg.analogsignals):
                trig_sig = seg.analogsignals[i]
            break
    if trig_sig is None:
        return {}

    stim_arr = np.asarray(trig_sig).flatten().astype(float)
    fs_trig  = float(trig_sig.sampling_rate.rescale("Hz").magnitude)
    t0_trig  = float(trig_sig.t_start.rescale("s").magnitude)
    max_val  = stim_arr.max()
    if max_val <= 0:
        return {}
    thr   = max_val * 0.5
    edges = np.where(np.diff((stim_arr >= thr).astype(np.int8)) == 1)[0]
    label = marker_name[:1].upper() if len(marker_name) == 1 else marker_name
    times = ((edges + 1) / fs_trig + t0_trig - t0).tolist()
    return {label: [t for t in times if t >= 0]} if times else {}
