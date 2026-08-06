"""
mep_cmap.formats.mne_bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optional fallback reader backed by MNE-Python.

This module exists to widen format coverage *without* displacing any of the
native readers.  It is deliberately constrained in three ways:

1. **MNE is imported lazily, inside the functions.**  There is no module-level
   ``import mne``, so importing this module is free and the absence of MNE
   degrades to "format not supported" rather than an ImportError at startup.
   MNE is an optional extra; it is not in requirements.txt and is not bundled
   into the PyInstaller builds.

2. **It claims only an explicit allowlist of extensions.**  MNE's own
   ``read_raw`` auto-dispatch maps ``.txt`` to BOXY and ``.mat`` to FieldTrip,
   which would silently steal two of this tool's primary formats.  The
   allowlist below contains only extensions that no native reader handles.

3. **It is registered as the last branch in io.detect_format().**  Even if the
   allowlist were wrong, every native reader is consulted first, so a validated
   reader can never be overridden by this one.

Units
-----
MNE returns data in SI units — volts for EEG/EMG/EOG/ECG channels.  This module
reports ``'V'`` for those channels and lets ``io._to_mV()`` perform the single,
central conversion to millivolts.  No scaling is done here.  Channels whose
FIFF unit is not volts report ``None``, so they pass through unconverted.

Validation status
-----------------
The waveform, unit and annotation logic below is verified against BrainVision
recordings (by calling this module directly; in normal operation BrainVision is
handled by the native reader).  The individual MNE format readers are *not*
independently verified against real files of each type — no test data was
available.  Treat any newly claimed format as unvalidated until it has been
checked against a real recording from that system.

Public API (mirrors the io.py contract)
----------------------------------------
  is_available()                               -> bool
  is_mne_readable(file_path)                   -> bool
  list_waveform_channels(file_path)            -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Dependency: mne (optional; ``pip install mne``).
"""

import os

import numpy as np

# ── Extension allowlist ───────────────────────────────────────────────────────
#
# Only extensions with no native reader in this package.  Deliberately EXCLUDED
# because a native, validated reader already owns them:
#
#   .txt   Spike-2 / LabChart / Brainsight   (MNE would claim it as BOXY)
#   .mat   LabChart / AcqKnowledge exports   (MNE would claim it as FieldTrip)
#   .vhdr  BrainVision                       (native reader; verified vs MNE)
#   .edf   .bdf  EDF/BDF                     (native reader; BIDS-ify output)
#   .smr   .adibin   .acq   .csv             (native readers)
#
# Note on .mff (EGI): MNE reads it, but an .mff is a *directory*, and
# io.detect_format() guards on os.path.isfile().  It is therefore omitted
# rather than listed and silently broken.

_MNE_EXTS = (
    '.set',                                  # EEGLAB
    '.cnt',                                  # Neuroscan / ANT eego
    '.nxe',                                  # Nexstim eXimia (TMS)
    '.gdf',                                  # GDF
    '.lay',                                  # Persyst
    '.cdt', '.cdt.dpa', '.cef', '.dap', '.rs3',   # Curry
    '.data',                                 # Nicolet
    '.ns3',                                  # Blackrock NSx
    '.nedf',                                 # NEDF
    '.fif', '.fif.gz',                       # FIF
    '.eeg',                                  # Nihon Kohden — see guard below
)

# Annotation descriptions that mark structure, not stimulation.
_BOUNDARY_DESCRIPTIONS = (
    'new segment', 'boundary', 'bad boundary', 'edge boundary',
    'bad_acq_skip', 'ignored',
)

_RAW_CACHE = {}          # {cache_key: Raw}   bounded to _CACHE_LIMIT entries
_CACHE_LIMIT = 2


# ── Lazy MNE import ───────────────────────────────────────────────────────────

def _mne():
    """Import and return the mne module, or None if unavailable."""
    try:
        import mne as _m
    except Exception:
        return None
    try:
        _m.set_log_level('ERROR')
    except Exception:
        pass
    return _m


def is_available() -> bool:
    """True if MNE-Python is installed and importable."""
    return _mne() is not None


# ── Detection ─────────────────────────────────────────────────────────────────

def _ext_matches(file_path: str) -> bool:
    low = os.path.basename(file_path).lower()
    return any(low.endswith(e) for e in _MNE_EXTS)


def is_mne_readable(file_path: str) -> bool:
    """
    True if this file should be handed to MNE.

    Requires MNE to be installed AND the extension to be on the allowlist.
    The .eeg case is guarded: a .eeg with a sibling .vhdr is a BrainVision
    binary owned by the native reader, not a Nihon Kohden recording.
    """
    if not _ext_matches(file_path):
        return False
    if os.path.splitext(file_path)[1].lower() == '.eeg':
        if os.path.exists(os.path.splitext(file_path)[0] + '.vhdr'):
            return False                      # BrainVision — hands off
    return is_available()


# ── Raw loading (cached) ──────────────────────────────────────────────────────

def _cache_key(file_path: str):
    try:
        st = os.stat(file_path)
        return (os.path.abspath(file_path), st.st_mtime_ns, st.st_size)
    except OSError:
        return (os.path.abspath(file_path), None, None)


def _raw(file_path: str):
    """
    Return a cached MNE Raw for this file (preload=False).

    preload=False keeps memory flat for long recordings; get_data() reads the
    requested channel from disk on demand.  The cache is keyed on path + mtime
    + size, so an edited file is re-read rather than served stale.
    """
    m = _mne()
    if m is None:
        raise RuntimeError(
            "MNE-Python is required to read this file "
            "(pip install mne).")

    key = _cache_key(file_path)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]

    try:
        raw = m.io.read_raw(file_path, preload=False, verbose='ERROR')
    except Exception as exc:
        raise ValueError(
            f"MNE could not read {os.path.basename(file_path)}: {exc}") from exc

    if len(_RAW_CACHE) >= _CACHE_LIMIT:
        _RAW_CACHE.pop(next(iter(_RAW_CACHE)))
    _RAW_CACHE[key] = raw
    return raw


def _unit_for(raw, idx):
    """
    Return 'V' when MNE reports this channel in volts, else None.

    io._to_mV() performs the conversion; nothing is scaled here.  Returning
    None for non-volt channels means they pass through unconverted rather than
    being silently mis-scaled.
    """
    try:
        from mne.io.constants import FIFF
        ch = raw.info['chs'][idx]
        if int(ch['unit']) == int(FIFF.FIFF_UNIT_V):
            return 'V'
    except Exception:
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return all channel names, in file order."""
    return list(_raw(file_path).ch_names)


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Return one channel's waveform, sample rate, and unit string.

    Returns
    -------
    emg  : np.ndarray  waveform in MNE's native SI unit (volts for EEG/EMG)
    fs   : int         sampling rate in Hz
    unit : str | None  'V' for volt channels, else None
    """
    raw = _raw(file_path)
    n_chan = len(raw.ch_names)
    if not (0 <= channel_idx < n_chan):
        raise IndexError(
            f"channel_idx {channel_idx} out of range (0..{n_chan - 1})")

    data = raw.get_data(picks=[channel_idx])
    emg = np.asarray(data[0], dtype=float)
    fs = int(round(float(raw.info['sfreq'])))
    return emg, fs, _unit_for(raw, channel_idx)


def _clean_description(desc: str) -> str:
    """
    Normalise an MNE annotation description to a bare stimulus label.

    MNE composes BrainVision-style markers as '<type>/<description>', e.g.
    'Stimulus/S128'.  The tool's stim-type labels are the bare form, so the
    trailing segment is taken.  Labels without a '/' are returned unchanged.
    """
    s = str(desc).strip()
    if '/' in s:
        s = s.rsplit('/', 1)[-1].strip()
    return s


def extract_stim_times(file_path: str, marker_name: str = '') -> dict:
    """
    Return stim times grouped by label.

    Annotations are used when present.  If a file carries no usable
    annotations, a stim/trigger channel is looked for instead and rising edges
    are decoded via mne.find_events — this is what most EEG-system formats use
    for TMS triggers.

    ``marker_name`` is accepted for API parity but deliberately ignored — see
    formats/brainvision.py for the rationale: filtering here would silently
    drop every stim type except the one the marker dropdown happens to hold.

    Returns
    -------
    dict mapping label -> sorted list of timestamps (seconds).
    """
    raw = _raw(file_path)
    out = {}

    # 1) Annotations
    try:
        ann = raw.annotations
        for onset, desc in zip(ann.onset, ann.description):
            label = _clean_description(desc)
            if not label or label.lower() in _BOUNDARY_DESCRIPTIONS:
                continue
            out.setdefault(label, []).append(float(onset))
    except Exception:
        pass

    # 2) Fallback: decode a stim/trigger channel
    if not out:
        m = _mne()
        try:
            picks = m.pick_types(raw.info, stim=True, meg=False, eeg=False)
            if len(picks):
                events = m.find_events(raw, verbose='ERROR')
                sfreq = float(raw.info['sfreq'])
                first = int(raw.first_samp)
                for samp, _prev, code in events:
                    out.setdefault(f"S{int(code)}", []).append(
                        (int(samp) - first) / sfreq)
        except Exception:
            pass

    for label in out:
        out[label].sort()

    return out
