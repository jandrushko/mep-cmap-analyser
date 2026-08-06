"""
mep_cmap.io
~~~~~~~~~~~
Format-agnostic public API for reading EMG data files.

Supported formats (auto-detected from file header)
----------------------------------------------------
  Spike-2 text export  — header contains "SUMMARY" / "START" / "CHANNEL" blocks
  LabChart text export — header line 0 starts with "Interval="
  Generic TSV          — headerless / all-numeric tab/space/comma delimited text
                         (requires a one-time Format Wizard dialog on first open)

Adding a new format
-------------------
  1. Create mep_cmap/formats/<format>.py with the three public functions.
  2. Add detection logic to detect_format().
  3. Add a dispatch branch to each of the three public functions below.
  4. Nothing else in the codebase needs to change.

Public API
----------
  detect_format(file_path)                     -> 'spike2' | 'spike2_smr' | 'labchart' | 'cfwb' | 'generic_tsv'
  needs_wizard(file_path)                      -> bool
  list_waveform_channels(file_path)            -> list[str]
  list_event_channels(file_path)               -> list[str]
  extract_emg_waveform_and_fs(file_path, ch)   -> (np.ndarray, int, str|None)
  extract_stim_times(file_path, marker_name)   -> dict[str, list[float]]

Generic TSV — wizard integration
---------------------------------
When detect_format() returns 'generic_tsv' and no sidecar config exists yet,
the caller (app.py / _browse_file_path) must launch FormatWizard before
calling list_waveform_channels() or extract_*.

The recommended pattern in app.py is:

    _fmt = detect_format(fpath)
    if _fmt == 'generic_tsv' and needs_wizard(fpath):
        _launch_format_wizard(fpath, on_complete=lambda cfg: ...)
        return   # _browse_file_path will be called again from the callback
    ...
    chan_list = list_waveform_channels(fpath)
"""

import os as _os

from .formats import spike2      as _spike2
from .formats import spike2_smr  as _spike2_smr
from .formats import labchart    as _labchart
from .formats import labchart_mat as _labchart_mat
from .formats import brainsight  as _brainsight
from .formats import acqknowledge_mat as _acqknowledge_mat
from .formats import acqknowledge_acq as _acqknowledge_acq
from .formats import brainvision  as _brainvision
from .formats import edf          as _edf
from .formats import cfwb        as _cfwb
from .formats import generic_tsv as _generic_tsv
from .formats import mne_bridge  as _mne_bridge   # optional; lazy-imports mne

def _generic_has_config(file_path: str) -> bool:
    return _generic_tsv.has_config(file_path)


def _resolve_path(file_path: str) -> str:
    """
    Resolve a possibly-relative path to absolute.

    Paths stored in the dataset JSON may be relative (for cross-computer /
    OneDrive portability) and may use backslashes on Windows.  This function
    normalises the slashes and searches a cascade of candidate roots until the
    file is found.
    """
    # Normalise backslashes → OS separator
    file_path = _os.path.normpath(file_path.replace("\\", _os.sep))
    if _os.path.isabs(file_path) and _os.path.exists(file_path):
        return file_path
    if _os.path.isabs(file_path):
        return file_path  # absolute but missing — let open() raise clearly

    import sys as _sys
    candidates = [_os.getcwd()]
    try:
        candidates.append(_os.path.dirname(_os.path.abspath(__file__)))
        candidates.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    except Exception:
        pass
    try:
        candidates.append(_os.path.dirname(_os.path.abspath(_sys.argv[0])))
    except Exception:
        pass
    # Walk up from cwd looking for the study root (contains derivatives/)
    walk = _os.getcwd()
    for _ in range(8):
        if _os.path.isdir(_os.path.join(walk, "derivatives")):
            candidates.append(walk)
            break
        parent = _os.path.dirname(walk)
        if parent == walk:
            break
        walk = parent

    for root in candidates:
        resolved = _os.path.normpath(_os.path.join(root, file_path))
        if _os.path.isfile(resolved):
            return resolved

    # Nothing found — return joined to cwd so open() gives a clear error
    return _os.path.normpath(_os.path.join(_os.getcwd(), file_path))


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(file_path: str) -> str:
    """
    Inspect the file header and return a format identifier string.

    Returns
    -------
    'labchart'    — LabChart text export (line 0 starts with 'Interval=')
    'spike2'      — Spike-2 text export (contains SUMMARY/CHANNEL/START blocks)
    'generic_tsv' — Headerless numeric text file (no recognised format header)
    """
    file_path = _resolve_path(file_path)

    # Guard: reject missing or zero-byte files with a clear message
    if not _os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if _os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty (0 bytes): {_os.path.basename(file_path)}")

    # ── Extension-based detection for binary / Neo formats ────────────────────
    ext = _os.path.splitext(file_path)[1].lower()
    if ext == '.smr':
        return 'spike2_smr'
    if ext == '.acq':
        return 'acqknowledge_acq'

    # ── Binary formats: check magic bytes before opening as text ─────────────
    if _cfwb.is_cfwb(file_path):
        return 'cfwb'

    # LabChart MATLAB export (.mat) — verify signature vars without loading data
    if ext == '.mat' and _labchart_mat.is_labchart_mat(file_path):
        return 'labchart_mat'
    if ext == '.mat' and _acqknowledge_mat.is_acqknowledge_mat(file_path):
        return 'acqknowledge_mat'

    # Brainsight neuronavigation export (.txt) — header-signature sniff
    if _brainsight.is_brainsight(file_path):
        return 'brainsight'

    # BrainVision (.vhdr/.vmrk/.eeg) — resolves via the sibling .vhdr header
    if _brainvision.is_brainvision(file_path):
        return 'brainvision'

    # EDF / BDF (.edf/.bdf) - written by BIDS-ify; stim times from sibling _events.tsv
    if _edf.is_edf(file_path):
        return 'edf'

    # Optional MNE fallback — LAST resort, after every native reader has been
    # consulted, so a validated reader can never be displaced.  Claims only an
    # explicit allowlist of extensions no native reader owns, and only when
    # MNE is actually installed.
    if _mne_bridge.is_mne_readable(file_path):
        return 'mne'

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
        second_line = f.readline()

    # LabChart: first line starts with 'Interval='
    if first_line.startswith('Interval='):
        return 'labchart'

    # Spike2: SUMMARY block or quoted channel names in the first two lines
    if ('"SUMMARY"' in first_line or '"SUMMARY"' in second_line
            or first_line.startswith('"')
            or '"Waveform"' in first_line or '"Waveform"' in second_line):
        return 'spike2'

    # Heuristic: if the first non-empty line parses as all-numeric fields,
    # treat as a generic headerless TSV.
    test_line = first_line.strip()
    if not test_line:
        test_line = second_line.strip()
    if test_line:
        # Try splitting by common delimiters
        for sep in ('\t', ',', ' '):
            parts = [p.strip() for p in test_line.split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    [float(p) for p in parts]
                    return 'generic_tsv'
                except ValueError:
                    pass

    # Default fallback
    return 'spike2'


def needs_wizard(file_path: str) -> bool:
    """
    Return True if the file requires first-open configuration.

    - generic_tsv: True when no sidecar config exists yet.
    - spike2_smr:  True when no SMR channel assignment sidecar exists yet.
    """
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'generic_tsv':
        return not _generic_has_config(file_path)
    if fmt == 'spike2_smr':
        return not _spike2_smr.has_config(file_path)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API — dispatches to the correct format reader
# ─────────────────────────────────────────────────────────────────────────────

def list_waveform_channels(file_path: str) -> list:
    """Return channel names for display in the channel selector."""
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'spike2_smr':
        return _spike2_smr.list_waveform_channels(file_path)
    if fmt == 'acqknowledge_acq':
        return _acqknowledge_acq.list_waveform_channels(file_path)
    if fmt == 'acqknowledge_mat':
        return _acqknowledge_mat.list_waveform_channels(file_path)
    if fmt == 'brainvision':
        return _brainvision.list_waveform_channels(file_path)
    if fmt == 'edf':
        return _edf.list_waveform_channels(file_path)
    if fmt == 'mne':
        return _mne_bridge.list_waveform_channels(file_path)
    if fmt == 'brainsight':
        return _brainsight.list_waveform_channels(file_path)
    if fmt == 'labchart_mat':
        return _labchart_mat.list_waveform_channels(file_path)
    if fmt == 'labchart':
        return _labchart.list_waveform_channels(file_path)
    if fmt == 'cfwb':
        return _cfwb.list_waveform_channels(file_path)
    if fmt == 'generic_tsv':
        return _generic_tsv.list_waveform_channels(file_path)
    return _spike2.list_waveform_channels(file_path)


def list_event_channels(file_path: str) -> list:
    """
    Return the names of event / marker / epoch channels.

    Currently meaningful for native Spike2 SMR files where the pipeline
    needs to know which event channel carries stim times.  Returns an
    empty list for all other formats (stim detection is handled internally).
    """
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'spike2_smr':
        return _spike2_smr.list_event_channels(file_path)
    return []


# ── Amplitude unit normalisation ─────────────────────────────────────────────
#
# Readers return each channel in the file's *native* unit (BrainVision at
# 0.1 µV resolution returns µV; Spike-2 and LabChart typically return mV).
# The analysis pipeline, however, treats millivolts as the canonical unit:
# LAT_COLS / SUM_HDR hardcode column names such as "PTP(mV)", "AUC(mV·s)" and
# "cSP_MEP_Ratio(ms/mV)".  Without a conversion step a µV recording is written
# into a column labelled mV — a silent 1000x error that leaves ratios and
# Z-scores correct while every absolute amplitude is wrong.
#
# _to_mV() is the single conversion point.  It scales only when the reader's
# unit string is unambiguously recognised, and passes the waveform through
# untouched (preserving the original unit string) when the unit is unknown or
# None, so behaviour is unchanged for readers that do not report a unit.

_MV_SCALE = {
    'v':          1e3,   'volt':       1e3,   'volts':      1e3,
    'mv':         1.0,   'millivolt':  1.0,   'millivolts': 1.0,
    'uv':         1e-3,  'microvolt':  1e-3,  'microvolts': 1e-3,
    '\u00b5v':    1e-3,  # MICRO SIGN + V
    '\u03bcv':    1e-3,  # GREEK SMALL LETTER MU + V
    'nv':         1e-6,  'nanovolt':   1e-6,  'nanovolts':  1e-6,
}

# Records the most recent conversion as (native_unit, scale_factor) so callers
# (e.g. the GUI log pane) can report what was applied.  None when no scaling
# was needed or the unit was unrecognised.
LAST_UNIT_CONVERSION = None


def _to_mV(emg, unit):
    """
    Scale a waveform into millivolts based on its reader-reported unit.

    Returns
    -------
    (emg, unit) : the waveform in mV and the canonical unit string 'mV' when
                  the unit was recognised; otherwise the inputs unchanged.
    """
    global LAST_UNIT_CONVERSION
    LAST_UNIT_CONVERSION = None

    if unit is None:
        return emg, unit

    # Tolerate decoration seen in the wild: '*mV*', ' (µV) ', 'uV.'
    key = str(unit).strip().strip('*').strip().strip('()[]').strip().rstrip('.')
    scale = _MV_SCALE.get(key.lower())
    if scale is None:
        return emg, unit          # unrecognised — never guess, never scale
    if scale == 1.0:
        return emg, 'mV'          # already mV; canonicalise the label only

    import numpy as _np
    LAST_UNIT_CONVERSION = (str(unit).strip(), scale)
    return _np.asarray(emg, dtype=float) * scale, 'mV'


def extract_emg_waveform_and_fs(file_path: str, channel_idx: int = 0):
    """
    Load EMG waveform, sampling rate, and voltage unit for the given channel.

    The waveform is normalised to millivolts — the canonical unit assumed by
    the analysis pipeline and its hardcoded "(mV)" column headers — whenever
    the underlying reader reports a recognised unit.  Readers that report no
    unit are passed through unchanged.

    Parameters
    ----------
    file_path   : path to the data file
    channel_idx : 0-based channel index

    Returns
    -------
    emg  : np.ndarray  EMG samples, in mV where the unit was recognised
    fs   : int         sampling frequency in Hz
    unit : str | None  'mV' where normalised; the reader's own unit otherwise
    """
    emg, fs, unit = _extract_emg_native(file_path, channel_idx)
    emg, unit = _to_mV(emg, unit)
    return emg, fs, unit


def _extract_emg_native(file_path: str, channel_idx: int = 0):
    """Dispatch to the format reader; returns the file's native unit."""
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'spike2_smr':
        return _spike2_smr.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'acqknowledge_acq':
        return _acqknowledge_acq.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'acqknowledge_mat':
        return _acqknowledge_mat.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'brainvision':
        return _brainvision.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'edf':
        return _edf.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'mne':
        return _mne_bridge.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'brainsight':
        return _brainsight.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'labchart_mat':
        return _labchart_mat.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'labchart':
        return _labchart.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'cfwb':
        return _cfwb.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'generic_tsv':
        return _generic_tsv.extract_emg_waveform_and_fs(file_path, channel_idx)
    return _spike2.extract_emg_waveform_and_fs(file_path, channel_idx)


def extract_stim_times(file_path: str, marker_name: str, stim_channel: str = None) -> dict:
    """
    Return stimulation timestamps.

    For Spike-2 SMR : marker_name selects the event/epoch channel by name.
    For Spike-2 text: marker_name selects the DigMark channel
                      (e.g. 'Keyboard', 'TTL').
    For LabChart    : marker_name is used as the stim-type label
                      (single uppercase letter, e.g. 'A').
    For CFWB        : stim channel is auto-detected by title keyword.
    For Generic TSV : stim channel is set in the sidecar config.

    Returns
    -------
    dict mapping stim_type -> list[float]  (timestamps in seconds)
    """
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'spike2_smr':
        return _spike2_smr.extract_stim_times(file_path, marker_name, stim_channel=stim_channel)
    if fmt == 'acqknowledge_acq':
        return _acqknowledge_acq.extract_stim_times(file_path, marker_name)
    if fmt == 'acqknowledge_mat':
        return _acqknowledge_mat.extract_stim_times(file_path, marker_name)
    if fmt == 'brainvision':
        return _brainvision.extract_stim_times(file_path, marker_name)
    if fmt == 'edf':
        return _edf.extract_stim_times(file_path, marker_name)
    if fmt == 'mne':
        return _mne_bridge.extract_stim_times(file_path, marker_name)
    if fmt == 'brainsight':
        return _brainsight.extract_stim_times(file_path, marker_name)
    if fmt == 'labchart_mat':
        return _labchart_mat.extract_stim_times(file_path, marker_name)
    if fmt == 'labchart':
        return _labchart.extract_stim_times(file_path, marker_name)
    if fmt == 'cfwb':
        return _cfwb.extract_stim_times(file_path, marker_name)
    if fmt == 'generic_tsv':
        return _generic_tsv.extract_stim_times(file_path, marker_name)
    return _spike2.extract_stim_times(file_path, marker_name)
