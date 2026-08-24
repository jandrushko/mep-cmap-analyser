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
  Epoched MATLAB       — pre-cut TMS-EMG trials (Chans/Markers/Conds/Events)
                         (requires a one-time unit-confirmation dialog)

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
  get_epoch_bounds(file_path)                  -> (pre_ms, post_ms) | None
  units_assumed(file_path)                     -> bool
  get_clipped_trials(file_path, channel_idx)   -> list[int]

Pre-epoched formats
-------------------
Most readers return a continuous recording.  A pre-epoched reader instead
returns its trials stitched into a pseudo-continuous trace separated by
guard-band padding, plus synthetic stim times.  Such a format reports a
non-None get_epoch_bounds(); callers must clamp their analysis windows to it
(see that function for why silence here is dangerous).

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
from .formats import kinemg_csv  as _kinemg_csv
from .formats import labchart    as _labchart
from .formats import labchart_mat as _labchart_mat
from .formats import brainsight  as _brainsight
from .formats import acqknowledge_mat as _acqknowledge_mat
from .formats import acqknowledge_acq as _acqknowledge_acq
from .formats import epoched_mat  as _epoched_mat
from .formats import signal_mat   as _signal_mat
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

# Formats recognised by name but not readable, so the reason can be given
# rather than left as a parse failure in an unrelated reader.
UNREADABLE_FORMATS = {
    ".adicht": ("ADInstruments LabChart native format. Export the recording "
                "from LabChart as text (File → Export) or as an ADInstruments "
                "binary (.adibin), both of which this tool reads."),
    ".adidat": ("ADInstruments LabChart data file. Export as text or .adibin."),
    ".cfs":    ("CED Signal CFS format. In Signal, use File \u2192 Export As "
                "\u2192 MATLAB, which this tool reads directly, or export as "
                "Spike2 (.smr) or as text."),
    ".s2r":    ("Spike2 resource file, not a recording. Open the .smr instead."),
}


# Extensions any reader in this package can open.
#
# Kept here rather than restated at each call site: the filter preview carried
# its own list of (".txt", ".smr", ".adibin"), which had not been updated when
# EDF, BrainVision, MATLAB, AcqKnowledge and CSV support was added. A file
# outside that stale list silently failed to load there, the sampling rate was
# never set, and the preview asked the analyst to type in a rate the file had
# already declared.
SUPPORTED_EXTENSIONS = (".txt", ".smr", ".adibin", ".edf", ".bdf",
                        ".vhdr", ".acq", ".mat", ".csv")

#: Glob patterns for the same set, for folder walks.
#:
#: DERIVED, never restated. The dataset queue kept two hand-written copies of
#: this tuple and the "no files found" message a third; all three had to be
#: edited together whenever a reader was added, and nothing failed loudly when
#: they were not. Only the .vhdr member of a BrainVision triple appears here,
#: as it is the header that resolves the other two -- globbing .eeg and .vmrk
#: would enter one recording into the queue three times.
SUPPORTED_GLOBS = tuple(f"*{e}" for e in SUPPORTED_EXTENSIONS)

#: The same set as a human-readable list, for dialogue text.
SUPPORTED_EXT_TEXT = ", ".join(SUPPORTED_EXTENSIONS)


def supported_filetypes(include_all: bool = True) -> list:
    """``filetypes=`` for a tkinter open dialogue, derived from the readers."""
    combined = " ".join(SUPPORTED_GLOBS)
    out = [("All supported formats", combined)]
    if include_all:
        out.append(("All files", "*.*"))
    return out


def unsupported_reason(file_path: str):
    """Why this file cannot be queued, or None if it can.

    Name-based only: no header is read, so this is cheap enough to run over
    every file in a folder walk. A file that passes here may still fail to
    open -- ``detect_format`` is what settles that -- so the queue treats this
    as a filter, not a guarantee.

    Returning a REASON rather than a boolean is the point. Silently dropping a
    file is the worse failure: the analyst gets a study missing a participant
    and no indication why, which is far harder to notice than a file that
    fails loudly when it is analysed.
    """
    ext = _os.path.splitext(file_path)[1].lower()
    if ext in UNREADABLE_FORMATS:
        return UNREADABLE_FORMATS[ext]
    if ext not in SUPPORTED_EXTENSIONS:
        return (f"{ext or 'no extension'} is not a format any reader in this "
                f"tool opens. Supported: {SUPPORTED_EXT_TEXT}.")
    return None


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

    # CED Signal MATLAB export (.mat) — checked FIRST among the .mat readers
    # because it is the only one written as MATLAB v7.3. The others go through
    # scipy.io.loadmat, which refuses v7.3 outright, so a Signal export used to
    # fall past all three and report unsupported_binary. The test is eight
    # magic bytes before h5py is imported at all, so it costs nothing for the
    # .mat files that are not Signal exports.
    if ext == '.mat' and _signal_mat.is_signal_mat(file_path):
        return 'signal_mat'

    # LabChart MATLAB export (.mat) — verify signature vars without loading data
    if ext == '.mat' and _labchart_mat.is_labchart_mat(file_path):
        return 'labchart_mat'
    if ext == '.mat' and _acqknowledge_mat.is_acqknowledge_mat(file_path):
        return 'acqknowledge_mat'
    # Pre-epoched TMS-EMG MATLAB export (.mat) — trials already cut around the
    # stimulus.  Checked after the two continuous .mat readers; the three
    # signatures are disjoint, so order is for consistency, not correctness.
    if ext == '.mat' and _epoched_mat.is_epoched_mat(file_path):
        return 'epoched_mat'

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

    # A binary file that matched no magic above is not a text export, and
    # calling it one produces a parse error somewhere downstream that names the
    # wrong format. Say what it is instead.
    #
    # ADInstruments' native .adicht falls here: the tool reads their CFWB
    # binary (.adibin) and their text export, but not the native format, which
    # needs a vendor library. Someone handed a .adicht previously saw a Spike2
    # error and no indication that the format was simply not supported.
    try:
        with open(file_path, 'rb') as _fh:
            _head = _fh.read(4096)
        if b'\x00' in _head:
            return 'unsupported_binary'
    except Exception:
        pass

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

    # KinEMG CSV: named in its first line. Checked here because the reader
    # existed with the full three-function contract and detect_format never
    # returned it -- so the format was supported by every measure except being
    # reachable, and a KinEMG export fell through to the generic TSV path or to
    # the Spike2 fallback depending on its punctuation.
    if first_line.strip().lower().startswith("author,kinemg"):
        return 'kinemg_csv'

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

    # Default fallback: Spike2 text exports vary enough that the checks
    # above cannot be made exhaustive, so anything textual reaching here is
    # given to that reader.
    #
    # Anything, however, included a README, a pyproject.toml and any other
    # text a person might drop in by mistake -- each claimed as a recording
    # and failing several steps later with a bare ValueError from inside a
    # parser. A file with no numeric data at all in its opening lines is not
    # a recording in any format, and saying so here costs nothing: a real
    # export has numbers within the first few dozen lines by definition.
    if not _has_numeric_rows(file_path):
        return 'unsupported_text'
    # ...but the catch-all is for Spike2 TEXT exports, which are .txt. A .csv
    # reaching here is a table of numbers from something else -- a force trace,
    # a log -- and calling it a Spike2 export is a claim about the file that is
    # simply false. It also made those files offer a "Scan file" button that
    # could never find anything, because there are no Marker blocks to find.
    #
    # Extension rather than content, deliberately: the comment above is right
    # that the positive checks cannot be made exhaustive, so narrowing them
    # would risk rejecting real exports. Narrowing by suffix rejects only
    # files that were never Spike2 exports to begin with.
    if _os.path.splitext(file_path)[1].lower() == '.csv':
        return 'generic_tsv'
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
    if fmt == 'epoched_mat':
        return not _epoched_mat.has_config(file_path)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API — dispatches to the correct format reader
# ─────────────────────────────────────────────────────────────────────────────

def _has_numeric_rows(file_path: str, scan_lines: int = 80) -> bool:
    """True if any of the opening lines looks like a row of numbers.

    Deliberately generous: two numeric fields on one line anywhere in the
    first eighty is enough. The question is "could this be a recording at
    all", not "which format is it", and a false yes merely restores the
    previous behaviour rather than introducing a new failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for _i, line in enumerate(fh):
                if _i >= scan_lines:
                    break
                for sep in ("\t", ",", ";", " "):
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if len(parts) < 2:
                        continue
                    numeric = 0
                    for p in parts[:8]:
                        try:
                            float(p)
                            numeric += 1
                        except ValueError:
                            pass
                    if numeric >= 2:
                        return True
    except Exception:
        return True   # unreadable here is not the same as not a recording
    return False


def _unreadable_reason(file_path: str) -> str:
    """Why this file cannot be read, in words the analyst can act on."""
    ext = _os.path.splitext(file_path)[1].lower()
    if ext == ".mat" and _signal_mat.looks_like_mat73(file_path) \
            and not _signal_mat.h5py_available():
        # A missing package is not an unreadable file, and only one of those
        # is something the analyst can do anything about. is_signal_mat has to
        # swallow the ImportError to stay total, so the distinction is drawn
        # here instead.
        return ("This is a MATLAB v7.3 file, which is HDF5 rather than the "
                "older MATLAB format, and reading it needs the h5py package. "
                "Install it with:  pip install h5py")
    if ext == ".mat" and _is_signal_cfs_mat(file_path):
        # Signal writes two different MATLAB files. The frame-based export is
        # read by signal_mat; this one is a dump of the CFS container and has
        # no waveform matrix to read.
        return ("This is a CED Signal CFS-export MATLAB file, which holds the "
                "CFS container rather than the sweeps. In Signal, use "
                "File \u2192 Export As \u2192 MATLAB instead, which writes the "
                "frame data this tool reads.")
    if detect_format(file_path) == 'unsupported_text':
        return ("This is a text file with no data rows in it. If it should be "
                "a recording, check it exported completely.")
    return UNREADABLE_FORMATS.get(
        ext,
        "This file is not in a format the tool can read. The readable formats "
        "are listed in File \u2192 Open.")


def _is_signal_cfs_mat(file_path: str) -> bool:
    """True for Signal's CFS-container MATLAB dump (a 'CfsFile' variable)."""
    try:
        import scipy.io as _sio
        keys = _sio.whosmat(file_path)
        return any(str(name) == "CfsFile" for name, _shape, _cls in keys)
    except Exception:
        return False


def list_waveform_channels(file_path: str) -> list:
    """Return channel names for display in the channel selector."""
    file_path = _resolve_path(file_path)
    fmt = detect_format(file_path)
    if fmt == 'unsupported_text':
        raise ValueError(
            f"{_os.path.basename(file_path)}: this is a text file with no "
            f"data rows in it, so it is not a recording this tool can "
            f"read. The readable formats are listed in File \u2192 Open.")
    if fmt == 'unsupported_binary':
        # Raised here rather than left to a reader further down. This function
        # is what the file queue calls to warm a file up, before
        # _browse_file_path and its unsupported-format message are reached, so
        # without this the analyst saw whatever the text reader said when it
        # was handed binary -- on a build with the Rust extension, "stream did
        # not contain valid UTF-8", which names neither the file nor the fix.
        raise ValueError(
            f"{_os.path.basename(file_path)}: {_unreadable_reason(file_path)}")
    if fmt == 'spike2_smr':
        return _spike2_smr.list_waveform_channels(file_path)
    if fmt == 'acqknowledge_acq':
        return _acqknowledge_acq.list_waveform_channels(file_path)
    if fmt == 'acqknowledge_mat':
        return _acqknowledge_mat.list_waveform_channels(file_path)
    if fmt == 'kinemg_csv':
        return _kinemg_csv.list_waveform_channels(file_path)
    if fmt == 'signal_mat':
        return _signal_mat.list_waveform_channels(file_path)
    if fmt == 'epoched_mat':
        return _epoched_mat.list_waveform_channels(file_path)
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
    # Spike-2 TEXT exports carry the same marker blocks. Answering [] here
    # left the BIDS-ify scan with nothing to offer for a .txt export, so its
    # codes came from the shared marker label instead -- one code for a
    # recording that had several, with nothing saying so.
    if fmt == 'spike2':
        return _spike2.list_event_channels(file_path)
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
    if fmt == 'kinemg_csv':
        return _kinemg_csv.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'signal_mat':
        return _signal_mat.extract_emg_waveform_and_fs(file_path, channel_idx)
    if fmt == 'epoched_mat':
        return _epoched_mat.extract_emg_waveform_and_fs(file_path, channel_idx)
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


def get_epoch_bounds(file_path: str):
    """
    Return (pre_ms, post_ms) available around the stimulus, or None.

    Continuous formats return None: the recording runs either side of every
    stimulus, so no window is structurally unavailable.  Pre-epoched formats
    return the real extent of their epochs.

    Callers MUST clamp pre_ms / post_ms / prestim_ms to a non-None result.
    Without that clamp an over-long window silently reaches past the end of
    one epoch — with a default prestim_ms of 100 ms against a 25 ms epoch
    lead-in, the "baseline" used for bootstrap onset thresholds and RMS
    outlier gating would be drawn largely from the neighbouring trial's MEP.

    Returns
    -------
    (pre_ms, post_ms) : tuple[float, float] for pre-epoched formats
    None              : for continuous formats
    """
    file_path = _resolve_path(file_path)
    _fmt = detect_format(file_path)
    if _fmt == 'spike2_smr':
        # A Spike2 session paused and restarted around each stimulus is
        # pre-epoched without saying so. Returns None for a continuous run.
        return _spike2_smr.get_epoch_bounds(file_path)
    if _fmt == 'labchart_mat':
        # LabChart MATLAB exports are block-structured too; the reader
        # concatenates them with zero-filled gaps, so the same block edge
        # limits apply.
        return _labchart_mat.get_epoch_bounds(file_path)
    if _fmt == 'labchart':
        # A LabChart block export is pre-epoched without saying so: each block
        # is a trial cut about the stimulus. Returns None for a continuous
        # export, which is the ordinary case for this format.
        return _labchart.get_epoch_bounds(file_path)
    if _fmt == 'signal_mat':
        return _signal_mat.get_epoch_bounds(file_path)
    if _fmt == 'epoched_mat':
        return _epoched_mat.get_epoch_bounds(file_path)
    return None


def units_assumed(file_path: str) -> bool:
    """
    True when the amplitude unit is an assumption rather than a file fact.

    Formats that declare their own units always return False.  Formats where
    the analyst had to supply the unit return True when the answer was left
    unknown, so that output can record whether a column labelled "(mV)" is a
    measurement or an unverified assumption.
    """
    file_path = _resolve_path(file_path)
    _fmt = detect_format(file_path)
    if _fmt == 'signal_mat':
        # The export states its unit per channel, so nothing is inferred.
        return _signal_mat.units_assumed(file_path)
    if _fmt == 'epoched_mat':
        return _epoched_mat.units_assumed(file_path)
    return False


def get_clipped_trials(file_path: str, channel_idx: int = 0) -> list:
    """
    Return indices of trials containing ADC-saturated samples, if knowable.

    Empty for formats that cannot determine it.  A saturated trial's
    peak-to-peak amplitude is an underestimate of unknown size.
    """
    file_path = _resolve_path(file_path)
    if detect_format(file_path) == 'epoched_mat':
        return _epoched_mat.get_clipped_trials(file_path, channel_idx)
    return []


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
    if fmt == 'kinemg_csv':
        return _kinemg_csv.extract_stim_times(file_path, marker_name)
    if fmt == 'signal_mat':
        return _signal_mat.extract_stim_times(file_path, marker_name)
    if fmt == 'epoched_mat':
        return _epoched_mat.extract_stim_times(file_path, marker_name)
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


def probe_fs_and_unit(file_path: str, channel_idx: int = 0):
    """Sampling rate and amplitude unit, without keeping the waveform.

    Every reader already returns both from
    ``extract_emg_waveform_and_fs``, but nothing surfaced them until the
    analysis ran. Opening a file gave no way to confirm what had been
    detected, which is indistinguishable from nothing having been detected --
    and these are the two values most worth checking first, since a wrong rate
    silently rescales every latency and a wrong unit every amplitude.

    The array is discarded immediately; only the two scalars are kept, so this
    costs the read but not the memory. Readers that cannot answer without a
    channel assignment raise, and the caller reports that rather than guessing.

    Returns
    -------
    (fs, unit) : (int or None, str or None)
    """
    wave, fs, unit = extract_emg_waveform_and_fs(file_path, channel_idx)
    del wave
    return (int(fs) if fs else None), unit


# ── Event sources ────────────────────────────────────────────────────────────

def list_event_sources(file_path: str) -> dict:
    """What this file can supply events from.

    Returns ``{"embedded": [names], "analogue": [names]}``.

    ``embedded`` are the file's own events -- comments, markers, annotations,
    event channels -- named as that format names them. ``analogue`` are the
    waveform channels, any of which can carry a trigger to threshold.

    Neither list is a promise that events exist, only that the file can be
    asked. An empty embedded list on a format that has no notion of markers is
    the normal answer, not a failure.
    """
    out = {"embedded": [], "analogue": []}
    try:
        out["analogue"] = list(list_waveform_channels(file_path) or [])
    except Exception:
        pass
    try:
        out["embedded"] = list(list_event_channels(file_path) or [])
    except Exception:
        pass
    if not out["embedded"]:
        # Formats without a channel list still name their events by the labels
        # they return; asking costs a read but is the only way to know.
        try:
            out["embedded"] = sorted(extract_stim_times(file_path, "") or {})
        except Exception:
            pass
    return out


def extract_events(file_path: str, sources, channel_names=None):
    """Stimulus times from an explicit list of sources.

    Returns ``(events, warnings)`` where ``events`` is
    ``{stim_type: [t_seconds]}``.

    ``extract_stim_times`` is unchanged and is what an embedded source calls.
    Building on it rather than replacing it means no existing path runs through
    new code: a file configured the way every file is configured today produces
    a byte-identical result, and the round-trip tests hold that.

    Threshold and interval sources are format-independent -- they need a
    waveform and a time base, which every reader already provides -- so the
    detection itself lives in mep_cmap.event_sources and is written once.
    """
    from .event_sources import (EventSource, detect_threshold_crossings,
                                generate_interval_events, merge_event_sources)

    if not sources:
        return dict(extract_stim_times(file_path, "") or {}), []

    names = list(channel_names or [])
    if not names:
        try:
            names = list(list_waveform_channels(file_path) or [])
        except Exception:
            names = []

    def _channel_index(label):
        if label in names:
            return names.index(label)
        raise ValueError(
            f"channel {label!r} is not in this file "
            f"({', '.join(names) if names else 'no channels listed'})")

    per_source = []
    for src in sources:
        if not isinstance(src, EventSource):
            src = EventSource.from_dict(src)

        if src.kind == "embedded":
            got = dict(extract_stim_times(file_path, src.channel) or {})
            if src.codes:
                got = {k: v for k, v in got.items() if k in src.codes}

        elif src.kind == "threshold":
            wave, fs, _unit = extract_emg_waveform_and_fs(
                file_path, _channel_index(src.channel))
            got = {src.label: detect_threshold_crossings(
                wave, float(fs), src.level, src.edge, src.refractory_ms)}

        else:  # interval
            # The recording length bounds the events; the first channel is read
            # only for its length, and the array is discarded immediately.
            wave, fs, _unit = extract_emg_waveform_and_fs(file_path, 0)
            duration = len(wave) / float(fs)
            del wave
            got = {src.label: generate_interval_events(
                src.start_s, src.period_s, src.count, duration)}

        per_source.append((src.describe(), got))

    return merge_event_sources(per_source)
