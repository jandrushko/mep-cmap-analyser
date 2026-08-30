"""
mep_cmap.bidsify
~~~~~~~~~~~~~~~~
BIDS-ify ingestion stage: turn native EMG recordings into a valid BIDS
``rawdata/`` tree, preserving the originals in ``sourcedata/``.

Design
------
Single-recording-with-typing: every waveform channel from a source file is
written into one EDF+/BDF recording under the ``emg`` datatype, with each
channel's role recorded in ``channels.tsv`` (EMG / MISC for force / TRIG for the
stim channel). Nothing is split out or dropped; an analysis that only wants the
force or EMG channels selects them by type. NIBS stimulation metadata is written
separately under the ``nibs`` datatype (BEP037), and stim onsets become an
``_events.tsv``.

The work is split into two phases so the UI can show a dry run before anything
touches disk:

  plan_bidsify(items, layout, schema)  -> Plan     (pure; cheap header reads only)
  execute_plan(plan, log=...)          -> [FileResult]   (does the copying/writing)

execute_plan, per file: copy native -> sourcedata/, convert -> rawdata/.../emg/,
write _emg.json / _channels.tsv / _events.tsv, write nibs/ sidecar, then re-read
the EDF/BDF and verify channel count / fs / sample count / per-channel RMS before
declaring success. Dataset-level files (dataset_description.json, participants.tsv
+ .json, *_scans.tsv) are created/updated with row de-duplication, so re-running
is safe.

Dependencies: numpy, pyedflib (>= 0.1.30 for the 'sample_frequency' header key),
plus mep_cmap.recording / bids_schema / bids. No import of pipeline.py or app.py.
"""

import os
import json
import shutil
import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .bids import StudyMetadata, TOOL_VERSION, _sanitise_bids_label
from .recording import build_recording, compare_signatures, Recording

try:
    import pyedflib
    _PYEDFLIB = True
except ImportError:
    _PYEDFLIB = False

BIDS_VERSION = "1.10.0"   # base spec the rawdata tree targets (EMG/NIBS are BEPs)


# ── Channel typing ────────────────────────────────────────────────────────────
# Name-keyword classifier. Returns (bids_type, fallback_unit). The recording's
# own reported unit wins; the fallback is only used when the reader gave None.
_TYPE_RULES = [
    (("stim", "trig", "ttl", "ttl pulse", "marker"), "TRIG", "V"),
    (("grip", "force", "dynam", "load", "torque", "newton"), "MISC", "N"),
    (("acc", "gonio", "angle", "position"), "MISC", "n/a"),
]


def classify_channel(name: str) -> tuple:
    """Map a channel name to (BIDS type, fallback unit) by keyword."""
    lc = (name or "").lower()
    for keys, ctype, unit in _TYPE_RULES:
        if any(k in lc for k in keys):
            return ctype, unit
    return "EMG", "mV"      # default: treat as EMG


# ── Layout ────────────────────────────────────────────────────────────────────

#: The folder name that marks the "sibling" study layout.
#:
#: BIDS allows two arrangements. In the CANONICAL one the raw dataset is the
#: study root and sourcedata/ and derivatives/ sit inside it. In the SIBLING
#: one -- which is what this tool builds -- rawdata/, sourcedata/ and
#: derivatives/ are three folders side by side.
#:
#: Both are valid; MIXING them is not. Putting sourcedata/ inside rawdata/
#: while derivatives/ is a sibling leaves native files inside the dataset a
#: validator walks, and inside every recursive glob of rawdata/**, which is
#: why the file queue needs a rule to skip them at all.
_SIBLING_RAWDATA_DIRNAME = "rawdata"


def default_sourcedata_root(rawdata_root: str) -> str:
    """Where native copies belong for this study's layout.

    Beside rawdata/ in the sibling layout, inside the dataset root in the
    canonical one, so sourcedata/ always ends up at the same level as
    derivatives/ rather than one level in from it.

    An EXISTING <rawdata_root>/sourcedata always wins, whatever the layout
    says. A study half-converted under the old default and half under the new
    one is worse than a study consistently in the old place: the native copy of
    a recording would sit in whichever folder happened to exist the day it was
    BIDS-ified, and nothing would say which. Moving those files is a decision
    for whoever owns the study, not a side effect of upgrading the tool.
    """
    legacy = os.path.join(rawdata_root, "sourcedata")
    if os.path.isdir(legacy):
        return legacy
    root = os.path.normpath(rawdata_root)
    if os.path.basename(root).lower() == _SIBLING_RAWDATA_DIRNAME:
        return os.path.join(os.path.dirname(root), "sourcedata")
    return legacy


@dataclass
class DatasetLayout:
    """
    Where the BIDS tree lives.

    rawdata_root    : the BIDS raw-dataset root (holds dataset_description.json,
                      participants.tsv, and the sub-XX/ tree). In this project that
                      is normally <study>/rawdata.
    sourcedata_root : where untouched native copies go. Resolved by
                      ``default_sourcedata_root`` when not given.
    dataset_name    : value for dataset_description.json "Name".
    """
    rawdata_root:    str
    sourcedata_root: Optional[str] = None
    dataset_name:    str = "MEP-CMAP dataset"

    def __post_init__(self):
        if not self.sourcedata_root:
            self.sourcedata_root = default_sourcedata_root(self.rawdata_root)


# ── Inputs / plan structures ──────────────────────────────────────────────────
@dataclass
class BidsifyItem:
    """One source file to BIDS-ify, with its resolved metadata and NIBS values."""
    source_path:       str
    metadata:          StudyMetadata
    modality:          str = "TMS"
    sidecar_values:    dict = field(default_factory=dict)
    marker_names:      Optional[list] = None
    stim_channel:      Optional[str] = None
    participant_extra: dict = field(default_factory=dict)   # extra participants.tsv cols
    task_name:         str = ""    # for _emg.json TaskName; falls back to metadata.task
    prefix_override:   Optional[str] = None   # explicit BIDS prefix (preserves source-stem tokens)
    #: Session-level stimulation parameter sets (mep_cmap.stim_params), and this
    #: file's mapping of stim code -> set name. Together they are what makes
    #: *_nibs.tsv possible: the sets are the rows, the mapping is what lets
    #: *_events.tsv reference them.
    param_sets:        list = field(default_factory=list)
    code_sets:         dict = field(default_factory=dict)
    #: Conditions read from the recording's own session (see conditions_for).
    #: Not stored by BIDS-ify: they belong to the recording, and most analysts
    #: never open this tab.
    condition_rows:    list = field(default_factory=list)


@dataclass
class PlannedFile:
    item:              BidsifyItem
    bids_prefix:       str
    rel_dir:           str          # e.g. sub-o001/ses-01/emg
    sourcedata_path:   str
    edf_path:          str
    json_path:         str
    channels_tsv_path: str
    events_tsv_path:   str
    nibs_dir:          str
    nibs_json_path:    str
    channels:          list         # [(name, type, unit_or_None)]
    #: v6.3 splits stimulation across four files. The sidecar alone cannot
    #: describe a recording with more than one protocol in it.
    nibs_tsv_path:     str = ""
    markers_tsv_path:  str = ""
    markers_json_path: str = ""
    notes:             list = field(default_factory=list)


@dataclass
class Plan:
    layout:        DatasetLayout
    files:         list = field(default_factory=list)
    container:     str = "BDF"      # 'BDF' or 'EDF'
    powerline_hz:  int = 50
    warnings:      list = field(default_factory=list)
    #: Delete each native original once its archival copy is in sourcedata/
    #: AND the converted EDF has passed read-back verification.
    #:
    #: Off by default, and deliberately not a mere efficiency: copying leaves
    #: the study recoverable if anything downstream is wrong, and a native
    #: recording is the one artefact that cannot be regenerated. Opting in is
    #: a statement that the sourcedata copy is now the archive.
    move_sources:  bool = False

    def preview_text(self) -> str:
        lines = [f"BIDS-ify plan  —  {len(self.files)} file(s)  ->  "
                 f"{self.container}+  in  {self.layout.rawdata_root}",
                 f"native copies -> {self.layout.sourcedata_root}"]
        # Stated in the preview, because the preview is the last thing shown
        # before anything is written and this is the only irreversible part
        # of the run.
        if self.move_sources:
            lines.append("originals    -> DELETED after each file verifies")
        lines.append("")
        for pf in self.files:
            lines.append(f"• {os.path.basename(pf.item.source_path)}")
            lines.append(f"    rawdata : {pf.rel_dir}/{os.path.basename(pf.edf_path)}")
            types = ", ".join(f"{n}[{t}]" for n, t, _ in pf.channels)
            lines.append(f"    channels: {types}")
            lines.append(f"    nibs    : {pf.item.modality} sidecar "
                         f"({len(pf.item.sidecar_values)} field(s))")
            for note in pf.notes:
                lines.append(f"    note    : {note}")
        if self.warnings:
            lines.append("")
            lines += [f"! {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class FileResult:
    source_path:   str
    ok:            bool
    edf_path:      str = ""
    discrepancies: list = field(default_factory=list)
    error:         str = ""
    #: Where the native original now lives, when it was moved rather than
    #: copied. The caller needs this to re-point its file queue: an entry left
    #: pointing at a deleted path is the stale-path failure, and it would look
    #: exactly like the tool having lost the recording.
    moved_to:      str = ""


# ── Planning (pure; cheap) ────────────────────────────────────────────────────
def _suffix_for_modality(modality: str) -> str:
    """Always ``nibs``. The modality is a COLUMN, not a filename.

    This returned tms/tes/tus, so a TMS study produced *_tms.tsv and *_tms.json.
    The spec uses *_nibs.tsv throughout -- every worked example, including the
    TMS ones -- and carries the modality in the `nibs_type` column of that file.
    A per-modality suffix is not an alternative spelling of that: a validator, or
    anyone else's script, looks for *_nibs.tsv and simply does not find the
    stimulation description, which defeats most of the point of writing it.

    (v6.2 did distinguish the system in the filename, through a `stimsys-<label>`
    entity and a `StimulationSystem` field. v6.3 removed both in favour of the
    column, so the per-modality suffix is not that either.)

    Kept as a function rather than inlined so the one place this is decided
    stays findable, and so the reasoning above travels with it.
    """
    return "nibs"


def plan_bidsify(items: list,
                 layout: DatasetLayout,
                 container: str = "BDF",
                 powerline_hz: int = 50,
                 move_sources: bool = False,
                 io_module: Any = None) -> Plan:
    """
    Build a Plan without touching disk. Reads only channel *names* (cheap header
    read via io.list_waveform_channels) to classify channel types and show a
    preview; the heavy waveform/event read happens in execute_plan.
    """
    if io_module is None:
        from . import io as io_module

    if container not in ("EDF", "BDF"):
        raise ValueError("container must be 'EDF' or 'BDF'")

    plan = Plan(layout=layout, container=container, powerline_hz=powerline_hz,
                move_sources=move_sources)
    ext = ".edf" if container == "EDF" else ".bdf"

    seen_prefixes = {}
    for item in items:
        meta = item.metadata
        prefix = item.prefix_override or meta.bids_prefix()

        # Guard against two source files resolving to the same BIDS name.
        if prefix in seen_prefixes:
            seen_prefixes[prefix] += 1
            run = seen_prefixes[prefix]
            prefix = f"{prefix}_run-{run:02d}"
            plan.warnings.append(
                f"{os.path.basename(item.source_path)}: name collided with an "
                f"earlier file; disambiguated as run-{run:02d}.")
        else:
            seen_prefixes[prefix] = 1

        sub_ses = meta.sub_ses_path().replace(os.sep, "/")
        rel_dir = f"{sub_ses}/emg"
        emg_dir = os.path.join(layout.rawdata_root, *sub_ses.split("/"), "emg")
        nibs_dir = os.path.join(layout.rawdata_root, *sub_ses.split("/"), "nibs")

        # cheap header read for channel names → types
        chans = []
        try:
            names = list(io_module.list_waveform_channels(item.source_path))
            for n in names:
                ctype, _unit = classify_channel(n)
                chans.append((n, ctype, None))
        except Exception as exc:
            plan.warnings.append(
                f"{os.path.basename(item.source_path)}: could not list channels "
                f"({exc}); will retry at execute.")

        nibs_suffix = _suffix_for_modality(item.modality)
        pf = PlannedFile(
            item=item,
            bids_prefix=prefix,
            rel_dir=rel_dir,
            sourcedata_path=os.path.join(layout.sourcedata_root, *sub_ses.split("/"),
                                         os.path.basename(item.source_path)),
            edf_path=os.path.join(emg_dir, f"{prefix}_emg{ext}"),
            json_path=os.path.join(emg_dir, f"{prefix}_emg.json"),
            channels_tsv_path=os.path.join(emg_dir, f"{prefix}_channels.tsv"),
            events_tsv_path=os.path.join(emg_dir, f"{prefix}_events.tsv"),
            nibs_dir=nibs_dir,
            nibs_json_path=os.path.join(nibs_dir, f"{prefix}_{nibs_suffix}.json"),
            nibs_tsv_path=os.path.join(nibs_dir, f"{prefix}_{nibs_suffix}.tsv"),
            markers_tsv_path=os.path.join(nibs_dir, f"{prefix}_markers.tsv"),
            markers_json_path=os.path.join(nibs_dir, f"{prefix}_markers.json"),
            channels=chans,
        )
        if not item.marker_names:
            pf.notes.append("no stim marker label set — _events.tsv will be empty")
        plan.files.append(pf)

    return plan


# ── EDF/BDF conversion ────────────────────────────────────────────────────────
def _fit_edf_phys(value: float, round_up: bool) -> float:
    """
    Coerce a physical min/max to fit EDF's 8-character header field, rounding
    OUTWARD (min down, max up) so the true signal can never fall outside the
    stored range (which would clip on read-back). Avoids pyedflib's lossy
    auto-truncation.
    """
    import math
    if value == 0:
        return 0.0
    neg = value < 0
    int_digits = len(str(int(math.floor(abs(value)))))
    avail = 8 - (1 if neg else 0) - int_digits     # chars left for '.' + decimals
    if avail <= 1:                                 # no room for decimals
        return float(math.ceil(value) if round_up else math.floor(value))
    factor = 10 ** (avail - 1)
    v = (math.ceil(value * factor) if round_up else math.floor(value * factor)) / factor
    return v


def _phys_range(arr: np.ndarray) -> tuple:
    """
    Physical min/max bracketing the signal, fitted to EDF's 8-char field and
    rounded outward, with a guard for flat channels.
    """
    if arr.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if lo == hi:                       # flat signal — give EDF a non-zero span
        lo, hi = lo - 1.0, hi + 1.0
    lo, hi = _fit_edf_phys(lo, round_up=False), _fit_edf_phys(hi, round_up=True)
    if lo >= hi:                       # rounding collapsed the range — widen
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def _digital_range(container: str) -> tuple:
    if container == "BDF":
        return -8388608, 8388607      # 24-bit
    return -32768, 32767              # 16-bit EDF


def write_recording(path: str,
                    rec: Recording,
                    channel_types: list,
                    channel_units: list,
                    container: str = "BDF",
                    prefilter: str = "") -> None:
    """
    Write a Recording to an EDF+/BDF file. channel_types/channel_units are
    per-channel, aligned to rec.channels.
    """
    if not _PYEDFLIB:
        raise RuntimeError("pyedflib is required to write EDF/BDF "
                           "(pip install pyedflib).")

    n = rec.n_channels
    ftype = pyedflib.FILETYPE_BDFPLUS if container == "BDF" else pyedflib.FILETYPE_EDFPLUS
    dmin, dmax = _digital_range(container)

    data = rec.data_matrix(on_length_mismatch="truncate")
    writer = pyedflib.EdfWriter(path, n, file_type=ftype)
    try:
        headers = []
        for i, ch in enumerate(rec.channels):
            pmin, pmax = _phys_range(data[i])
            unit = channel_units[i] or "n/a"
            headers.append({
                "label":            _sanitise_bids_label(ch.name)[:16],
                "dimension":        str(unit)[:8],
                "sample_frequency": float(rec.sampling_frequency),
                "physical_max":     pmax,
                "physical_min":     pmin,
                "digital_max":      dmax,
                "digital_min":      dmin,
                "transducer":       channel_types[i],
                "prefilter":        prefilter,
            })
        writer.setSignalHeaders(headers)
        writer.writeSamples([np.ascontiguousarray(data[i]) for i in range(n)])

        # stim events → EDF+ annotations (also written to _events.tsv separately)
        for ev in rec.events_table():
            writer.writeAnnotation(ev["onset"], ev["duration"],
                                   str(ev["trial_type"]))
    finally:
        writer.close()


def read_back_signature(path: str) -> dict:
    """Re-read a written EDF/BDF into a signature dict matching Recording.signature()."""
    if not _PYEDFLIB:
        raise RuntimeError("pyedflib is required for the read-back check.")
def read_back_signature(path: str, ref_counts=None) -> dict:
    """
    Re-read a written EDF/BDF into a signature dict matching Recording.signature().

    ``ref_counts`` (per-channel source sample counts) lets RMS be computed over
    the real, pre-padding region only — EDF/BDF zero-pads to a whole record, and
    including that tail would dilute the RMS and cause a false mismatch. The
    reported ``n_samples`` is still the full written length (for the padding-aware
    sample-count check).
    """
    if not _PYEDFLIB:
        raise RuntimeError("pyedflib is required for the read-back check.")
    r = pyedflib.EdfReader(path)
    try:
        n = r.signals_in_file
        chans = []
        fs = r.getSampleFrequency(0) if n else 0.0
        spr = 0
        if n:
            try:
                spr = int(r.samples_in_datarecord(0))   # EDF record size (pad ceiling)
            except Exception:
                spr = int(round(fs))                     # fallback: 1-second record
        for i in range(n):
            x = np.asarray(r.readSignal(i), dtype=np.float64)
            n_full = int(x.shape[0])
            if ref_counts and i < len(ref_counts):       # RMS over real data only
                x_rms = x[:min(int(ref_counts[i]), n_full)]
            else:
                x_rms = x
            rms = float(np.sqrt(np.mean(np.square(x_rms)))) if x_rms.size else 0.0
            # Quantisation step of THIS channel, so the comparison can predict
            # how much the RMS should legitimately move rather than guessing a
            # relative tolerance. It matters most exactly where a fixed
            # tolerance fails: a channel carrying a large stimulus artefact has
            # a wide physical range, so its step is coarse relative to the
            # small EMG the RMS is actually measuring.
            try:
                pmin = float(r.getPhysicalMinimum(i))
                pmax = float(r.getPhysicalMaximum(i))
                dmin = float(r.getDigitalMinimum(i))
                dmax = float(r.getDigitalMaximum(i))
                lsb = abs(pmax - pmin) / max(1.0, abs(dmax - dmin))
            except Exception:               # noqa: BLE001 — older pyedflib
                lsb = 0.0
            chans.append({"name": r.getLabel(i), "n_samples": n_full,
                          "rms": rms, "lsb": lsb})
        return {"n_channels": n, "sampling_frequency": float(fs),
                "samples_per_record": spr, "channels": chans}
    finally:
        r.close()


# ── Multi-file source formats ─────────────────────────────────────────────────
#
# BrainVision splits one recording across a .vhdr header, a .vmrk marker file
# and a binary data file (.eeg, or .dat/.seg for some exports).  Copying only
# the file the user selected leaves an orphan in sourcedata/ that no reader can
# open.  sourcedata_path preserves the original basename, so the header's
# DataFile= / MarkerFile= pointers stay valid and need no rewriting — the
# siblings simply have to travel with it.

_BV_EXTS = ('.vhdr', '.vmrk', '.eeg', '.dat', '.seg')


def _brainvision_members(src: str) -> list:
    """Return every file belonging to a BrainVision recording, incl. `src`."""
    base, ext = os.path.splitext(src)
    vhdr = src if ext.lower() == '.vhdr' else base + '.vhdr'
    members = {src}
    if os.path.isfile(vhdr):
        members.add(vhdr)
        # Prefer the header's own DataFile=/MarkerFile= entries
        try:
            with open(vhdr, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    s = line.strip()
                    if s.startswith(';'):
                        continue
                    for key in ('DataFile=', 'MarkerFile='):
                        if s.startswith(key):
                            cand = os.path.join(os.path.dirname(vhdr),
                                                s[len(key):].strip())
                            if os.path.isfile(cand):
                                members.add(cand)
        except Exception:
            pass
    # Basename fallback for anything the header did not name
    for e in _BV_EXTS:
        cand = os.path.splitext(vhdr)[0] + e
        if os.path.isfile(cand):
            members.add(cand)
    return sorted(members)


def _copy_source_siblings(src: str, dst: str) -> list:
    """
    Copy any companion files a multi-file format needs alongside `src`.

    `src` has already been copied to `dst`; siblings are placed in the same
    directory under their own basenames.  Returns the sibling paths copied.
    """
    ext = os.path.splitext(src)[1].lower()
    if ext not in _BV_EXTS:
        return []

    dst_dir = os.path.dirname(dst)
    copied = []
    for member in _brainvision_members(src):
        if os.path.abspath(member) == os.path.abspath(src):
            continue                       # already copied as the primary
        target = os.path.join(dst_dir, os.path.basename(member))
        if os.path.abspath(member) == os.path.abspath(target):
            continue                       # source == destination; nothing to do
        try:
            shutil.copy2(member, target)
            copied.append(target)
        except Exception:
            pass                           # a missing sibling must not abort BIDS-ify
    return copied


# ── Sidecar / TSV writers ─────────────────────────────────────────────────────
def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _write_tsv(path: str, header: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(_tsv_cell(row.get(c)) for c in header) + "\n")


def _tsv_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_emg_sidecar(pf: PlannedFile, rec: Recording, powerline_hz: int) -> None:
    n_emg = sum(1 for _, t, _ in pf.channels if t == "EMG")
    task = pf.item.task_name or pf.item.metadata.task or "n/a"
    sidecar = {
        "TaskName":          task,
        "SamplingFrequency": float(rec.sampling_frequency),
        "RecordingDuration": round(rec.duration_s, 6),
        "RecordingSampleCount": rec.n_samples,   # true source length (pre-EDF-padding)
        "RecordingType":     "continuous",
        "EMGChannelCount":   n_emg,
        "PowerLineFrequency": powerline_hz,
        "SoftwareFilters":   "n/a",
        "EMGReference":      "n/a",
        "Manufacturer":      pf.item.sidecar_values.get("Manufacturer", "n/a"),
        "SourceFile":        os.path.basename(pf.item.source_path),
        "GeneratedBy":       [{"Name": "MEP-CMAP Analyser", "Version": TOOL_VERSION}],
    }
    _write_json(pf.json_path, sidecar)


def write_channels_tsv(pf: PlannedFile, rec: Recording,
                       channel_types: list, channel_units: list) -> None:
    rows = []
    for i, ch in enumerate(rec.channels):
        rows.append({
            "name":  ch.name,
            "type":  channel_types[i],
            "units": channel_units[i] or "n/a",
            "sampling_frequency": float(rec.sampling_frequency),
        })
    _write_tsv(pf.channels_tsv_path,
               ["name", "type", "units", "sampling_frequency"], rows)


def find_session_for(source_path: str, metadata=None,
                     derivatives_root: str = "") -> str:
    """The session JSON belonging to one recording, or "".

    Two routes, because the obvious one is not always enough.

    The direct construction via session_path_for is exact and cheap, and is
    tried first. It relies on the participant being derivable from the file's
    name, which is true of a BIDS-organised study and false of, say,
    `rawdata/Spike/Example Data 1.smr`. There the name yields nothing, the
    lookup resolves to `sub-unknown`, and a session saved under the metadata
    the analyst typed is missed entirely -- the conditions exist and are simply
    not found.

    So the fallback matches on what the session itself records. Every session
    stores the `file_path` it was written for, and its filename ends with the
    source stem, so candidates are found by name and then CONFIRMED by reading
    that field. Matching on the stem alone would be a guess; confirming makes
    it a fact.

    Where several sessions claim the same recording -- which happens when the
    participant id was corrected, leaving the old one orphaned -- the most
    recently written wins, since that is the one the analyst last worked in.
    """
    from .bids import session_path_for

    direct = session_path_for(source_path, metadata, derivatives_root)
    if direct and os.path.isfile(direct):
        return direct
    if not source_path:
        return ""

    root = derivatives_root or os.path.dirname(source_path)
    if os.path.basename(os.path.normpath(root)).lower() != "derivatives":
        root = os.path.join(root, "derivatives")
    if not os.path.isdir(root):
        return ""

    stem = os.path.splitext(os.path.basename(source_path))[0]
    want = os.path.normcase(os.path.basename(source_path))
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith("_session.json"):
                continue
            if not name[:-len("_session.json")].endswith(stem):
                continue
            cand = os.path.join(dirpath, name)
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    stored = json.load(fh).get("file_path") or ""
            except Exception:           # noqa: BLE001 — unreadable, skip
                continue
            if os.path.normcase(os.path.basename(stored)) != want:
                continue
            try:
                hits.append((os.path.getmtime(cand), cand))
            except OSError:
                hits.append((0.0, cand))
    if not hits:
        return ""
    hits.sort()
    return hits[-1][1]


def recorded_metadata_for(source_path: str, metadata=None,
                          derivatives_root: str = ""):
    """The StudyMetadata the analyst entered for this recording, or None.

    Taken from the recording's own session, which stores `study_metadata`
    verbatim. That entry is the only place a participant is recorded: nothing
    else in the tool holds one, so a recording whose FILENAME says nothing --
    `rawdata/Spike/Example Data 1.smr` -- has no other source.

    Without this, BIDS-ify parsed the filename, found no `sub-`, and wrote the
    conversion to `sub-unknown` while the analyst had plainly typed sub-333 in
    the metadata window. The output disagreed with the tool's own record of
    who the recording belonged to.

    Returns None when there is no session or no metadata in it, leaving the
    caller's filename-derived guess in place.
    """
    from .bids import StudyMetadata

    try:
        path = find_session_for(source_path, metadata, derivatives_root)
        if not path:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            sm = json.load(fh).get("study_metadata") or {}
        if not sm.get("participant_id"):
            return None
        fields = getattr(StudyMetadata, "__dataclass_fields__", {})
        return StudyMetadata(**{k: v for k, v in sm.items() if k in fields})
    except Exception:                   # noqa: BLE001 — fall back to the guess
        return None


def conditions_for(source_path: str, metadata=None,
                   derivatives_root: str = "") -> list:
    """The conditions assigned to a recording, or [] if none were.

    Read from the recording's session JSON, which is where the Conditions tab
    already stores them. BIDS-ify owns no copy: conditions belong to the
    recording rather than to a conversion, and most analysts never open this
    tab at all -- their work must not end up filed under a feature they do not
    use.

    Reads `condition_rows`, the table the analyst edited, NOT
    `condition_event_rows`, which is that table already projected into events.
    Both are stored, and projecting the model here rather than trusting the
    stored projection is what keeps the Conditions tab and BIDS-ify from
    describing the same recording differently.

    Never raises. A missing, unreadable or half-written session means no
    conditions, and no conditions is a valid state -- it writes the stimulus
    codes exactly as recorded.
    """
    from .conditions import ConditionRow

    try:
        path = find_session_for(source_path, metadata, derivatives_root)
        if not path or not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            sess = json.load(fh)
        out = []
        for r in (sess.get("condition_rows") or []):
            if not isinstance(r, dict):
                continue
            out.append(ConditionRow(
                stim_type=r.get("stim_type", ""),
                condition=r.get("condition", ""),
                trials=tuple(r.get("trials") or ()),
                excluded=bool(r.get("excluded")),
                pre_ms=r.get("pre_ms"),
                post_ms=r.get("post_ms")))
        return out
    except Exception:                   # noqa: BLE001 — see docstring
        return []


def write_events_tsv(pf: PlannedFile, rec: Recording) -> None:
    """The timeline: one row per delivery.

    Projected by events_model, which the Conditions tab uses too. Each built
    its own answer before, so a recording could carry two _events.tsv files
    that disagreed with nothing to say which was right.

    Conditions come from the recording's session rather than from BIDS-ify's
    own state, so a file grouped in the Conditions tab carries that grouping
    here, and a file that was never grouped writes its stimulus codes exactly
    as recorded.
    """
    from . import events_model as _em

    # trial_type: prefer something meaningful over the cosmetic marker label.
    fallback_type = (pf.item.metadata.measure or pf.item.metadata.acq
                     or "stim")

    raw = []
    for ev in rec.events_table():
        code = ev["trial_type"]
        # The fallback applies only when the code is genuinely empty. Single-
        # character DigMark codes (A/B/C/D) are valid labels - do NOT collapse.
        if code in (None, "", "n/a"):
            code = fallback_type
        raw.append({"onset": ev["onset"], "code": code,
                    "duration": ev.get("duration", 0)})

    columns, rows = _em.project(
        raw,
        condition_rows=pf.item.condition_rows,
        code_sets=pf.item.code_sets,
        param_sets=pf.item.param_sets)
    _write_tsv(pf.events_tsv_path, columns, rows)

    # Provenance beside it: a reader needs to know whether the grouping came
    # from the recording or from somebody's judgement about it.
    _write_json(os.path.splitext(pf.events_tsv_path)[0] + ".json", {
        "trial_type": {"Description": _em.describe_source(pf.item.condition_rows)},
    })


def write_nibs_sidecar(pf: PlannedFile, schema, log_callback=None) -> None:
    """Write the NIBS-BIDS v6.3 stimulation description.

    Four files, not one. The old single flat *_nibs.json could hold exactly one
    value per field for the whole recording, so a file containing a peripheral
    M-wave on one code and a TMS MEP on another could describe neither. v6.3
    separates them:

        *_nibs.tsv      one row per stimulation parameter set
        *_nibs.json     device, dosing references, and the column definitions
        *_markers.tsv   one row per element placement
        *_events.tsv    one row per delivery, naming a set and a position
                        (written by write_events_tsv, in the emg/ folder)

    Everything about WHERE a value goes is read from the schema's `block` and
    `emits`, so adding a field is a schema edit rather than a code change here.
    """
    from . import stim_params as _sp

    sets = _sp.sets_in_use(pf.item.code_sets, pf.item.param_sets)

    # No parameter sets: keep writing the flat sidecar so a study that has not
    # adopted them still converts. It is the old shape, and honestly so --
    # inventing an empty *_nibs.tsv would claim a description that is absent.
    if not sets:
        values = dict(pf.item.sidecar_values)
        values.setdefault("StimulationModality", pf.item.modality)
        sidecar = schema.ordered_sidecar(values, modality=pf.item.modality)
        sidecar["SourceFile"] = os.path.basename(pf.item.source_path)
        _write_json(pf.nibs_json_path, sidecar)
        return

    _write_nibs_tsv(pf, schema, sets)
    _write_markers(pf, schema, sets)
    _write_nibs_json(pf, schema, sets)

    # Two spellings of one device is a typo, not a second stimulator, and it
    # splits the rows between two entries describing the same box. Said here
    # because it is only visible once every parameter set has been resolved.
    if log_callback:
        for _msg in _vocabulary_problems(sets, schema):
            log_callback(_msg)
        _p, _stims, _elems = _device_ids(pf, schema, sets)
        for _label, _entries, _key in (("stimulator", _stims, "StimulatorID"),
                                       ("element", _elems, "ElementID")):
            for _lower, _spellings in _near_duplicate_devices(_entries, _key).items():
                log_callback(
                    f"   \u26a0\ufe0f  Two {_label} entries differ only by case "
                    f"or spacing: {', '.join(repr(s) for s in sorted(set(_spellings)))}. "
                    f"They are written as separate devices. If they are the "
                    f"same one, make the spelling match in the parameter sets.")


def _vocabulary_problems(sets, schema):
    """Values outside a field's closed vocabulary, as messages.

    coerce_value checks TYPES, not vocabularies, so a value stored before a
    vocabulary changed is written verbatim. The spec's closed sets are the
    whole point of a closed set: `current_direction` took PA/AP/LM/ML while
    the tool offered them, and after the field was corrected to the v6.3
    winding vocabulary the old values stayed in the saved parameter sets and
    went into *_nibs.tsv unremarked, where a reader following the spec would
    misread them.

    Reported, not corrected. PA cannot be translated into a winding direction
    -- they describe different things -- so the tool cannot fix this without
    inventing a measurement. Naming the field, the value and what is allowed is
    everything the analyst needs to fix it themselves.
    """
    out = []
    for s in sets:
        for key, raw in (s.values or {}).items():
            if raw in (None, "") or str(raw).strip() == "":
                continue
            fld = schema.field(key)
            if not fld or not fld.enum:
                continue
            if str(raw) in [str(v) for v in fld.enum]:
                continue
            out.append(
                f"   \u26a0\ufe0f  '{s.name}': {key} is {raw!r}, which is not "
                f"one of {', '.join(repr(str(v)) for v in fld.enum)}. It is "
                f"written as entered; re-pick it in the parameter set to make "
                f"the file valid.")
    return out


def _resolved(pf, schema, block, s, modality=None):
    """Session defaults for one block, with this parameter set's overrides on top.

    A device is not necessarily a property of the recording. M-waves delivered
    by a Digitimer and MEPs by a Magstim in one file are two stimulators, and a
    coil swapped mid-session to reach a different target is two elements. v6.3
    expects that: stimulator_id and nibs_element_id are columns of *_nibs.tsv
    pointing into arrays, so the device is referenced per row.

    Returned keyed by `emits`, with a stable identity string so two sets sharing
    a device share its entry rather than duplicating it.
    """
    out = {}
    for fld in schema.fields_for(modality or s.nibs_type, block=block):
        raw = s.values.get(fld.key)
        if raw in (None, "") or str(raw).strip() == "":
            raw = pf.item.sidecar_values.get(fld.key)
        if raw in (None, "") or str(raw).strip() == "":
            continue
        coerced, err = schema.coerce_value(fld, raw)
        out[fld.emits or fld.key] = raw if err else coerced
    return out


def _identity(values) -> str:
    """A stable key for a block's values, so identical devices share one entry."""
    return "|".join(f"{k}={values[k]}" for k in sorted(values))


def _blocked_values(pf, schema, block, modality=None):
    """Session-level values whose schema `block` matches, keyed by `emits`."""
    out = {}
    for fld in schema.fields_for(modality or pf.item.modality, block=block):
        raw = pf.item.sidecar_values.get(fld.key)
        if raw in (None, "") or str(raw).strip() == "":
            continue
        coerced, err = schema.coerce_value(fld, raw)
        out[fld.emits or fld.key] = raw if err else coerced
    return out


def _device_ids(pf, schema, sets):
    """``({set_name: (stimulator_id, element_id)}, stimulators, elements)``.

    Computed once and used by both the table and the sidecar, so a row cannot
    reference an id the sidecar does not define. Sets sharing a device share its
    id; sets that override it get their own.
    """
    stim_by_key, elem_by_key = {}, {}
    per_set = {}
    for s in sets:
        sv = _resolved(pf, schema, "StimulatorSet", s)
        ev = _resolved(pf, schema, "ElementSet", s)
        sid = eid = ""
        if sv:
            k = _identity(sv)
            if k not in stim_by_key:
                # Named after the manufacturer where there is one, because an
                # id a human can read is worth more in a shared dataset than a
                # serial number nobody recognises.
                base = str(sv.get("Manufacturer") or "stimulator")
                name = base
                n = 2
                while any(v[0] == name for v in stim_by_key.values()):
                    name = f"{base}_{n}"
                    n += 1
                stim_by_key[k] = (name, sv)
            sid = stim_by_key[k][0]
        if ev:
            k = _identity(ev)
            if k not in elem_by_key:
                base = str(ev.get("ModelName") or ev.get("Manufacturer")
                           or "element")
                name = base
                n = 2
                while any(v[0] == name for v in elem_by_key.values()):
                    name = f"{base}_{n}"
                    n += 1
                elem_by_key[k] = (name, ev)
            eid = elem_by_key[k][0]
        per_set[s.name] = (sid, eid)

    stimulators = [dict(v, StimulatorID=nm) for nm, v in stim_by_key.values()]
    elements = [dict(v, ElementID=nm) for nm, v in elem_by_key.values()]
    return per_set, stimulators, elements


def _near_duplicate_devices(entries, id_key):
    """Ids that differ only by case or surrounding space.

    'Magstim' and 'magstim' are two devices to a dictionary and one device to
    everyone else, so a typo in one parameter set silently doubles the
    StimulatorSet and splits the rows between two entries describing the same
    box. Reported rather than merged: the tool cannot know which spelling was
    meant, and quietly picking one would rewrite what the analyst entered.
    """
    seen = {}
    for e in entries:
        raw = str(e.get(id_key, ""))
        key = raw.strip().lower()
        if not key:
            continue
        seen.setdefault(key, []).append(raw)
    return {k: v for k, v in seen.items() if len(set(v)) > 1}


def _write_nibs_tsv(pf: PlannedFile, schema, sets) -> None:
    from . import stim_params as _sp
    columns, rows = _sp.nibs_rows(sets, schema)

    # Device columns, only where they say something. A study on one stimulator
    # does not need a column repeating its name on every row; a file mixing a
    # Digitimer and a Magstim cannot be read without one.
    per_set, stimulators, elements = _device_ids(pf, schema, sets)
    if len(stimulators) > 1 or len(elements) > 1:
        at = columns.index("nibs_type") + 1
        if stimulators:
            columns.insert(at, "stimulator_id"); at += 1
        if elements:
            columns.insert(at, "nibs_element_id")
        for r in rows:
            sid, eid = per_set.get(r["nibs_event_id"], ("", ""))
            if stimulators:
                r["stimulator_id"] = sid or "n/a"
            if elements:
                r["nibs_element_id"] = eid or "n/a"
    _write_tsv(pf.nibs_tsv_path, columns, rows)


def _placement_values(pf, schema, s):
    """One placement's markers.tsv columns, per set with session fallback.

    `position_description` is free text and THREE fields emit into it: the
    target region, the montage description, and the induced current direction.
    Keyed by `emits` alone they overwrite each other and whichever the schema
    happens to list last silently wins. Free text is the one case where more
    than one contributor is not a conflict, so they are joined instead --
    "M1 hand hotspot; PA-induced current" says both things, which is what the
    analyst entered.

    Joined in schema order so the wording is stable between runs rather than
    depending on dict iteration.
    """
    out, parts = {}, []
    for fld in schema.fields_for(s.nibs_type, block="markers.tsv"):
        raw = s.values.get(fld.key)
        if raw in (None, "") or str(raw).strip() == "":
            raw = pf.item.sidecar_values.get(fld.key)
        if raw in (None, "") or str(raw).strip() == "":
            continue
        name = fld.emits or fld.key
        coerced, err = schema.coerce_value(fld, raw)
        val = raw if err else coerced
        if name == "position_description":
            parts.append(str(val).strip())
        else:
            out[name] = val
    if parts:
        out["position_description"] = "; ".join(parts)
    return out


def _write_markers(pf: PlannedFile, schema, sets) -> None:
    """One row per element placement, referenced from *_events.tsv.

    A set with no position still gets a row: nibs_position_id is what the
    timeline points at, and a delivery with nowhere to point is unreadable.
    The placement is named rather than located, which is the usual case for
    non-navigated work and exactly what the spec's own motor example does.
    """
    seen, rows = {}, []
    for s in sets:
        pos = s.position or f"{s.name}_position"
        if pos in seen:
            continue
        seen[pos] = True
        row = {"nibs_position_id": pos, "nibs_element_id": s.name}
        # Resolved PER SET, with the session values underneath, exactly as the
        # device blocks are. Reading only session values meant a placement
        # entered against one protocol -- a coil moved to a second site
        # mid-recording -- was never written.
        row.update(_placement_values(pf, schema, s))
        rows.append(row)

    columns = ["nibs_position_id", "nibs_element_id"]
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    for r in rows:
        for k in columns:
            r.setdefault(k, "n/a")
    _write_tsv(pf.markers_tsv_path, columns, rows)

    _write_json(pf.markers_json_path, {
        "nibs_position_id": {
            "LongName": "Position identifier",
            "Description": "Placement of the stimulating element. Referenced "
                           "from nibs_position_id in *_events.tsv."},
        "nibs_element_id": {
            "LongName": "Element identifier",
            "Description": "Element delivering the stimulation. Links to "
                           "ElementSet.ElementID in *_nibs.json."},
    })


def _write_nibs_json(pf: PlannedFile, schema, sets) -> None:
    """Device, dosing references, and what every column in *_nibs.tsv means.

    The spec is explicit that units are always stated in the sidecar and never
    assumed from the numbers in the table: 58 could be percent of maximum
    stimulator output or milliamps, and a numeric column without a declared
    unit is not merely undocumented, it is unreadable.
    """
    from . import stim_params as _sp

    out = {}
    desc = pf.item.sidecar_values.get("StimulationDescription")
    if desc:
        out["NIBSDescription"] = desc
    out["ConcurrentModalities"] = ["emg"]
    # The timeline lives with the recording, so the stimulation description
    # points at it rather than duplicating onsets.
    out["IntendedFor"] = (f"bids::{pf.rel_dir}/"
                          f"{os.path.basename(pf.events_tsv_path)}")

    # Devices, from the same computation the table used, so a stimulator_id in
    # a row always has an entry here to resolve against.
    _per_set, _stimulators, _elements = _device_ids(pf, schema, sets)
    if _stimulators:
        out["StimulatorSet"] = (_stimulators[0] if len(_stimulators) == 1
                                else _stimulators)
    if _elements:
        out["ElementSet"] = _elements[0] if len(_elements) == 1 else _elements

    # IntensitySet: the measured value of each dosing reference, given once.
    intensity = []
    _ref_field = {"RestingMotorThreshold": ("rMT", "resting_motor"),
                  "ActiveMotorThreshold": ("aMT", "active_motor")}
    _used = {str(s.values.get("IntensityReference") or "") for s in sets}
    for key, (ref_id, ref_type) in _ref_field.items():
        # A set may carry its own threshold: one measured for a different
        # target, or with a different coil. The session default is the
        # fallback, not the only answer.
        raw = next((s.values.get(key) for s in sets
                    if s.values.get(key) not in (None, "")), None)
        if raw in (None, ""):
            raw = pf.item.sidecar_values.get(key)
        if raw in (None, "") or ref_id not in _used:
            continue
        fld = schema.field(key)
        coerced, err = schema.coerce_value(fld, raw) if fld else (raw, None)
        entry = {"IntensityID": ref_id,
                 "Value": raw if err else coerced,
                 "Units": (fld.units if fld and fld.units else "%MSO"),
                 "Type": ref_type}
        method = pf.item.sidecar_values.get("MotorThresholdMethod")
        if method:
            entry["Algorithm"] = method
        intensity.append(entry)
    if intensity:
        out["IntensitySet"] = intensity

    nav = _blocked_values(pf, schema, "NavigationSystem")
    if nav:
        out["NavigationSystem"] = nav

    # Column definitions for *_nibs.tsv, including the units the spec requires.
    columns, _ = _sp.nibs_rows(sets, schema)
    out["nibs_event_id"] = {
        "LongName": "Stimulation parameter set identifier",
        "Description": "Identifier of a stimulation parameter set. Referenced "
                       "from nibs_event_id in *_events.tsv. Unique within this "
                       "file."}
    out["nibs_type"] = {
        "LongName": "Stimulation modality",
        "Description": "Non-invasive or peripheral stimulation modality.",
        "Levels": {t: t for t in sorted({s.nibs_type for s in sets})}}

    by_emit = {}
    for fld in schema.fields_for(pf.item.modality, block="nibs.tsv"):
        by_emit.setdefault(fld.emits or fld.key, fld)
    for col in columns:
        if col in out:
            continue
        fld = by_emit.get(col)
        if fld is None:
            continue
        entry = {"Description": fld.description or fld.key}
        if fld.units:
            entry["Units"] = fld.units
        if fld.enum:
            entry["Levels"] = {v: v for v in fld.enum}
        out[col] = entry

    # Units declared per set rather than per column when the analyst stated
    # them: a PNS row in mA and a TMS row in %MSO share one column.
    units = {str(s.values.get("StimulationIntensityUnits") or "")
             for s in sets} - {""}
    if units and "stimulus_intensity" in out:
        out["stimulus_intensity"]["Units"] = (
            units.pop() if len(units) == 1
            else "mixed — see StimulationIntensityUnits per set")

    out["NIBSSchema"]        = schema.schema_name
    out["NIBSSchemaVersion"] = schema.schema_version
    out["SourceFile"]        = os.path.basename(pf.item.source_path)
    _write_json(pf.nibs_json_path, out)


# ── Dataset-level files (idempotent) ──────────────────────────────────────────
def ensure_dataset_description(layout: DatasetLayout) -> None:
    path = os.path.join(layout.rawdata_root, "dataset_description.json")
    if os.path.isfile(path):
        return                       # never overwrite an existing description
    _write_json(path, {
        "Name": layout.dataset_name,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "MEP-CMAP Analyser", "Version": TOOL_VERSION}],
    })


def _read_tsv(path: str) -> tuple:
    if not os.path.isfile(path):
        return [], []
    with open(path, encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = [dict(zip(header, l.split("\t"))) for l in lines[1:]]
    return header, rows


#: The optional participants.tsv columns BIDS gives a defined meaning to.
#:
#: Emitted as n/a rather than omitted. A file carrying only participant_id is
#: valid BIDS and useless: nothing downstream can tell "this study did not
#: record age" from "nobody thought to add the column", and the analyst has no
#: prompt to add it. Present-but-n/a is an invitation to fill it in, and it is
#: what makes the reporting-checklist participant items answerable at all.
#:
#: Restricted to the columns BIDS DEFINES. A guessed column name is worse than
#: no column: a reader that expects `sex` gains nothing from `gender`, and the
#: study is left describing itself in private vocabulary.
PARTICIPANT_TEMPLATE_COLUMNS = ("age", "sex", "handedness")

#: What each standard column means, for participants.json. Levels are the
#: values BIDS itself lists, so a study that fills them in is machine-readable
#: rather than merely populated.
PARTICIPANT_COLUMN_DESCRIPTIONS = {
    "age": {"Description": "Age of the participant at the time of scanning",
            "Units": "years"},
    "sex": {"Description": "Sex of the participant as reported",
            "Levels": {"M": "male", "F": "female", "O": "other"}},
    "handedness": {"Description": "Handedness of the participant as reported",
                   "Levels": {"L": "left", "R": "right",
                              "A": "ambidextrous"}},
}


def upsert_participant(layout: DatasetLayout, participant_id: str,
                       extra: dict = None) -> None:
    """Add (or merge) a participants.tsv row, de-duplicated by participant_id.

    The standard optional columns are always present, as n/a when unknown, so
    the file is a form to complete rather than a bare list of identifiers.
    Existing values are never overwritten by the template.
    """
    extra = extra or {}
    path = os.path.join(layout.rawdata_root, "participants.tsv")
    header, rows = _read_tsv(path)

    cols = ["participant_id"] + [k for k in extra if k != "participant_id"]
    for c in PARTICIPANT_TEMPLATE_COLUMNS:
        if c not in cols:
            cols.append(c)
    for c in header:
        if c not in cols:
            cols.append(c)

    by_id = {r.get("participant_id"): r for r in rows}
    row = by_id.get(participant_id, {"participant_id": participant_id})
    row.update({"participant_id": participant_id, **extra})
    by_id[participant_id] = row

    ordered = sorted(by_id.values(), key=lambda r: r.get("participant_id", ""))
    _write_tsv(path, cols, ordered)

    # Written whenever there are columns to describe, not only when a caller
    # supplied extras. Gated on `extra` it never appeared at all, because the
    # BIDS-ify path calls this with the participant id alone -- so the standard
    # columns went out undocumented and any custom column stayed that way.
    json_path = os.path.join(layout.rawdata_root, "participants.json")
    if not os.path.isfile(json_path):
        described = {}
        for c in cols:
            if c == "participant_id":
                continue        # BIDS defines it; describing it adds nothing
            described[c] = dict(
                PARTICIPANT_COLUMN_DESCRIPTIONS.get(c)
                # A custom column the tool cannot describe gets a placeholder
                # naming itself, so the gap is visible in the file rather than
                # absent from it.
                or {"Description": f"TODO: describe the {c!r} column"})
        if described:
            _write_json(json_path, described)


def append_scan(layout: DatasetLayout, meta: StudyMetadata,
                scan_relpath: str, acq_time: str = "n/a") -> None:
    """Append a row to sub-XX[_ses-YY]_scans.tsv, de-duplicated by filename."""
    sub = meta.participant_id or "sub-unknown"
    ses = meta.session or ""
    sub_dir = os.path.join(layout.rawdata_root, sub)
    if ses:
        fname = f"{sub}_{ses}_scans.tsv"
    else:
        fname = f"{sub}_scans.tsv"
    path = os.path.join(sub_dir, fname)

    header, rows = _read_tsv(path)
    if not header:
        header = ["filename", "acq_time"]
    by_file = {r.get("filename"): r for r in rows}
    by_file[scan_relpath] = {"filename": scan_relpath, "acq_time": acq_time}
    _write_tsv(path, header, list(by_file.values()))


def _remove_verified_source(pf, log=print) -> str:
    """Delete the native original, but only once its copy is provably safe.

    Returns the archival path on success, "" if the original was left alone.

    Every check here answers "could this delete the only copy?". A move is the
    one step in BIDS-ify that cannot be undone by re-running it, so it refuses
    on any doubt rather than reporting a problem afterwards -- by which time
    the recording is gone.
    """
    src, dst = pf.item.source_path, pf.sourcedata_path
    try:
        if os.path.abspath(src) == os.path.abspath(dst):
            return ""                       # already the archive; nothing to move
        if not os.path.isfile(dst):
            log("  ! original kept: the sourcedata copy is not there")
            return ""
        # Size, not mere existence. A copy interrupted by a full disk or a
        # sync client leaves a short file that exists and is not the recording.
        if os.path.getsize(src) != os.path.getsize(dst):
            log(f"  ! original kept: the sourcedata copy is "
                f"{os.path.getsize(dst)} bytes against the original's "
                f"{os.path.getsize(src)}")
            return ""

        # Siblings first. A .vhdr removed while its .eeg stays behind leaves
        # a BrainVision recording that is neither in sourcedata nor readable
        # where it was.
        for member in _brainvision_members(src):
            if os.path.abspath(member) == os.path.abspath(src):
                continue
            mirror = os.path.join(os.path.dirname(dst),
                                  os.path.basename(member))
            if os.path.isfile(mirror) and os.path.isfile(member):
                os.remove(member)
                log(f"  − removed sibling: {os.path.basename(member)}")

        os.remove(src)
        log(f"  → original moved to {dst}")
        return dst
    except Exception as exc:                # noqa: BLE001
        # Never fatal. The conversion succeeded; failing the file now would
        # report a good recording as bad because a delete was refused.
        log(f"  ! original kept ({type(exc).__name__}: {exc})")
        return ""


# ── Execution ─────────────────────────────────────────────────────────────────
def execute_plan(plan: Plan,
                 schema=None,
                 log=print,
                 io_module: Any = None) -> list:
    """
    Execute a Plan. Returns a list of FileResult. Per-file failures are caught and
    recorded so one bad file never aborts the batch (and never leaves a worker
    thread blocked — the caller gets a clean list back either way).
    """
    if schema is None:
        from .bids_schema import load_schema
        schema = load_schema()
    if io_module is None:
        from . import io as io_module

    ensure_dataset_description(plan.layout)
    results = []

    for pf in plan.files:
        try:
            log(f"BIDS-ify: {os.path.basename(pf.item.source_path)}")

            # 1) copy native → sourcedata (copy, never move)
            os.makedirs(os.path.dirname(pf.sourcedata_path), exist_ok=True)
            shutil.copy2(pf.item.source_path, pf.sourcedata_path)
            for _sib in _copy_source_siblings(pf.item.source_path,
                                              pf.sourcedata_path):
                log(f"  + sibling: {os.path.basename(_sib)}")

            # 2) full read into a Recording
            rec = build_recording(pf.item.source_path,
                                  marker_names=pf.item.marker_names,
                                  stim_channel=pf.item.stim_channel,
                                  io_module=io_module)

            # resolve per-channel type + unit (reader unit wins over fallback)
            ctypes, cunits = [], []
            for ch in rec.channels:
                ctype, fallback_unit = classify_channel(ch.name)
                ctypes.append(ctype)
                cunits.append(ch.unit or fallback_unit)

            # 3) convert → EDF/BDF
            os.makedirs(os.path.dirname(pf.edf_path), exist_ok=True)
            write_recording(pf.edf_path, rec, ctypes, cunits,
                            container=plan.container)

            # 4) read-back fidelity check (tolerate EDF's whole-record zero-padding;
            #    compare RMS over the real pre-padding region only)
            ref = rec.signature()
            ref_counts = [c["n_samples"] for c in ref["channels"]]
            test = read_back_signature(pf.edf_path, ref_counts=ref_counts)
            ok, disc = compare_signatures(
                ref, test,
                sample_pad_tolerance=test.get("samples_per_record", 1))

            # 5) sidecars + TSVs
            write_emg_sidecar(pf, rec, plan.powerline_hz)
            write_channels_tsv(pf, rec, ctypes, cunits)
            write_events_tsv(pf, rec)
            write_nibs_sidecar(pf, schema, log)

            # 6) dataset-level files
            upsert_participant(plan.layout, pf.item.metadata.participant_id or
                               "sub-unknown", pf.item.participant_extra)
            scan_rel = f"{pf.rel_dir}/{os.path.basename(pf.edf_path)}"
            append_scan(plan.layout, pf.item.metadata, scan_rel)

            moved_to = ""
            if ok:
                log(f"  ✓ verified ({rec.n_channels} ch, {len(rec.events)} events)")
                # Only here. The original is deleted after the copy is in
                # place AND the EDF has been read back and matched -- a
                # mismatch means the conversion is not trustworthy, and that
                # is exactly when the native file must still exist.
                if plan.move_sources:
                    moved_to = _remove_verified_source(pf, log)
            else:
                log("  ! read-back mismatch: " + "; ".join(disc))
                if plan.move_sources:
                    log("  ! original kept: this file did not verify")
            results.append(FileResult(pf.item.source_path, ok,
                                      edf_path=pf.edf_path, discrepancies=disc,
                                      moved_to=moved_to))
        except Exception as exc:
            log(f"  ✗ failed: {type(exc).__name__}: {exc}")
            results.append(FileResult(pf.item.source_path, False, error=str(exc)))

    return results
