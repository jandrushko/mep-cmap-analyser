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
@dataclass
class DatasetLayout:
    """
    Where the BIDS tree lives.

    rawdata_root    : the BIDS raw-dataset root (holds dataset_description.json,
                      participants.tsv, and the sub-XX/ tree). In this project that
                      is normally <study>/rawdata.
    sourcedata_root : where untouched native copies go. Defaults to
                      <rawdata_root>/sourcedata (a BIDS-reserved location).
    dataset_name    : value for dataset_description.json "Name".
    """
    rawdata_root:    str
    sourcedata_root: Optional[str] = None
    dataset_name:    str = "MEP-CMAP dataset"

    def __post_init__(self):
        if not self.sourcedata_root:
            self.sourcedata_root = os.path.join(self.rawdata_root, "sourcedata")


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
    notes:             list = field(default_factory=list)


@dataclass
class Plan:
    layout:        DatasetLayout
    files:         list = field(default_factory=list)
    container:     str = "EDF"      # 'EDF' or 'BDF'
    powerline_hz:  int = 50
    warnings:      list = field(default_factory=list)

    def preview_text(self) -> str:
        lines = [f"BIDS-ify plan  —  {len(self.files)} file(s)  ->  "
                 f"{self.container}+  in  {self.layout.rawdata_root}",
                 f"native copies -> {self.layout.sourcedata_root}", ""]
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


# ── Planning (pure; cheap) ────────────────────────────────────────────────────
def _suffix_for_modality(modality: str) -> str:
    return {"TMS": "tms", "tES": "tes", "TUS": "tus"}.get(modality, "nibs")


def plan_bidsify(items: list,
                 layout: DatasetLayout,
                 container: str = "EDF",
                 powerline_hz: int = 50,
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

    plan = Plan(layout=layout, container=container, powerline_hz=powerline_hz)
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
                    container: str = "EDF",
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
            chans.append({"name": r.getLabel(i), "n_samples": n_full, "rms": rms})
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


def write_events_tsv(pf: PlannedFile, rec: Recording) -> None:
    # trial_type: prefer something meaningful over the cosmetic marker label.
    fallback_type = (pf.item.metadata.measure or pf.item.metadata.acq
                     or "stim")
    rows = []
    for ev in rec.events_table():
        tt = ev["trial_type"]
        # Use the fallback only when the code is genuinely empty. Single-
        # character DigMark codes (A/B/C/D) are valid labels - do NOT collapse.
        if tt in (None, "", "n/a"):
            tt = fallback_type
        rows.append({"onset": ev["onset"], "duration": ev["duration"],
                     "trial_type": tt})
    _write_tsv(pf.events_tsv_path, ["onset", "duration", "trial_type"], rows)


def write_nibs_sidecar(pf: PlannedFile, schema) -> None:
    values = dict(pf.item.sidecar_values)
    values.setdefault("StimulationModality", pf.item.modality)
    sidecar = schema.ordered_sidecar(values, modality=pf.item.modality)
    sidecar["SourceFile"] = os.path.basename(pf.item.source_path)
    _write_json(pf.nibs_json_path, sidecar)


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


def upsert_participant(layout: DatasetLayout, participant_id: str,
                       extra: dict = None) -> None:
    """Add (or merge) a participants.tsv row, de-duplicated by participant_id."""
    extra = extra or {}
    path = os.path.join(layout.rawdata_root, "participants.tsv")
    header, rows = _read_tsv(path)

    cols = ["participant_id"] + [k for k in extra if k != "participant_id"]
    for c in header:
        if c not in cols:
            cols.append(c)

    by_id = {r.get("participant_id"): r for r in rows}
    row = by_id.get(participant_id, {"participant_id": participant_id})
    row.update({"participant_id": participant_id, **extra})
    by_id[participant_id] = row

    ordered = sorted(by_id.values(), key=lambda r: r.get("participant_id", ""))
    _write_tsv(path, cols, ordered)

    json_path = os.path.join(layout.rawdata_root, "participants.json")
    if not os.path.isfile(json_path) and extra:
        _write_json(json_path,
                    {k: {"Description": k} for k in extra})


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
            write_nibs_sidecar(pf, schema)

            # 6) dataset-level files
            upsert_participant(plan.layout, pf.item.metadata.participant_id or
                               "sub-unknown", pf.item.participant_extra)
            scan_rel = f"{pf.rel_dir}/{os.path.basename(pf.edf_path)}"
            append_scan(plan.layout, pf.item.metadata, scan_rel)

            if ok:
                log(f"  ✓ verified ({rec.n_channels} ch, {len(rec.events)} events)")
            else:
                log("  ! read-back mismatch: " + "; ".join(disc))
            results.append(FileResult(pf.item.source_path, ok,
                                      edf_path=pf.edf_path, discrepancies=disc))
        except Exception as exc:
            log(f"  ✗ failed: {type(exc).__name__}: {exc}")
            results.append(FileResult(pf.item.source_path, False, error=str(exc)))

    return results
