"""
mep_cmap.dataset_session
~~~~~~~~~~~~~~~~~~~~~~~~
Persistent dataset-level session state for multi-file processing.

A DatasetSession tracks a queue of source files, their processing status,
shared analysis settings, and stim-label-keyed design configuration.

One mep_cmap_dataset.json lives at the derivatives root and acts as the
single source of truth for what has been processed and what remains.
The filename is intentionally project-specific to distinguish it from
other analysis pipelines (e.g. twitch torque) sharing the same derivatives
folder.
"""

from __future__ import annotations
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_NOT_STARTED  = "not_started"
STATUS_IN_PROGRESS  = "in_progress"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_COMPLETE     = "complete"
STATUS_STALE        = "stale"

STATUS_LABELS = {
    STATUS_NOT_STARTED:  "⏳ Not started",
    STATUS_IN_PROGRESS:  "🔄 In progress",
    STATUS_NEEDS_REVIEW: "⚠️  Needs review",
    STATUS_COMPLETE:     "✅ Complete",
    STATUS_STALE:        "🔁 Stale",
}

STATUS_COLOURS = {
    STATUS_NOT_STARTED:  "#888888",
    STATUS_IN_PROGRESS:  "#f0a500",
    STATUS_NEEDS_REVIEW: "#d9534f",
    STATUS_COMPLETE:     "#5cb85c",
    STATUS_STALE:        "#8b6914",
}


# ── Path portability helpers ──────────────────────────────────────────────────

def _study_root_from_deriv(derivatives_root: str) -> str:
    """Return the study root (parent of derivatives/) given derivatives_root."""
    norm = os.path.normpath(derivatives_root)
    if os.path.basename(norm).lower() == "derivatives":
        return os.path.dirname(norm)
    return norm


def _to_relative(abs_path: str, study_root: str) -> str:
    """
    Convert abs_path to a path relative to study_root.
    Returns the original string if already relative or if the path escapes
    the study tree (starts with '..').
    """
    if not abs_path or not os.path.isabs(abs_path):
        return abs_path
    try:
        rel = os.path.relpath(abs_path, study_root)
        if rel.startswith(".."):
            return abs_path
        return rel
    except ValueError:
        return abs_path   # cross-drive on Windows


def _to_absolute(stored_path: str, study_root: str) -> str:
    """
    Resolve stored_path (relative or absolute) to an absolute path.

    Resolution order
    ----------------
    1. Absolute path that still exists on this machine → use as-is.
    2. Relative path joined with study_root → use if file exists.
    3. Search by filename inside derivatives/ then the full study tree.
    4. Fall back: return joined path so open() gives a clear error.
    """
    if not stored_path:
        return stored_path
    # Normalise both slash styles to the OS separator
    stored_path = stored_path.replace("/", os.sep).replace("\\", os.sep)
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    candidate = os.path.normpath(os.path.join(study_root, stored_path))
    if os.path.exists(candidate):
        return candidate
    basename = os.path.basename(stored_path)
    if basename:
        for search_root in [os.path.join(study_root, "derivatives"), study_root]:
            if os.path.isdir(search_root):
                for dirpath, _dirs, filenames in os.walk(search_root):
                    if basename in filenames:
                        return os.path.join(dirpath, basename)
    return candidate


@dataclass
class FileEntry:
    """One source file in the dataset queue."""
    id:               str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    path:             str   = ""
    label:            str   = ""          # display label (from BIDS or user)
    status:           str   = STATUS_NOT_STARTED
    last_processed:   str   = ""          # ISO timestamp
    derivatives_json: str   = ""          # path to per-file autosave JSON
    crop_range:       Optional[list] = None   # [t_start, t_end] or None = full file
    stim_letters:     list  = field(default_factory=list)
    stim_label_map:   dict  = field(default_factory=dict)  # {letter: label}
    include_in_group: bool  = True
    review_flags:     dict  = field(default_factory=dict)  # {letter: status}
    is_external_ref:  bool  = False       # True if added as external normalisation ref

    def to_dict(self, study_root: str = "") -> dict:
        path_out  = _to_relative(self.path,             study_root) if study_root else self.path
        deriv_out = _to_relative(self.derivatives_json, study_root) if study_root else self.derivatives_json
        return {
            "id":               self.id,
            "path":             path_out,
            "label":            self.label,
            "status":           self.status,
            "last_processed":   self.last_processed,
            "derivatives_json": deriv_out,
            "crop_range":       self.crop_range,
            "stim_letters":     self.stim_letters,
            "stim_label_map":   self.stim_label_map,
            "include_in_group": self.include_in_group,
            "review_flags":     self.review_flags,
            "is_external_ref":  self.is_external_ref,
        }

    @classmethod
    def from_dict(cls, d: dict, study_root: str = "") -> "FileEntry":
        raw_path  = d.get("path", "")
        raw_deriv = d.get("derivatives_json", "")
        path_abs  = _to_absolute(raw_path,  study_root) if study_root else raw_path
        deriv_abs = _to_absolute(raw_deriv, study_root) if study_root and raw_deriv else raw_deriv
        return cls(
            id               = d.get("id", str(uuid.uuid4())[:8]),
            path             = path_abs,
            label            = d.get("label", ""),
            status           = d.get("status", STATUS_NOT_STARTED),
            last_processed   = d.get("last_processed", ""),
            derivatives_json = deriv_abs,
            crop_range       = d.get("crop_range"),
            stim_letters     = d.get("stim_letters", []),
            stim_label_map   = d.get("stim_label_map", {}),
            include_in_group = d.get("include_in_group", True),
            review_flags     = d.get("review_flags", {}),
            is_external_ref  = d.get("is_external_ref", False),
        )

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)

    def mark_complete(self):
        self.status = STATUS_COMPLETE
        self.last_processed = datetime.now().isoformat(timespec="seconds")

    def mark_in_progress(self):
        self.status = STATUS_IN_PROGRESS
        self.last_processed = datetime.now().isoformat(timespec="seconds")

    def check_stale(self) -> bool:
        """Mark as stale if source file modified after last processing."""
        if self.status != STATUS_COMPLETE or not self.last_processed:
            return False
        try:
            mtime = os.path.getmtime(self.path)
            processed = datetime.fromisoformat(self.last_processed).timestamp()
            if mtime > processed:
                self.status = STATUS_STALE
                return True
        except Exception:
            pass
        return False


class DatasetSession:
    """
    Top-level dataset state — persisted as mep_cmap_dataset.json
    at the derivatives root.

    The filename is project-specific so MEP-CMAP dataset files are
    clearly distinguishable from other pipelines (e.g. twitch torque)
    that may share the same derivatives folder.
    """

    FILENAME = "mep_cmap_dataset.json"
    SCHEMA   = "1.0"

    def __init__(self, derivatives_root: str = ""):
        self.derivatives_root: str  = derivatives_root
        self.created:          str  = datetime.now().isoformat(timespec="seconds")
        self.last_modified:    str  = self.created
        self.files:            list[FileEntry] = []

        # Shared settings — serialised from app GUI vars at save time
        self.shared_settings:   dict = {}

        # Stim design keyed by Stim_Label string
        # {"CSE": {"colour": ..., "gap_ms": ..., ...}}
        self.stim_design:       dict = {}

        # Normalisation map keyed by Stim_Label
        # {"SICI": "CSE", "ICF": "CSE", "CSE": "Mmax"}
        self.normalisation_map: dict = {}
        # Paths explicitly removed by the user — never re-added by auto-scan
        self.excluded_paths:    set  = set()

    # ── Persistence ───────────────────────────────────────────────────────────

    @property
    def _deriv_dir(self) -> str:
        """The derivatives/ subfolder — where the JSON is stored."""
        # If derivatives_root already ends in 'derivatives', use it directly.
        # Otherwise append 'derivatives/' so the JSON lives inside the folder
        # that also contains sub-XXX session subfolders.
        if os.path.basename(self.derivatives_root).lower() == "derivatives":
            return self.derivatives_root
        candidate = os.path.join(self.derivatives_root, "derivatives")
        if os.path.isdir(candidate):
            return candidate
        # Neither exists yet — will be created on first save
        return candidate

    @property
    def json_path(self) -> str:
        return os.path.join(self._deriv_dir, self.FILENAME)

    def save(self) -> bool:
        self.last_modified = datetime.now().isoformat(timespec="seconds")
        # Study root is the parent of the derivatives/ folder.
        # Always derive it from _deriv_dir (where the JSON actually lives)
        # so it stays correct even when the folder is on OneDrive and the
        # absolute path differs between computers.
        study_root = os.path.dirname(self._deriv_dir)
        data = {
            "schema_version":    self.SCHEMA,
            "created":           self.created,
            "last_modified":     self.last_modified,
            "derivatives_root":  "derivatives",   # always store as relative sentinel
            "shared_settings":   self.shared_settings,
            "stim_design":       self.stim_design,
            "normalisation_map": self.normalisation_map,
            "excluded_paths":    [
                _to_relative(p, study_root)
                for p in getattr(self, "excluded_paths", set())
            ],
            "files":             [f.to_dict(study_root=study_root) for f in self.files],
        }
        try:
            os.makedirs(self._deriv_dir, exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, derivatives_root: str) -> "Optional[DatasetSession]":
        """Try to load from derivatives_root or its derivatives/ subfolder."""
        ds = cls(derivatives_root=derivatives_root)
        path = ds.json_path  # uses _deriv_dir logic
        # Also check the root itself in case an old file lives there
        alt_path = os.path.join(derivatives_root, cls.FILENAME)
        if not os.path.isfile(path) and os.path.isfile(alt_path):
            path = alt_path
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            ds.created           = data.get("created", "")
            ds.last_modified     = data.get("last_modified", "")
            ds.shared_settings   = data.get("shared_settings", {})
            ds.stim_design       = data.get("stim_design", {})
            ds.normalisation_map = data.get("normalisation_map", {})

            # KEY: derive study root from where the JSON file actually lives,
            # not from the stored derivatives_root string (which may be a stale
            # absolute OneDrive path from a different computer).
            study_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))

            ds.excluded_paths = {
                _to_absolute(p, study_root)
                for p in data.get("excluded_paths", [])
            }
            ds.files = [
                FileEntry.from_dict(f, study_root=study_root)
                for f in data.get("files", [])
            ]
            for fe in ds.files:
                fe.check_stale()
            return ds
        except Exception:
            return None

    @classmethod
    def load_or_create(cls, derivatives_root: str) -> "DatasetSession":
        existing = cls.load(derivatives_root)
        return existing if existing else cls(derivatives_root=derivatives_root)

    # ── File queue management ─────────────────────────────────────────────────

    def add_file(self, path: str, label: str = "",
                 is_external_ref: bool = False) -> FileEntry:
        """Add a file to the queue. Returns existing entry if already present."""
        for fe in self.files:
            if os.path.normpath(fe.path) == os.path.normpath(path):
                return fe
        fe = FileEntry(
            path=path,
            label=label or os.path.basename(path),
            is_external_ref=is_external_ref,
        )
        self.files.append(fe)
        return fe

    def remove_file(self, file_id: str):
        self.files = [f for f in self.files if f.id != file_id]

    def get_file(self, file_id: str) -> Optional[FileEntry]:
        for f in self.files:
            if f.id == file_id:
                return f
        return None

    def get_by_path(self, path: str) -> Optional[FileEntry]:
        norm = os.path.normpath(path)
        for f in self.files:
            if os.path.normpath(f.path) == norm:
                return f
        return None

    def move_up(self, file_id: str):
        idx = next((i for i, f in enumerate(self.files) if f.id == file_id), None)
        if idx and idx > 0:
            self.files[idx-1], self.files[idx] = self.files[idx], self.files[idx-1]

    def move_down(self, file_id: str):
        idx = next((i for i, f in enumerate(self.files) if f.id == file_id), None)
        if idx is not None and idx < len(self.files) - 1:
            self.files[idx], self.files[idx+1] = self.files[idx+1], self.files[idx]

    # ── Convenience queries ───────────────────────────────────────────────────

    @property
    def n_total(self) -> int:
        return len(self.files)

    @property
    def n_complete(self) -> int:
        return sum(1 for f in self.files if f.status == STATUS_COMPLETE)

    @property
    def n_remaining(self) -> int:
        return sum(1 for f in self.files
                   if f.status in (STATUS_NOT_STARTED, STATUS_IN_PROGRESS,
                                   STATUS_NEEDS_REVIEW, STATUS_STALE))

    @property
    def all_complete(self) -> bool:
        return self.n_total > 0 and self.n_remaining == 0

    def next_unprocessed(self) -> Optional[FileEntry]:
        """Return the first file that still needs processing."""
        for f in self.files:
            if f.status in (STATUS_NOT_STARTED, STATUS_IN_PROGRESS,
                            STATUS_NEEDS_REVIEW, STATUS_STALE):
                return f
        return None

    def label_from_bids(self, path: str) -> str:
        """
        Extract a human-readable label from a BIDS-style filename.

        Examples
        --------
        sub-015_ses-2_limb-left_CSE_260116_000.txt  → 'limb-left CSE'
        sub-015_ses-1_limb-left_260114_095222_000.smr → 'limb-left 260114 095222 000'

        Falls back to the bare stem if the BIDS prefix pattern is not found.

        Date stripping logic
        --------------------
        Trailing _XXXXXX_NNN and _XXXXXX patterns are stripped only when the
        remaining portion has more than one component (e.g. 'limb-left_CSE').
        If stripping would reduce the label to a bare hyphenated entity such as
        'limb-left', the date is kept — it is the only information that
        distinguishes files with the same subject, session, and limb label.
        """
        import re
        bn = os.path.splitext(os.path.basename(path))[0]
        # Strip the sub-XXX_ses-XXX_ prefix
        m = re.match(r'sub-[^_]+_ses-[^_]+_(.*)', bn)
        if m:
            label = m.group(1)
            # Tentatively strip trailing date/counter patterns
            stripped = re.sub(r'_\d{6}_\d+$', '', label)
            stripped = re.sub(r'_\d{6}$', '', stripped)
            # Only use the stripped version if it has at least two underscore-
            # separated components (e.g. 'limb-left_CSE').  A bare entity like
            # 'limb-left' with no other content means the date was the only
            # differentiator — keep it.
            if stripped and '_' in stripped:
                label = stripped
            return label.replace('_', ' ')
        return os.path.splitext(os.path.basename(path))[0]
