"""
mep_cmap.layout_migration
~~~~~~~~~~~~~~~~~~~~~~~~~
Move outputs from the legacy derivatives layout into the BIDS one.

    legacy (before 1.4.8)   <study>/derivatives/sub-XX/ses-YY/...
    current                 <study>/derivatives/mep-cmap-analyser/sub-XX/ses-YY/emg/...

Two operations, both of which move and never delete:

archive_legacy_for_recording
    After a recording has been re-run and its new session saved, its
    legacy-layout outputs are moved to the archive (``superseded_derivatives/``
    beside ``derivatives/``, as archive_superseded does). Without this a
    re-run study holds two copies of the recording and Stage 2, which walks
    the whole derivatives tree, merges it twice.

plan_migration / apply_migration
    Move a whole study's legacy outputs into the current layout in one pass,
    unchanged. A recording that already has a session in the current layout
    has been re-run since, so its legacy copy is archived instead of moved.

WHAT BELONGS TO A RECORDING

A legacy ``ses-`` folder holds every recording of that session side by side.
A session JSON's filename carries the prefix its outputs were written under
(the rule archive_superseded and the batch re-runner use), so a recording's
files are the session JSON plus every file or folder, anywhere below the
``ses-`` folder, whose name starts with ``<prefix>_``. A longer prefix claimed
by another session in the same folder (``<prefix>_run2``) is that session's,
not this one's.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import shutil
from dataclasses import dataclass, field

from . import archive_superseded as AR
from .bids import (DATATYPE, PIPELINE_NAME, derivatives_dir,
                   legacy_session_output_dir, pipeline_dir)

REASON_LEGACY = "legacy-layout"
_SESSION_SUFFIX = "_session.json"


def _session_prefixes(folder: str) -> list:
    """Prefixes claimed by the session JSONs directly in ``folder``."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(n[:-len(_SESSION_SUFFIX)] for n in names
                  if n.endswith(_SESSION_SUFFIX))


def _belongs(name: str, prefix: str, others: list) -> bool:
    """True if a file or folder called ``name`` is ``prefix``'s."""
    if name == prefix + _SESSION_SUFFIX:
        return True
    if not name.startswith(prefix + "_"):
        return False
    # A longer prefix of another session in the same folder claims it.
    return not any(o != prefix and o.startswith(prefix + "_")
                   and (name == o + _SESSION_SUFFIX or name.startswith(o + "_"))
                   for o in others)


def legacy_items(session_dir: str, prefix: str) -> list:
    """Every file or folder below a legacy ``ses-`` folder that is ``prefix``'s.

    A folder whose own name belongs (``<prefix>_onset_methods_figures``) is
    returned whole rather than file by file. Never descends into an ``emg``
    folder, which is the current layout's, or into an archive.
    """
    others = _session_prefixes(session_dir)
    out = []
    for dirpath, dirnames, filenames in os.walk(session_dir):
        keep_dirs = []
        for d in sorted(dirnames):
            if d == DATATYPE or d.startswith("_archive"):
                continue
            if _belongs(d, prefix, others):
                out.append(os.path.join(dirpath, d))
            else:
                keep_dirs.append(d)
        dirnames[:] = keep_dirs
        for f in sorted(filenames):
            if _belongs(f, prefix, others):
                out.append(os.path.join(dirpath, f))
    return out


def _recording_of(session_json: str) -> str:
    """The recording a session was written for, by file name; "" if unknown."""
    try:
        with open(session_json, encoding="utf-8") as fh:
            stored = json.load(fh).get("file_path") or ""
    except Exception:                       # noqa: BLE001 - unreadable
        return ""
    return os.path.normcase(os.path.basename(stored.replace("\\", "/")))


# ── (a) after a re-run ───────────────────────────────────────────────────────

def archive_legacy_for_recording(source_path: str, metadata, study_root: str,
                                 new_session_path: str, log_callback=None):
    """Archive one recording's legacy-layout outputs once it has been re-run.

    Only when ``new_session_path`` exists: that is the proof the re-run
    finished and saved. Before it, the legacy session is the only record of
    the analyst's inspector edits on disk, and moving it would lose them if
    the run then failed.

    Returns apply_archive's summary, or None when there was nothing to move.
    """
    log = log_callback or (lambda _m: None)
    if not (source_path and new_session_path
            and os.path.isfile(new_session_path)):
        return None
    root = study_root or os.path.dirname(source_path)
    legacy_dir = legacy_session_output_dir(root, metadata)
    if not os.path.isdir(legacy_dir):
        return None
    if os.path.normcase(os.path.abspath(legacy_dir)) == os.path.normcase(
            os.path.abspath(os.path.dirname(new_session_path))):
        return None                         # same folder: nothing is legacy

    want = os.path.normcase(os.path.basename(source_path))
    plan = AR.ArchivePlan(derivatives_root=os.path.abspath(derivatives_dir(root)))
    for prefix in _session_prefixes(legacy_dir):
        sess = os.path.join(legacy_dir, prefix + _SESSION_SUFFIX)
        if _recording_of(sess) != want:
            continue
        for path in legacy_items(legacy_dir, prefix):
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            plan.items.append(AR.ArchiveItem(
                path=path, reason=REASON_LEGACY, session_dir=legacy_dir,
                prefix=prefix, mtime=mtime))
    if not plan.items:
        return None
    log(f"   Moving {len(plan.items)} output(s) from the old folder layout "
        f"aside, so this recording is not counted twice:")
    out = AR.apply_archive(plan, AR.default_archive_root(plan.derivatives_root),
                           log_callback=log)
    _remove_empty_dirs(legacy_dir)
    return out


def _remove_empty_dirs(top: str):
    """Remove folders left empty by a move, bottom up, including ``top``."""
    for dirpath, _dirs, _files in os.walk(top, topdown=False):
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass


# ── (b) migrating a whole study ──────────────────────────────────────────────

@dataclass
class MigrationPlan:
    """What a migration would do. Nothing has moved until apply_migration."""
    study_root: str = ""
    moves: list = field(default_factory=list)       # (src, dst, prefix)
    archive: list = field(default_factory=list)     # AR.ArchiveItem, re-run since
    conflicts: list = field(default_factory=list)   # (src, dst) target exists
    leftovers: list = field(default_factory=list)   # legacy files no session claims
    session_dirs: list = field(default_factory=list)  # legacy ses- folders seen

    @property
    def n_sessions(self) -> int:
        return len({p for _s, _d, p in self.moves})

    def describe(self) -> str:
        lines = [
            f"Study folder: {self.study_root}",
            f"Recordings to move into derivatives/{PIPELINE_NAME}/.../{DATATYPE}/: "
            f"{self.n_sessions} ({len(self.moves)} file(s) or folder(s))",
            f"Re-run since, old copy archived instead: "
            f"{len({(i.session_dir, i.prefix) for i in self.archive})}",
        ]
        if self.conflicts:
            lines.append(f"Already present in the new layout, left where they "
                         f"are: {len(self.conflicts)}")
        if self.leftovers:
            lines.append(f"Files no session claims, left where they are: "
                         f"{len(self.leftovers)}")
        return "\n".join(lines)


def _legacy_session_dirs(dderiv: str) -> list:
    """Folders under derivatives/ holding session JSONs, outside the pipeline
    folder and outside any current-layout ``emg`` folder."""
    pdir = os.path.join(dderiv, PIPELINE_NAME)
    out = []
    for dirpath, dirnames, filenames in os.walk(dderiv):
        if os.path.normcase(dirpath) == os.path.normcase(pdir):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if d != DATATYPE and not d.startswith("_archive")
                       and os.path.normcase(os.path.join(dirpath, d))
                       != os.path.normcase(pdir)]
        if any(f.endswith(_SESSION_SUFFIX) for f in filenames):
            out.append(dirpath)
    return sorted(out)


def plan_migration(study_root: str) -> MigrationPlan:
    """Decide every move for a study, without moving anything."""
    dderiv = derivatives_dir(study_root)
    # Always the folder that HOLDS derivatives/, whichever of the three the
    # analyst pointed at, so the manifest lands beside derivatives/ and never
    # inside the tree Stage 2 walks.
    plan = MigrationPlan(study_root=os.path.dirname(os.path.abspath(dderiv)))
    if not os.path.isdir(dderiv):
        return plan
    pdir = pipeline_dir(study_root)
    for sdir in _legacy_session_dirs(dderiv):
        plan.session_dirs.append(sdir)
        rel = os.path.relpath(sdir, dderiv)
        target = os.path.join(pdir, rel, DATATYPE)
        claimed = set()
        for prefix in _session_prefixes(sdir):
            items = legacy_items(sdir, prefix)
            claimed.update(items)
            rerun = os.path.isfile(os.path.join(target, prefix + _SESSION_SUFFIX))
            for src in items:
                if rerun:
                    try:
                        mtime = os.path.getmtime(src)
                    except OSError:
                        mtime = 0.0
                    plan.archive.append(AR.ArchiveItem(
                        path=src, reason=REASON_LEGACY, session_dir=sdir,
                        prefix=prefix, mtime=mtime))
                    continue
                dst = os.path.join(target, os.path.relpath(src, sdir))
                if os.path.exists(dst):
                    plan.conflicts.append((src, dst))
                else:
                    plan.moves.append((src, dst, prefix))
        for dirpath, _d, files in os.walk(sdir):
            if any(os.path.normcase(dirpath).startswith(os.path.normcase(c))
                   for c in claimed if os.path.isdir(c)):
                continue
            for f in files:
                p = os.path.join(dirpath, f)
                if p not in claimed:
                    plan.leftovers.append(p)
    return plan


MIGRATION_MANIFEST_HEADER = ["OriginalPath", "NewPath", "Prefix", "Outcome"]


def apply_migration(plan: MigrationPlan, log_callback=None) -> dict:
    """Carry out a plan: manifest first, then every move, then the archive.

    A move that fails is reported and the rest continue, as in
    archive_superseded: one file locked by OneDrive should not leave the study
    half migrated with no record of which half.
    """
    log = log_callback or (lambda _m: None)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    manifest = os.path.join(plan.study_root, f"layout_migration_{stamp}.csv")
    moved, failed = 0, []
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MIGRATION_MANIFEST_HEADER)
        for src, dst, prefix in plan.moves:
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                moved += 1
                w.writerow([src, dst, prefix, "moved"])
            except Exception as exc:                  # noqa: BLE001 - reported
                failed.append((src, f"{type(exc).__name__}: {exc}"))
                w.writerow([src, dst, prefix, f"FAILED: {exc}"])
                log(f"   ⚠️  Could not move {os.path.basename(src)}: {exc}")
        for src, dst in plan.conflicts:
            w.writerow([src, dst, "", "left: already in the new layout"])
    log(f"   Moved {moved} file(s) or folder(s) into the BIDS layout. "
        f"Record: {os.path.basename(manifest)}")

    archived = None
    if plan.archive:
        aplan = AR.ArchivePlan(
            derivatives_root=os.path.abspath(derivatives_dir(plan.study_root)),
            items=list(plan.archive))
        archived = AR.apply_archive(
            aplan, AR.default_archive_root(aplan.derivatives_root),
            log_callback=log)

    from .bids import write_derivatives_description
    write_derivatives_description(plan.study_root)
    for sdir in plan.session_dirs:
        _remove_empty_dirs(sdir)
    if failed:
        log(f"   ⚠️  {len(failed)} item(s) could not be moved and are still "
            f"in the old layout.")
    return {"moved": moved, "failed": failed, "manifest": manifest,
            "archived": archived, "conflicts": len(plan.conflicts),
            "leftovers": len(plan.leftovers)}


def remap_session_paths(entries, plan: MigrationPlan) -> int:
    """Point dataset-queue entries at their session's new location.

    The queue stores each recording's session path and restores from it when
    the file is opened; after a migration that path no longer exists, and the
    recording would open with none of its saved work. Returns how many
    entries changed. ``entries`` are FileEntry-like (``derivatives_json``).
    """
    moved = {os.path.normcase(os.path.abspath(s)): d
             for s, d, _p in plan.moves if s.endswith(_SESSION_SUFFIX)}
    n = 0
    for fe in entries or []:
        old = getattr(fe, "derivatives_json", "") or ""
        new = moved.get(os.path.normcase(os.path.abspath(old))) if old else None
        if new:
            fe.derivatives_json = new
            n += 1
    return n
