"""
mep_cmap.archive_superseded
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Move outputs a later run has replaced out of the way, so a group merge reads
one copy of each result rather than several.

WHY THIS EXISTS

``stage2._s2_scan`` walks the whole derivatives tree and makes a row of every
``*_trials.json`` it finds. Nothing reconciles two files that describe the same
recording, so a study part-written in the flat layout and part in
``results/trial-level/`` merges as though it had twice the participants. The
same is true of outputs left behind by a run whose naming rules have since
changed: they are not overwritten, they accumulate beside the current set.

WHAT COUNTS AS CURRENT

The session JSON is the authority. Its own filename carries the prefix the
recording's outputs were written under -- the same rule the batch re-runner
uses to pin its output prefix (``reprocess_ui._reprocess_saved_prefix``), so
the two cannot disagree about which files belong to a session.

A result file is CURRENT when it sits in the foldered layout under a prefix
some session JSON in that folder claims. Everything else is superseded:

``flat-layout``
    In ``results/`` directly, under a claimed prefix. The foldered copy is the
    one a current run wrote.

``unclaimed-prefix``
    Under a prefix no session claims, in either layout. Written by a run whose
    naming rules have since changed; nothing points at it any more.

WHAT IS NEVER TOUCHED

A session with NO file in the foldered layout has not been re-run. Its flat
output is the only copy that exists, and archiving it would leave the session
with no results at all -- a merge would then silently drop it rather than
double-count it, which is worse. Those sessions are reported as
:attr:`ArchivePlan.stale` and left exactly where they are. They want
re-running, not archiving.

MOVED, NOT DELETED

Everything goes to an archive root OUTSIDE ``derivatives/``, keeping its path
relative to the derivatives root so where it came from is readable from where
it ended up. Outside because ``_s2_scan`` walks the derivatives tree directly
rather than through ``results_layout.search_roots``, so the
``EXCLUDED_FROM_SEARCH`` mechanism would not cover an archive placed inside it.

The manifest records every move. Nothing here reads or rewrites a result file:
a superseded output is moved intact, so restoring one is a copy back.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import shutil
from dataclasses import dataclass, field

from . import results_layout as RL

#: Result families whose files are candidates for archiving. Keyed off
#: results_layout so a family added there is covered here without a second
#: list to keep in step.
_SUFFIXES = tuple(sorted(RL.FAMILIES, key=len, reverse=True))

#: Reasons a file is superseded, in the manifest's own words.
REASON_FLAT = "flat-layout"
REASON_UNCLAIMED = "unclaimed-prefix"

MANIFEST_NAME = "archive_manifest.csv"

MANIFEST_HEADER = [
    "OriginalPath",     # relative to the derivatives root
    "ArchivePath",      # relative to the archive root
    "Reason",
    "Modified",
    "SessionFolder",    # relative to the derivatives root
    "Prefix",           # the prefix this file was written under
    "Bytes",
]


@dataclass
class ArchiveItem:
    """One superseded file and why it is superseded."""
    path: str            # absolute
    reason: str          # REASON_FLAT | REASON_UNCLAIMED
    session_dir: str     # absolute, the ses- folder
    prefix: str          # the prefix the file was written under
    mtime: float

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)


@dataclass
class StaleSession:
    """A session whose only outputs are in the flat layout.

    Not archived. Named so the caller can say which sessions still need a
    re-run rather than leaving the study silently mixed-vintage.
    """
    session_dir: str
    prefix: str
    flat_files: list = field(default_factory=list)


@dataclass
class ArchivePlan:
    """What would move, what would not, and what is missing a current copy."""
    items: list = field(default_factory=list)      # ArchiveItem
    stale: list = field(default_factory=list)      # StaleSession
    current: list = field(default_factory=list)    # absolute paths, kept
    derivatives_root: str = ""

    @property
    def n_items(self) -> int:
        return len(self.items)

    def by_reason(self) -> dict:
        out = {}
        for it in self.items:
            out[it.reason] = out.get(it.reason, 0) + 1
        return out


def _claimed_prefixes(session_dir: str) -> set:
    """Prefixes the session JSONs in one ses- folder claim.

    The session filename is the record of the prefix its outputs carry. Taken
    from the name rather than from a field inside, because a session written
    before that field existed still has a name.
    """
    out = set()
    try:
        names = os.listdir(session_dir)
    except OSError:
        return out
    for name in names:
        if name.endswith("_session.json"):
            out.add(name[: -len("_session.json")])
    return out


def _prefix_of(filename: str) -> str:
    """The BIDS prefix a result file was written under, or "" if unrecognised.

    Longest suffix first, for the same reason family_for does it: a name
    ending ``_summary_with_outliers.csv`` also ends ``_summary.csv``.
    """
    for suffix in _SUFFIXES:
        if filename.endswith("_" + suffix):
            return filename[: -len(suffix) - 1]
    return ""


def _session_dirs(derivatives_root: str) -> list:
    """Every folder holding at least one ``*_session.json``.

    Found by walking rather than by assuming ``sub-*/ses-*``: a study whose
    folders are named otherwise still has its sessions found, and a folder with
    no session JSON has nothing to judge its outputs against and is skipped
    rather than guessed at.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(derivatives_root):
        if any(f.endswith("_session.json") for f in filenames):
            out.append(dirpath)
        # Never descend into an archive that happens to sit inside.
        dirnames[:] = [d for d in dirnames if not d.startswith("_archive")]
    return sorted(out)


def _result_files(session_dir: str) -> list:
    """Every result file under this session's ``results/``, both layouts.

    Not results_layout.find_results: that deduplicates by basename and skips
    the report family, both of which are right for finding ONE current result
    and wrong here, where the duplicates are exactly what is being looked for.
    """
    res = os.path.join(session_dir, "results")
    out = []
    if not os.path.isdir(res):
        return out
    for dirpath, dirnames, filenames in os.walk(res):
        # One level of family folders, as the layout documents. Anything
        # deeper is not ours -- figures/, or whatever an analyst has left.
        if os.path.relpath(dirpath, res).count(os.sep) >= 1:
            dirnames[:] = []
        for fn in sorted(filenames):
            if _prefix_of(fn):
                out.append(os.path.join(dirpath, fn))
    return out


def _is_foldered(path: str, session_dir: str) -> bool:
    """True when the file sits in a family subfolder rather than flat.

    Asked of the path rather than of the filename, because the same name is
    valid in both layouts -- which is the whole reason duplicates exist.
    """
    res = os.path.join(session_dir, "results")
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return (os.path.abspath(os.path.dirname(path)) != os.path.abspath(res)
            and parent in set(RL.FAMILIES.values()) | {RL.ADDON_DIR})


def plan_archive(derivatives_root: str) -> ArchivePlan:
    """Decide what is superseded, without moving anything.

    Judged one session folder at a time. A prefix is current only against the
    sessions in its OWN folder: two participants can carry the same task names,
    and a prefix claimed across the study would let one session's outputs vouch
    for another's.
    """
    plan = ArchivePlan(derivatives_root=os.path.abspath(derivatives_root))
    if not os.path.isdir(derivatives_root):
        return plan

    for session_dir in _session_dirs(derivatives_root):
        prefixes = _claimed_prefixes(session_dir)
        if not prefixes:
            continue

        files = _result_files(session_dir)
        # Which prefixes have a current (foldered) file at all. A prefix with
        # none has not been re-run, and its flat copy is the only one there is.
        foldered_prefixes = {
            _prefix_of(os.path.basename(p))
            for p in files
            if _is_foldered(p, session_dir)
            and _prefix_of(os.path.basename(p)) in prefixes
        }

        stale_flat = {}
        for path in files:
            name = os.path.basename(path)
            prefix = _prefix_of(name)
            claimed = prefix in prefixes
            foldered = _is_foldered(path, session_dir)

            if claimed and foldered:
                plan.current.append(path)
                continue

            if claimed and not foldered:
                if prefix not in foldered_prefixes:
                    # The only copy. Left alone, reported instead.
                    stale_flat.setdefault(prefix, []).append(path)
                    continue
                reason = REASON_FLAT
            else:
                reason = REASON_UNCLAIMED

            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            plan.items.append(ArchiveItem(
                path=path, reason=reason, session_dir=session_dir,
                prefix=prefix, mtime=mtime))

        for prefix, paths in sorted(stale_flat.items()):
            plan.stale.append(StaleSession(
                session_dir=session_dir, prefix=prefix,
                flat_files=sorted(paths)))

    plan.items.sort(key=lambda i: i.path)
    plan.current.sort()
    return plan


def default_archive_root(derivatives_root: str, when=None) -> str:
    """Where an archive made now would go: BESIDE derivatives, never inside.

    Inside would be walked by ``stage2._s2_scan``, which recurses the whole
    derivatives tree directly rather than through
    ``results_layout.search_roots`` -- so the folder it was moved out of would
    find it again and the double-count this exists to stop would survive the
    move.
    """
    when = when or _dt.datetime.now()
    parent = os.path.dirname(os.path.abspath(derivatives_root))
    stamp = when.strftime("%Y%m%dT%H%M%S")
    return os.path.join(parent, "superseded_derivatives", stamp)


def _archive_target(item: ArchiveItem, plan: ArchivePlan,
                    archive_root: str) -> str:
    """Where one item lands, keeping its path relative to the derivatives root.

    Relative rather than flattened so provenance reads off the location: a file
    restored by hand goes back where it plainly came from. Two files with the
    same basename in different layouts therefore stay distinct, which a flat
    archive folder would not manage.
    """
    rel = os.path.relpath(item.path, plan.derivatives_root)
    return os.path.join(archive_root, rel)


def write_manifest(plan: ArchivePlan, archive_root: str, path: str = "") -> str:
    """Record every move, before any of them happen.

    Written first so an interrupted archive still says what it intended. The
    manifest is the only record linking a moved file to the prefix it lost to,
    since the file itself carries nothing about why it was superseded.
    """
    path = path or os.path.join(archive_root, MANIFEST_NAME)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_HEADER)
        for item in plan.items:
            try:
                size = os.path.getsize(item.path)
            except OSError:
                size = ""
            w.writerow([
                os.path.relpath(item.path, plan.derivatives_root),
                os.path.relpath(_archive_target(item, plan, archive_root),
                                archive_root),
                item.reason,
                _dt.datetime.fromtimestamp(item.mtime).isoformat(
                    timespec="seconds") if item.mtime else "",
                os.path.relpath(item.session_dir, plan.derivatives_root),
                item.prefix,
                size,
            ])
    return path


def apply_archive(plan: ArchivePlan, archive_root: str,
                  log_callback=None) -> dict:
    """Move every planned item, manifest first.

    A move that fails is reported and the rest continue: a single locked file
    -- ordinary under OneDrive -- should not leave the study half archived with
    no record of which half.

    Returns a summary dict rather than raising, so a caller can report partial
    success honestly.
    """
    log = log_callback or (lambda _m: None)
    os.makedirs(archive_root, exist_ok=True)
    manifest = write_manifest(plan, archive_root)

    moved, failed = 0, []
    for item in plan.items:
        target = _archive_target(item, plan, archive_root)
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target):
                # Never overwrite inside the archive: two runs archiving the
                # same path would otherwise lose the earlier copy, which is
                # the one thing an archive must not do.
                stem, ext = os.path.splitext(target)
                n = 2
                while os.path.exists(f"{stem}__{n}{ext}"):
                    n += 1
                target = f"{stem}__{n}{ext}"
            shutil.move(item.path, target)
            moved += 1
        except Exception as exc:                      # noqa: BLE001 — reported
            failed.append((item.path, f"{type(exc).__name__}: {exc}"))
            log(f"   ⚠️  Could not archive {item.basename}: {exc}")

    log(f"   Archived {moved} superseded file(s) to {archive_root}")
    if failed:
        log(f"   ⚠️  {len(failed)} file(s) could not be moved and are still "
            f"in place.")
    if plan.stale:
        log(f"   ⚠️  {len(plan.stale)} session output(s) have no current "
            f"copy and were left alone. They need re-running.")

    return {"moved": moved, "failed": failed, "manifest": manifest,
            "archive_root": archive_root, "stale": list(plan.stale)}


def describe(plan: ArchivePlan) -> str:
    """A dry-run summary, for showing before anything moves."""
    by = plan.by_reason()
    lines = [
        f"Current outputs kept:        {len(plan.current)}",
        f"Superseded, would archive:   {plan.n_items}",
    ]
    for reason in (REASON_FLAT, REASON_UNCLAIMED):
        if by.get(reason):
            lines.append(f"    {reason:<20} {by[reason]}")
    lines.append(f"No current copy, left alone: {len(plan.stale)}")
    for st in plan.stale[:10]:
        lines.append(f"    {st.prefix}")
    if len(plan.stale) > 10:
        lines.append(f"    ... and {len(plan.stale) - 10} more")
    return "\n".join(lines)
