"""
mep_cmap.sidecars
~~~~~~~~~~~~~~~~~
Where a format reader keeps what it was told about a recording.

Three readers hold a small configuration per file: which channel Spike2 should
treat as EMG, how a generic TSV's columns map onto channels, what unit a
pre-epoched MATLAB file is in. Each wrote it beside the recording, as
``<stem>.smr_config.json`` and friends.

That put tool state into the raw data. In a BIDS study ``rawdata/`` is what the
acquisition system wrote, and everything derived from it belongs under
``derivatives/``: a folder that can be deleted and rebuilt, backed up
separately, or handed to a collaborator without carrying one program's
settings along. It also meant "reset from scratch" had to know three filename
rules to clean up after itself, and a read-only or archived rawdata tree could
not be configured at all.

Migration
---------
Changing where a file is looked for makes every existing one invisible, and a
tool that silently forgets what it was told is worse than one that never
remembered: the analyst is asked to configure files they configured last week,
with no explanation and no clue that an answer exists somewhere. So the old
location is still read, and the first time a sidecar is found there it is moved.

The move is best-effort. A rawdata tree mounted read-only is exactly the
situation this change is meant to help, and failing to tidy it up is not a
reason to refuse to read it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

#: Where sidecars live under the derivatives root.
_SUBDIR = "sourcedata_config"

#: Set by the application when a derivatives folder is chosen. Without one
#: there is nowhere to put a sidecar, and the old location beside the recording
#: is used -- which is what every earlier version did, so a study with no
#: derivatives folder configured behaves exactly as it always has.
_deriv_root = [None]


def set_derivatives_root(path):
    """Tell the sidecar layer where derivatives live, or None to unset."""
    _deriv_root[0] = str(path) if path else None


def derivatives_root():
    return _deriv_root[0]


def legacy_path(file_path: str, suffix: str) -> Path:
    """The old location: beside the recording."""
    return Path(file_path).with_suffix(suffix)


def sidecar_path(file_path: str, suffix: str) -> Path:
    """Where this recording's sidecar should be written.

    Falls back to the old location when no derivatives root is set, so a study
    that has not chosen one is not silently deprived of its configuration.
    """
    root = _deriv_root[0]
    if not root:
        return legacy_path(file_path, suffix)
    # The study's derivatives/ folder whichever of the study folder, the
    # derivatives folder or the pipeline folder was configured.
    from .bids import derivatives_dir
    parent = derivatives_dir(root)
    return Path(parent) / _SUBDIR / (Path(file_path).stem + suffix)


def resolve(file_path: str, suffix: str) -> Path:
    """The sidecar to read, migrating one from the old location if found.

    Returns the new path whether or not anything was migrated, so callers can
    treat it as the single answer to "where is it".
    """
    new = sidecar_path(file_path, suffix)
    if new.exists():
        return new
    old = legacy_path(file_path, suffix)
    if old.exists() and str(old) != str(new):
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
        except Exception:
            # Best-effort: a read-only rawdata tree is precisely the case this
            # change exists to help, and being unable to tidy it is not a
            # reason to refuse to read it.
            return old
    return new


def read(file_path: str, suffix: str):
    """The sidecar's contents, or None. Never raises."""
    try:
        p = resolve(file_path, suffix)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def write(file_path: str, suffix: str, payload: dict) -> bool:
    """Write the sidecar. Returns False rather than raising."""
    try:
        p = sidecar_path(file_path, suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return True
    except Exception:
        return False


def exists(file_path: str, suffix: str) -> bool:
    try:
        return resolve(file_path, suffix).exists()
    except Exception:
        return False


def remove(file_path: str, suffix: str) -> list:
    """Delete the sidecar from both locations. Returns what was removed.

    Both, because a migration that failed leaves one behind, and a reset that
    cleared only the new location would leave the old one to be found and
    migrated straight back.
    """
    gone = []
    for p in (sidecar_path(file_path, suffix), legacy_path(file_path, suffix)):
        try:
            if p.exists():
                p.unlink()
                gone.append(p.name)
        except Exception:
            pass
    return gone
