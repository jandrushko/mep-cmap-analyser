"""
mep_cmap.addons — modular add-on discovery, context building, and safe execution.

Add-ons are built-in or user Python modules that post-process a saved results
bundle and write NEW files only. See design/ADDON_ARCHITECTURE.md and
design/addon_template.py for the contract an add-on must satisfy:

    ADDON_NAME, ADDON_DESCRIPTION, ADDON_VERSION, ADDON_AUTHOR   (module constants)
    run(context) -> list[str]                                    (entry point)

Discovery sources (both scanned; built-in first):
    • built-in:  mep_cmap/add_ons/         (shipped; home of the MEPFeatX add-on)
    • user:      the folder set in Preferences  (prefs.addons_path)

User modules are imported BY FILE PATH via importlib, so they load even inside a
frozen (PyInstaller) build — their own `import numpy/scipy/pandas` resolve against
the bundled libraries.

This module has no dependency on the GUI or the pipeline; it only reads the saved
results bundle, so it is fully testable in isolation.
"""
from __future__ import annotations

import os
import sys
import importlib.util
import traceback
import numpy as np

try:
    import pandas as pd
except Exception:                       # pandas always present in the app; guard anyway
    pd = None


# ─────────────────────────────────────────────────────────────────────────────
# Context handed to an add-on's run()
# ─────────────────────────────────────────────────────────────────────────────
class AddonContext:
    """Read-only view of ONE recording's results bundle.

    Fields mirror design/ADDON_ARCHITECTURE.md §5. `fs` and `unit` vary by source
    format and must be taken from here (never assumed); `time_ms` is 0 at the
    stimulus.

    WRITE OUTPUTS INTO ``addons_dir``, named with ``bids_prefix``.
    ``results_dir`` is the results ROOT, for READING what the pipeline wrote;
    it is no longer where an add-on should put its own files. The two were the
    same folder while everything was written flat, so an add-on that still
    writes to ``results_dir`` produces a correct file in an untidy place rather
    than a broken one.
    """
    __slots__ = ("trials", "segments", "fs", "unit", "time_ms",
                 "time_ms_by_type",
                 "config", "results_dir", "figures_dir", "bids_prefix", "log",
                 "_addons_dir")

    def time_ms_for(self, stim_type):
        """The time axis for one stimulus type.

        Windows are per stimulus type, so a recording has no single axis. An
        add-on that took one and applied it to every type met an axis of one
        length and traces of another -- a broadcast error where the numbers
        happened to differ, and a silently misaligned measurement where they
        did not.
        """
        axis = self.time_ms_by_type.get(stim_type)
        return self.time_ms if axis is None else axis

    @property
    def addons_dir(self):
        """Where this add-on's own output files go, created on first access.

        Created here rather than when the context is built, so a run with no
        add-ons enabled does not leave an empty folder in every session.
        """
        os.makedirs(self._addons_dir, exist_ok=True)
        return self._addons_dir

    def __init__(self, *, trials, segments, fs, unit, time_ms, config,
                 results_dir, bids_prefix, log, figures_dir=None,
                 time_ms_by_type=None, addons_dir=None):
        self.trials      = trials         # pandas.DataFrame | None
        self.segments    = segments       # {stim_type: ndarray[n_trials, n_samples]}
        self.fs          = fs             # float
        self.unit        = unit           # str | None
        self.time_ms     = time_ms        # 1-D ndarray (ms rel. stim, 0 = stim)
        # {stim_type: 1-D ndarray}. Stimulus types may be epoched over
        # different windows, so there is no single axis for a recording; use
        # time_ms_for(). `time_ms` is retained as the first group's axis, which
        # is what it always was and is still correct when the windows agree.
        self.time_ms_by_type = dict(time_ms_by_type or {})
        self.config      = config         # dict
        self.results_dir = results_dir    # str  (…/ses-*/results) — READ from
        self.figures_dir = figures_dir    # str  (…/ses-*/figures) sibling of results
        self.bids_prefix = bids_prefix    # str
        self.log         = log            # callable(str)
        # Defaulted rather than required, so a caller building a context
        # directly (a test, or an add-on harness) still gets a working one.
        from .results_layout import ADDON_DIR
        self._addons_dir = (addons_dir
                            or os.path.join(results_dir, ADDON_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Load a results bundle into one context per recording
# ─────────────────────────────────────────────────────────────────────────────
def load_contexts(segments_npz_path, config=None, log=print):
    """Build one AddonContext per file contained in a `<prefix>_segments.npz`.

    The bundle is format-agnostic (normalised waveforms + fs + unit + window),
    so the returned contexts are identical in shape regardless of which importer
    produced the data. Loads without `allow_pickle`.
    """
    # The results ROOT, not the folder the bundle happens to sit in. Under the
    # family layout the bundle is in results/segments/, so taking its dirname
    # gave a results_dir one level too deep: figures/ resolved inside results/,
    # and the per-trial table below was looked for somewhere it never is, which
    # left context.trials silently None rather than raising.
    from .results_layout import results_root_for
    # In Windows' extended-length form, as the pipeline's own outputs are:
    # add-on figures sit as deep and carry the same long prefix, so they hit
    # the 260-character limit first. Every path an add-on joins onto these
    # inherits the exemption.
    from .utils import long_path
    segments_npz_path = long_path(segments_npz_path)
    results_dir = results_root_for(segments_npz_path)
    # figures/ is the sibling of results/ in the BIDS derivatives layout
    figures_dir = os.path.join(os.path.dirname(results_dir), "figures")
    base        = os.path.basename(segments_npz_path)
    suffix      = "_segments.npz"
    run_prefix  = base[:-len(suffix)] if base.endswith(suffix) else base

    z = np.load(segments_npz_path, allow_pickle=False)
    files = [str(x) for x in z["manifest_file"]]
    stims = [str(x) for x in z["manifest_stim"]]
    fss   = z["manifest_fs"]
    units = [str(x) for x in z["manifest_unit"]]
    pres  = z["manifest_pre_ms"]

    # Optional per-trial table (for context.trials).
    trials_df = None
    if pd is not None:
        from .results_layout import result_path as _rp
        _tp = _rp(results_dir, f"{run_prefix}_trials.csv",
                  create=False)
        if not os.path.isfile(_tp):
            _tp = os.path.join(results_dir,
                               f"{run_prefix}_trials.csv")
        if os.path.exists(_tp):
            try:
                trials_df = pd.read_csv(_tp)
            except Exception as e:                       # non-fatal
                log(f"add-ons: could not read {os.path.basename(_tp)}: {e}")

    # Group bundle entries by source file.
    by_file = {}
    for i, f in enumerate(files):
        by_file.setdefault(f, []).append(i)

    multi = len(by_file) > 1
    contexts = []
    for fname, idxs in by_file.items():
        segments = {stims[i]: z[f"wav_{i}"] for i in idxs}
        fs   = float(fss[idxs[0]])
        unit = units[idxs[0]] or None

        # One axis per stimulus type, from that group's own pre and length.
        # The bundle has always recorded both per group; collapsing them to the
        # first group's values was harmless only while every type shared a
        # window.
        def _axis(_pre, _n):
            return (np.arange(_n) - int(float(_pre) * fs / 1000)) / fs * 1000.0

        time_ms_by_type = {}
        for i in idxs:
            _seg = segments[stims[i]]
            time_ms_by_type[stims[i]] = _axis(pres[i], _seg.shape[1])

        pre  = float(pres[idxs[0]])
        L    = next(iter(segments.values())).shape[1]
        time_ms = _axis(pre, L)
        _shapes = {(float(pres[i]), segments[stims[i]].shape[1]) for i in idxs}
        if len(_shapes) > 1:
            log(f"add-ons: {fname} has stimulus types epoched over different "
                f"windows; add-ons using context.time_ms rather than "
                f"context.time_ms_for(stim_type) will see only "
                f"{stims[idxs[0]]}'s axis")

        tdf = trials_df
        if trials_df is not None and "File" in trials_df.columns:
            tdf = trials_df[trials_df["File"] == fname].reset_index(drop=True)

        prefix = f"{run_prefix}_{fname}" if multi else run_prefix
        contexts.append(AddonContext(
            trials=tdf, segments=segments, fs=fs, unit=unit, time_ms=time_ms,
            time_ms_by_type=time_ms_by_type,
            config=dict(config or {}), results_dir=results_dir,
            figures_dir=figures_dir, bids_prefix=prefix, log=log))
    return contexts


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────
def _load_module_from_path(path):
    mod_name = "mep_cmap_addon_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # runs top-level code of the add-on
    return mod


def discover_addons(dirs, log=print):
    """Scan folders for valid add-on modules.

    Returns a list of dicts: {name, description, version, author, module, path}.
    Modules that fail to import or lack the contract are skipped with a log note,
    never raised. First occurrence of a given ADDON_NAME wins (built-in dirs
    should be listed first).
    """
    found, seen, seen_dirs = [], set(), set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        _rp = os.path.realpath(d)
        if _rp in seen_dirs:          # same folder listed twice → scan once
            continue
        seen_dirs.add(_rp)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            path = os.path.join(d, fn)
            try:
                mod = _load_module_from_path(path)
            except Exception as e:
                log(f"add-ons: skipped {fn} (import error: {e})")
                continue
            name = getattr(mod, "ADDON_NAME", None)
            if not name or not callable(getattr(mod, "run", None)):
                log(f"add-ons: skipped {fn} (missing ADDON_NAME or run())")
                continue
            if name in seen:
                log(f"add-ons: duplicate add-on '{name}' in {fn} ignored")
                continue
            seen.add(name)
            found.append({
                "name":        str(name),
                "description": str(getattr(mod, "ADDON_DESCRIPTION", "")),
                "version":     str(getattr(mod, "ADDON_VERSION", "")),
                "author":      str(getattr(mod, "ADDON_AUTHOR", "")),
                "settings":    list(getattr(mod, "ADDON_SETTINGS", []) or []),
                "scope":       str(getattr(mod, "ADDON_SCOPE", "")),
                "module":      mod,
                "path":        path,
            })
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Safe execution
# ─────────────────────────────────────────────────────────────────────────────
def run_addon(entry, context):
    """Run one discovered add-on on one context.

    Never raises: returns {ok: bool, paths: list[str], error: str|None}. A broken
    add-on therefore cannot crash the caller (or the GUI).
    """
    try:
        paths = entry["module"].run(context) or []
        # Contexts carry Windows' extended-length paths so add-ons can write
        # past 260 characters; what they report back is for people and logs,
        # so it is returned in the ordinary form.
        from .utils import display_path
        return {"ok": True, "paths": [display_path(str(p)) for p in paths],
                "error": None}
    except Exception:
        return {"ok": False, "paths": [], "error": traceback.format_exc()}


# ─────────────────────────────────────────────────────────────────────────────
# Built-in add-ons folder (shipped inside the package)
# ─────────────────────────────────────────────────────────────────────────────
def builtin_addons_dir():
    """Absolute path to the shipped add-ons folder (mep_cmap/add_ons).

    Resolves in a normal install via this module's __file__, and in a frozen
    (PyInstaller) build via sys._MEIPASS as a fallback, so the built-in add-ons
    are found whether the package is on disk or bundled.
    """
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "add_ons")
    if os.path.isdir(here):
        return here
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        frozen = os.path.join(meipass, "mep_cmap", "add_ons")
        if os.path.isdir(frozen):
            return frozen
    return here


def _scope_dirs(scope, user_dir=None):
    """Folders to scan for a given add-on scope ('single_file' | 'group_level')."""
    root = builtin_addons_dir()
    dirs = [os.path.join(root, scope)]
    if user_dir:
        dirs.append(os.path.join(user_dir, scope))
        if scope == "single_file":
            dirs.append(user_dir)   # backward-compat: flat user add-ons = single-file
    return dirs


def discover_all(scope="single_file", user_dir=None, log=print):
    """Discover add-ons for a scope: the built-in <scope>/ folder first, then the
    user folder's <scope>/ subfolder (and, for single_file, its root too, for
    backward-compatibility with flat user add-ons)."""
    return discover_addons(_scope_dirs(scope, user_dir), log=log)


# ─────────────────────────────────────────────────────────────────────────────
# Group-level (second-level) context + loader
# ─────────────────────────────────────────────────────────────────────────────
class GroupAddonContext:
    """Read-only view of the second-level (group) table handed to a group-level
    add-on's run(context). Built from group_level_LME_ready.csv (every included
    trial across sessions/participants, with design columns prepended)."""
    __slots__ = ("group_table", "group_csv_path", "design_columns", "metric_columns",
                 "results_dir", "figures_dir", "config", "bids_prefix", "log")

    def __init__(self, *, group_table, group_csv_path, design_columns, metric_columns,
                 results_dir, config, bids_prefix, log, figures_dir=None):
        self.group_table    = group_table        # pandas.DataFrame
        self.group_csv_path = group_csv_path      # str
        self.design_columns = design_columns      # list[str]  (participant/condition/...)
        self.metric_columns = metric_columns      # list[str]  (numeric measurements)
        self.results_dir    = results_dir         # str  (where to write outputs)
        self.figures_dir    = figures_dir         # str | None
        self.config         = config              # dict
        self.bids_prefix    = bids_prefix         # str
        self.log            = log                 # callable(str)


# Design/identifier columns that Second Level ▸ Group Analysis prepends/carries.
_GROUP_DESIGN_NAMES = {
    "File", "participant_id", "session", "task", "timepoint", "Limb",
    "StimType", "Stim_Label", "Stim_Role", "Segment", "Segment_Overall",
    "Trial", "Outlier_Decision",
}


def load_group_contexts(group_csv_path, config=None, log=print):
    """Build a GroupAddonContext from a group_level_LME_ready.csv.

    Returns a single-element list (mirroring load_contexts) so the GUI can treat
    single- and group-level runs uniformly. Splits columns into design vs metric
    heuristically (known design names + non-numeric = design; numeric = metric);
    the add-on always has the full group_table for its own column selection.
    """
    if pd is None:
        raise RuntimeError("group-level add-ons require pandas.")
    results_dir = os.path.dirname(group_csv_path)
    df = pd.read_csv(group_csv_path)

    design_columns, metric_columns = [], []
    for c in df.columns:
        if c in _GROUP_DESIGN_NAMES or not pd.api.types.is_numeric_dtype(df[c]):
            design_columns.append(c)
        else:
            metric_columns.append(c)

    base = os.path.basename(group_csv_path)
    prefix = base[:-4] if base.lower().endswith(".csv") else base

    ctx = GroupAddonContext(
        group_table=df, group_csv_path=group_csv_path,
        design_columns=design_columns, metric_columns=metric_columns,
        results_dir=results_dir,
        figures_dir=os.path.join(results_dir, "figures"),
        config=dict(config or {}), bids_prefix=prefix, log=log)
    return [ctx]
