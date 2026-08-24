"""
mep_cmap.bidsify_tab
~~~~~~~~~~~~~~~~~~~~
BidsifyTabMixin — the BIDS-ify worklist tab for TMSAnalysisApp.

Model (see bidsify_state.py): shared session defaults ⊕ per-file overrides,
persisted to bidsify_state.json so the worklist is resumable across sessions.

Tab layout, top to bottom:
  • Shared defaults panel — edit the session-constant metadata once; choose the
    output rawdata folder.
  • Status tree — one row per queued file, colour-coded not_started / incomplete
    / ready / converted, sharing the analysis queue's file list.
  • Per-file editor (on double-click) — file-scoped fields foregrounded,
    session defaults shown inherited-but-overridable.
  • Convert ready files — runs the (idempotent) writer on everything Ready.

This mixin assumes the host provides: self.root, self.msg_q, self.notebook,
self._dataset / self._get_or_create_dataset(), self.derivatives_path,
self._rawdata_path, self._parse_bids_from_filename(), self._show_bidsify_preview(),
self._bidsify_done_gui(), self.log(). app.py wires the tab in and adds the
msg_q "bidsify-convert-done" branch that calls self._bidsify_convert_done().
"""

import os
import datetime
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .bids import StudyMetadata
from .preferences import accent_button_kw
from .tooltips import Tooltip
from .bidsify import (DatasetLayout, BidsifyItem, plan_bidsify, execute_plan,
                      conditions_for, recorded_metadata_for)
from . import events_model as _em
from .bidsify_dialog import BidsifyDialog
from .bidsify_state import (BidsifyState, STATUS_LABELS, STATUS_COLOURS,
                            STATUS_NOT_STARTED, STATUS_INCOMPLETE,
                            STATUS_READY, STATUS_CONVERTED)

_ACCENT = "#2196F3"

#: Formats whose stim codes can be read out of the file itself, mapped to a
#: lazy import of the reader that does it.
#:
#: A format that is NOT here falls back to the shared stim marker label, which
#: names one code for a recording that may carry several -- and says nothing
#: about the difference. So the gate is worth widening wherever a reader can
#: actually enumerate the codes, rather than treating .smr as the only case.
#:
#: A reader qualifies by exposing ``list_event_codes(path) -> {channel:
#: {code: count}}``. ``load_config``/``has_config`` are optional.
_SCANNABLE_FORMATS = {
    "spike2_smr": lambda: __import__(
        "mep_cmap.formats.spike2_smr", fromlist=["spike2_smr"]),
    "spike2": lambda: __import__(
        "mep_cmap.formats.spike2", fromlist=["spike2"]),
}

#: The pseudo-channel used when codes are read from the file as a whole rather
#: than from a named event channel.
#:
#: Most formats have no channel to pick: a LabChart .mat, a BrainVision header
#: and an EDF each carry their events in one place. They still HAVE codes --
#: 'Start Task' and 'Trigger', 'S128', 'A' through 'G' -- and before this the
#: dialogue offered none of them, falling back to the shared stim marker label
#: and showing a single code for a recording that had six. Nothing said the
#: list was a fallback, so the missing codes were invisible.
#:
#: Never written to rec.stim_channel: it is a label for the analyst, not a
#: channel name any reader would recognise.
_AUTO_EVENT_CHANNEL = "(events in this file)"


def scan_event_codes(path, io_module, marker_name=""):
    """``{channel: {code: n_events}}`` for any format that can answer.

    A reader with its own enumerator is preferred: it reads headers only, and
    for Spike2 it can distinguish several marker channels. Everything else
    goes through ``extract_stim_times``, which every format implements and
    which returns exactly the codes the ANALYSIS will later find -- so the
    table cannot offer a code the pipeline would not see, nor hide one it
    would.

    Returns ({}, reason) when nothing can be enumerated, so the caller can say
    why rather than showing an empty table.
    """
    fmt = io_module.detect_format(path)
    if fmt in _SCANNABLE_FORMATS:
        mod = _SCANNABLE_FORMATS[fmt]()
        codes = mod.list_event_codes(path) or {}
        cfg = (mod.load_config(path)
               if getattr(mod, "has_config", None) and mod.has_config(path)
               else {})
        # A channel carrying no codes cannot be assigned a parameter set, so
        # offering it in the dropdown offers an empty table. The .smr reader
        # returns such channels (an unused Keyboard channel, say).
        codes = {c: v for c, v in codes.items() if v}
        if codes:
            return codes, cfg, ""
        return {}, cfg, "No event channels with codes were found in this file."

    # The whole-file route. Costs a full read rather than a header sniff,
    # which is why it is second: for a large .smr the dedicated reader above
    # answers from the header alone.
    #
    # Failures are REPORTED, not raised. A format needing first-open
    # configuration (generic TSV) and one the tool cannot open at all both
    # reach here, and "Scan failed: [Errno 2] No generic-TSV config found"
    # tells the analyst nothing they can act on.
    try:
        times = io_module.extract_stim_times(path, marker_name or "A") or {}
    except FileNotFoundError as exc:
        if "config" in str(exc).lower():
            return ({}, {}, "This file needs its columns configured before "
                            "its events can be read. Open it on the First "
                            "Level tab once, then scan again.")
        return {}, {}, f"Could not read this file: {exc}"
    except Exception as exc:                # noqa: BLE001
        return ({}, {},
                f"Could not read events from this file, which the tool reads "
                f"as \u201c{fmt}\u201d: {type(exc).__name__}: {exc}")

    counts = {str(k): len(v or ()) for k, v in times.items() if k is not None}
    if not counts:
        return ({}, {},
                f"No stim events found in this file (read as \u201c{fmt}\u201d).")
    return {_AUTO_EVENT_CHANNEL: counts}, {}, ""

#: BIDS datatype directory names. A recording inside one of these, under a
#: `sub-` or `ses-` parent, is BIDS output rather than a source file. Listed
#: beyond what this tool writes, so a dataset converted by other means, or one
#: holding EEG alongside EMG, is recognised too.
_BIDS_DATATYPE_DIRS = {"emg", "eeg", "ieeg", "meg", "nibs", "beh",
                       "func", "anat", "dwi", "fmap", "perf", "pet", "motion"}


# ── Pure helper (unit-testable) ───────────────────────────────────────────────
def compute_overrides(file_vals: dict, session_vals: dict, defaults: dict) -> dict:
    """
    Build the minimal per-file override set.

      • file-scoped fields   → stored if non-blank
      • session-scoped fields→ stored only if non-blank AND different from the
                               shared default (so files don't just duplicate defaults)
    """
    ov = {}
    for k, v in (file_vals or {}).items():
        if v is not None and str(v).strip() != "":
            ov[k] = v
    for k, v in (session_vals or {}).items():
        if v is None or str(v).strip() == "":
            continue
        if str(v) != str(defaults.get(k, "") or ""):
            ov[k] = v
    return ov


class BidsifyTabMixin:
    # ---- state access ---------------------------------------------------------
    def _bidsify_state_root(self) -> str:
        """
        Where bidsify_state.json lives: the study root (parent of derivatives/,
        sibling to rawdata/), so its relative-path keys line up with the rawdata
        files. Falls back to the derivatives/rawdata folder, then home.
        """
        ds = self._get_or_create_dataset()
        root = ""
        try:
            dd = ds._deriv_dir                      # derivatives/ folder
            if dd and os.path.isabs(dd):
                root = os.path.dirname(dd)          # study root
        except Exception:
            root = ""
        if not root or not os.path.isabs(root):
            root = (getattr(ds, "derivatives_root", "") or self.derivatives_path.get()
                    or self._rawdata_path.get() or os.path.expanduser("~"))
        return root

    def _get_bidsify_state(self) -> BidsifyState:
        root = self._bidsify_state_root()
        st = getattr(self, "_bidsify_state", None)
        if st is None or st.root != root:
            self._bidsify_state = BidsifyState.load_or_create(root)
        return self._bidsify_state

    def _bidsify_nibs_record(self, path: str):
        """This recording's NIBS-BIDS description, for the analysis record.

        Resolved here rather than in the pipeline because finding
        bidsify_state.json means knowing where the study root is, which is a
        question only the application can answer, and because reading it on
        the GUI thread at snapshot time means the record describes the state
        the run started from rather than whatever the analyst edits while it
        is running.

        Returns None when there is no state to read. None and an empty
        description are different: the first says the tool never looked, the
        second says it looked and found nothing, and the log distinguishes
        them.
        """
        if not path:
            return None
        # A folder is not a recording. effective_values() would answer for one
        # anyway -- there is no per-file record to find, so it returns the
        # session defaults -- and those would then be attributed to every file
        # in the folder as though each had been reviewed.
        if not os.path.isfile(path):
            return None
        try:
            return self._get_bidsify_state().nibs_record(path)
        except Exception as _e:             # noqa: BLE001 — never block a run
            self.log(f"   ⚠️  Could not read the BIDS-ify metadata for the "
                     f"analysis record: {_e}")
            return None

    def _bidsify_all_paths(self) -> list:
        """Files shared with the analysis queue (dataset order).

        Excluding the tool's own output. Conversion copies each native file
        into <rawdata>/sourcedata/, so when the output root sits inside the
        folder being scanned -- which is the normal layout when someone points
        it at their study -- the next refresh finds that copy and offers to
        BIDS-ify it again. The list grows by one file per conversion, and the
        duplicate reaches the analysis queue too, where it looks like a second
        recording for the same participant.
        """
        ds = self._dataset
        if ds is None:
            return []
        return [fe.path for fe in ds.files
                if os.path.isfile(fe.path) and not self._bidsify_is_own_output(fe.path)]

    def _bidsify_is_own_output(self, path: str) -> bool:
        """True for anything inside a folder this tool writes.

        Two shapes, because conversion produces two.

        `sourcedata` and `derivatives` are matched anywhere in the path: the
        first holds the native copies made during conversion, the second the
        analysis output, and neither is ever a source recording.

        The converted recordings are matched by BIDS LAYOUT rather than by
        name or extension: a file sitting in a datatype folder whose parent is
        a `sub-` or `ses-` directory. Extension cannot be used, because EDF and
        BDF are input formats this tool legitimately reads -- excluding them
        wholesale would hide real data. Requiring the sub/ses parent is what
        keeps an analyst's own folder called `emg` from disappearing.
        """
        try:
            parts = os.path.normpath(path).split(os.sep)
        except Exception:                       # noqa: BLE001 — odd path
            return False
        lower = [p.lower() for p in parts]
        if {"sourcedata", "derivatives"} & set(lower):
            return True
        # .../sub-001/ses-01/emg/file.edf  or  .../sub-001/emg/file.edf
        if len(lower) >= 3:
            datatype, parent = lower[-2], lower[-3]
            if datatype in _BIDS_DATATYPE_DIRS and (
                    parent.startswith("sub-") or parent.startswith("ses-")):
                return True
        return False

    # ---- tab construction -----------------------------------------------------
    def _build_bidsify_tab(self, parent: tk.Frame):
        from .bids_schema import load_schema
        self._bidsify_schema = load_schema()

        # ── Shared defaults panel ─────────────────────────────────────────────
        top = tk.LabelFrame(parent, text="Shared stimulation metadata  (applies to every file)",
                            padx=8, pady=6)
        top.pack(fill="x", padx=10, pady=(10, 4))

        row1 = tk.Frame(top); row1.pack(fill="x", pady=(0, 4))
        # Numbered, because the order is not optional and nothing else says
        # so: parameter sets must exist before a file's codes can be assigned
        # to them, and a first-time user has no way to infer that from three
        # buttons that look like peers. The status label after each says
        # whether that step has been done.
        tk.Button(row1, text="1 · ✎  Shared defaults…",
                  command=self._bidsify_edit_defaults).pack(side="left")
        # Separate button, not a section of the defaults dialogue: the defaults
        # are one value per field for the whole session, and a parameter set is
        # one of several. Putting them in the same window invites the reading
        # that stimulation intensity is a session constant, which is the exact
        # mistake this replaces.
        tk.Button(row1, text="2 · ⚡  Stimulation parameter sets…",
                  command=self._bidsify_edit_param_sets).pack(side="left", padx=(6, 0))
        self._bidsify_defaults_lbl = tk.Label(row1, text="Not set yet", fg="#888")
        self._bidsify_defaults_lbl.pack(side="left", padx=(10, 0))
        self._bidsify_sets_lbl = tk.Label(row1, text="", fg="#888")
        self._bidsify_sets_lbl.pack(side="left", padx=(10, 0))

        row2 = tk.Frame(top); row2.pack(fill="x", pady=(2, 0))
        tk.Label(row2, text="Output rawdata folder:", width=20, anchor="w").pack(side="left")
        self._bidsify_rawroot_var = tk.StringVar()
        tk.Button(row2, text="Browse…",
                  command=self._bidsify_browse_rawroot).pack(side="right", padx=(4, 0))
        tk.Entry(row2, textvariable=self._bidsify_rawroot_var, state="readonly",
                 fg="#555").pack(side="left", fill="x", expand=True, padx=(4, 4))

        # ── File status tree ──────────────────────────────────────────────────
        mid = tk.LabelFrame(parent,
            text="Files  (double-click to review / edit metadata)", padx=6, pady=4)
        mid.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        bar = tk.Frame(mid); bar.pack(fill="x", pady=(0, 4))
        tk.Button(bar, text="🔄 Refresh", command=self._bidsify_tab_refresh).pack(side="left", padx=(0, 4))
        tk.Button(bar, text="3 · ✎ Review selected…",
                  command=self._bidsify_review_selected).pack(side="left", padx=(0, 4))
        tk.Button(bar, text="✓ Accept defaults for selected",
                  command=self._bidsify_accept_defaults_selected).pack(side="left", padx=(0, 4))
        tk.Button(bar, text="↻ Re-open converted",
                  command=self._bidsify_reopen_selected).pack(side="left", padx=(0, 4))
        tk.Button(bar, text="▶  Convert ready files", command=self._bidsify_convert_ready,
                  **accent_button_kw("green")).pack(side="right", padx=(0, 4))

        # Sits beside Convert, because it changes what Convert DOES. Off by
        # default and never remembered from another study: this is the only
        # part of BIDS-ify that destroys anything, and a destructive setting
        # inherited silently is one nobody chose.
        self._bidsify_move_var = tk.BooleanVar(value=False)
        _mv = tk.Checkbutton(
            bar, text="Move originals (don't copy)",
            variable=self._bidsify_move_var,
            command=self._bidsify_move_toggled)
        _mv.pack(side="right", padx=(0, 10))
        Tooltip(_mv,
                "Off: the native file is COPIED into sourcedata/ and left "
                "where it is.\n"
                "On: the original is DELETED once its copy is in sourcedata/ "
                "and the converted EDF has been read back and verified. A "
                "file that fails to verify keeps its original.\n\n"
                "Saves disk, but the native recording is the one thing that "
                "cannot be regenerated. Only turn this on when sourcedata/ is "
                "the archive you intend to keep and it is backed up.")

        cols = ("status", "sub", "ses", "cond", "modality", "missing", "path")
        tree_wrap = tk.Frame(mid); tree_wrap.pack(fill="both", expand=True)
        self._bidsify_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings",
                                          height=14, selectmode="extended")
        for c, txt, w in [("status", "Status", 110), ("sub", "Subject", 90),
                          ("ses", "Session", 70), ("cond", "Condition/Task", 200),
                          ("modality", "Modality", 80), ("missing", "Missing required", 180),
                          ("path", "Path", 900)]:
            self._bidsify_tree.heading(c, text=txt)
            # No column stretches, and Path is wide enough to overflow.
            #
            # stretch=(c == "path") made Path absorb whatever the other six
            # left over, so the columns totalled exactly the widget width,
            # never more, and the horizontal scrollbar below had nothing to
            # scroll to. A path longer than the column was simply unreadable --
            # and these are files inside a OneDrive tree, so the part that
            # identifies the file is at the end, which is the part that got
            # cut. Same fault as the Stage 1 file queue and the Stage 2
            # assignment table, and fixed the same way.
            self._bidsify_tree.column(c, width=w, stretch=False, minwidth=40)
        vs = ttk.Scrollbar(tree_wrap, orient="vertical", command=self._bidsify_tree.yview)
        hs = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self._bidsify_tree.xview)
        self._bidsify_tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self._bidsify_tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        tree_wrap.grid_rowconfigure(0, weight=1)
        tree_wrap.grid_columnconfigure(0, weight=1)

        # Shift+wheel scrolls sideways, which is what anyone reaches for before
        # dragging a scrollbar. Bound to the tree rather than the window so it
        # does not hijack the wheel elsewhere. Same idiom as the Stage 1 file
        # queue and the Stage 2 table, deliberately: three tables that scroll
        # differently is worse than any one of the behaviours on its own.
        def _bids_hwheel(event):
            _d = event.delta
            if _d:
                self._bidsify_tree.xview_scroll(int(-_d / 120) or
                                                (-1 if _d > 0 else 1), "units")
            return "break"

        self._bidsify_tree.bind("<Shift-MouseWheel>", _bids_hwheel)
        # X11 reports the wheel as buttons 6/7 horizontally. Windows Tk does
        # not know those numbers and REFUSES THE BIND -- "bad button number 6"
        # at construction, which stops the tab being built at all. So it is
        # attempted rather than assumed: a binding that cannot exist on this
        # platform is not an error, it is simply not that platform.
        for _seq, _dir in (("<Button-6>", -1), ("<Button-7>", 1)):
            try:
                self._bidsify_tree.bind(
                    _seq, lambda _e, _d=_dir:
                    self._bidsify_tree.xview_scroll(_d, "units"))
            except tk.TclError:
                pass
        for status, colour in STATUS_COLOURS.items():
            self._bidsify_tree.tag_configure(status, foreground=colour)
        self._bidsify_tree.bind("<Double-1>", lambda _e: self._bidsify_review_selected())

        self._bidsify_counts_var = tk.StringVar(value="No files")
        tk.Label(parent, textvariable=self._bidsify_counts_var, fg="grey",
                 anchor="w").pack(fill="x", padx=10, pady=(0, 4))

        self._bidsify_iid_to_path = {}

    def _bidsify_select_source(self, path: str) -> bool:
        """Select and reveal one recording in the worklist.

        Used when arriving from the Conditions tab: landing on a list of
        fifteen rows and having to find the file just conditioned loses the
        connection between what was done and what to do next.

        Compared with normcase and normpath, because the path arrives from
        another tab and Windows will happily hand over the same file with a
        different case or separator.
        """
        if not hasattr(self, "_bidsify_tree") or not path:
            return False
        want = os.path.normcase(os.path.normpath(path))
        for iid, p in (self._bidsify_iid_to_path or {}).items():
            if os.path.normcase(os.path.normpath(p)) != want:
                continue
            try:
                self._bidsify_tree.selection_set(iid)
                self._bidsify_tree.focus(iid)
                self._bidsify_tree.see(iid)
            except Exception:               # noqa: BLE001 — row may have gone
                return False
            return True
        return False

    # ---- refresh --------------------------------------------------------------
    def _bidsify_tab_refresh(self):
        if not hasattr(self, "_bidsify_tree"):
            return
        state = self._get_bidsify_state()
        schema = self._bidsify_schema

        # sync the small header widgets from state
        self._bidsify_rawroot_var.set(state.rawdata_root or self._rawdata_path.get() or "")
        # Set directly, not through the toggle handler, which would raise the
        # confirmation dialogue on every refresh of a study that has already
        # answered it.
        if hasattr(self, "_bidsify_move_var"):
            self._bidsify_move_var.set(bool(getattr(state, "move_sources",
                                                    False)))
        if state.defaults or state.modality:
            n = len(state.defaults)
            self._bidsify_defaults_lbl.config(
                text=f"{state.modality} · {n} field(s) set · {state.container}+ · marker '{state.marker_name}'",
                fg="#333")
        else:
            self._bidsify_defaults_lbl.config(text="Not set yet", fg="#888")

        # Step 2's own status. "No sets yet" is the state a first-time user is
        # in and cannot otherwise tell from a button that looks the same
        # either way -- and every file's codes will report as unassigned until
        # it changes.
        if hasattr(self, "_bidsify_sets_lbl"):
            _n = len(state.param_sets)
            self._bidsify_sets_lbl.config(
                text=(f"· {_n} set(s): "
                      + ", ".join(s.name for s in state.param_sets[:4])
                      + (" …" if _n > 4 else "")) if _n else "· No sets yet",
                fg="#333" if _n else "#c77")

        self._bidsify_tree.delete(*self._bidsify_tree.get_children())
        self._bidsify_iid_to_path = {}
        paths = self._bidsify_all_paths()
        # One session read per file per refresh, not two.
        self._bidsify_splits_cache = {}
        self._bidsify_meta_cache = {}
        tally = {STATUS_NOT_STARTED: 0, STATUS_INCOMPLETE: 0,
                 STATUS_READY: 0, STATUS_CONVERTED: 0}
        for p in paths:
            parsed = self._parse_bids_from_filename(p)
            # Shown from the same route the conversion uses, so the Subject
            # column cannot say one thing and the output folder another.
            _meta = self._bidsify_metadata_for(p)
            status = state.status(p, schema, self._bidsify_splits_for(p))
            tally[status] = tally.get(status, 0) + 1
            missing = ", ".join(state.missing_required(p, schema)) if status == STATUS_INCOMPLETE else ""
            cond = parsed.get("acq") or parsed.get("task") or ""
            # Named in the list, not just inside the file's own dialogue. A
            # deviation nobody can see without opening every recording is one
            # that resurfaces months later as an unexplained divergence in the
            # group report.
            _dev = state.deviations(p)
            if _dev:
                _n = sum(len(v) for v in _dev.values())
                cond = (cond + "  " if cond else "") + f"⚠ {_n} deviation(s)"
            iid = self._bidsify_tree.insert(
                "", "end",
                values=(STATUS_LABELS[status], _meta.participant_id,
                        _meta.session, cond, state.modality, missing, p),
                tags=(status,))
            self._bidsify_iid_to_path[iid] = p

        # Tallied from the rows rather than recomputed. state.counts() would
        # derive the statuses a second way, without the splits, so the footer
        # could say "ready" about a file the list showed as incomplete.
        counts = tally
        self._bidsify_counts_var.set(
            f"{len(paths)} file(s):  "
            f"{counts[STATUS_NOT_STARTED]} not started · "
            f"{counts[STATUS_INCOMPLETE]} incomplete · "
            f"{counts[STATUS_READY]} ready · "
            f"{counts[STATUS_CONVERTED]} converted")

    def _bidsify_metadata_for(self, path: str):
        """The metadata for one recording: what was recorded, else the filename.

        One route, used by the worklist, the review dialogue and the conversion
        alike, so all three name the same participant.

        The recording's own session wins. The Study Metadata window is the only
        place a participant is entered, and nothing else in the tool holds one,
        so a file whose name says nothing -- `rawdata/Spike/Example Data 1.smr`
        -- has no other source. Parsing the filename alone found no `sub-` and
        converted it to `sub-unknown`, contradicting the sub-333 the analyst
        had typed and the folder their derivatives were already in.

        The filename is the fallback, not the authority: it is right for a
        BIDS-organised study and silent for everything else.
        """
        parsed = self._parse_bids_from_filename(path)
        guess = StudyMetadata(
            participant_id=parsed['participant_id'] or "sub-unknown",
            session=parsed['session'] or "ses-01", task=parsed['task'],
            timepoint=parsed['timepoint'], limb=parsed['limb'],
            measure=parsed['measure'], acq=parsed['acq'])

        cache = getattr(self, "_bidsify_meta_cache", None)
        if cache is None:
            cache = self._bidsify_meta_cache = {}
        if path not in cache:
            deriv = (self.derivatives_path.get()
                     if hasattr(self, "derivatives_path") else "")
            cache[path] = recorded_metadata_for(path, guess, deriv)
        return cache[path] or guess

    def _bidsify_conditions_for(self, path: str) -> list:
        """Conditions assigned to one recording, via its own metadata."""
        return conditions_for(
            path, self._bidsify_metadata_for(path),
            (self.derivatives_path.get()
             if hasattr(self, "derivatives_path") else ""))

    def _bidsify_splits_for(self, path: str) -> list:
        """(code, condition) pairs the Conditions tab created for one recording.

        Cached per refresh: the worklist asks for every file, and each answer
        costs a session JSON read.
        """
        cache = getattr(self, "_bidsify_splits_cache", None)
        if cache is None:
            cache = self._bidsify_splits_cache = {}
        if path not in cache:
            try:
                cache[path] = _em.split_codes(
                    self._bidsify_conditions_for(path))
            except Exception:               # noqa: BLE001 — no conditions
                cache[path] = []
        return cache[path]

    def _bidsify_selected_paths(self) -> list:
        return [self._bidsify_iid_to_path[i]
                for i in self._bidsify_tree.selection()
                if i in self._bidsify_iid_to_path]

    # ---- shared defaults ------------------------------------------------------
    def _bidsify_edit_defaults(self):
        state = self._get_bidsify_state()
        defaults = {
            "modality": state.modality,
            "container": state.container,
            "powerline_hz": state.powerline_hz,
            "marker_name": state.marker_name,
            "sidecar_values": state.defaults,
        }
        BidsifyDialog(self.root, schema=self._bidsify_schema,
                      on_complete=self._bidsify_defaults_saved, defaults=defaults)

    def _bidsify_defaults_saved(self, result):
        if result is None:
            return
        state = self._get_bidsify_state()
        # keep only session-scoped values as shared defaults; file-scoped values a
        # user happened to type here would wrongly apply to all files.
        session_keys = {f.key for f in self._bidsify_schema.fields_for(
            result.modality, scope="session")}
        shared = {k: v for k, v in result.sidecar_values.items()
                  if k in session_keys or k == "StimulationModality"}
        state.set_defaults(result.modality, shared,
                           result.container, result.powerline_hz, result.marker_name)
        state.save()
        self._bidsify_tab_refresh()

    def _bidsify_edit_param_sets(self):
        """Session-level stimulation parameter sets."""
        from .stim_params_dialog import StimParamSetsDialog
        state = self._get_bidsify_state()
        StimParamSetsDialog(self.root, schema=self._bidsify_schema,
                            sets=state.param_sets,
                            on_complete=self._bidsify_param_sets_saved)

    def _bidsify_param_sets_saved(self, sets):
        if sets is None:
            return
        state = self._get_bidsify_state()
        state.param_sets = list(sets)
        state.save()
        self._bidsify_tab_refresh()

    def _bidsify_reopen_selected(self):
        """Put converted files back into the worklist so they can be corrected.

        Conversion is not a one-way door: a wrong intensity or a mis-assigned
        stim code is only discovered after looking at the output, and without
        this the only route back was deleting the file's state and re-entering
        everything.
        """
        paths = self._bidsify_selected_paths()
        if not paths:
            messagebox.showinfo("BIDS-ify", "Select one or more converted files "
                                "first.", parent=self.root)
            return
        state = self._get_bidsify_state()
        done = [p for p in paths if state.reopen_for_edit(p)]
        if not done:
            messagebox.showinfo("BIDS-ify", "None of the selected files have "
                                "been converted.", parent=self.root)
            return
        state.save()
        self._bidsify_tab_refresh()
        self.log(f"BIDS-ify: re-opened {len(done)} converted file(s) for editing. "
                 f"Converting again overwrites the previous output.")
        messagebox.showinfo(
            "BIDS-ify",
            f"{len(done)} file(s) re-opened.\n\nEdit them as usual, then "
            f"Convert ready files. The existing output is overwritten in "
            f"place; nothing is deleted first.", parent=self.root)

    def _bidsify_browse_rawroot(self):
        state = self._get_bidsify_state()
        init = state.rawdata_root or self._rawdata_path.get() or os.path.expanduser("~")
        d = filedialog.askdirectory(title="Choose the BIDS rawdata output folder",
                                    initialdir=init)
        if d:
            state.rawdata_root = d
            state.save()
            self._bidsify_rawroot_var.set(d)

    # ---- per-file editor ------------------------------------------------------
    def _bidsify_review_selected(self):
        paths = self._bidsify_selected_paths()
        if not paths:
            messagebox.showinfo("BIDS-ify", "Select a file to review.", parent=self.root)
            return
        self._bidsify_open_file_editor(paths[0])

    def _bidsify_accept_defaults_selected(self):
        """Mark selected files reviewed with no overrides (inherit defaults as-is)."""
        paths = self._bidsify_selected_paths()
        if not paths:
            return
        state = self._get_bidsify_state()
        for p in paths:
            state.set_overrides(p, {}, reviewed=True)
        state.save()
        self._bidsify_tab_refresh()

    def _bidsify_open_file_editor(self, path):
        state = self._get_bidsify_state()
        schema = self._bidsify_schema
        modality = state.modality
        rec = state.record_for(path, create=False)
        existing = dict(rec.overrides) if rec else {}

        win = tk.Toplevel(self.root)
        win.title(f"Review — {os.path.basename(path)}")
        win.configure(bg="#f5f5f5")
        win.grab_set()

        tk.Label(win, text=os.path.basename(path), bg="#f5f5f5",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        tk.Label(win, text=f"Modality: {modality}   (set in shared defaults)",
                 bg="#f5f5f5", fg="#777").pack(anchor="w", padx=12, pady=(0, 6))

        file_vars, session_vars = {}, {}

        def _add_field(parent, fld, value, r):
            lbl = f"{fld.key}{' *' if fld.level == 'required' else ''}" \
                  + (f"  [{fld.units}]" if fld.units else "")
            tk.Label(parent, text=lbl, bg=parent["bg"], anchor="w").grid(
                row=r, column=0, sticky="w", padx=(4, 8), pady=1)
            var = tk.StringVar(value=value or "")
            if fld.enum:
                w = ttk.Combobox(parent, textvariable=var, values=[""] + list(fld.enum),
                                 state="readonly")
            else:
                w = tk.Entry(parent, textvariable=var)
            w.grid(row=r, column=1, sticky="ew", padx=(0, 4), pady=1)
            parent.columnconfigure(1, weight=1)
            return var

        # This recording (file-scoped) — foregrounded
        f1 = tk.LabelFrame(win, text="This recording", bg="#f5f5f5", padx=8, pady=6)
        f1.pack(fill="x", padx=12, pady=4)
        for r, fld in enumerate(schema.fields_for(modality, scope="file")):
            file_vars[fld.key] = _add_field(f1, fld, existing.get(fld.key, ""), r)

        # ── Stim events (this file) — scan & pick ─────────────────────────────
        fe = tk.LabelFrame(win, text="Stim events (this file)", bg="#f5f5f5",
                           padx=8, pady=6)
        fe.pack(fill="x", padx=12, pady=4)
        # Conditions for THIS recording, read once. They supply the stim codes
        # when the file cannot be scanned, and the splits within them.
        _conds = self._bidsify_conditions_for(path)
        _scan = {"cache": {}, "traced": False,
                 "chan": tk.StringVar(value=(rec.stim_channel if rec else "")),
                 "codes": list(rec.marker_names if rec else []),
                 "code_vars": {},
                 # Session-level sets, and this file's assignment of codes to
                 # them. Read once here rather than per redraw so the dropdown
                 # list cannot change halfway through editing.
                 "param_sets": list(state.param_sets),
                 "code_sets": dict(rec.code_sets) if rec else {},
                 # (code, condition) pairs for codes the Conditions tab split.
                 # Read from the recording's own session, so the review
                 # dialogue offers a parameter set per half without BIDS-ify
                 # storing a second copy of the grouping.
                 "splits": _em.split_codes(_conds),
                 "conditions": _conds,
                 "set_vars": {}}
        _scan_info = tk.Label(fe, justify="left", bg="#f5f5f5", fg="#777", wraplength=560,
            text=("Click \u201cScan file\u201d to see the stim markers in this recording, "
                  "then pick your stim channel and tick the codes that are your stimuli. "
                  "Leave unset to use the shared Stim marker label."))
        _scan_info.pack(anchor="w")
        if rec and rec.marker_names:
            _scan_info.config(fg="#333",
                text=("Saved for this file \u2014 channel \u201c%s\u201d, codes: %s.  "
                      "Click \u201cScan file\u201d to change." %
                      (rec.stim_channel, ", ".join(rec.marker_names))))
        _chan_row  = tk.Frame(fe, bg="#f5f5f5")
        _codes_row = tk.Frame(fe, bg="#f5f5f5")

        def _codes_for_assignment():
            """``{code: n_events}`` from the best source this file has.

            The assignment table used to be built only from a Spike2 scan, so a
            LabChart .mat or a BrainVision file reached the "scanning supports
            .smr" message and then had nowhere to assign a parameter set at
            all. The codes are knowable without scanning: a recording that has
            been through the Conditions tab has already named its stim types,
            and a file relying on the shared marker label has named them there.
            """
            scanned = _scan["cache"].get(_scan["chan"].get(), {})
            if scanned:
                return dict(scanned)
            counts = {}
            for _r in (_scan.get("conditions") or []):
                counts[_r.stim_type] = (counts.get(_r.stim_type, 0)
                                        + len(_r.trials or ()))
            if counts:
                return counts
            names = list((rec.marker_names if rec else None) or [])
            if not names and state.marker_name:
                names = [m.strip() for m in state.marker_name.split(",")
                         if m.strip()]
            return {n: 0 for n in names}

        def _render_codes(*_a):
            for w in _codes_row.winfo_children():
                w.destroy()
            _scan["code_vars"] = {}
            codes = _codes_for_assignment()
            if not codes:
                tk.Label(_codes_row, text="(no stim codes known for this file "
                                          "— scan it, or set a shared Stim "
                                          "marker label)",
                         bg="#f5f5f5", fg="#999").pack(anchor="w")
                return
            # A split with an unnamed half cannot be described: the group has
            # nothing to be referenced by. Said here rather than silently
            # dropping it, which is what made it invisible.
            _unnamed = _em.unnamed_splits(_scan.get("conditions") or [])
            if _unnamed:
                tk.Label(_codes_row, bg="#f5f5f5", fg="#b00020", justify="left",
                         wraplength=540,
                         text=("Code(s) %s are split into groups and at least "
                               "one group has no name. Name every condition in "
                               "the Conditions tab before assigning protocols."
                               % ", ".join(_unnamed))).pack(anchor="w")
            tk.Label(_codes_row, text="Stim codes — tick your stimuli, "
                                      "then say which protocol each one was:",
                     bg="#f5f5f5").pack(anchor="w")
            grid = tk.Frame(_codes_row, bg="#f5f5f5"); grid.pack(anchor="w", pady=(2, 0))

            # One row per code rather than a grid of tickboxes.
            #
            # A file can contain more than one protocol -- a peripheral M-wave
            # on one code and a TMS MEP on another -- and a single intensity
            # panel for the whole recording could describe neither. The set
            # named here becomes this code's nibs_event_id in *_nibs.tsv.
            _set_names = [s.name for s in (_scan.get("param_sets") or [])]
            # Codes the analyst has split into conditions get a row per half.
            #
            # A split is the case where a condition changes the STIMULUS rather
            # than only what the trial meant: half of 'A' at 100 mA and half at
            # 150 mA is two protocols, and the spec is explicit that a
            # recruitment curve at five intensities is five rows of _nibs.tsv.
            # One dropdown per code cannot say that, so a split code is offered
            # once per condition instead.
            _splits = {}
            for _c, _cond in _scan.get("splits") or []:
                _splits.setdefault(_c, []).append(_cond)
            if not _set_names:
                tk.Label(grid, bg="#f5f5f5", fg="#b00020", justify="left",
                         wraplength=520,
                         text=("No stimulation parameter sets defined yet — "
                               "the codes below cannot be described until "
                               "there is at least one. Use “＋ New set…” "
                               "just below.")).grid(row=0, column=0,
                                                    columnspan=3, sticky="w")
            tk.Label(grid, text="Code", bg="#f5f5f5", fg="#666").grid(
                row=1, column=0, sticky="w", padx=(0, 8))
            tk.Label(grid, text="Events", bg="#f5f5f5", fg="#666").grid(
                row=1, column=1, sticky="w", padx=(0, 8))
            tk.Label(grid, text="Parameter set", bg="#f5f5f5", fg="#666").grid(
                row=1, column=2, sticky="w")

            for i, (code, n) in enumerate(sorted(codes.items())):
                r = i + 2
                # Ticked by default when nothing was saved for this file: a
                # code that came from the conditions is one the analyst has
                # already said is a stimulus, so making them tick it again is
                # asking the same question twice.
                _saved = _scan["codes"]
                v = tk.BooleanVar(value=(code in _saved) if _saved else True)
                _scan["code_vars"][code] = v
                tk.Checkbutton(grid, text=str(code), variable=v,
                               bg="#f5f5f5").grid(row=r, column=0, sticky="w")
                tk.Label(grid, text=("(%d)" % n) if n else "",
                         bg="#f5f5f5", fg="#666").grid(
                    row=r, column=1, sticky="w", padx=(0, 8))

                _conds = _splits.get(code) or []
                if not _conds:
                    sv = tk.StringVar(
                        value=(_scan.get("code_sets") or {}).get(code, ""))
                    _scan["set_vars"][code] = sv
                    # Resolved when the trace fires, not now: _render_devs is
                    # defined below and this runs before it exists.
                    sv.trace_add("write", lambda *_a: _render_devs())
                    ttk.Combobox(grid, textvariable=sv,
                                 values=[""] + _set_names, state="readonly",
                                 width=24).grid(row=r, column=2, sticky="w",
                                                pady=1)
                    continue

                # Split: one dropdown per condition, keyed by the pair. The
                # bare code keeps no assignment of its own -- an assignment on
                # the whole code would silently override one of the halves.
                _sub = tk.Frame(grid, bg="#f5f5f5")
                _sub.grid(row=r, column=2, sticky="w", pady=1)
                for j, _cond in enumerate(_conds):
                    _key = _em.pair_key(code, _cond)
                    sv = tk.StringVar(
                        value=(_scan.get("code_sets") or {}).get(_key, ""))
                    _scan["set_vars"][_key] = sv
                    sv.trace_add("write", lambda *_a: _render_devs())
                    tk.Label(_sub, text=f"{_cond}:", bg="#f5f5f5",
                             fg="#444").grid(row=j, column=0, sticky="e",
                                             padx=(0, 4))
                    ttk.Combobox(_sub, textvariable=sv,
                                 values=[""] + _set_names, state="readonly",
                                 width=20).grid(row=j, column=1, sticky="w",
                                                pady=1)

        def _do_scan():
            from . import io as _io
            _scan_info.config(fg="#777", text="Scanning\u2026")
            _scan_info.update_idletasks()
            try:
                _scan["cache"], cfg, _why_not = scan_event_codes(
                    path, _io, state.marker_name)
            except Exception as exc:
                _scan_info.config(fg="#d9534f",
                                  text="Scan failed: %s" % exc)
                return
            if not _scan["cache"]:
                _scan_info.config(fg="#d9534f", text=_why_not or
                                  "No stim events found in this file.")
                return
            chans = list(_scan["cache"].keys())
            if _scan["chan"].get() not in chans:
                best = (cfg.get("stim_channel") if cfg.get("stim_channel") in chans
                        else max(chans, key=lambda c: len(_scan["cache"][c])))
                _scan["chan"].set(best)
            _scan_info.config(fg="#333",
                text=("Pick the stim channel, then tick the codes that are your stimuli "
                      "(counts in brackets). Saved for THIS file only."))
            for w in _chan_row.winfo_children():
                w.destroy()
            # No picker where there is nothing to pick: a dropdown holding one
            # pseudo-entry invites the analyst to choose a channel this format
            # does not have.
            if chans == [_AUTO_EVENT_CHANNEL]:
                _scan_info.config(
                    fg="#333",
                    text=("Codes read from the file (counts in brackets). "
                          "This format keeps its events in one place, so "
                          "there is no stim channel to choose. Saved for "
                          "THIS file only."))
                _chan_row.pack_forget()
            else:
                tk.Label(_chan_row, text="Stim channel:", bg="#f5f5f5").pack(side="left")
                ttk.Combobox(_chan_row, textvariable=_scan["chan"], values=chans,
                             state="readonly", width=18).pack(side="left", padx=(4, 8))
                _chan_row.pack(anchor="w", pady=(4, 2))
            _codes_row.pack(anchor="w")
            if not _scan["traced"]:
                _scan["chan"].trace_add("write", _render_codes)
                _scan["traced"] = True
            _render_codes()

        # Offer Scan file only where scanning can work, and say where the codes
        # came from otherwise.
        #
        # The button used to be shown for every format and answer, for anything
        # but Spike2, with a red "scanning supports .smr" message -- an error
        # about a button the analyst was invited to press, for a situation they
        # cannot change and which is not a fault. Worse, it advised falling back
        # to the shared marker label even when the recording's own conditions
        # already named the codes.
        try:
            from . import io as _io0
            _fmt0 = _io0.detect_format(path)
        except Exception:                   # noqa: BLE001 — treat as scannable
            _fmt0 = "spike2_smr"
        _scan_btn = tk.Button(fe, text="\U0001F50D  Scan file", command=_do_scan)
        _scan_btn.pack(anchor="w", pady=(4, 0))
        if _fmt0 not in _SCANNABLE_FORMATS:
            # Offered for every format now: extract_stim_times answers for all
            # of them, and it returns exactly the codes the analysis will find.
            # Withholding the button was what left a LabChart .mat showing one
            # code from the shared marker label when the file held two.
            _why = ("the conditions assigned to this recording"
                    if _conds else "the shared Stim marker label")
            _scan_info.config(
                fg="#777",
                text=("Stim codes below come from %s. Click \u201cScan "
                      "file\u201d to read them from the recording itself "
                      "instead." % _why))

        # Show the assignment table straight away, without waiting for a scan.
        #
        # Scanning only works on Spike2 files, and it used to be the only route
        # to this table -- so a LabChart .mat or a BrainVision recording got the
        # "scanning supports .smr" message and then had nowhere to assign a
        # parameter set at all. The codes are knowable from the conditions or
        # from the shared marker label, and where they are, the table belongs
        # on screen from the outset.
        if _codes_for_assignment():
            _codes_row.pack(anchor="w", pady=(6, 0))
            _render_codes()

        # Create a set without leaving the dialogue.
        #
        # Sets had to exist before a file could be reviewed, but nothing said
        # so and the only route to making one was to close this window, find
        # a button on the tab behind it, and come back. Worse, the numbers
        # being entered are ones you know while looking at a recording, not
        # before you have opened any.
        _sets_row = tk.Frame(fe, bg="#f5f5f5")
        _sets_row.pack(anchor="w", pady=(6, 0))

        def _new_set():
            from .stim_params_dialog import StimParamSetsDialog

            def _done(sets):
                if sets is None:
                    return
                state.param_sets = list(sets)
                state.save()
                # The dropdowns are built from this list, so it has to be the
                # one they read, not a copy taken when the dialogue opened.
                _scan["param_sets"] = list(sets)
                _render_codes()
                _render_devs()
                self._bidsify_tab_refresh()
            StimParamSetsDialog(win, schema=schema,
                                sets=state.param_sets, on_complete=_done)

        tk.Button(_sets_row, text="＋  New set…", command=_new_set).pack(side="left")
        tk.Label(_sets_row, bg="#f5f5f5", fg="#777",
                 text="  (sets are shared by every file in the study)").pack(
                     side="left")

        # ── Deviations for this recording ────────────────────────────────────
        #
        # Keyed by SET, not by code, because that is how a deviation is stored:
        # *_nibs.tsv carries one row per set, so two codes sharing a set share
        # one definition. A button per code row would suggest otherwise and
        # let the same value be edited from two places.
        fdev = tk.LabelFrame(
            win, text="Deviations for this recording  (differs from the protocol)",
            bg="#f5f5f5", padx=8, pady=6)
        fdev.pack(fill="x", padx=12, pady=4)
        _dev_body = tk.Frame(fdev, bg="#f5f5f5")
        _dev_body.pack(fill="x")
        # Working copy. Written to the record only on Save, so Cancel really
        # cancels.
        _devs = {k: dict(v) for k, v in
                 ((rec.set_overrides if rec else None) or {}).items()}

        def _sets_assigned_now():
            """Set names this file's ticked codes currently reference."""
            out = []
            for _key, _var in _scan.get("set_vars", {}).items():
                _code = _key.split("\u00b7", 1)[0]
                _tick = _scan.get("code_vars", {}).get(_code)
                if _tick is not None and not _tick.get():
                    continue
                _name = _var.get()
                if _name and _name not in out:
                    out.append(_name)
            return out

        def _edit_dev(set_name):
            """Adjust one set's values for this recording only."""
            base = next((s for s in (_scan.get("param_sets") or [])
                         if s.name == set_name), None)
            if base is None:
                return
            sub = tk.Toplevel(win)
            sub.title(f"{set_name} — this recording only")
            sub.configure(bg="#f5f5f5")
            sub.grab_set()
            tk.Label(sub, bg="#f5f5f5", fg="#555", justify="left",
                     wraplength=520,
                     text=("Pre-filled with the protocol's values. Change one "
                           "and it is written for THIS recording only; the "
                           "set itself is unchanged and every other file "
                           "keeps the protocol value.\n"
                           "Anything left as the protocol says is not "
                           "recorded as a deviation. Clearing a field returns "
                           "it to the protocol.")).pack(
                               anchor="w", padx=12, pady=(10, 6))
            body = tk.Frame(sub, bg="#f5f5f5")
            body.pack(fill="both", expand=True, padx=12)
            cur = dict(_devs.get(set_name) or {})
            vars_ = {}
            # Parameter-set scope only. A session-scope field belongs to the
            # shared defaults and a file-scope one to the panel above; offering
            # them here would give the same field two editors that disagree.
            for r, fld in enumerate(schema.fields_for(modality,
                                                      scope="parameter_set")):
                proto = base.values.get(fld.key, "")
                tk.Label(body, bg="#f5f5f5", anchor="w",
                         text=f"{fld.key}" + (f"  [{fld.units}]"
                                              if fld.units else "")).grid(
                    row=r, column=0, sticky="w", padx=(0, 8), pady=1)
                tk.Label(body, bg="#f5f5f5", fg="#777", anchor="w",
                         text=(f"protocol: {proto}" if proto not in (None, "")
                               else "protocol: —")).grid(
                    row=r, column=1, sticky="w", padx=(0, 8))
                # Pre-filled with the deviation if there is one, else the
                # protocol's own value. Editing a populated form is the task;
                # a column of blanks makes the analyst retype the value they
                # are NOT changing in order to change the one they are.
                #
                # Safe because _ok() compares against the protocol: a field
                # left at the pre-filled value is not stored, so pre-filling
                # cannot manufacture a deviation.
                v = tk.StringVar(value=str(cur.get(fld.key,
                                                   proto if proto is not None
                                                   else "")))
                vars_[fld.key] = v
                if fld.enum:
                    ttk.Combobox(body, textvariable=v,
                                 values=[""] + list(fld.enum),
                                 state="readonly", width=22).grid(
                        row=r, column=2, sticky="ew", pady=1)
                else:
                    tk.Entry(body, textvariable=v, width=24).grid(
                        row=r, column=2, sticky="ew", pady=1)
            body.columnconfigure(2, weight=1)

            def _ok():
                # Only what actually DIFFERS is kept. Storing a value equal to
                # the protocol would list the recording as deviating when it
                # did not, and every such false entry makes the real ones
                # easier to dismiss.
                keep = {}
                for k, v in vars_.items():
                    val = v.get().strip()
                    if val and str(base.values.get(k, "")) != val:
                        keep[k] = val
                if keep:
                    _devs[set_name] = keep
                else:
                    _devs.pop(set_name, None)
                _render_devs()
                sub.destroy()

            row = tk.Frame(sub, bg="#f5f5f5")
            row.pack(fill="x", padx=12, pady=10)
            tk.Button(row, text="Cancel", command=sub.destroy).pack(side="right")
            tk.Button(row, text="Apply", **accent_button_kw("blue"),
                      command=_ok).pack(side="right", padx=(0, 8))
            tk.Button(row, text="Clear deviations",
                      command=lambda: (_devs.pop(set_name, None),
                                       _render_devs(), sub.destroy())
                      ).pack(side="left")
            sub.update_idletasks()
            sub.minsize(520, 260)

        def _render_devs(*_a):
            for w in _dev_body.winfo_children():
                w.destroy()
            names = _sets_assigned_now()
            if not names:
                tk.Label(_dev_body, bg="#f5f5f5", fg="#999", justify="left",
                         wraplength=540,
                         text=("Assign a parameter set to a code above, then "
                               "adjust it here if this recording differed "
                               "from the protocol.")).pack(anchor="w")
                return
            for i, name in enumerate(names):
                r = tk.Frame(_dev_body, bg="#f5f5f5")
                r.pack(fill="x", pady=1)
                tk.Label(r, text=name, bg="#f5f5f5", width=18,
                         anchor="w").pack(side="left")
                tk.Button(r, text="Adjust…",
                          command=lambda n=name: _edit_dev(n)).pack(side="left")
                diff = _devs.get(name) or {}
                base = next((s for s in (_scan.get("param_sets") or [])
                             if s.name == name), None)
                if diff and base is not None:
                    txt = ", ".join(
                        f"{k}: {base.values.get(k, '—')} → {v}"
                        for k, v in sorted(diff.items()))
                    tk.Label(r, text=txt, bg="#f5f5f5", fg="#b8860b",
                             anchor="w", wraplength=380, justify="left").pack(
                                 side="left", padx=(8, 0))
                else:
                    tk.Label(r, text="as the protocol", bg="#f5f5f5",
                             fg="#999").pack(side="left", padx=(8, 0))

        _render_devs()

        # Inherited defaults (session-scoped) — collapsible override section
        show_over = tk.BooleanVar(value=False)
        f2 = tk.LabelFrame(win, text="Shared defaults (inherited — override only if this file differs)",
                           bg="#f5f5f5", padx=8, pady=6)
        holder = tk.Frame(f2, bg="#f5f5f5")

        def _toggle():
            (holder.pack(fill="x") if show_over.get() else holder.pack_forget())
        tk.Checkbutton(f2, text="Show / override inherited fields", variable=show_over,
                       bg="#f5f5f5", command=_toggle).pack(anchor="w")
        for r, fld in enumerate(schema.fields_for(modality, scope="session")):
            inherited = existing.get(fld.key, state.defaults.get(fld.key, ""))
            session_vars[fld.key] = _add_field(holder, fld, inherited, r)
        f2.pack(fill="x", padx=12, pady=4)

        status_lbl = tk.Label(win, text="", bg="#f5f5f5", fg="#d9534f", wraplength=520,
                              justify="left")
        status_lbl.pack(anchor="w", padx=12, pady=(2, 0))

        def _collect():
            fv = {k: v.get() for k, v in file_vars.items()}
            sv = {k: v.get() for k, v in session_vars.items()}
            return compute_overrides(fv, sv, state.defaults)

        def _validate():
            ov = _collect()
            merged = dict(state.defaults); merged.update(ov)
            merged["StimulationModality"] = modality
            vr = schema.validate(merged, modality=modality)
            # Unassigned codes block Ready just as a missing required field
            # does, so Validate has to say so -- otherwise the dialogue reports
            # "Ready" and the worklist then reports "Incomplete", with nothing
            # explaining the disagreement.
            _unassigned = []
            if _scan["code_vars"]:
                _splits_by_code = {}
                for _c, _cond in _scan.get("splits") or []:
                    _splits_by_code.setdefault(_c, []).append(_cond)
                for _c, _v in _scan["code_vars"].items():
                    if not _v.get():
                        continue
                    _conds = _splits_by_code.get(_c) or []
                    # A split code needs every half assigned: one half without
                    # a set writes events that reference nothing.
                    _keys = ([_em.pair_key(_c, _x) for _x in _conds]
                             if _conds else [_c])
                    for _k in _keys:
                        _var = _scan["set_vars"].get(_k)
                        if not (_var and _var.get()):
                            _unassigned.append(
                                _k.replace("\u00b7", " / ") if _conds else _c)
            if not vr.ok:
                status_lbl.config(
                    text=("Missing required: "
                          + ", ".join(e.split(" ", 1)[0] for e in vr.errors)),
                    fg="#d9534f")
            elif _unassigned:
                status_lbl.config(
                    text=("No parameter set for code(s): "
                          + ", ".join(sorted(_unassigned))),
                    fg="#d9534f")
            else:
                status_lbl.config(text="✓ Ready", fg="#5cb85c")

        def _save():
            state.set_overrides(path, _collect(), reviewed=True)
            _r = state.record_for(path, create=True)
            if _scan["code_vars"]:          # user scanned this session -> apply pick
                picked = [c for c, v in _scan["code_vars"].items() if v.get()]
                _r.marker_names = picked
                # The pseudo-channel is a label, not a channel any reader
                # knows. Saving it would hand extract_stim_times a name that
                # resolves to nothing on the next run.
                _chan = _scan["chan"].get()
                _r.stim_channel = ("" if (not picked
                                          or _chan == _AUTO_EVENT_CHANNEL)
                                   else _chan)
                # Only ticked codes carry an assignment. An unticked code is
                # not a stimulus, so a set left on it from an earlier pass
                # would describe stimulation that is not being written.
                #
                # Pair keys are kept against their CODE being ticked, not
                # against the key itself: 'A·100mA' belongs to code A, and
                # unticking A must drop both halves rather than leaving one
                # behind to be written with no events referencing it.
                _picked = set(picked)
                _cs = {}
                for _key, _var in _scan["set_vars"].items():
                    _val = _var.get()
                    if not _val:
                        continue
                    _code = _key.split("\u00b7", 1)[0]
                    if _code in _picked:
                        _cs[_key] = _val
                _r.code_sets = _cs
            # Deviations, kept only for sets this file still references. One
            # left behind on a set the analyst has since unassigned would
            # describe stimulation that is not being written, and would show
            # up later as an unexplained divergence in the group report.
            _live = set(_r.code_sets.values()) if _r.code_sets else set()
            _r.set_overrides = {k: dict(v) for k, v in _devs.items()
                                if v and (not _live or k in _live)}
            state.save()
            self._bidsify_tab_refresh()
            win.destroy()

        btns = tk.Frame(win, bg="#f5f5f5"); btns.pack(fill="x", padx=12, pady=10)
        tk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
        tk.Button(btns, text="Save", **accent_button_kw("blue"),
                  command=_save).pack(side="right", padx=(0, 8))
        tk.Button(btns, text="Validate", command=_validate).pack(side="right", padx=(0, 8))

        win.update_idletasks()
        win.minsize(560, 400)

    # ---- conversion -----------------------------------------------------------
    def _bidsify_marker_names_for(self, state, path):
        """Per-file scan-and-pick codes take priority; else the shared marker
        label (comma-separated codes or a channel name)."""
        rec = state.record_for(path, create=False)
        if rec and getattr(rec, "marker_names", None):
            return list(rec.marker_names)
        if state.marker_name:
            return [m.strip() for m in state.marker_name.split(",") if m.strip()]
        return None

    def _bidsify_convert_ready(self):
        state = self._get_bidsify_state()
        schema = self._bidsify_schema
        paths = self._bidsify_all_paths()
        ready = state.ready_paths(paths, schema, self._bidsify_splits_for)
        if not ready:
            messagebox.showinfo(
                "BIDS-ify",
                "No files are marked Ready. Review files (double-click) so they "
                "validate, then convert.", parent=self.root)
            return

        rawroot = state.rawdata_root or self._rawdata_path.get()
        if not rawroot:
            rawroot = filedialog.askdirectory(
                title="Choose the BIDS rawdata output folder")
            if not rawroot:
                return
            state.rawdata_root = rawroot
            state.save()

        from .bids import make_bids_prefix as _make_bids_prefix
        from pathlib import Path
        items = []
        _dev_notes = []
        for p in ready:
            # The shared route, so the conversion writes to the same
            # participant the worklist and the review dialogue showed.
            meta = self._bidsify_metadata_for(p)
            prefix = _make_bids_prefix(meta.bids_prefix(), Path(p).stem)
            _rec = state.record_for(p, create=False)
            items.append(BidsifyItem(
                source_path=p, metadata=meta, modality=state.modality,
                sidecar_values=state.effective_values(p),
                marker_names=self._bidsify_marker_names_for(state, p),
                stim_channel=(_rec.stim_channel if _rec else None),
                # Sets are session-level, the assignment is this file's. Both
                # are needed: the sets become the rows of *_nibs.tsv and the
                # assignment is what lets *_events.tsv reference them.
                # effective_sets, not state.param_sets: a recording that
                # deviated from the protocol publishes the value it was
                # actually delivered at. Passing the session sets here would
                # write the protocol's number into this file's *_nibs.tsv.
                param_sets=state.effective_sets(p),
                code_sets=(dict(_rec.code_sets) if _rec else {}),
                # From the recording's own session, so a file grouped in the
                # Conditions tab carries that grouping into the events file,
                # and one that was never grouped writes its codes as recorded.
                condition_rows=conditions_for(
                    p, meta,
                    (self.derivatives_path.get()
                     if hasattr(self, "derivatives_path") else "")),
                prefix_override=prefix))

            # Named in the preview, not merely applied. A deviation the
            # analyst cannot see before converting is one they will not
            # remember afterwards, and it will surface later as an unexplained
            # divergence in the group-level report.
            for _set_name, _diff in state.deviations(p).items():
                for _field, (_was, _now) in _diff.items():
                    _dev_notes.append(
                        f"{os.path.basename(p)}: {_set_name}.{_field} "
                        f"{_was!r} → {_now!r} for this recording")

        ds_name = (os.path.basename(os.path.dirname(rawroot.rstrip("/\\")))
                   or "MEP-CMAP dataset")
        layout = DatasetLayout(rawdata_root=rawroot, dataset_name=ds_name)
        plan = plan_bidsify(items, layout, container=state.container,
                            powerline_hz=state.powerline_hz,
                            move_sources=bool(getattr(state, "move_sources",
                                                      False)))
        if _dev_notes:
            plan.warnings.append(
                f"{len(_dev_notes)} per-recording deviation(s) from the "
                f"stimulation parameter sets will be written as delivered:")
            plan.warnings.extend(f"  {n}" for n in _dev_notes)
        if not self._show_bidsify_preview(plan):
            return

        self.log(f"BIDS-ify: converting {len(plan.files)} ready file(s) → {rawroot}")

        def _worker():
            try:
                results = execute_plan(plan, log=lambda m: self.msg_q.put(("log", m)))
            except Exception as exc:
                self.msg_q.put(("log", f"BIDS-ify failed: {exc}"))
                results = []
            self.msg_q.put(("bidsify-convert-done", results))
        threading.Thread(target=_worker, daemon=True).start()

    def _bidsify_move_toggled(self):
        """Confirm before arming the only destructive part of BIDS-ify.

        Confirmed on TURNING ON, not at Convert time. A dialogue at Convert
        would arrive with the analyst's hand already on the button and be
        dismissed with everything else; asked here it is the whole content of
        the interaction. Turning it off needs no confirmation, because the
        safe state never does.
        """
        state = self._get_bidsify_state()
        if self._bidsify_move_var.get():
            ok = messagebox.askyesno(
                "Move originals into sourcedata?",
                "Each native recording will be DELETED from where it is now, "
                "once its copy is in sourcedata/ and the converted EDF has "
                "been read back and verified.\n\n"
                "A file that fails to verify keeps its original.\n\n"
                "The native recording is the one artefact this tool cannot "
                "regenerate. Only continue if sourcedata/ is the archive you "
                "mean to keep, and it is backed up.\n\n"
                "Turn this on?",
                icon="warning", default="no", parent=self.root)
            if not ok:
                self._bidsify_move_var.set(False)
                return
        state.move_sources = bool(self._bidsify_move_var.get())
        state.save()
        self.log("BIDS-ify: originals will be "
                 + ("MOVED into sourcedata after verification"
                    if state.move_sources else "copied into sourcedata"))

    def _bidsify_convert_done(self, results):
        """Main-thread handler: mark converted files, refresh, summarise."""
        state = self._get_bidsify_state()
        when = datetime.datetime.now().isoformat(timespec="seconds")
        for r in results:
            if getattr(r, "ok", False):
                state.mark_converted(r.source_path, when=when)

        # A moved original leaves every entry that named it pointing at a path
        # that no longer exists. Re-pointed here rather than left to the next
        # Refresh, which only ADDS files and would never notice: the analyst
        # would see a queue full of recordings the tool could no longer open,
        # which looks exactly like the tool having lost them.
        moved = [(r.source_path, r.moved_to) for r in results
                 if getattr(r, "moved_to", "")]
        if moved:
            ds = getattr(self, "_dataset", None)
            for old, new in moved:
                if ds is not None:
                    fe = ds.get_by_path(old)
                    if fe is not None:
                        fe.path = new
                state.repoint(old, new)
            if ds is not None:
                ds.save()
                self._queue_refresh()
            self.log(f"   ↪ {len(moved)} queue entr(ies) re-pointed to "
                     f"sourcedata")
        state.save()
        self._bidsify_tab_refresh()
        self._bidsify_done_gui(results)
