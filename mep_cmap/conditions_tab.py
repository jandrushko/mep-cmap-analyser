"""
mep_cmap.conditions_tab
~~~~~~~~~~~~~~~~~~~~~~~
Setup ▸ Conditions — assigning a recording's events to named conditions.

The rules live in :mod:`mep_cmap.conditions`, which holds no Tk and is tested
without a display. This module is the surface: a table, a trial selector, and a
pane for looking at what has been selected. Keeping the two apart is what allows
the parsing, splitting and completeness rules to be exercised properly, since
almost none of what can go wrong here is a drawing problem.

The workflow
------------
The table opens populated: one row per stimulus type the event sources found,
each holding all of that type's trials. That state produces exactly the analysis
the recording produced before conditions existed, so the tab is somewhere to go
when a file needs it rather than a step every file must pass through.

From there a row is selected, its trials appear in the list and its epochs in
the review pane, and the row can be split: select ten of twenty ``A`` trials,
name them, and the remaining ten stay behind. The point of doing it here rather
than by typing two trial lists is that a split cannot lose or duplicate a
trial, which two independent assignments can.

Why look at the epochs at all
-----------------------------
Because the question the tab answers is not one the file can answer. Deciding
that trials 1-10 were the baseline block is the analyst's knowledge, and the
only check available is whether the waveforms look like they belong together.
Hence overlay, average, or both: the overlay shows spread and outliers, the
average shows what the condition will contribute, and comparing two conditions
side by side is how a split is judged before it is applied.

No detection markers are drawn here, deliberately. This is raw epochs before
any analysis; onset and amplitude belong to Preview detection, where they are
shown one trial at a time because a dozen sets of markers on one axes cannot be
read.
"""

from __future__ import annotations

import os
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, simpledialog, ttk

import numpy as np

from . import conditions as C
from .tooltips import attach_info_icon

#: Colours for conditions drawn together. Chosen to stay distinguishable when
#: several are overlaid translucently, which the default cycle does not.
_CYCLE = ("#1F3864", "#B03A2E", "#1E8449", "#7D3C98",
          "#B9770E", "#117864", "#6C3483", "#943126")

#: How many trials may be overlaid before the pane draws only the average.
#: Two hundred translucent lines is neither readable nor quick, and the average
#: is what the analyst is looking for by that point.
OVERLAY_LIMIT = 120

#: How many edits back the undo stack reaches. Deep enough to recover from a
#: run of mistaken splits, shallow enough that it cannot grow without bound in
#: a long session.
UNDO_DEPTH = 50


HELP = {
    "table": (
        "One row per condition. The table opens with a row for each stimulus "
        "type the event sources found, holding all of that type's trials, "
        "which reproduces the analysis this recording would have had before "
        "conditions existed.\n\n"
        "Split a row to separate trials that the recording cannot distinguish "
        "but you can: twenty pulses labelled A may be ten before an "
        "intervention and ten after. The stimulus type stays A — it decides "
        "how the response is detected — and the condition records what the "
        "trial was for."
    ),
    "trials": (
        "Trials of the selected condition, numbered as the trial file and the "
        "Data Inspector number them.\n\n"
        "Select some and use Split to move them into a new condition, or Set "
        "to make them the whole of this one. The review pane follows the "
        "selection."
    ),
    "review": (
        "The selected trials, drawn from the recording as they will be cut.\n\n"
        "Overlay shows spread and outliers; Average shows what the condition "
        "will contribute, with one standard deviation shaded. Selecting two "
        "conditions draws both, in different colours, which is how a split is "
        "checked before it is applied.\n\n"
        "No onset or amplitude markers are drawn: these are raw epochs, and "
        "detection is reviewed in Preview detection where one trial is shown "
        "at a time."
    ),
    "apply": (
        "Writes the conditions to a BIDS events file beside the recording, "
        "with a column for the stimulus type and a column for the condition.\n\n"
        "It is an ordinary events file, not a private format: this tool reads "
        "it back on the next load, and so can anything else that understands "
        "BIDS.\n\n"
        "Every event must belong to a condition or be explicitly excluded. A "
        "trial in neither would simply disappear between the recording and the "
        "analysis, with nothing saying so."
    ),
}


class ConditionsTabMixin:
    """Setup ▸ Conditions."""

    # ── construction ─────────────────────────────────────────────────────────

    def _build_conditions_tab(self, parent):
        self._cond_rows = []          # list[C.ConditionRow]
        self._cond_stim_times = {}    # {stim_type: [onset, ...]}
        self._cond_segments = {}      # {stim_type: ndarray (n_trials, n_samples)}
        self._cond_axis = None        # ms, shared by the drawn epochs
        self._cond_selected_rows = () # table rows currently shown
        self._cond_source_path = None # the recording the table was built from
        self._cond_confirmed_path = None  # the recording last confirmed here
        self._cond_chan_idx = None    # channel the review pane is drawing
        self._cond_clamped = None     # epoch bounds, when the file is stitched
        # {channel_idx: {group_key: (pre_ms, post_ms)}}.
        #
        # A condition is a property of the TRIAL -- trial 5 is "pre" whichever
        # muscle is being looked at -- so the table itself is the same for
        # every channel. An epoch is a property of the RESPONSE, and a hand
        # muscle and a leg muscle legitimately want different windows, so the
        # epoch columns belong to the channel they were judged on.
        self._cond_epochs = {}
        self._cond_unit = None
        # Undo/redo stacks of (rows, what-was-done). A snapshot is the list
        # itself rather than a copy that has to be kept honest: ConditionRow is
        # frozen and every editing helper returns a new list.
        self._cond_undo_stack = []
        self._cond_redo_stack = []
        self._cond_list_index = []    # (row index, trial index) per list entry

        head = tk.Frame(parent)
        head.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(head, text="Conditions",
                 font=("TkDefaultFont", 11, "bold")).pack(side="left")
        attach_info_icon(head, HELP["apply"]).pack(side="left", padx=(4, 0))
        self._cond_status = tk.Label(
            head, text="No recording loaded — open one from Setup ▸ Dataset.",
            fg="#888")
        self._cond_status.pack(side="left", padx=(12, 0))
        tk.Button(head, text="⟳ Reload from file",
                  command=self._cond_reload).pack(side="right")

        # A paned window, not three packed frames.
        #
        # With pack, every child is given its REQUESTED width first and only
        # the remainder is shared between the expanding ones. The requested
        # width of the table pane is set by its widest child, which is the row
        # of action buttons, and the width of a row of buttons is a property of
        # the font: the same nine labels are substantially wider in macOS's
        # system font than in Segoe UI at the same point size. So the table
        # claimed most of the window and the review pane -- the one part of
        # this tab you actually have to look at -- was squeezed to nothing,
        # while on Windows the identical layout looked correct.
        #
        # Sash positions do not care what anything requests, and the weights
        # keep the proportions when the window is resized. Dragging is a
        # bonus, but the point is that no widget can take over the tab again.
        body = ttk.PanedWindow(parent, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        # ── the table ────────────────────────────────────────────────────────
        left = tk.Frame(body)
        _lh = tk.Frame(left); _lh.pack(fill="x")
        tk.Label(_lh, text="Conditions table").pack(side="left")
        attach_info_icon(_lh, HELP["table"]).pack(side="left", padx=(3, 0))

        cols = ("stim", "cond", "trials", "n", "pre", "post")
        self._cond_tree = ttk.Treeview(left, columns=cols, show="headings",
                                       selectmode="extended", height=12)
        for c, txt, w in (("stim", "Stim", 70), ("cond", "Condition", 120),
                          ("trials", "Trials", 180), ("n", "N", 40),
                          ("pre", "Pre (ms)", 65), ("post", "Post (ms)", 68)):
            self._cond_tree.heading(c, text=txt)
            self._cond_tree.column(c, width=w,
                                   anchor="w" if c != "n" else "e")
        _tsb = ttk.Scrollbar(left, orient="vertical",
                             command=self._cond_tree.yview)
        self._cond_tree.configure(yscrollcommand=_tsb.set)
        _tsb.pack(side="right", fill="y")
        self._cond_tree.pack(fill="both", expand=True, pady=(2, 4))
        self._cond_tree.bind("<<TreeviewSelect>>",
                             lambda _e: self._cond_on_row_selected())
        self._cond_tree.bind("<Double-1>", lambda _e: self._cond_rename())

        btns = tk.Frame(left); btns.pack(fill="x")
        # Gridded across two rows rather than packed into one. Nine buttons in
        # a line is a lot to read at the best of times, and in a proportional
        # font it is also the thing that decides how wide this pane wants to
        # be. Two rows halves that, and the columns share the pane's width
        # evenly instead of each button demanding its label's length.
        _per_row = 5
        for _i, (txt, cmd) in enumerate(
                (("Split selection…", self._cond_split),
                 ("Combine selected", self._cond_merge),
                 ("Set from selection", self._cond_set_from_selection),
                 ("Rename…", self._cond_rename),
                 ("Auto fill…", self._cond_autofill),
                 ("Set epoch from view", self._cond_set_epoch),
                 ("Clear epoch", self._cond_clear_epoch),
                 ("Add unassigned", self._cond_add_unassigned),
                 ("Exclude", self._cond_toggle_exclude),
                 ("Delete", self._cond_delete))):
            tk.Button(btns, text=txt, command=cmd).grid(
                row=_i // _per_row, column=_i % _per_row,
                sticky="ew", padx=(0, 4), pady=(0, 2))
        for _c in range(_per_row):
            btns.columnconfigure(_c, weight=1)

        _ep = tk.Frame(left)
        _ep.pack(fill="x", pady=(3, 0))
        self._cond_epoch_all_chans = tk.BooleanVar(value=True)
        tk.Checkbutton(
            _ep, variable=self._cond_epoch_all_chans,
            command=self._cond_refresh_table,
            text="Epoch applies to every analysed channel").pack(side="left")
        self._cond_epoch_scope = tk.Label(_ep, fg="grey", text="")
        self._cond_epoch_scope.pack(side="left", padx=(8, 0))

        hist = tk.Frame(left)
        hist.pack(fill="x", pady=(4, 0))
        # Wide enough for the longest label these buttons take. They are not
        # static: the text names what will be undone ("Undo rename", "Undo
        # set trials"), so a width chosen for the word "Undo" clips the thing
        # the analyst actually needs to read.
        _hist_w = max(len(f"\u21b6 Undo {w}")
                      for w in ("split", "rename", "delete", "set trials",
                                "autofill", "merge", "epoch", "exclude"))
        self._cond_undo_btn = tk.Button(hist, text="\u21b6 Undo",
                                        width=_hist_w, state="disabled",
                                        command=self._cond_undo)
        self._cond_undo_btn.pack(side="left")
        self._cond_redo_btn = tk.Button(hist, text="\u21b7 Redo",
                                        width=_hist_w, state="disabled",
                                        command=self._cond_redo)
        self._cond_redo_btn.pack(side="left", padx=(4, 0))
        tk.Label(hist, fg="grey", text="  Ctrl+Z / Ctrl+Y").pack(side="left")

        # ── trials ───────────────────────────────────────────────────────────
        mid = tk.Frame(body)
        _mh = tk.Frame(mid); _mh.pack(fill="x")
        tk.Label(_mh, text="Trials").pack(side="left")
        attach_info_icon(_mh, HELP["trials"]).pack(side="left", padx=(3, 0))
        _holder = tk.Frame(mid); _holder.pack(fill="both", expand=True)
        _sb = ttk.Scrollbar(_holder, orient="vertical")
        # exportselection=False, or selecting in the table clears this list:
        # Tk hands the X selection to whichever widget was touched last.
        self._cond_list = tk.Listbox(_holder, selectmode="extended", width=14,
                                     height=16, exportselection=False,
                                     yscrollcommand=_sb.set)
        _sb.config(command=self._cond_list.yview)
        _sb.pack(side="right", fill="y")
        self._cond_list.pack(side="left", fill="both", expand=True, pady=(2, 4))
        self._cond_list.bind("<<ListboxSelect>>",
                            lambda _e: self._cond_draw())
        tk.Button(mid, text="Select all",
                  command=self._cond_select_all_trials).pack(fill="x")

        # Bound here rather than beside the table: both widgets must exist, and
        # the listbox is built after it. Bound to the two widgets rather than
        # the window, so that Ctrl+Z in a dialogue's entry field is still the
        # entry's own undo.
        for _seq, _fn in (("<Control-z>", self._cond_undo),
                          ("<Control-Z>", self._cond_undo),
                          ("<Control-y>", self._cond_redo),
                          ("<Control-Y>", self._cond_redo)):
            for _w in (self._cond_tree, self._cond_list):
                _w.bind(_seq, lambda _e, f=_fn: (f(), "break")[1])

        # ── review ───────────────────────────────────────────────────────────
        right = tk.Frame(body)
        _rh = tk.Frame(right); _rh.pack(fill="x")
        tk.Label(_rh, text="Review epochs").pack(side="left")
        attach_info_icon(_rh, HELP["review"]).pack(side="left", padx=(3, 0))
        self._cond_mode = tk.StringVar(value="both")
        for txt, val in (("Overlay", "overlay"), ("Average", "average"),
                         ("Both", "both")):
            tk.Radiobutton(_rh, text=txt, value=val,
                           variable=self._cond_mode,
                           command=self._cond_draw).pack(side="left", padx=(8, 0))
        _ctl = tk.Frame(right)
        _ctl.pack(fill="x", pady=(4, 0))
        tk.Label(_ctl, text="Channel:").pack(side="left")
        # The channels ticked in Channel Assignment. Whether trials belong
        # together is judged on the muscle they were recorded from, and on a
        # recording where several were analysed there is no one right answer
        # -- the pane opened on whichever channel the analysis happened to
        # start with, which on a file whose first channel is not the one under
        # study showed a trace with nothing to do with the question.
        self._cond_chan = tk.StringVar(value="")
        self._cond_chan_dd = ttk.Combobox(_ctl, textvariable=self._cond_chan,
                                          state="readonly", width=18, values=[])
        self._cond_chan_dd.pack(side="left", padx=(4, 12))
        self._cond_chan.trace_add("write", lambda *_a: self._cond_channel_changed())

        tk.Label(_ctl, text="Window (ms):  pre").pack(side="left")
        self._cond_pre = tk.StringVar(value="")
        tk.Entry(_ctl, textvariable=self._cond_pre, width=6).pack(side="left",
                                                                  padx=(4, 6))
        tk.Label(_ctl, text="post").pack(side="left")
        self._cond_post = tk.StringVar(value="")
        tk.Entry(_ctl, textvariable=self._cond_post, width=6).pack(side="left",
                                                                   padx=(4, 6))
        tk.Button(_ctl, text="Apply window",
                  command=self._cond_window_changed).pack(side="left")
        tk.Label(_ctl, fg="grey",
                 text="  for looking only — the analysis window is on tab 1a"
                 ).pack(side="left")

        self._cond_note = tk.Label(right, text="", fg="#1F3864", anchor="w")
        self._cond_note.pack(fill="x", pady=(2, 0))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self._cond_fig = Figure(figsize=(6.4, 4.2), dpi=100)
        self._cond_ax = self._cond_fig.add_subplot(111)
        self._cond_fig.subplots_adjust(left=0.12, right=0.99, top=0.96,
                                       bottom=0.14)
        self._cond_canvas = FigureCanvasTkAgg(self._cond_fig, master=right)
        self._cond_canvas.get_tk_widget().pack(fill="both", expand=True,
                                               pady=(2, 0))

        # Added last, once all three exist. Weights govern how EXTRA space is
        # shared on resize; the initial sash positions do not follow from them
        # and are set below.
        body.add(left, weight=3)
        body.add(mid, weight=0)
        body.add(right, weight=4)

        def _place_sashes(_event=None):
            """Set the opening split by proportion, once, when width is known.

            Without this the panes open at their requested widths, which is the
            behaviour being fixed: the table pane asks for the width of its
            button grid and takes whatever that happens to be on this font and
            platform. A proportion is the same on every machine.

            Once only, and never again, so that a sash the analyst has dragged
            is not snapped back on the next resize or tab switch.
            """
            if getattr(self, "_cond_sashes_set", False):
                return
            total = body.winfo_width()
            if total < 400:               # not laid out yet
                return
            trials_w = max(mid.winfo_reqwidth(), 120)
            table_w = max(320, int((total - trials_w) * 0.42))
            try:
                body.sashpos(0, table_w)
                body.sashpos(1, table_w + trials_w)
            except Exception:             # noqa: BLE001 — not yet mapped
                return
            self._cond_sashes_set = True

        body.bind("<Configure>", _place_sashes)
        self._cond_body = body

        # ── apply ────────────────────────────────────────────────────────────
        foot = tk.Frame(parent)
        foot.pack(fill="x", padx=10, pady=(4, 10))
        self._cond_allow_unassigned = tk.BooleanVar(value=False)
        tk.Checkbutton(
            foot, variable=self._cond_allow_unassigned,
            text="Exclude any trial left out of every condition",
            command=self._cond_refresh_status).pack(side="left")
        # Two destinations, one commit.
        #
        # The work behind these buttons is substantial -- validation, the
        # per-channel epoch check, writing the events, merging the window maps
        # -- and duplicating it would be two callers of one rule, which is how
        # a rule ends up enforced in one place and quietly skipped in the
        # other. _cond_apply takes a destination; these only choose it.
        #
        # First Level keeps the accent because it is the ordinary route: most
        # analysts never BIDS-ify at all, and a plain second button reads as
        # the option it is rather than as an equal fork.
        self._cond_apply_btn = tk.Button(
            foot, text="✔  Confirm & continue to First Level",
            command=lambda: self._cond_apply(destination="first_level"))
        self._cond_apply_btn.pack(side="right")
        self._cond_apply_bids_btn = tk.Button(
            foot, text="Confirm & continue to BIDS-ify",
            command=lambda: self._cond_apply(destination="bidsify"))
        self._cond_apply_bids_btn.pack(side="right", padx=(0, 6))
        # Why Confirm is unavailable, beside Confirm.
        #
        # It shared the review pane's label, which the drawing code rewrites on
        # every selection -- so the reason appeared and was overwritten by
        # "10 trial(s) from 1 condition(s)" before it could be read, leaving a
        # disabled button with no explanation anywhere on the tab.
        self._cond_block = tk.Label(foot, text="", fg="#B03A2E", anchor="e",
                                    justify="right", wraplength=900)
        self._cond_block.pack(side="right", padx=(0, 10))

    # ── loading ──────────────────────────────────────────────────────────────

    def _cond_tab_shown(self):
        """Populate on first sight of a recording, and after the file changes.

        Not on every visit: an assignment already made must survive going away
        to check something and coming back, which is why Reload is a button.
        The table is rebuilt only when it is empty, or when the recording it
        was built from is no longer the one loaded.
        """
        path = (self.file_path.get() if hasattr(self, "file_path") else "") or ""
        if not path:
            return
        if self._cond_rows and getattr(self, "_cond_source_path", None) == path:
            return
        # A session carries its conditions. Rebuilding from the file would
        # discard them the moment the tab was opened, which is what made them
        # look as though they had never been saved.
        if self._cond_rows and getattr(self, "_cond_source_path", None) is None:
            self._cond_source_path = path
            try:
                self._cond_stim_times = {}
                _ev, _w = self._configured_events(path)
                from .pipeline import crop_stim_times
                _p = self._snapshot_analysis_params()
                self._cond_stim_times = crop_stim_times(
                    _ev, _p.get("crop_ranges"), _p.get("crop_start"),
                    _p.get("crop_end"))
            except Exception:
                pass
            self._cond_refresh_channel_picker()
            self._cond_refresh_table()
            # Say which recording these belong to. Restoring the rows without
            # touching the header left "No recording loaded" above a table
            # full of conditions, which contradicts itself and reads as the
            # restore having half worked.
            self._cond_status.config(
                text=(os.path.basename(path) + "  \u2014  conditions restored "
                      "from the session"), fg="#555")
            self.log("🏷  Conditions restored from the session")
            return
        self._cond_source_path = path
        self._cond_reload()

    def _cond_reload(self):
        """Rebuild the table from the recording's own events.

        Discards edits, which is why it is a button rather than something that
        happens on every visit to the tab: returning to check a waveform should
        not silently undo an assignment.
        """
        path = (self.file_path.get() if hasattr(self, "file_path")
                else "") or ""
        if not path or not os.path.isfile(path):
            self._cond_status.config(
                text="No recording loaded — open one from Setup ▸ Dataset.",
                fg="#888")
            self._cond_rows, self._cond_stim_times = [], {}
            self._cond_refresh_table()
            return
        try:
            events, _warn = self._configured_events(path)
        except Exception as exc:                      # noqa: BLE001 — reported
            self._cond_status.config(text=f"Could not read events: {exc}",
                                     fg="#B03A2E")
            return
        if not events:
            self._cond_status.config(
                text="No stimulus events found in this recording.", fg="#B03A2E")
            self._cond_rows, self._cond_stim_times = [], {}
            self._cond_refresh_table()
            return

        # Only the stimuli inside the selected range, by the same rule the
        # analysis applies. Conditions are assigned by trial INDEX, so a table
        # built from the whole recording numbers its trials differently from
        # the analysis and every assignment would land on the wrong trial --
        # and the waveforms drawn beside the list are already cropped, so the
        # two halves of this tab would disagree with each other as well.
        from .pipeline import crop_stim_times
        try:
            _p = self._snapshot_analysis_params()
            _cr, _cs, _ce = self._cond_crop_for(_p)
            events = crop_stim_times(events, _cr, _cs, _ce)
        except Exception:
            pass
        if not events:
            self._cond_status.config(
                text="No stimulus events fall inside the selected data range.",
                fg="#B03A2E")
            self._cond_rows, self._cond_stim_times = [], {}
            self._cond_refresh_table()
            return

        self._cond_source_path = path
        self._cond_stim_times = {k: list(v) for k, v in events.items()}
        self._cond_refresh_channel_picker()
        # A converted recording carries its grouping in its own events file.
        #
        # Rebuilding from stim types alone threw it away: the conditions were
        # written into *_events.tsv at conversion and sitting there on reopen,
        # and the table came back as if the recording had never been grouped.
        # Falls through to the plain one-row-per-stim-type table when the file
        # says nothing, which is every unconverted recording.
        self._cond_rows = []
        try:
            from .formats.edf import read_events_rows
            from . import events_model as _em
            _side = events_tsv_path(path)
            if os.path.isfile(_side):
                self._cond_rows = _em.rows_from_events(read_events_rows(_side))
                if self._cond_rows:
                    self.log(f"🏷  Conditions restored from "
                             f"{os.path.basename(_side)}: "
                             + ", ".join(f"{r.stim_type}/{r.condition}"
                                         for r in self._cond_rows))
        except Exception as exc:            # noqa: BLE001 — fall back below
            self.log(f"   ⚠️  Could not read conditions from the events file: "
                     f"{exc}")
            self._cond_rows = []
        if not self._cond_rows:
            self._cond_rows = C.rows_from_events(self._cond_stim_times)
        # Undoing past a reload would restore rows belonging to a recording
        # that is no longer open.
        self._cond_undo_stack.clear()
        self._cond_redo_stack.clear()
        self._cond_undo_stack.clear()
        self._cond_redo_stack.clear()
        self._cond_segments, self._cond_axis = {}, None
        self._cond_chan_idx = None
        _cropped = ""
        try:
            _p = self._snapshot_analysis_params()
            _cr, _cs, _ce = self._cond_crop_for(_p)
            if _cr or _cs is not None:
                # Named, because with per-channel ranges "the selected data
                # range" is ambiguous: two channels of one recording can be
                # cut differently and the table would look simply wrong on
                # the second unless it says which channel it is describing.
                _cropped = ("  (within the range selected for "
                            f"{self._cond_chan.get() or 'this channel'})")
        except Exception:
            pass
        self._cond_status.config(
            text=f"{os.path.basename(path)} — "
                 + ", ".join(f"{k} ({len(v)})"
                             for k, v in sorted(events.items())) + _cropped,
            fg="#555")
        self._cond_refresh_table()

    def _cond_event_rows_for_recording(self, rows_for_shown_channel):
        """Events-file rows covering EVERY event, not only the shown channel's.

        The table shows one channel's selected data, which is what the analyst
        asked for. The events FILE cannot be that narrow.
        ``pipeline_load_file`` uses these rows exclusively when they exist, so
        an event with no row is not an unconditioned trial -- it is not a trial
        at all. With per-channel ranges that silently emptied every channel
        whose range differed from the one last displayed: conditions assigned
        while viewing Sol-CL gave TA six onsets from a range it does not cover,
        and TA measured nothing.

        So each event in the recording gets a row. Conditions already assigned
        are kept, including for events outside the range on screen, because a
        condition is a property of the trial rather than of the channel that
        happened to be displayed when it was assigned. An event nobody has
        conditioned carries its stimulus type and no condition, which is what
        an unassigned recording has always looked like.
        """
        def _key(onset):
            # A tenth of a millisecond: finer than any jitter between the
            # readers, coarser than float error on a re-read.
            return round(float(onset), 4)

        merged = {}
        # Whatever earlier channels established, so switching channel and
        # applying again accumulates rather than replaces.
        #
        # Earlier channels OF THIS RECORDING. This read every row on the app,
        # and after opening another file those were the previous recording's
        # events -- which were merged in here and written into this file's
        # events file, where they were read back on every later load.
        _prior = (self._owned_condition_rows()
                  if hasattr(self, "_owned_condition_rows")
                  else C.rows_for_recording(
                      getattr(self, "condition_event_rows", None),
                      getattr(self, "_cond_rows_path", ""),
                      self.file_path.get()))
        for row in _prior:
            try:
                merged[_key(row["onset"])] = dict(row)
            except (KeyError, TypeError, ValueError):
                continue
        for row in (rows_for_shown_channel or []):
            try:
                merged[_key(row["onset"])] = dict(row)
            except (KeyError, TypeError, ValueError):
                continue

        # Every event in the recording, uncropped, so no channel's range can
        # leave its own trials without a row.
        try:
            events, _warn = self._configured_events(
                self.file_path.get(),
                channel_idx=self._cond_selected_channel_idx())
        except Exception:                             # noqa: BLE001
            events = {}
        for stim, times in (events or {}).items():
            for t in (times or []):
                merged.setdefault(_key(t), {
                    "onset": float(t), "duration": 0.0,
                    "trial_type": stim, "condition": C.NA,
                })
        return sorted(merged.values(), key=lambda r: r["onset"])

    def _cond_crop_for(self, params, channel_idx=None):
        """The ranges in force for the channel this tab is showing.

        Ranges are chosen per channel, so reading the snapshot's file-wide
        list here showed one channel's selection against another channel's
        trace. Conditions are assigned by trial INDEX, so that is not a
        cosmetic difference: the table would number its trials from one
        channel's selection while the analysis numbered them from another's,
        and every assignment would land on the wrong trial.

        Returns (crop_ranges, crop_start, crop_end) ready to pass on. A
        channel with no ranges of its own falls back to the file-wide pair,
        which is what a single-channel study and every older session carry.
        """
        _ch = (self._cond_selected_channel_idx() if channel_idx is None
               else channel_idx)
        from .pipeline import crop_for_channel
        return crop_for_channel(getattr(self, "_chan_settings", {}), _ch,
                                params)

    def _cond_groups_in_range(self, groups):
        """The groups that have at least one trial the analysis will measure.

        The events file deliberately gives every event in the recording a
        row, including events outside the selected range, so that per-channel
        ranges cannot leave a channel without its trials. Tab 1a was built
        from all of them, so a stimulus type whose in-range trials had all
        been excluded still got a row there -- from its out-of-range events
        alone -- and the analysis, which crops by the same rule, then measured
        nothing for it.

        Kept if in range for ANY analysed channel, because tab 1a is rebuilt
        from this one list when switching channel. Returns ``groups``
        unchanged if the ranges cannot be resolved, or if nothing would be
        left (a row too many is a nuisance; an empty setup table is not).
        """
        try:
            from .pipeline import crop_stim_times
            params = self._snapshot_analysis_params()
            chans = []
            try:
                chans = list(self._analysis_channel_indices() or [])
            except Exception:                         # noqa: BLE001
                chans = []
            if not chans:
                chans = [self._cond_selected_channel_idx()]
            keep = set()
            for ch in chans:
                _cr, _cs, _ce = self._cond_crop_for(params, channel_idx=ch)
                keep.update(crop_stim_times(groups, _cr, _cs, _ce).keys())
        except Exception:                             # noqa: BLE001
            return groups
        if not keep:
            return groups
        dropped = sorted(k for k in groups if k not in keep)
        if dropped:
            self.log("   Not in tab 1a (no trials inside the selected range): "
                     + ", ".join(dropped))
        return {k: v for k, v in groups.items() if k in keep}

    def _cond_load_segments(self):
        """Read and cut the recording once, for the review pane only.

        Separate from the analysis path: this is a look at raw epochs, and the
        window used is the one the analysis would use so that what is judged
        here is what will be measured.
        """
        if self._cond_segments:
            return True
        path = (self.file_path.get() if hasattr(self, "file_path")
                else "") or ""
        if not path or not self._cond_stim_times:
            return False
        try:
            from .pipeline import (FILTER_CFG_FIELDS, PipelineConfig,
                                   clamp_window_map, pipeline_apply_filters,
                                   pipeline_load_file, window_samples)
            from .io import get_epoch_bounds
            params = self._snapshot_analysis_params()
            _ch = self._cond_selected_channel_idx()
            self._cond_chan_idx = _ch
            _cr, _cs, _ce = self._cond_crop_for(params, _ch)
            from .pipeline import epoch_pad_seconds as _epoch_pad
            # Absent from this channel's setup means fall back; an EMPTY map
            # means this channel has no per-type windows and must not borrow
            # another channel's.
            _cond_setup = (getattr(self, "_chan_settings", {}).get(_ch) or {})
            _cond_wm = _cond_setup.get("window_map",
                                       params.get("window_map")) or {}
            emg, time, fs, _unit, _stim = pipeline_load_file(
                params["input_path"], _ch,
                params["marker_choice"],
                crop_ranges=_cr,
                crop_start=_cs,
                crop_end=_ce,
                # Padded as the run pads, so the review pane shows every trial
                # the analysis will measure. Without it the last trial of each
                # range was missing from the pane but present in the results.
                epoch_pad_s=_epoch_pad(pre_ms=params.get("pre_ms"),
                                       post_ms=params.get("post_ms"),
                                       window_map=_cond_wm,
                                       prestim_ms=params.get("prestim_ms")),
                sources=None, event_rows=None)

            # Filtered as the analysis filters. Raw epochs carry whatever DC
            # offset and drift the amplifier had, which stacks the trials at
            # different baselines and makes the overlay unreadable -- and shows
            # the analyst something the analysis never sees, which is the
            # failure a review pane exists to prevent rather than commit.
            try:
                _fcfg = PipelineConfig(**{f: params[f]
                                          for f in FILTER_CFG_FIELDS})
                emg = pipeline_apply_filters(emg, fs, _fcfg)
            except Exception as _fe:
                self._cond_note.config(
                    text=f"Showing unfiltered data: {_fe}", fg="#B03A2E")
            wincfg = PipelineConfig(pre_ms=float(params["pre_ms"]),
                                    post_ms=float(params["post_ms"]),
                                    window_map=params.get("window_map") or {})
            _vpre, _vpost = self._cond_view_window_ms()

            # A pre-epoched recording is stitched from its own epochs with
            # mirror-padded guard bands between them. A window longer than an
            # epoch reads that padding -- it draws as plausible signal and is
            # not signal at all -- or reaches into the neighbouring trial. The
            # analysis clamps for exactly this reason; a typed viewing window
            # would otherwise walk straight past that protection.
            _bounds = None
            try:
                _bounds = get_epoch_bounds(params["input_path"])
            except Exception:
                _bounds = None
            if _bounds:
                _bp, _bq = float(_bounds[0]), float(_bounds[1])
                if _vpre is not None and _vpre > _bp:
                    _vpre = _bp
                if _vpost is not None and _vpost > _bq:
                    _vpost = _bq
                _wm, _ = clamp_window_map(params.get("window_map") or {},
                                          _bp, _bq)
                wincfg = PipelineConfig(
                    pre_ms=min(float(params["pre_ms"]), _bp),
                    post_ms=min(float(params["post_ms"]), _bq),
                    window_map=_wm)
                self._cond_clamped = (_bp, _bq)
            else:
                self._cond_clamped = None

            out = {}
            for stim, times in self._cond_stim_times.items():
                before, after = window_samples(wincfg, stim, fs)
                # A typed viewing window overrides the analysis one, per side,
                # so widening only the tail is possible without restating both.
                if _vpre is not None:
                    before = int(_vpre * fs / 1000)
                if _vpost is not None:
                    after = int(_vpost * fs / 1000)
                segs = []
                for t in times:
                    ix = int(np.argmin(np.abs(time - t)))
                    seg = emg[max(0, ix - before): ix + after]
                    segs.append(seg if len(seg) == before + after else None)
                keep = [s for s in segs if s is not None]
                if keep:
                    out[stim] = (np.vstack(keep),
                                 np.arange(len(keep[0])) * 1000.0 / fs
                                 - before * 1000.0 / fs)
            self._cond_segments = out
            self._cond_unit = _unit
            return bool(out)
        except Exception as exc:                      # noqa: BLE001 — reported
            self._cond_note.config(
                text=f"Could not read the waveforms: {exc}", fg="#B03A2E")
            return False

    # ── the table ────────────────────────────────────────────────────────────

    def _cond_refresh_table(self):
        # The epoch columns show the channel the review pane is on, since that
        # is the channel they were judged against.
        _eps = self._cond_epochs_for_channel()

        def _win_cells(row):
            win = _eps.get(row.group_key)
            if not win:
                return ("", "")
            p, q = win
            return ("" if p is None else f"{p:g}",
                    "" if q is None else f"{q:g}")

        sel = set(self._cond_tree.selection()) if self._cond_tree else set()
        self._cond_tree.delete(*self._cond_tree.get_children())
        for i, row in enumerate(self._cond_rows):
            self._cond_tree.insert(
                "", "end", iid=str(i), tags=(),
                values=(row.stim_type,
                        ("— excluded —" if row.excluded
                         else (row.condition or "")),
                        C.format_trials(row.trials), row.n,
                        *_win_cells(row)))
        for iid in sel:
            if self._cond_tree.exists(iid):
                self._cond_tree.selection_add(iid)
        try:
            _nm = self._cond_chan.get() or "this channel"
            _n_set = len(self._cond_epochs_for_channel())
            self._cond_epoch_scope.config(
                text=(f"showing epochs for {_nm}"
                      + (f" \u2014 {_n_set} set" if _n_set else "")))
        except Exception:
            pass
        self._cond_refresh_history_buttons()
        self._cond_refresh_status()

    def _cond_apply_state(self, state: str):
        """Both Confirm buttons together.

        They commit the same work and differ only in where they land, so one
        being available while the other is not would mean an invalid table
        could still be committed by choosing the other destination.
        """
        for _name in ("_cond_apply_btn", "_cond_apply_bids_btn"):
            _btn = getattr(self, _name, None)
            if _btn is not None:
                _btn.config(state=state)

    def _cond_refresh_status(self):
        """Say whether the table can be applied, and why not when it cannot."""
        if not self._cond_rows:
            self._cond_apply_state("disabled")
            self._cond_block.config(text="No conditions to apply.",
                                    fg="#B03A2E")
            return
        try:
            C.validate(self._cond_rows, self._cond_stim_times,
                       allow_unassigned=bool(self._cond_allow_unassigned.get()))
        except C.ConditionError as exc:
            self._cond_block.config(text=str(exc))
            self._cond_apply_state("disabled")
            return
        self._cond_apply_state("normal")
        n = sum(r.n for r in self._cond_rows if not r.excluded)
        self._cond_block.config(
            text=f"{len(self._cond_rows)} condition(s), {n} trial(s) — ready",
            fg="#1E8449")

    def _cond_selected_indices(self):
        return tuple(int(i) for i in self._cond_tree.selection()
                     if str(i).isdigit())

    def _cond_on_row_selected(self):
        """Fill the trial list from the selected rows and draw them."""
        rows = self._cond_selected_indices()
        self._cond_selected_rows = rows
        self._cond_list.delete(0, "end")
        self._cond_list_index = []
        for ri in rows:
            row = self._cond_rows[ri]
            for t in row.trials:
                self._cond_list.insert("end", f"Trial {t + 1}")
                self._cond_list_index.append((ri, t))
        self._cond_list.selection_set(0, "end")
        self._cond_draw()

    def _cond_select_all_trials(self):
        self._cond_list.selection_set(0, "end")
        self._cond_draw()

    def _cond_current_selection(self):
        """{row index: (trial index, ...)} for what is selected in the list."""
        out = {}
        for i in self._cond_list.curselection():
            ri, t = self._cond_list_index[int(i)]
            out.setdefault(ri, []).append(t)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    def _cond_channel_names(self):
        """(index, name) for every channel the analysis will run over."""
        try:
            names = list(self.channel_dd["values"]) or []
        except Exception:
            names = []
        out = []
        for i in self._analysis_channel_indices():
            out.append((i, names[i] if i < len(names) else f"Channel {i + 1}"))
        return out

    def _cond_refresh_channel_picker(self):
        """Offer the analysed channels, opening on the current one."""
        pairs = self._cond_channel_names()
        labels = [nm for _i, nm in pairs]
        self._cond_chan_dd.config(values=labels)
        if not labels:
            return
        current = dict(pairs).get(getattr(self, "channel_idx", 0))
        if self._cond_chan.get() not in labels:
            self._cond_chan.set(current if current in labels else labels[0])

    def _cond_selected_channel_idx(self):
        for i, nm in self._cond_channel_names():
            if nm == self._cond_chan.get():
                return i
        return getattr(self, "channel_idx", 0)

    def _cond_channel_changed(self):
        """Redraw on the newly chosen channel.

        The cut epochs are per channel, so the cache is dropped rather than
        reused: keeping it would draw the previous channel's waveforms under
        the new channel's name, which is the failure this control exists to
        fix rather than a new one to introduce.

        The TRIAL LIST is rebuilt too, not only the waveforms. Data ranges are
        per channel, and the list is cropped to them, so leaving it alone
        showed the previous channel's trials beside the new channel's traces.
        Conditions are assigned by trial index, so that is not a display
        problem: an assignment made against the wrong list lands on the wrong
        trial.
        """
        idx = self._cond_selected_channel_idx()
        if idx == self._cond_chan_idx:
            return
        self._cond_segments = {}
        if self._cond_ranges_differ_by_channel():
            # Only when they actually differ. Rebuilding otherwise would
            # discard hand-made condition assignments to reach an identical
            # list, which is a far worse trade than a redundant redraw.
            self._cond_reload_events_for_channel()
        self._cond_refresh_table()
        self._cond_draw()

    def _cond_ranges_differ_by_channel(self) -> bool:
        """Whether any two analysed channels carry different data ranges."""
        seen = set()
        for i, _nm in self._cond_channel_names():
            _own = (getattr(self, "_chan_settings", {}).get(i, {}) or {}).get(
                "crop_ranges")
            seen.add(None if _own is None else tuple(tuple(r) for r in _own))
        return len(seen) > 1

    def _cond_reload_events_for_channel(self):
        """Re-cut the trial list to the channel now on screen.

        Assignments are kept where the trial still exists: a condition is a
        property of the trial, and re-cutting must not throw away work the
        analyst has done on trials both channels share.
        """
        path = getattr(self, "_cond_source_path", None) or (
            self.file_path.get() if hasattr(self, "file_path") else "")
        if not path:
            return
        from .pipeline import crop_stim_times
        try:
            events, _warn = self._configured_events(
                path, channel_idx=self._cond_selected_channel_idx())
            _p = self._snapshot_analysis_params()
            _cr, _cs, _ce = self._cond_crop_for(_p)
            events = crop_stim_times(dict(events or {}), _cr, _cs, _ce)
        except Exception as exc:                      # noqa: BLE001 — reported
            self._cond_status.config(
                text=f"Could not re-read events for this channel: {exc}",
                fg="#B03A2E")
            return
        if not events:
            self._cond_status.config(
                text="No stimulus events fall inside the range selected for "
                     f"{self._cond_chan.get()}.", fg="#B03A2E")
            self._cond_stim_times, self._cond_rows = {}, []
            return
        self._cond_stim_times = {k: list(v) for k, v in events.items()}
        self._cond_rows = C.rows_from_events(self._cond_stim_times)
        # Undo history belongs to the previous list; a trial index in it may
        # not exist in this one.
        self._cond_undo_stack.clear()
        self._cond_redo_stack.clear()
        self._cond_status.config(
            text=f"{os.path.basename(path)} — "
                 + ", ".join(f"{k} ({len(v)})"
                             for k, v in sorted(events.items()))
                 + f"  (within the range selected for {self._cond_chan.get()})",
            fg="#555")

    def _cond_window_changed(self):
        """Redraw with the typed viewing window.

        Deliberately does not touch the analysis window on tab 1a. Widening the
        view to see whether a late component belongs to a condition is a
        question about looking, and silently re-epoching the analysis to answer
        it would be a surprising thing for a viewing control to do.
        """
        self._cond_segments = {}
        self._cond_draw()

    def _cond_view_window_ms(self):
        """(pre, post) for the review pane, or None to use the analysis window."""
        def _num(var):
            try:
                v = float(str(var.get()).strip())
            except (TypeError, ValueError):
                return None
            return v if v > 0 else None
        return _num(self._cond_pre), _num(self._cond_post)

    # ── history ──────────────────────────────────────────────────────────────

    def _cond_push_epochs(self, what):
        """Record the epoch book before it changes.

        Held on the same stack as the table edits, so one Undo means the last
        thing done rather than the last thing done to one of two structures.
        """
        import copy
        self._cond_undo_stack.append(
            (list(self._cond_rows), copy.deepcopy(self._cond_epochs), what))
        del self._cond_undo_stack[:-UNDO_DEPTH]
        self._cond_redo_stack.clear()

    def _cond_commit(self, new_rows, what):
        """Adopt an edited table, remembering what it replaced.

        Called only once an edit has succeeded, so a refused one -- a split
        that would move every trial, a name containing the separator -- leaves
        nothing behind. An undo that does nothing visible is worse than no
        undo: it teaches the analyst the button is unreliable, and the next
        press is the one that goes too far.
        """
        import copy
        self._cond_undo_stack.append(
            (list(self._cond_rows), copy.deepcopy(self._cond_epochs), what))
        del self._cond_undo_stack[:-UNDO_DEPTH]
        # A new edit invalidates the redo path: redoing onto a table that has
        # since changed would apply an edit to rows it was never made against.
        self._cond_redo_stack.clear()
        self._cond_rows = list(new_rows)
        self._cond_refresh_table()

    def _cond_undo(self):
        if not self._cond_undo_stack:
            return
        import copy
        rows, eps, what = self._cond_undo_stack.pop()
        self._cond_redo_stack.append(
            (list(self._cond_rows), copy.deepcopy(self._cond_epochs), what))
        self._cond_rows, self._cond_epochs = rows, eps
        self._cond_refresh_table()
        self.log(f"\u21b6 Undid: {what}")

    def _cond_redo(self):
        if not self._cond_redo_stack:
            return
        import copy
        rows, eps, what = self._cond_redo_stack.pop()
        self._cond_undo_stack.append(
            (list(self._cond_rows), copy.deepcopy(self._cond_epochs), what))
        self._cond_rows, self._cond_epochs = rows, eps
        self._cond_refresh_table()
        self.log(f"\u21b7 Redid: {what}")

    def _cond_refresh_history_buttons(self):
        """Name what will be undone, rather than offering a bare Undo.

        "Undo split" is a different proposition from "Undo delete", and the
        analyst should not have to remember which came last to know what the
        button will do.
        """
        if self._cond_undo_stack:
            self._cond_undo_btn.config(
                state="normal",
                text=f"\u21b6 Undo {self._cond_undo_stack[-1][-1]}")
        else:
            self._cond_undo_btn.config(state="disabled", text="\u21b6 Undo")
        if self._cond_redo_stack:
            self._cond_redo_btn.config(
                state="normal",
                text=f"\u21b7 Redo {self._cond_redo_stack[-1][-1]}")
        else:
            self._cond_redo_btn.config(state="disabled", text="\u21b7 Redo")

    # ── editing ──────────────────────────────────────────────────────────────

    def _cond_split(self):
        sel = self._cond_current_selection()
        if len(sel) != 1:
            messagebox.showinfo(
                "Split", "Select trials from one condition to split it.",
                parent=self.root)
            return
        (ri, trials), = sel.items()
        name = simpledialog.askstring(
            "Split condition",
            f"{len(trials)} trial(s) selected.\n\n"
            f"Name for the new condition:", parent=self.root)
        if not name:
            return
        try:
            new_rows = C.split_row(self._cond_rows, ri, trials,
                                   new_condition=name)
        except C.ConditionError as exc:
            messagebox.showwarning("Split", str(exc), parent=self.root)
            return
        self._cond_commit(new_rows, "split")

    def _cond_merge(self):
        """Combine the selected conditions back into one.

        The way out of a split made on the wrong boundary. Undo only reaches
        the last action, so once the analyst had done anything else -- or
        reopened the file, which clears the stacks -- the halves were
        permanent and the only route back was Reload from file, which discards
        every other edit made to the table.
        """
        rows = self._cond_selected_indices()
        if len(rows) < 2:
            messagebox.showinfo(
                "Combine",
                "Select two or more conditions to combine.\n\n"
                "Ctrl+click or Shift+click the rows in the table.",
                parent=self.root)
            return

        chosen = [self._cond_rows[i] for i in rows]
        names = [r.condition or "(unnamed)" for r in chosen]
        total = len({t for r in chosen for t in r.trials})
        # Named before it happens. Merging is not destructive of trials, but
        # it does discard the condition labels of every row but the first, and
        # those are the analyst's own work.
        if not messagebox.askyesno(
                "Combine conditions",
                f"Combine {len(rows)} conditions into one?\n\n"
                + "\n".join(f"  • {n} ({len(r.trials)} trial(s))"
                            for n, r in zip(names, chosen))
                + f"\n\nThe result keeps the name '{names[0]}' and holds "
                  f"{total} trial(s). The other names are discarded.",
                parent=self.root):
            return
        try:
            new_rows = C.merge_rows(self._cond_rows, rows)
        except C.ConditionError as exc:
            messagebox.showwarning("Combine", str(exc), parent=self.root)
            return
        self._cond_commit(new_rows, "merge")

    def _cond_set_from_selection(self):
        """Make the selection the whole of its row.

        The complement becomes unassigned rather than silently dropped, which
        the status line then reports: removing trials from a condition is a
        decision, and where they went should be visible.
        """
        sel = self._cond_current_selection()
        if len(sel) != 1:
            messagebox.showinfo(
                "Set", "Select trials from one condition.", parent=self.root)
            return
        (ri, trials), = sel.items()
        new_rows = list(self._cond_rows)
        new_rows[ri] = replace(new_rows[ri], trials=trials)
        self._cond_commit(new_rows, "set trials")

    def _cond_rename(self):
        rows = self._cond_selected_indices()
        if not rows:
            return
        current = self._cond_rows[rows[0]].condition
        name = simpledialog.askstring("Condition name", "Name:",
                                      initialvalue=current, parent=self.root)
        if name is None:
            return
        try:
            clean = C.sanitise_name(name)
        except C.ConditionError as exc:
            messagebox.showwarning("Condition name", str(exc), parent=self.root)
            return
        new_rows = list(self._cond_rows)
        for ri in rows:
            new_rows[ri] = replace(new_rows[ri], condition=clean)
        self._cond_commit(new_rows, "rename")

    def _cond_autofill(self):
        rows = self._cond_selected_indices()
        if len(rows) != 1:
            messagebox.showinfo("Auto fill", "Select one condition to divide.",
                                parent=self.root)
            return
        n = simpledialog.askinteger(
            "Auto fill", "Trials per condition:", minvalue=1, parent=self.root)
        if not n:
            return
        stim = self._cond_rows[rows[0]].stim_type
        try:
            new_rows = C.autofill(
                self._cond_rows, rows[0], per_row=n,
                n_available=len(self._cond_stim_times.get(stim) or []))
        except C.ConditionError as exc:
            messagebox.showwarning("Auto fill", str(exc), parent=self.root)
            return
        self._cond_commit(new_rows, "auto fill")

    def _cond_epochs_for_channel(self, ch=None):
        if ch is None:
            ch = self._cond_selected_channel_idx()
        return self._cond_epochs.setdefault(ch, {})

    def _cond_set_epoch(self):
        """Give the selected conditions the window currently being viewed.

        The point of setting it here rather than on the labels tab is that the
        decision can be seen: whether the response is truncated, whether a
        silent period runs past the end of the window, whether a long tail is
        empty. On the labels tab the same numbers are typed blind.
        """
        rows = self._cond_selected_indices()
        if not rows:
            messagebox.showinfo(
                "Epoch", "Select the condition(s) to give this window to.",
                parent=self.root)
            return
        pre, post = self._cond_view_window_ms()
        if pre is None and post is None:
            messagebox.showinfo(
                "Epoch",
                "Type a viewing window above the plot first, then use this to "
                "make it this condition's epoch.", parent=self.root)
            return
        # Whatever is actually being drawn, not what was typed: on a stitched
        # file the view is clamped to the stored epoch, and handing the
        # analysis a window the recording cannot supply would undo that.
        clamp = getattr(self, "_cond_clamped", None)
        if clamp:
            if pre is not None:
                pre = min(pre, clamp[0])
            if post is not None:
                post = min(post, clamp[1])
        targets = ([i for i, _n in self._cond_channel_names()]
                   if self._cond_epoch_all_chans.get()
                   else [self._cond_selected_channel_idx()])
        self._cond_push_epochs("set epoch")
        for ch in targets:
            book = self._cond_epochs_for_channel(ch)
            for ri in rows:
                book[self._cond_rows[ri].group_key] = (pre, post)
        self._cond_refresh_table()

    def _cond_clear_epoch(self):
        """Return the selected conditions to the file-wide window."""
        rows = self._cond_selected_indices()
        if not rows:
            return
        targets = ([i for i, _n in self._cond_channel_names()]
                   if self._cond_epoch_all_chans.get()
                   else [self._cond_selected_channel_idx()])
        self._cond_push_epochs("clear epoch")
        for ch in targets:
            book = self._cond_epochs_for_channel(ch)
            for ri in rows:
                book.pop(self._cond_rows[ri].group_key, None)
        self._cond_refresh_table()

    def _cond_add_unassigned(self):
        """Give every loose trial a row of its own.

        The commonest way to arrive at a table that cannot be applied is a
        stimulus type nobody split, left with no row after the others were
        edited. Reporting that and offering no way to act on it means retyping
        trial ranges the tool already knows.
        """
        loose = C.unassigned(self._cond_rows, self._cond_stim_times)
        if not loose:
            messagebox.showinfo("Conditions",
                                "Every trial already belongs to a condition.",
                                parent=self.root)
            return
        new_rows = list(self._cond_rows)
        for stim, trials in sorted(loose.items()):
            new_rows.append(C.ConditionRow(stim_type=stim, condition="",
                                           trials=tuple(trials)))
        self._cond_commit(new_rows, "add unassigned")

    def _cond_toggle_exclude(self):
        rows = self._cond_selected_indices()
        if not rows:
            return
        new_rows = list(self._cond_rows)
        for ri in rows:
            new_rows[ri] = replace(new_rows[ri],
                                   excluded=not new_rows[ri].excluded)
        self._cond_commit(new_rows, "exclude")

    def _cond_delete(self):
        rows = set(self._cond_selected_indices())
        if not rows:
            return
        self._cond_commit([r for i, r in enumerate(self._cond_rows)
                           if i not in rows], "delete")

    # ── drawing ──────────────────────────────────────────────────────────────

    def _cond_draw(self):
        ax = self._cond_ax
        ax.clear()
        sel = self._cond_current_selection()
        if not sel or not self._cond_load_segments():
            ax.text(0.5, 0.5, "Select a condition to see its epochs.",
                    ha="center", va="center", color="0.5",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self._cond_canvas.draw_idle()
            return

        mode = self._cond_mode.get()
        drawn = 0
        for n, (ri, trials) in enumerate(sorted(sel.items())):
            row = self._cond_rows[ri]
            pack = self._cond_segments.get(row.stim_type)
            if pack is None:
                continue
            stack, axis = pack
            usable = [t for t in trials if t < stack.shape[0]]
            if not usable:
                continue
            block = stack[usable, :]
            colour = _CYCLE[n % len(_CYCLE)]
            label = row.describe().split(":")[0]

            # Above the limit only the average is drawn: two hundred
            # translucent lines is neither readable nor quick, and by then the
            # average is what is being looked at anyway.
            if mode in ("overlay", "both") and len(usable) <= OVERLAY_LIMIT:
                for s in block:
                    ax.plot(axis, s, color=colour, alpha=0.18, linewidth=0.6)
            if mode in ("average", "both") or len(usable) > OVERLAY_LIMIT:
                mean = block.mean(axis=0)
                ax.plot(axis, mean, color=colour, linewidth=2.0, label=label)
                if block.shape[0] > 1:
                    sd = block.std(axis=0)
                    ax.fill_between(axis, mean - sd, mean + sd,
                                    color=colour, alpha=0.15, linewidth=0)
            elif mode == "overlay":
                ax.plot([], [], color=colour, linewidth=2.0, label=label)
            drawn += len(usable)

        ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time about the stimulus (ms)", fontsize=8)
        _u = getattr(self, "_cond_unit", None) or "a.u."
        ax.set_ylabel(f"EMG ({_u})", fontsize=8)
        ax.tick_params(labelsize=8)
        if len(sel) > 1:
            ax.legend(fontsize=7, loc="upper right")
        _msg = f"{drawn} trial(s) from {len(sel)} condition(s)"
        if getattr(self, "_cond_clamped", None):
            _bp, _bq = self._cond_clamped
            _msg += (f"  \u2014  pre-epoched file: the view is limited to the "
                     f"stored epoch, -{_bp:g} to +{_bq:g} ms")
        self._cond_note.config(text=_msg, fg="#1F3864")
        self._cond_canvas.draw_idle()

    # ── apply ────────────────────────────────────────────────────────────────

    def _cond_apply(self, destination: str = "first_level"):
        """Validate, write the events file, and hand the groups to the analysis."""
        try:
            rows = C.validate(
                self._cond_rows, self._cond_stim_times,
                allow_unassigned=bool(self._cond_allow_unassigned.get()))
        except C.ConditionError as exc:
            messagebox.showwarning("Conditions", str(exc), parent=self.root)
            return
        self._cond_rows = rows
        self._cond_visited_chans = set()

        # With the epoch applying to one channel at a time, every analysed
        # channel needs visiting before the setup table is built: an epoch set
        # on EMG 1 alone leaves EMG 2 on the file-wide window, and the analyst
        # would not find that out until the results disagreed.
        if not self._cond_epoch_all_chans.get():
            _chans = [i for i, _n in self._cond_channel_names()]
            _seen = getattr(self, "_cond_visited_chans", None) or set()
            _seen.add(self._cond_selected_channel_idx())
            self._cond_visited_chans = _seen
            _left = [c for c in _chans if c not in _seen]
            if _left:
                _names = dict(self._cond_channel_names())
                self._cond_chan.set(_names.get(_left[0], ""))
                self._cond_channel_changed()
                self._cond_block.config(
                    text=(f"{len(_left)} channel(s) still to review — now "
                          f"showing {_names.get(_left[0], '')}"), fg="#B03A2E")
                self.log(f"   Epochs are per channel: {len(_left)} channel(s) "
                         f"still to review.")
                return

        event_rows = self._cond_event_rows_for_recording(
            C.to_event_rows(rows, self._cond_stim_times))
        groups, decoded = C.group_events(event_rows)

        path = self.file_path.get()
        try:
            written = write_events_tsv_beside(path, event_rows)
        except Exception as exc:                      # noqa: BLE001 — reported
            messagebox.showerror(
                "Conditions",
                f"The conditions are applied to this session but the events "
                f"file could not be written:\n\n{exc}", parent=self.root)
            written = None

        # Held on the app, and read by _snapshot_analysis_params, so that the
        # analysis and the preview group trials by these conditions.
        self.condition_event_rows = event_rows
        # And which recording they describe, so no other file can use them.
        self._cond_rows_path = C.recording_key(path)
        self.condition_map = decoded

        # Per-condition epochs become the per-type windows the labels tab and
        # the analysis already understand, keyed by group key. Merged rather
        # than replacing: a window set on the labels tab for a stimulus type
        # this table did not give one to is still the analyst's setting.
        # window_map is PER CHANNEL state, held in _chan_settings. Writing
        # self.window_map alone reaches only the channel currently selected,
        # which is why an epoch set here appeared on one channel and not the
        # other. Each channel's book is written into that channel's snapshot.
        # Every channel's book goes into that channel's SNAPSHOT, including the
        # channel currently on screen.
        #
        # The current channel used to be special-cased: its epochs were merged
        # into the live window_map while the others went into their snapshots.
        # But _restore_chan_settings, called a few lines below to select the
        # first analysed channel, loads that channel's snapshot over the live
        # map -- so on the usual path, where the channel in the Conditions tab
        # IS the first analysed one, the epochs were written to the live map and
        # then immediately overwritten by a snapshot that never received them.
        # Tab 1a fell back to the file-wide window and the analyst's epochs were
        # silently gone. It appeared to work whenever those two channels
        # happened to differ.
        #
        # One direction of travel now: conditions write snapshots,
        # _restore_chan_settings loads the live map from a snapshot, and the
        # special case is gone.
        _cur = getattr(self, "channel_idx", 0)
        if _cur is not None and _cur not in self._chan_settings:
            # A channel edited but never left has no snapshot yet, and writing
            # epochs into a snapshot that does not exist would lose the rest of
            # its setup when the restore reads it back.
            try:
                self._snapshot_chan_settings(_cur)
            except Exception:               # noqa: BLE001 — best effort
                pass
        for ch, book in (self._cond_epochs or {}).items():
            if not book:
                continue
            snap = self._chan_settings.setdefault(ch, {})
            merged = dict(snap.get("window_map") or {})
            merged.update(book)
            snap["window_map"] = merged
            if ch == _cur:
                # The live map too, so the table is right even if nothing
                # restores before it is drawn.
                _live = dict(getattr(self, "window_map", None) or {})
                _live.update(book)
                self.window_map = _live
            self.log(f"   Epochs for channel {ch + 1}: "
                     + ", ".join(f"{k} {p if p is not None else 'default'}"
                                 f"/{q if q is not None else 'default'} ms"
                                 for k, (p, q) in sorted(book.items())))

        self._cond_confirmed_path = path
        self.log("🏷  Conditions applied: "
                 + ", ".join(f"{k} ({len(v)})" for k, v in sorted(groups.items())))
        if written:
            self.log(f"   → {os.path.basename(written)}")

        # Persist the conditions, not just apply them.
        #
        # They lived on the app until the session happened to be saved, so
        # anything reading them from disk -- BIDS-ify, a later run, this file
        # reopened tomorrow -- found the table empty and silently described the
        # recording as ungrouped. Confirming is the moment the analyst has
        # decided, so it is the moment worth writing.
        try:
            self._autosave_session()
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Conditions applied but the session could not be "
                     f"saved: {exc}")
        self._cond_refresh_status()
        # Select the first analysed channel BEFORE the table is built.
        #
        # It was selected afterwards, so the table was built for whichever
        # channel happened to be current and then rebuilt again by the switch
        # -- and the switch was skipped when the index already matched, which
        # left the table showing the wrong channel with no way to notice. The
        # setup starts at the first selected channel and the advance walks
        # forward from there; starting anywhere else means the advance has
        # nothing to advance to.
        try:
            _chans = self._analysis_channel_indices()
            if _chans:
                self._chan_confirmed = set()
                _cnames = (list(self.channel_dd["values"])
                           if hasattr(self, "channel_dd") else [])
                _first = _chans[0]
                self.channel_idx = _first
                if _first < len(_cnames):
                    self.channel_var.set(_cnames[_first])
                try:
                    self._restore_chan_settings(_first)
                except Exception:
                    pass
        except Exception:
            pass

        # Rebuild the labels tab from the conditions, then go where asked.
        #
        # The navigation is done HERE rather than inside _build_labels_tab.
        # That method runs on every rebuild -- opening a file, switching
        # channel, confirming a channel -- so navigating from it sent all of
        # those somewhere unexpected. Each caller now says where it wants to
        # end up, and this one wants wherever the analyst asked for.
        #
        # The rebuild happens either way. Conditions change the per-type
        # windows the labels tab is built from, so skipping it on the way to
        # BIDS-ify would leave 1a describing the previous grouping until
        # something else happened to rebuild it.
        try:
            # Applying conditions splits every stimulus type into new rows --
            # A becomes A-pre and A-post -- so whatever was confirmed no
            # longer describes the table. Discarded HERE, at the point the
            # rows actually change, rather than left to _build_labels_tab:
            # that method rebuilds for several reasons and cannot tell which
            # of them invalidates a confirmation.
            #
            # Both records are cleared together. _chan_confirmed is the one
            # the per-channel run guard reads, and clearing only the flag left
            # that guard still passing for this channel.
            try:
                self._chan_confirmed.discard(self.channel_idx)
            except Exception:
                pass
            self._build_labels_tab(sorted(self._cond_groups_in_range(groups)))
            self.log("   Tab 1a rebuilt from the conditions.")
        except Exception as exc:                      # noqa: BLE001 — reported
            self.log(f"   ⚠️  Could not rebuild the labels tab: {exc}")
            return

        if destination == "bidsify":
            try:
                self.root.update_idletasks()
                self.notebook.select(self.setup_outer)
                self.nb_setup.select(self.tab_bidsify)
                self._bidsify_tab_refresh()
                # Select the recording just conditioned. Landing on a list of
                # fifteen rows and having to find it again loses the connection
                # between what was just done and what to do next. After the
                # refresh, because that rebuilds the rows.
                try:
                    self._bidsify_select_source(path)
                except Exception:                     # noqa: BLE001 — cosmetic
                    pass
                self.log("   → BIDS-ify. The conditions are written into the "
                         "events file when this recording is converted.")
            except Exception as exc:                  # noqa: BLE001 — reported
                self.log(f"   ⚠️  Could not open BIDS-ify: {exc}")
            return

        try:
            self.root.update_idletasks()
            self.notebook.select(self.stage1_outer)
            # tab1b_frame is, confusingly, the 1a tab.
            self.nb_stage1.select(self.tab1b_frame)
        except Exception:
            pass


# ── writing ──────────────────────────────────────────────────────────────────

def events_tsv_path(recording_path: str) -> str:
    """``<entities>_emg.edf`` → ``<entities>_events.tsv``.

    Mirrors the reader's own rule, so that what this writes is what it reads.
    """
    d = os.path.dirname(recording_path)
    stem = os.path.splitext(os.path.basename(recording_path))[0]
    for suffix in ("_emg", "_eeg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return os.path.join(d, stem + "_events.tsv")


def write_events_tsv_beside(recording_path: str, event_rows) -> str:
    """Write a BIDS events file next to the recording, and its sidecar.

    An ordinary events file rather than a private format: this tool reads it
    back on the next load through a path that already existed, and so can
    anything else that understands BIDS. The sidecar exists because ``condition``
    is not a BIDS-defined column, and an undocumented extra column is a column
    only its author can interpret.
    """
    import csv
    import json

    path = events_tsv_path(recording_path)
    cols = ["onset", "duration", "trial_type", "condition"]

    # Columns this writer does not own are carried through, not overwritten.
    #
    # For a CONVERTED recording the "beside the recording" path IS the BIDS
    # events file: sub-X/ses-Y/emg/..._emg.edf sits next to ..._events.tsv,
    # which BIDS-ify wrote with nibs_event_id and nibs_position_id linking each
    # delivery to the stimulation description. Rewriting the file with only
    # these four columns silently deleted those links, leaving *_nibs.tsv
    # describing protocols nothing referenced.
    #
    # Matched by onset, which is the one field both writers agree on and which
    # identifies a delivery.
    extra_by_onset, extra_cols = {}, []
    try:
        from .formats.edf import read_events_rows
        for _old in read_events_rows(path):
            _extras = {k: v for k, v in _old.items() if k not in cols}
            if not _extras:
                continue
            extra_by_onset[round(float(_old["onset"]), 6)] = _extras
            for k in _extras:
                if k not in extra_cols:
                    extra_cols.append(k)
    except Exception:               # noqa: BLE001 — no file, or unreadable
        extra_by_onset, extra_cols = {}, []

    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + extra_cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for row in event_rows:
            out = {k: row.get(k, C.NA) for k in cols}
            if extra_cols:
                try:
                    _kept = extra_by_onset.get(round(float(row.get("onset")), 6), {})
                except (TypeError, ValueError):
                    _kept = {}
                for k in extra_cols:
                    out[k] = _kept.get(k, C.NA)
            w.writerow(out)

    side = os.path.splitext(path)[0] + ".json"
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({
            "onset": {"Description": "Onset of the stimulus, in seconds from "
                                     "the start of the recording.",
                      "Units": "s"},
            "duration": {"Description": "Duration of the stimulus. Zero for an "
                                        "instantaneous event.",
                         "Units": "s"},
            "trial_type": {"Description": "Stimulus type as recorded by the "
                                          "acquisition system. n/a marks a "
                                          "trial excluded from analysis, "
                                          "retained here so that the file "
                                          "accounts for every event."},
            "condition": {"Description": "Condition the trial was assigned to "
                                         "by the analyst: an experimental "
                                         "factor the recording itself does not "
                                         "distinguish, such as a timepoint or "
                                         "a block. n/a where none was "
                                         "assigned."},
        }, fh, indent=2)
    return path
