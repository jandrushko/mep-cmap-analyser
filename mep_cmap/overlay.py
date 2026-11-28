"""
mep_cmap.overlay
~~~~~~~~~~~~~~~~
Every chosen trial of a condition on one set of axes.

The preview shows one trial at a time, which answers "did the markers land
sensibly on this trial" and cannot answer "is this setting right for this
condition". The second question is about a DISTRIBUTION: whether onsets cluster
or scatter, whether the amplitude window contains the response on most trials
or only the large ones, whether one trial is unlike the rest. No single trial
shows any of that, and stepping through eighty of them turns a distribution
into a memory test.

So this draws them together, with the amplitude window the analysis will
actually use marked across them, and a strip of detected onsets beneath. The
single-trial view is not replaced: a trace can be clicked to open it, because
"which trial is that outlier" is the question an overlay always raises next.

Nothing here detects anything. Both the traces and the onsets come from what
the preview already computed, so the overlay cannot disagree with the preview
about what the detector did.

Approach adopted from the TMS Analysis ToolBox (Cunningham et al.), which
overlays a condition's trials for exactly this purpose. Written independently;
no code was taken.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

#: Above this many traces, drawing each one separately is slow enough to be
#: felt, and the plot is unreadable anyway. Beyond it the traces are drawn as a
#: band -- the per-sample minimum and maximum across trials -- with the median
#: over the top.
#:
#: A band rather than a subsample of the traces, for the same reason the
#: threshold preview draws min/max columns: a subsample of trials hides the
#: outlier, and an outlier is the thing an overlay is being read for.
MAX_INDIVIDUAL_TRACES = 60


def trace_alpha(n: int) -> float:
    """Opacity for one trace when n are drawn on top of each other.

    Fixed opacity fails at both ends: at five traces a faint line is hard to
    see, and at sixty an opaque one is a solid block. Scaled so the stack
    stays about as dark whatever the count, with a floor because a trace too
    faint to see is not a trace.
    """
    if n <= 1:
        return 0.9
    return max(0.08, min(0.9, 2.4 / float(n)))


def band_of(traces) -> tuple:
    """(min, max, median) across trials, per sample.

    The min and max are what make an outlier visible without drawing every
    trace: a single trial unlike the rest widens the band exactly where it
    differs, which a mean and standard deviation would smooth away.
    """
    arr = np.asarray(traces, dtype=float)
    return (np.min(arr, axis=0), np.max(arr, axis=0), np.median(arr, axis=0))


class OverlayPanel:
    """Trials of one or more conditions on shared axes, inside a container.

    A PANEL, not a window. The overlay and the trial-by-trial view answer two
    halves of one question -- what the condition does, and what happened on
    this trial -- and two windows made the analyst arrange them by hand every
    time. Both now live in one window; see :class:`CombinedPreviewWindow`.

    ``groups`` is ``{group_key: {"traces": 2-D array, "onsets_ms": [...],
    "trial_numbers": [...], "colour": str}}``. Every group must already have
    been checked to share an epoch: this draws what it is given, and the
    decision about what may share axes belongs to
    :func:`pipeline.overlay_groups`, which the caller consults. Drawing it here
    as well would be a second answer to a question with one right answer.
    """

    def __init__(self, container, groups, fs, prestim_ms, unit,
                 ptp_window_ms=None, on_pick_trial=None,
                 prestim_window_ms=None):
        self.groups = groups
        self.fs = float(fs)
        self.prestim_ms = float(prestim_ms)
        self.unit = unit or "mV"
        self.ptp_window_ms = ptp_window_ms
        #: (start_ms, end_ms) of the baseline the detectors are given. Shaded
        #: because every threshold in the tool is expressed relative to it: an
        #: onset criterion is so many SDs above THIS interval, so a baseline
        #: that has caught the tail of the previous response or the stimulus
        #: artefact raises every threshold at once, and the symptom appears
        #: everywhere except where the cause is.
        self.prestim_window_ms = prestim_window_ms
        self.on_pick_trial = on_pick_trial
        self._picker = []

        self.frame = tk.Frame(container)
        self._build()
        self.draw()

    def pack(self, **kw):
        self.frame.pack(**kw)
        return self

    # ── layout ───────────────────────────────────────────────────────────

    def _build(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        bar = tk.Frame(self.frame)
        bar.pack(fill="x", padx=8, pady=(6, 2))

        self.show_band = tk.BooleanVar(value=False)
        self.show_rug = tk.BooleanVar(value=True)
        self.show_median = tk.BooleanVar(value=True)
        #: Display only. The traces are rectified on the way to the axes and
        #: nothing else changes: the amplitude window, the baseline band and
        #: the onset strip are all drawn from values computed on the RAW
        #: signal, because that is what the analysis measures. Rectifying is a
        #: way of looking at a response, not a way of measuring one.
        self.rectify = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Median", variable=self.show_median,
                       command=self.draw).pack(side="left")
        tk.Checkbutton(bar, text="Rectify", variable=self.rectify,
                       command=self.draw).pack(side="left", padx=(8, 0))
        tk.Checkbutton(bar, text="Onset strip", variable=self.show_rug,
                       command=self.draw).pack(side="left", padx=(8, 0))
        tk.Checkbutton(bar, text="Draw as band", variable=self.show_band,
                       command=self.draw).pack(side="left", padx=(8, 0))

        self.note = tk.Label(bar, text="", fg="grey")
        self.note.pack(side="left", padx=(16, 0))

        # Two axes sharing an x: the traces, and a short strip beneath holding
        # one tick per detected onset. The strip is what turns "the markers
        # look about right" into a statement about spread, and it has to share
        # the axis or it says nothing about where the onsets fall.
        self.fig = Figure(figsize=(9.4, 3.4), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.06)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_rug = self.fig.add_subplot(gs[1], sharex=self.ax)
        # Explicit margins rather than tight_layout. The plotting AREA has to
        # line up with the trial view below, because the two are read together
        # and an eye travelling down the window compares positions, not axis
        # values; a few percent of horizontal offset makes a marker look
        # misplaced when it is not. The bottom margin is what stops the axis
        # label being clipped, which tight_layout would fix by moving the axes
        # and so breaking the alignment.
        self.fig.subplots_adjust(left=0.075, right=0.985,
                                 top=0.97, bottom=0.22)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=8, pady=(0, 4))
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def set_groups(self, groups, ptp_window_ms=None, prestim_window_ms=None):
        """Draw a different condition without rebuilding the panel.

        The baseline band comes with the groups because the blanking gap is
        per stimulus type: holding it fixed for the window's lifetime drew one
        type's gap over every other type's trials.
        """
        self.groups = groups
        self.ptp_window_ms = ptp_window_ms
        self.prestim_window_ms = prestim_window_ms
        self.draw()

    # ── drawing ──────────────────────────────────────────────────────────

    def _time_axis(self, n_samples):
        """ms about the stimulus, from the segment's own length.

        Derived from the array rather than from the configured window, so a
        segment shortened to fit a pre-epoched recording is drawn against the
        axis it actually has.
        """
        return (np.arange(n_samples) - self.prestim_ms * self.fs / 1000.0) \
            * 1000.0 / self.fs

    def draw(self):
        self.ax.clear()
        self.ax_rug.clear()
        self._picker = []

        n_total = sum(len(g.get("traces") or []) for g in self.groups.values())
        as_band = self.show_band.get() or n_total > MAX_INDIVIDUAL_TRACES

        for key, g in sorted(self.groups.items()):
            traces = np.asarray(g.get("traces") or [], dtype=float)
            if not len(traces):
                continue
            # Rectified BEFORE the median and before the band, which is the
            # order that makes each of them mean what it usually means: a
            # rectified average is the average of the rectified trials, not
            # the rectified average of the raw ones. The two differ wherever
            # trials disagree in sign, which on a biphasic response is most of
            # the epoch.
            if self.rectify.get():
                traces = np.abs(traces)
            t = self._time_axis(traces.shape[1])
            colour = g.get("colour") or None
            nums = g.get("trial_numbers") or list(range(1, len(traces) + 1))

            if as_band:
                lo, hi, med = band_of(traces)
                self.ax.fill_between(t, lo, hi, alpha=0.25, color=colour,
                                     linewidth=0, label=f"{key} (n={len(traces)})")
                if self.show_median.get():
                    self.ax.plot(t, med, color=colour, linewidth=1.6)
            else:
                a = trace_alpha(len(traces))
                for i, tr in enumerate(traces):
                    ln, = self.ax.plot(t, tr, color=colour, alpha=a,
                                       linewidth=0.8, picker=4)
                    # The trial NUMBER, not the row index: a click has to
                    # name the trial as the recording numbers it, or it
                    # cannot be found again after the run.
                    self._picker.append((ln, key, nums[i] if i < len(nums)
                                         else i + 1))
                if self.show_median.get():
                    self.ax.plot(t, np.median(traces, axis=0), color=colour,
                                 linewidth=1.8)
                # One legend entry per group rather than per trace.
                self.ax.plot([], [], color=colour,
                             label=f"{key} (n={len(traces)})")

            if self.show_rug.get():
                # Onsets and offsets on one strip, at different heights and in
                # different colours. Together they say what the DURATION
                # distribution looks like, which neither says alone: a tight
                # cluster of onsets under a scattered set of offsets is a
                # detector finding the response reliably and losing its end.
                onsets = [o for o in (g.get("onsets_ms") or [])
                          if o is not None]
                if onsets:
                    self.ax_rug.plot(onsets, np.full(len(onsets), 0.68),
                                     "|", color=colour, markersize=9,
                                     markeredgewidth=1.2)
                offsets = [o for o in (g.get("offsets_ms") or [])
                           if o is not None]
                if offsets:
                    self.ax_rug.plot(offsets, np.full(len(offsets), 0.5),
                                     "|", color="#0072B2", markersize=9,
                                     markeredgewidth=1.2)
                # The silent period's END. Its start is the MEP offset row
                # above, the two being one event, so drawing it again here
                # would show one finding as two.
                csp = [o for o in (g.get("csp_end_ms") or [])
                       if o is not None]
                if csp:
                    self.ax_rug.plot(csp, np.full(len(csp), 0.18),
                                     "|", color="#CC79A7", markersize=9,
                                     markeredgewidth=1.2)

        self._draw_ptp_window()
        self._draw_prestim_window()

        self.ax.axvline(0.0, color="#888888", linewidth=0.8)
        self.ax.set_ylabel(f"Amplitude ({self.unit})")
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.tick_params(labelbottom=False)

        self.ax_rug.set_ylim(0, 1)
        # One row per landmark, labelled, so which is which is readable
        # without a legend and without colour vision.
        self.ax_rug.set_yticks([0.18, 0.5, 0.68])
        self.ax_rug.set_yticklabels(["cSP end", "offset", "onset"], fontsize=7)
        self.ax_rug.set_xlabel("Time from stimulus (ms)")
        self.ax_rug.set_ylabel("landmarks", fontsize=8)

        self._update_note(n_total, as_band)
        self.canvas.draw_idle()

    def _draw_ptp_window(self):
        """Mark the amplitude window, or say nothing if there isn't one.

        Drawn from the window the ANALYSIS resolved, which for an anchored
        type is derived from the median onset across every trial of the type
        rather than the ones shown. A window computed from the displayed
        subset would move as the selection changed and would not be the window
        the run measures in.
        """
        if not self.ptp_window_ms:
            return
        w0, w1 = self.ptp_window_ms
        for x in (w0, w1):
            self.ax.axvline(x, color="#B03A2E", linestyle="--", linewidth=1.2)
        self.ax.axvspan(w0, w1, color="#B03A2E", alpha=0.06, linewidth=0)

    def _draw_prestim_window(self):
        """Shade the baseline the detectors are given.

        Worth its own marking because every threshold in the tool is relative
        to it: an onset criterion is so many SDs above THIS interval. A
        baseline that has caught the stimulus artefact, or the tail of the
        previous response, raises every threshold at once, and the symptom
        then appears in onset, offset and duration together while the cause is
        in none of them.
        """
        if not self.prestim_window_ms:
            return
        b0, b1 = self.prestim_window_ms
        self.ax.axvspan(b0, b1, color="#4C72B0", alpha=0.08, linewidth=0)
        for x in (b0, b1):
            self.ax.axvline(x, color="#4C72B0", linestyle=":", linewidth=1.0)

    def _update_note(self, n_total, as_band):
        bits = [f"{n_total} trial(s)"]
        # How many landmarks the strip actually has. Without this an empty
        # offset row is ambiguous between "the detector found none", "they
        # were never seeded" and "the strip is not drawing them", and the
        # three are told apart by reading a log the analyst has no reason to
        # think of looking at.
        n_on = sum(1 for g in self.groups.values()
                   for v in (g.get("onsets_ms") or []) if v is not None)
        n_off = sum(1 for g in self.groups.values()
                    for v in (g.get("offsets_ms") or []) if v is not None)
        bits.append(f"{n_on} onset(s), {n_off} offset(s)")
        # A latency profile that does not fit the data (pipeline.
        # profile_fit_warning): the responses lie after the onset search
        # window. Said here, beside the onset count it explains, because a
        # low count with no reason given reads as a detector failure.
        _notes = [g.get("profile_note") for g in self.groups.values()
                  if g.get("profile_note")]
        if _notes:
            bits.append("⚠ " + "; ".join(_notes))
        if self.rectify.get():
            # Said on the plot, not only on the tick box. A rectified overlay
            # pasted into a slide is otherwise indistinguishable from a
            # monophasic response, and the markers around it were computed on
            # the raw signal.
            bits.append("RECTIFIED (display only)")
        if as_band:
            bits.append("drawn as a band (min-max across trials)")
        if self.ptp_window_ms:
            bits.append(f"amplitude window "
                        f"{self.ptp_window_ms[0]:.1f} to "
                        f"{self.ptp_window_ms[1]:.1f} ms")
        if self.prestim_window_ms:
            bits.append(f"baseline {self.prestim_window_ms[0]:.0f} to "
                        f"{self.prestim_window_ms[1]:.0f} ms")
        self.note.config(text="   ".join(bits))

    # ── interaction ──────────────────────────────────────────────────────

    def _on_pick(self, event):
        if not callable(self.on_pick_trial):
            return
        for line, key, number in self._picker:
            if line is event.artist:
                self.on_pick_trial(key, number)
                return


class CombinedPreviewWindow:
    """The overlay and the trial-by-trial view, in one window.

    They were two Toplevels, which meant arranging them by hand on every
    preview and losing the condition-level picture the moment the trial view
    was raised. They answer two halves of one question and belong together:
    the overlay says what the condition does, the trial view says what
    happened on the trial the overlay just made you suspicious of.

    The Inspector is used AS IS, hosted in the lower pane through its
    ``container`` argument. It is not reimplemented and not subclassed: it
    still calls the analysis detector for every trial it draws, which is the
    property that makes a preview worth trusting. Rebuilding a trial view here
    would be a second implementation of marker placement and therefore a second
    answer.

    A resizable split rather than a fixed layout, because which half matters
    depends on the question: judging a detector setting is mostly the overlay,
    and chasing one bad trial is mostly the trial view.
    """

    def __init__(self, parent, groups_for, keys, options, fs, prestim_ms,
                 unit, inspector_factory, on_close=None):
        # groups_for(members) -> (groups, ptp_window_ms, prestim_window_ms).
        # The baseline window comes back WITH the groups rather than being
        # fixed here, because the blanking gap is per stimulus type.
        self.groups_for = groups_for
        self.options = options
        self.fs = fs
        self.prestim_ms = prestim_ms
        self.unit = unit
        self.on_close = on_close

        self.win = tk.Toplevel(parent)
        self.win.title("Preview detection")
        try:
            self.win.state("zoomed")
        except Exception:
            try:
                self.win.attributes("-zoomed", True)
            except Exception:
                self.win.geometry("1280x900")

        bar = tk.Frame(self.win)
        bar.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(bar, text="Event type:").pack(side="left")
        self.choice = tk.StringVar(value=options[0][0] if options else "")
        self.chooser = ttk.Combobox(
            bar, textvariable=self.choice, state="readonly", width=34,
            values=[lbl for lbl, _m, reason in options if not reason])
        self.chooser.pack(side="left", padx=(6, 8))
        self.chooser.bind("<<ComboboxSelected>>",
                          lambda _e: self._redraw(refill=True))

        # Combinations that cannot share a time axis are named rather than
        # omitted. An option that is simply absent reads as a limitation of
        # the tool; one that says why reads as a property of the recording.
        refused = [f"{lbl}: {reason}" for lbl, _m, reason in options if reason]
        if refused:
            tk.Label(bar, text="  ".join(refused), fg="#B03A2E",
                     wraplength=760, justify="left").pack(side="left")

        panes = ttk.PanedWindow(self.win, orient="vertical")
        panes.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        top = tk.Frame(panes)
        bottom = tk.Frame(panes)
        panes.add(top, weight=2)
        panes.add(bottom, weight=3)

        first = self._members_for(self.choice.get())
        groups, window, baseline = self.groups_for(first)

        # Trials down the left, multi-select. The overlay's whole value is
        # comparing a chosen set: "these eight look alike and that one does
        # not" is the observation, and it cannot be made if the set is fixed
        # at whatever the trial chooser picked. Selecting none means all,
        # since an empty plot is never what was wanted.
        side = tk.Frame(top)
        side.pack(side="left", fill="y", padx=(8, 4), pady=(6, 4))
        tk.Label(side, text="Trials").pack(anchor="w")
        self.trial_list = tk.Listbox(side, selectmode="extended",
                                     exportselection=False, width=10,
                                     height=14)
        self.trial_list.pack(fill="y", expand=True)
        self.trial_list.bind("<<ListboxSelect>>", lambda _e: self._redraw())
        tk.Button(side, text="All", width=8,
                  command=self._select_all_trials).pack(pady=(4, 0))

        self.overlay = OverlayPanel(
            top, groups, fs, prestim_ms, unit, ptp_window_ms=window,
            prestim_window_ms=baseline,
            on_pick_trial=self._on_pick_trial).pack(
                side="left", fill="both", expand=True)
        self._fill_trial_list(groups)

        # Built last, and given the frame rather than a parent window. Its own
        # "Close preview" button is removed: this window owns closing now, and
        # two close buttons in one window is an invitation to press the wrong
        # one.
        self.inspector = inspector_factory(bottom)
        try:
            self.inspector.btn_row.pack_forget()
        except Exception:
            pass
        # ONE control for one decision. The Inspector has its own event-type
        # dropdown, and with the overlay's beside it the two could disagree --
        # the overlay showing B while the trial view showed A, which reads as
        # the trial view contradicting the summary above it rather than as two
        # controls being out of step. The Inspector's is disabled and driven
        # from the overlay's instead of being removed, so nothing about how it
        # redraws changes.
        try:
            self.inspector.dd_event.configure(state="disabled")
        except Exception:
            pass
        self._sync_inspector_type()

        foot = tk.Frame(self.win)
        foot.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(foot,
                 text="Click a trace in the overlay to jump to that trial "
                      "below. Markers are fixed and nothing is saved.",
                 fg="grey").pack(side="left")
        tk.Button(foot, text="Close preview", width=16,
                  command=self.close).pack(side="right")

        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.grab_set()

        # Left/Right step through trials from ANYWHERE in the window.
        #
        # bind() on the window only fires when the focused widget lets the
        # event propagate, and the matplotlib canvas, the listbox and the
        # combobox all consume arrows, so it worked only after clicking the
        # lower plot. bind_all catches the keystroke wherever focus sits;
        # released on close, so it cannot outlive this window and start
        # stepping trials in the main one.
        #
        # Still guarded on the focused widget, but only for TEXT entry, where
        # arrows move a cursor and stealing them would make typing
        # impossible. The listbox loses its arrow navigation deliberately: its
        # selection now drives the trial below, so the two would fight.
        self.win.bind_all("<Left>", lambda e: self._step_trial(-1))
        self.win.bind_all("<Right>", lambda e: self._step_trial(+1))
        self.win.focus_set()

    def _step_trial(self, direction):
        try:
            focused = self.win.focus_get()
            if isinstance(focused, (tk.Entry, tk.Text, ttk.Entry)):
                return
        except Exception:
            pass
        insp = getattr(self, "inspector", None)
        if insp is None:
            return
        try:
            (insp._next if direction > 0 else insp._prev)()
        except Exception:
            pass
        return "break"

    def _members_for(self, label):
        for lbl, members, reason in self.options:
            if lbl == label and not reason:
                return members
        return self.options[0][1] if self.options else []

    def _fill_trial_list(self, groups):
        """List every trial in the drawn groups, by recording number."""
        self._list_entries = []
        for key in sorted(groups):
            for n in (groups[key].get("trial_numbers") or []):
                self._list_entries.append((key, n))
        self.trial_list.delete(0, "end")
        multi = len({k for k, _n in self._list_entries}) > 1
        for key, n in self._list_entries:
            self.trial_list.insert(
                "end", f"{key} {n}" if multi else f"Trial {n}")

    def _select_all_trials(self):
        self.trial_list.selection_clear(0, "end")
        self._redraw()

    def _selected_pairs(self):
        """(key, number) pairs ticked in the list, or None for all of them."""
        idxs = self.trial_list.curselection()
        if not idxs:
            return None
        return {self._list_entries[i] for i in idxs
                if 0 <= i < len(self._list_entries)}

    @staticmethod
    def _filter_groups(groups, keep):
        """Groups reduced to the chosen trials, keeping every list in step.

        Traces, onsets, offsets and trial numbers are parallel lists; filtering
        one and not the others would attach a trial's onset to a different
        trial's trace, which is the kind of fault that looks like a detection
        problem.
        """
        if keep is None:
            return groups
        out = {}
        for key, g in groups.items():
            nums = list(g.get("trial_numbers") or [])
            idxs = [i for i, n in enumerate(nums) if (key, n) in keep]
            if not idxs:
                continue
            def _pick(seq):
                seq = list(seq or [])
                return [seq[i] for i in idxs if i < len(seq)]
            out[key] = dict(g,
                            traces=_pick(g.get("traces")),
                            onsets_ms=_pick(g.get("onsets_ms")),
                            offsets_ms=_pick(g.get("offsets_ms")),
                            csp_end_ms=_pick(g.get("csp_end_ms")),
                            trial_numbers=_pick(nums))
        return out or groups

    def _sync_inspector_type(self):
        """Point the trial view at the stimulus type the overlay is showing.

        Driven through the Inspector's own attributes and redraw, the same
        route _on_pick_trial uses, so the trial view changes type exactly as
        it always did.

        Where several conditions of a type are overlaid the trial view can
        only show one, so it shows the first. The overlay says which
        conditions it holds in its legend, and a trial view claiming to show
        "all of them" would be the misleading option.
        """
        insp = getattr(self, "inspector", None)
        if insp is None:
            return
        members = self._members_for(self.choice.get())
        if not members:
            return
        want = members[0]
        try:
            values = list(insp.dd_event["values"])
            if want not in values:
                return
            insp.dd_event.set(want)
            insp.cur_type = want
            insp._select_axis()
            insp.cur_idx = 0
            insp._plot()
        except Exception:
            pass

    def _redraw(self, refill=False):
        groups, window, baseline = self.groups_for(
            self._members_for(self.choice.get()))
        if refill:
            self._fill_trial_list(groups)
        self.overlay.set_groups(self._filter_groups(groups,
                                                    self._selected_pairs()),
                                window, baseline)
        if refill:
            # Only when the CONDITION changed. A change of trial selection
            # must not reset the trial view to the first segment, or picking
            # trials in the list would keep throwing away the one being read.
            self._sync_inspector_type()
            return
        # Exactly ONE trial picked is an unambiguous request to look at it, so
        # the trial view follows. Several is a request to compare them, and
        # there is no single trial to show; the view is left where it is
        # rather than jumping to an arbitrary member of the selection.
        pairs = self._selected_pairs()
        if pairs and len(pairs) == 1:
            key, number = next(iter(pairs))
            self._on_pick_trial(key, number)

    def _on_pick_trial(self, key, number):
        """Drive the Inspector to the picked trial.

        Through its own event-type dropdown and index, so the Inspector
        re-draws by the route it always uses. Reaching past that into its
        drawing internals would couple this window to the one file that is
        meant to stay untouched.
        """
        insp = getattr(self, "inspector", None)
        if insp is None:
            return
        try:
            values = list(insp.dd_event["values"])
            if key in values:
                insp.dd_event.set(key)
                insp.cur_type = key
                insp._select_axis()
            # The overlay numbers trials as the recording does; the Inspector
            # numbers only the ones it was given. Map through the displayed
            # order, or the click lands on the wrong trial.
            shown = list(insp.segments.get(insp.cur_type) or [])
            nums = (self.trial_numbers or {}).get(key) or []
            insp.cur_idx = (nums.index(number)
                            if number in nums else 0)
            insp.cur_idx = max(0, min(insp.cur_idx, max(0, len(shown) - 1)))
            insp._plot()
        except Exception:
            pass

    trial_numbers = None

    def close(self):
        # Released FIRST. bind_all is application-wide, so a binding left
        # behind would go on stepping trials in a window that no longer has an
        # inspector to step.
        for seq in ("<Left>", "<Right>"):
            try:
                self.win.unbind_all(seq)
            except Exception:
                pass
        try:
            self.win.grab_release()
        except Exception:
            pass
        try:
            self.win.destroy()
        except Exception:
            pass
        if callable(self.on_close):
            self.on_close()
