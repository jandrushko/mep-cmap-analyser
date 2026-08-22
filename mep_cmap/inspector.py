"""
mep_cmap.inspector
~~~~~~~~~~~~~~~~~~
Interactive per-trial data inspector.

  • DraggablePoint       — draggable scatter marker on a matplotlib axes
  • DataInspectorWindow  — Tkinter toplevel for reviewing/editing segments
"""

import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext

from .compat import _np_trapz, _np_ptp
from .detection import (dispatch_onset,
                        CspSettings,
                        detect_csp_for_trial,
                        detect_mep_offset,
                        offset_marker_field)
from .detection.defaults import DETECTION_DEFAULTS

class DraggablePoint:
    """
    A draggable scatter point.  When the user lets go,
    it snaps to the sample that best matches its *role*:
        • 'ptp_min_idx' → local minimum in a ±radius window  
        • 'ptp_max_idx' → local maximum in a ±radius window  
        • anything else → nearest sample (previous behaviour)
    """
    def __init__(self, point, time_axis, emg, idx0, update_cb, role='generic',
                 radius=8, read_only=False):
        # read_only: draw the marker, ignore the mouse. Preview detection shows
        # what the configured detector produced; a marker the analyst can drag
        # there is an invitation to correct the answer by hand, and nothing in
        # a preview is saved, so the correction would silently evaporate.
        self.read_only = read_only
        self.point = point
        self.t = time_axis
        self.emg = emg
        self.idx = idx0
        self.update_cb = update_cb
        self._dragging = False
        self.role = role
        self.radius = radius
        canvas = point.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------------
    def _on_press(self, event):
        if self.read_only:
            return
        if event.inaxes is not self.point.axes:
            return
        # Do not start a drag when the matplotlib toolbar is in zoom or pan
        # mode — those modes need exclusive mouse control for their own interaction.
        try:
            toolbar = self.point.figure.canvas.toolbar
            if toolbar is not None and toolbar.mode != '':
                return
        except Exception:
            pass
        contains, _ = self.point.contains(event)
        if contains:
            # Disable dragging for all other points
            for dp in getattr(self.point.figure, '_draggables', []):
                dp._dragging = False
            self._dragging = True


    def _on_motion(self, event):
        if not self._dragging or event.inaxes is not self.point.axes:
            return
        self.point.set_offsets([[event.xdata, event.ydata]])
        self.point.figure.canvas.blit(self.point.axes.bbox)

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        idx_cand = int(np.argmin(np.abs(self.t - event.xdata)))

        if self.role in ('ptp_min_idx', 'ptp_max_idx'):
            w0 = max(0, idx_cand - self.radius)
            w1 = min(len(self.emg), idx_cand + self.radius + 1)
            win = self.emg[w0:w1]
            if self.role == 'ptp_min_idx':
                idx_new = w0 + int(np.argmin(win))
            else:  # 'ptp_max_idx'
                idx_new = w0 + int(np.argmax(win))
        else:
            idx_new = idx_cand

        x_new, y_new = self.t[idx_new], self.emg[idx_new]

        self.idx = idx_new
        self.point.set_offsets([[x_new, y_new]])
        self.update_cb(idx_new)
        self.point.figure.canvas.draw_idle()


class DraggableLine:
    """
    A draggable vertical line on a matplotlib axes.
    Only responds to clicks within pick_radius_ms of the line —
    clicking elsewhere has no effect whatsoever.
    On release, snaps to nearest sample and fires update_cb(new_idx).
    """
    def __init__(self, ax, time_axis, idx0, update_cb,
                 color="tab:blue", lw=1.8, ls="--", pick_radius_ms=4.0,
                 read_only=False):
        self.read_only   = read_only
        self.ax          = ax
        self.t           = time_axis
        self.idx         = idx0
        self.update_cb   = update_cb
        self.pick_radius = pick_radius_ms
        self._dragging   = False
        self.line = ax.axvline(time_axis[idx0], color=color,
                               lw=lw, ls=ls, alpha=0.9, zorder=4)
        canvas = ax.figure.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event",   self._on_press),
            canvas.mpl_connect("motion_notify_event",  self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def remove(self):
        canvas = self.ax.figure.canvas
        for cid in self._cids:
            try: canvas.mpl_disconnect(cid)
            except Exception: pass
        try: self.line.remove()
        except Exception: pass

    def set_idx(self, new_idx):
        """Move line programmatically without firing update_cb."""
        self.idx = new_idx
        self.line.set_xdata([self.t[new_idx], self.t[new_idx]])

    def _on_press(self, event):
        if self.read_only:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        try:
            tb = self.ax.figure.canvas.toolbar
            if tb is not None and tb.mode != '':
                return
        except Exception:
            pass
        if abs(event.xdata - self.t[self.idx]) <= self.pick_radius:
            self._dragging = True

    def _on_motion(self, event):
        if not self._dragging or event.inaxes is not self.ax:
            return
        if event.xdata is None:
            return
        self.line.set_xdata([event.xdata, event.xdata])
        self.ax.figure.canvas.draw_idle()

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        if event.xdata is None:
            return
        new_idx = int(np.argmin(np.abs(self.t - event.xdata)))
        self.idx = new_idx
        self.line.set_xdata([self.t[new_idx], self.t[new_idx]])
        self.update_cb(new_idx)
        self.ax.figure.canvas.draw_idle()

class _EmbeddedTop(tk.Frame):
    """A Frame that answers to the Toplevel calls the Inspector makes.

    The Inspector owns its own window: it titles it, grabs it, maximises it
    and resizes it to fit its widgets. All of that is right for a window and
    wrong for a panel sitting inside one, but the alternative to this class was
    editing seventeen call sites in a 1755-line file that is verified working.

    A REAL Frame, subclassed rather than a wrapper object, because ``self.top``
    is used as the parent of half a dozen widgets and Tk needs a genuine
    widget there.

    ``state()`` reports "zoomed" deliberately. Both resize paths skip geometry
    changes when the window is maximised, so reporting it is the honest answer
    for a panel whose size belongs to its container, and it disables the
    auto-resize without touching either code path.
    """

    def title(self, *_a):
        return ""

    def grab_set(self):
        pass

    def grab_release(self):
        pass

    def protocol(self, *_a, **_k):
        pass

    def geometry(self, *_a):
        return ""

    def state(self, *_a):
        return "zoomed"

    def attributes(self, *_a):
        return True


class DataInspectorWindow:
    """
    Interactive reviewer for single-trial EMG segments.

    New in v2
    ----------
    • “Silent period” toggle                     (unchanged)
    • “AUC selector”   toggle                    ← new
        – shows a 2nd subplot with |EMG|
        – drag a blue span to mark the window
        – stores ‘auc_start_idx’ / ‘auc_end_idx’
    """
    FIG_H_RAW = 4      # inches – height when only raw trace is shown
    FIG_H_EXTRA = 2      # inches – extra height for |EMG| panel

    DOT_COLOURS = {
        "ptp_min_idx":      "#56B4E9",
        "ptp_max_idx":      "#D55E00",
        "onset_idx":        "#009E73",
        "silent_start_idx": "#F0E442",
        "silent_end_idx":   "#CC79A7",
        "mep_offset_idx":   "#0072B2",
    }

    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, master, segments_dict, time_axis, metadata_dict,
                 label_map=None, color_map=None, emg_unit=None,
                 ptp_start_ms=10, ptp_end_ms=50,
                 ptp_windows_by_type=None,
                 delay_ms_map=None,
                 visible_pre_ms=None,
                 onset_method="peak_fraction",
                 onset_bootstrap_crit=1.96, onset_bootstrap_n=500,
                 onset_bigoni_smooth_ms=0.5, onset_bigoni_min_run_ms=0.5,
                 onset_bigoni_walkback_sd=1.0,
                 detection_params=None,
                 enable_auc=True,
                 csp_search_end_ms=400,
                 csp_min_silence_ms=25, csp_min_return_ms=40,
                 csp_criterion=1.96, csp_significance=0.99,
                 csp_n_boot=1000, csp_rms_window_ms=10,
                 # 100 to match the interface default and PipelineConfig. It
                 # stood at 40 here, so anything constructing the Inspector
                 # without passing it explicitly reviewed trials under a
                 # stricter onset limit than the analysis used.
                 csp_max_mep_offset_ms=100,
                 latency_map=None,
                 csp_types=None, analysis_pre_ms=None,
                 extra_segs=None, wide_window_s=3.0, underlays=None,
                 read_only=False, container=None):

        # --------- book-keeping -----------------------------------------
        # ``container`` makes this a PANEL rather than a window. Given one,
        # every Toplevel-only call the class makes is absorbed by _EmbeddedTop
        # and nothing else in this file changes: the same widgets are built,
        # the same detector runs, the same markers are drawn. Used by the
        # preview, where the overlay and the trial view are two halves of one
        # question and belonged in one window.
        if container is not None:
            self.top = _EmbeddedTop(container)
            self.top.pack(fill="both", expand=True)
        else:
            self.top = tk.Toplevel(master)
            self.top.title("Data Inspector – review")
            # Note: transient(master) is intentionally NOT set here — it removes
            # the minimise/maximise/restore buttons from the title bar on Windows.
            self.top.grab_set()
        self.embedded = container is not None

        # read_only: a viewing window. Markers are drawn but fixed, the
        # editing controls are inert, and nothing is committed on close. Used
        # by Preview detection, where the question is what the configured
        # detector does -- not what the analyst would rather it had done.
        # Everything else about the window is unchanged, deliberately: the
        # preview is only worth trusting if it looks and measures exactly like
        # the review that follows it.
        self.read_only = bool(read_only)
        self.segments  = segments_dict
        # stim_type -> median waveform over retained trials; see
        # _condition_template. Invalidated when an exclusion changes.
        self._template_cache = {}
        self._is_closing = False
        # time_axis may be one array, or {stim_type: array} where stimulus
        # types are epoched over different windows. Held as a mapping either
        # way and re-selected whenever the displayed type changes, so every
        # index-to-latency conversion below -- marker placement, the AUC
        # window, the reported latencies -- uses the axis belonging to the
        # trial on screen rather than to whichever type happened to be first.
        if isinstance(time_axis, dict):
            self._axes_by_type = dict(time_axis)
            self.t = next(iter(time_axis.values())) if time_axis else None
        else:
            self._axes_by_type = {}
            self.t = time_axis
        self.meta      = metadata_dict
        self.snap_radius = 8
        self.label_map = label_map or {}
        self.color_map = color_map or {}
        self.emg_unit  = emg_unit
        self.ptp_start_ms         = ptp_start_ms
        self.ptp_end_ms           = ptp_end_ms
        # {stim_type: (start_ms, end_ms)} from the analysis. With amplitude
        # window anchoring the window is per stimulus type, and the file-wide
        # pair above is only a fallback.
        self.ptp_windows_by_type  = dict(ptp_windows_by_type or {})
        # Part of what positions a landmark (see _segment_geometry). Applying a
        # delay cuts the epoch elsewhere in the recording WITHOUT changing the
        # axis, so nothing else on this object can tell that the response has
        # moved under a stored index.
        self.delay_ms_map         = dict(delay_ms_map or {})
        # visible_pre_ms: how much pre-stim to SHOW (xlim)
        # _analysis_pre_ms: full pre-stim used for detection (may be larger)
        # May be one number, or {stim_type: pre_ms} where types are epoched
        # over different windows. Resolved per trial rather than at
        # construction, since the displayed type changes.
        self._visible_pre_map = (dict(visible_pre_ms)
                                 if isinstance(visible_pre_ms, dict) else {})
        self.visible_pre_ms = (None if self._visible_pre_map
                               else visible_pre_ms)
        self.onset_method              = onset_method
        self.onset_bootstrap_crit      = onset_bootstrap_crit
        self.onset_bootstrap_n         = onset_bootstrap_n
        self.onset_bigoni_smooth_ms    = onset_bigoni_smooth_ms
        self.onset_bigoni_min_run_ms   = onset_bigoni_min_run_ms
        self.onset_bigoni_walkback_sd  = onset_bigoni_walkback_sd

        # Detection parameters as one dict, keyed by PipelineConfig field name,
        # for detection.dispatch_onset. Built from the canonical defaults, then
        # `detection_params` (everything the analysis ran with), then the
        # individual keyword arguments above -- which are retained so existing
        # call sites keep working, and which win because a caller that names a
        # parameter explicitly means it.
        #
        # Passing the whole dict rather than adding a keyword per parameter is
        # what lets the inspector honour settings it previously ignored:
        # min_peak_amplitude, peak_fraction and slope_threshold were never
        # forwarded here, so re-detection silently used detector defaults while
        # the pipeline used the configured values.
        self.detection_params = dict(DETECTION_DEFAULTS)
        if detection_params:
            self.detection_params.update(
                {k: v for k, v in detection_params.items() if v is not None})
        self.detection_params.update({
            "onset_method":             onset_method,
            "onset_bootstrap_crit":     onset_bootstrap_crit,
            "onset_bootstrap_n":        onset_bootstrap_n,
            "onset_bigoni_smooth_ms":   onset_bigoni_smooth_ms,
            "onset_bigoni_min_run_ms":  onset_bigoni_min_run_ms,
            "onset_bigoni_walkback_sd": onset_bigoni_walkback_sd,
        })
        self._enable_auc_global        = enable_auc
        self.csp_search_end_ms    = csp_search_end_ms
        self.csp_min_silence_ms   = csp_min_silence_ms
        self.csp_min_return_ms    = csp_min_return_ms
        self.csp_criterion        = csp_criterion
        self.csp_significance     = csp_significance
        self.csp_n_boot           = csp_n_boot
        self.csp_rms_window_ms    = csp_rms_window_ms
        self.csp_max_mep_offset_ms = csp_max_mep_offset_ms
        self.latency_map           = latency_map or {}
        self._analysis_pre_ms     = analysis_pre_ms
        # extra_segs: {chan_name: {stim_type: [wide_seg_array]}}
        self._extra_segs          = extra_segs or {}
        self._wide_window_s       = wide_window_s
        # Averaged-mode individual traces to underlay behind the mean:
        # {stim_type: ndarray[n_trials, L]}. Empty on the normal path.
        self._underlays           = underlays or {}
        self._extra_axes          = []   # subplot axes for extra channels
        # Pre-populate silent period state from caller-specified csp_types.
        # Types in csp_types start ticked; all others start unticked.
        _csp_set = set(csp_types) if csp_types else set()
        self._silent_per_type = {
            k: (k in _csp_set) for k in segments_dict
        }
        # Per-segment user-override flag: True = user explicitly set this
        # checkbox themselves, so auto-sync logic must not overwrite it.
        self._silent_user_override = {}   # {(stim_type, idx): bool}

        # --------- header bar -------------------------------------------
        self.hdr = tk.Frame(self.top)              
        self.hdr.pack(fill="x", pady=6, padx=10)

        tk.Label(self.hdr, text="Event type:").pack(side="left")
        self.dd_event = ttk.Combobox(self.hdr, state="readonly",
                                    values=sorted(self.segments))
                                    # values=list(self.segments))
        self.dd_event.pack(side="left", padx=6)
        self.dd_event.bind("<<ComboboxSelected>>", lambda e: self._first())

        self.btn_prev = tk.Button(self.hdr, text="◀ Prev", width=9, command=self._prev)
        self.btn_prev.pack(side="right")
        self.btn_next = tk.Button(self.hdr, text="Next ▶", width=9, command=self._next)
        self.btn_next.pack(side="right", padx=(0, 4))

        # --------- matplotlib figure + toolbar in a dedicated frame --------
        # Use plt.Figure (not plt.subplots) to avoid registering the figure
        # with the interactive TkAgg backend, which would create a ghost window.
        self.fig_frame = tk.Frame(self.top)
        self.fig_frame.pack(fill="both", expand=True)

        self.fig = plt.Figure(figsize=(12, 6))
        self.ax_raw = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Zoom/pan toolbar — pack_toolbar=False so we control placement.
        # Must be packed AFTER canvas inside the same frame.
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self._toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame,
                                             pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(fill="x", side="bottom")

        # --------- status bar -------------------------------------------
        self.status = tk.Label(self.top, anchor="w")
        # Whatever the theme gives a label, captured once so the status line
        # can be turned red for a fault and put back afterwards without
        # hardcoding a colour that would be wrong on another platform.
        try:
            self._status_fg_default = self.status.cget("fg")
        except Exception:               # noqa: BLE001 — stubbed Tk in tests
            self._status_fg_default = "black"
        self.status.pack(fill="x", padx=10, pady=4)

        # --------- toggles ----------------------------------------------
        self.btn_bar = tk.Frame(self.top)                # << keep a handle 👉 self.btn_bar
        self.btn_bar.pack(pady=(0, 6))

        self.enable_silent = tk.BooleanVar(value=True)
        self.enable_auc    = tk.BooleanVar(value=enable_auc)
        self.exclude_var = tk.BooleanVar(value=False)
        # Draw this event type's median waveform behind the trial, so an odd
        # trial can be judged against its own condition rather than from memory.
        self.show_median_var = tk.BooleanVar(value=False)
        self._median_line = None
        self.note_enable_var = tk.BooleanVar(value=True)

        _edit_state = "disabled" if self.read_only else "normal"

        tk.Checkbutton(self.btn_bar, text="Silent period",
                    variable=self.enable_silent, state=_edit_state,
                    command=self._on_silent_toggle).pack(side="left", padx=10)

        def _on_auc_toggle():
            key = (self.cur_type, self.cur_idx)
            self.meta.setdefault(key, {})["auc_enabled"] = self.enable_auc.get()
            self._plot()
        # (auc_enabled is restored in _plot from segment metadata)
        tk.Checkbutton(self.btn_bar, text="AUC selector",
                    variable=self.enable_auc, state=_edit_state,
                    command=_on_auc_toggle).pack(side="left")
        self.link_onset_auc = tk.BooleanVar(value=True)
        tk.Checkbutton(self.btn_bar, text="Link AUC to onset & offset",
                    variable=self.link_onset_auc, state=_edit_state,
                    command=lambda: self._plot()).pack(side="left", padx=(8, 0))

        # Exclusion and notes exist only to write metadata, and a preview
        # keeps none, so they are omitted rather than shown dead.
        if not self.read_only:
            tk.Checkbutton(self.btn_bar, text="Exclude this segment",
                        variable=self.exclude_var,
                        command=lambda: self._set_exclude()).pack(side="left", padx=12) 
        tk.Checkbutton(self.btn_bar, text="Show event-type median",
                    variable=self.show_median_var,
                    command=lambda: self._plot()).pack(side="left", padx=12)

        if not self.read_only:
            tk.Checkbutton(self.btn_bar, text="Make a note",
                        variable=self.note_enable_var,
                        command=self._toggle_note_box).pack(side="left", padx=6)

        # ── Extra channel controls ─────────────────────────────────────
        if self._extra_segs:
            tk.Frame(self.btn_bar, width=2, bg="grey").pack(
                side="left", fill="y", padx=8)
            tk.Label(self.btn_bar, text="Extra channel:").pack(
                side="left")
            self._extra_chan_var = tk.StringVar(value="None")
            chan_opts = ["None"] + sorted(self._extra_segs.keys())
            self._extra_chan_dd = ttk.Combobox(
                self.btn_bar, textvariable=self._extra_chan_var,
                values=chan_opts, state="readonly", width=16)
            self._extra_chan_dd.pack(side="left", padx=(4,8))
            self._extra_chan_dd.bind(
                "<<ComboboxSelected>>", lambda e: self._plot())
            tk.Label(self.btn_bar, text="±(s):").pack(side="left")
            self._wide_var = tk.DoubleVar(value=self._wide_window_s)
            tk.Spinbox(
                self.btn_bar, from_=0.5, to=10.0, increment=0.5,
                textvariable=self._wide_var, width=5,
                command=self._plot).pack(side="left", padx=(2,0))
            self._flip_var = tk.BooleanVar(value=False)
            tk.Checkbutton(self.btn_bar, text="Flip",
                variable=self._flip_var,
                command=self._plot_extra_channel).pack(side="left", padx=(6,0))
        else:
            self._extra_chan_var = tk.StringVar(value="None")
            self._wide_var      = tk.DoubleVar(value=self._wide_window_s)
            self._flip_var      = tk.BooleanVar(value=False)
        
        # Note box — visible by default. Built either way so the methods that
        # reference it need no read_only branch of their own; simply never
        # shown in a preview.
        self.note_box = scrolledtext.ScrolledText(self.top, height=3)
        if self.read_only:
            self.note_enable_var.set(False)
            self.note_box_is_shown = False
        else:
            self.note_box.pack(fill="x", padx=10, pady=(4, 6))
            self.note_box_is_shown = True

        # --------- keyboard navigation ----------------------------------
        self.top.bind("<Right>", lambda e: self._next())
        self.top.bind("<Left>",  lambda e: self._prev())

        # --------- close ------------------------------------------------
        # "Save edits & close" would be a lie in a preview: _close_and_save
        # commits the note box on the way out, and the caller discards the
        # metadata regardless.
        self.btn_row = tk.Button(
            self.top,
            text="Close preview" if self.read_only else "Save edits & close",
            width=20,
            command=self._close_preview if self.read_only
                    else self._close_and_save)            
        self.btn_row.pack(side="bottom", pady=(0, 8))    # 👈 ALWAYS at the bottom
        # --------- internal state ---------------------------------------
        self.cur_type = self.dd_event["values"][0]
        self._select_axis()
        self.cur_idx  = 0
        self.dd_event.current(0)

        # AUC draggable lines and fill shading
        self._auc_lines = []   # [DraggableLine start, DraggableLine end]
        self._auc_fill  = []

        # Maximise the inspector window on open — gives the most room for
        # the figure and makes marker placement much easier.
        # _resize_window will skip geometry changes while the window is maximised.
        import sys as _sys
        try:
            if _sys.platform in ("win32", "darwin"):
                self.top.state("zoomed")
            else:
                self.top.attributes("-zoomed", True)
        except Exception:
            pass   # fallback: window opens at default size

        self._plot()      # first draw
    # ──────────────────────────────────────────────────────────────────────
    def _on_silent_toggle(self):
        key = (self.cur_type, self.cur_idx)
        new_state = self.enable_silent.get()
        self._silent_per_type[self.cur_type] = new_state
        # Record that the user explicitly set this segment's state.
        self._silent_user_override[key] = new_state
        if not new_state:
            # User ticked OFF — remove markers so they don't persist in output.
            m = self.meta.get(key, {})
            m.pop('silent_start_idx',    None)
            m.pop('silent_end_idx',      None)
            m.pop('csp_detection_failed', None)
        else:
            # User ticked ON — clear any previous failure flag so detection
            # will run fresh when _plot redraws below.
            m = self.meta.get(key, {})
            m.pop('csp_detection_failed', None)
            m.pop('silent_start_idx',    None)
            m.pop('silent_end_idx',      None)
            # Turning cSP off or on changes WHICH marker carries the offset, so
            # a value stored under the other rule no longer applies.
            m.pop('mep_offset_idx',      None)
        self._plot()

    def _first(self):
        self._save_note_from_widget()
        self._silent_per_type[self.cur_type] = self.enable_silent.get()
        self.cur_type, self.cur_idx = self.dd_event.get(), 0
        self._select_axis()
        self._plot()

    def _csp_settings(self):
        """
        The cSP settings this review is running under, as the same value the
        pipeline builds.

        Assembled from the attributes the constructor was given, which come
        from the interface, so changing a field in Settings changes review and
        analysis together. Built here rather than stored so that a settings
        change between draws is picked up without the Inspector caching a
        stale copy of it.
        """
        return CspSettings.from_source({
            "csp_min_silence_ms":    self.csp_min_silence_ms,
            "csp_min_return_ms":     self.csp_min_return_ms,
            "csp_criterion":         self.csp_criterion,
            "csp_significance":      self.csp_significance,
            "csp_n_boot":            self.csp_n_boot,
            "csp_search_end_ms":     self.csp_search_end_ms,
            "csp_max_mep_offset_ms": self.csp_max_mep_offset_ms,
            "csp_rms_window_ms":     self.csp_rms_window_ms,
        })

    def _next(self):
        if self._closed():
            return
        self._save_note_from_widget()                    
        self.cur_idx = (self.cur_idx + 1) % len(self.segments[self.cur_type])
        self._plot()

    def _prev(self):
        if self._closed():
            return
        self._save_note_from_widget()                       
        self.cur_idx = (self.cur_idx - 1) % len(self.segments[self.cur_type])
        self._plot()

    # ---------------------------------------------------------------- helper
    def _update_meta(self, field, new_idx):
        key = (self.cur_type, self.cur_idx)
        m   = self.meta.setdefault(key, {})
        m[field] = new_idx
        if field == "onset_idx":
            # A placed marker is a measurement, whatever the detector managed.
            m.pop("onset_auto_failed", None)
        _linked = (getattr(self, "link_onset_auc", None)
                   and self.link_onset_auc.get()
                   and self.enable_auc.get())

        # When onset moves and link is active, sync AUC start line to match
        if field == "onset_idx" and _linked:
            m["auc_start_idx"] = new_idx
            if self._auc_lines:
                self._auc_lines[0].set_idx(new_idx)
                self._redraw_auc_fill(m)

        # The same for the other end. Area under the curve is integrated from
        # onset to the end of the response, so whichever marker carries the
        # offset must move the AUC end with it -- otherwise the window shown in
        # this review and the window the pipeline integrated differ, and the
        # AUC in the results file is not the one on screen.
        elif _linked and field == offset_marker_field(
                self.enable_silent.get(),
                'silent_start_idx' in m and 'silent_end_idx' in m):
            m["auc_end_idx"] = new_idx
            if len(self._auc_lines) >= 2:
                self._auc_lines[1].set_idx(new_idx)
                self._redraw_auc_fill(m)
        self._refresh_status()
    
    def _redraw_auc_fill(self, m):
        """Redraw fill_between shading for current AUC window."""
        for fc in self._auc_fill:
            try: fc.remove()
            except Exception: pass
        self._auc_fill = []
        a0 = m.get("auc_start_idx")
        a1 = m.get("auc_end_idx")
        if a0 is None or a1 is None or a0 >= a1:
            self.canvas.draw_idle()
            return
        seg = self.segments[self.cur_type][self.cur_idx]
        t_win   = self.t[a0:a1]
        emg_win = seg[a0:a1]
        self._auc_fill.append(self.ax_raw.fill_between(
            t_win, 0, emg_win, where=(emg_win >= 0),
            alpha=0.35, color="tab:blue", zorder=2))
        self._auc_fill.append(self.ax_raw.fill_between(
            t_win, 0, emg_win, where=(emg_win < 0),
            alpha=0.35, color="tab:blue", zorder=2))
        self.canvas.draw_idle()

    def _ylab(self, base="EMG"):
            return f"{base} ({self.emg_unit})" if self.emg_unit else base
    
    def _set_exclude(self):                                                
        key = (self.cur_type, self.cur_idx)
        self.meta.setdefault(key, {})
        self.meta[key]['exclude'] = self.exclude_var.get()
        # The condition template is built from the retained trials, so an
        # exclusion change invalidates it.
        self._template_cache.pop(self.cur_type, None)
        self._refresh_status()

    def _segment_geometry(self, stim_type=None):
        """What positions a landmark within a segment, as a comparable string.

        A stored index means nothing on its own: it is a position, and it is
        only the same position while the segment is cut the same way and the
        amplitude window falls in the same place. Three things move it --

          * the EVENT DELAY, which shifts where t=0 sits in the recording, so
            the whole response moves against the axis;
          * the EPOCH WINDOW, which changes how much precedes the stimulus and
            therefore renumbers every sample;
          * the AMPLITUDE WINDOW, which is where the peaks were searched for.

        Recorded per stimulus type, because all three are per type.

        Deliberately not a hash: the value is written into the session JSON and
        read by a human when something looks wrong, and "d=17.5,e=100.0/300.0,
        a=12.0/50.0" says what changed while a digest says only that something
        did. Rounded so that float noise does not read as a change.
        """
        st = self.cur_type if stim_type is None else stim_type
        # getattr, because this must not depend on a caller remembering to
        # pass the map. Absent, the delay reads as 0.0 -- which is wrong in
        # exactly the case this exists for, so it is passed at both call sites
        # and the default is a fallback rather than the normal path.
        try:
            delay = float((getattr(self, "delay_ms_map", None) or {})
                          .get(st, 0.0) or 0.0)
        except Exception:                   # noqa: BLE001 — odd map contents
            delay = 0.0
        pre = float(self.t[0]) if len(self.t) else 0.0
        post = float(self.t[-1]) if len(self.t) else 0.0
        a0, a1 = self._ptp_window_ms(st)
        return (f"d={delay:.2f},e={pre:.1f}/{post:.1f},"
                f"a={float(a0 or 0):.1f}/{float(a1 or 0):.1f}")

    def _ptp_window_ms(self, stim_type=None):
        """Amplitude window for a stimulus type, in ms relative to the stimulus.

        Returns the window the analysis measured with when one was supplied,
        otherwise the file-wide pair from tab 1c. Anchoring makes the window
        per type, so using the file-wide values during review would measure a
        different interval from the one that produced the results.
        """
        stim_type = self.cur_type if stim_type is None else stim_type
        win = self.ptp_windows_by_type.get(stim_type)
        if win:
            try:
                a, b = float(win[0]), float(win[1])
                if b > a:
                    return a, b
            except Exception:
                pass
        return self.ptp_start_ms, self.ptp_end_ms

    def _condition_template(self, stim_type=None):
        """Median waveform across the retained trials of one stimulus type.

        The derivative-ratio detector uses this to reject a trial whose peak
        sits implausibly far from where the condition's peak normally falls.
        It is the only detector that consults it, and it is the only gate the
        template feeds -- it plays no part in locating the onset.

        Excluded trials are left out, matching pipeline_detect_onsets. If the
        two used different trial sets the analysis and this window would judge
        the same trial against different landmarks, so a trial rejected during
        analysis could be accepted on review, or the reverse -- the divergence
        between the two detection paths that sharing one dispatch was meant to
        end.

        Returns None when fewer than two trials remain, since a median over one
        trial is that trial.
        """
        stim_type = self.cur_type if stim_type is None else stim_type
        if stim_type in self._template_cache:
            return self._template_cache[stim_type]

        segs = self.segments.get(stim_type)
        template = None
        try:
            if segs is not None and len(segs) >= 2:
                keep = [np.asarray(segs[i], dtype=float)
                        for i in range(len(segs))
                        if not self.meta.get((stim_type, i), {}).get("exclude", False)]
                if len(keep) >= 2:
                    template = np.median(np.vstack(keep), axis=0)
        except Exception:
            template = None
        self._template_cache[stim_type] = template
        return template

    def _seed_offset_idx(self, emg, m, stim_idx, dt_ms):
        """Initial MEP-offset index from the envelope detector, or None.

        Only ever used to place the marker the first time a trial is viewed.
        Once the analyst has moved it, the stored value is authoritative and
        this is not consulted again -- re-seeding would silently undo a manual
        decision on the next redraw.
        """
        try:
            onset_ms = (m['onset_idx'] - stim_idx) * dt_ms
            pre_ms = -float(self.t[0])
            off_ms = detect_mep_offset(
                np.asarray(emg, dtype=float), 1000.0 / dt_ms,
                onset_ms=onset_ms, pre_ms=pre_ms,
                search_end_ms=float(self.t[-1]),
                max_duration_ms=float(self.detection_params.get(
                    "mep_offset_max_duration_ms", 100.0)),
                min_duration_ms=float(self.detection_params.get(
                    "mep_offset_min_duration_ms", 5.0)),
                min_return_ms=float(self.detection_params.get(
                    "mep_offset_min_return_ms", 10.0)),
                env_window_ms=float(self.detection_params.get(
                    "mep_offset_env_window_ms", 5.0)),
                criterion=float(self.detection_params.get(
                    "mep_offset_criterion", 2.5)),
                peak_frac=float(self.detection_params.get(
                    "mep_offset_peak_frac", 0.12)))
            if off_ms is None:
                return None
            idx = int(round(stim_idx + off_ms / dt_ms))
            return idx if 0 <= idx < len(emg) else None
        except Exception:
            return None

    def _toggle_note_box(self):                                            
        # show/hide the note box widget
        if self.note_enable_var.get():
            if not self.note_box_is_shown:
                self.note_box.pack(fill="x", padx=10, pady=(4, 6))
                self.note_box_is_shown = True
                self._resize_window()
        else:
            if self.note_box_is_shown:
                # persist current note
                self._save_note_from_widget()
                self.note_box.pack_forget()
                self.note_box_is_shown = False
                self._resize_window()

    def _save_note_from_widget(self):
        if self._closed():
            return
        """Save the note box content to metadata for the current segment."""
        key = (self.cur_type, self.cur_idx)
        # Only save if the note box is currently shown — if hidden, the widget
        # may contain stale text from a previous segment.
        if not self.note_box_is_shown:
            return
        txt = self.note_box.get("1.0", "end").strip()
        if txt:
            self.meta.setdefault(key, {})['note'] = txt
        else:
            # wipe if empty
            if key in self.meta and 'note' in self.meta[key]:
                del self.meta[key]['note']
    
    def _select_axis(self):
        """Point self.t and the x-limit at the current type's window."""
        if self._visible_pre_map:
            _vp = self._visible_pre_map.get(self.cur_type)
            if _vp is not None:
                self.visible_pre_ms = float(_vp)
        if not self._axes_by_type:
            return
        axis = self._axes_by_type.get(self.cur_type)
        if axis is not None:
            self.t = axis

    def _resize_window(self):
        """Resize the Toplevel so every widget (note box included) is visible.
        Skipped when the window is maximised — geometry() would un-maximise it."""
        try:
            state = self.top.state()
            if state == "zoomed":
                return
        except Exception:
            pass
        try:
            if self.top.attributes("-zoomed"):
                return
        except Exception:
            pass

        self.top.update_idletasks()

        pieces = [self.hdr, self.fig_frame, self.status, self.btn_bar, self.btn_row]
        if self.note_box_is_shown:
            pieces.append(self.note_box)

        need_h = sum(p.winfo_reqheight() for p in pieces) + 40
        need_w = max(p.winfo_reqwidth()  for p in pieces) + 40
        self.top.geometry(f"{need_w}x{need_h}")

    # ---------------------------------------------------------------- plot
    def _plot(self):
        if self._closed():
            return
        """Redraw the inspector for the currently‑selected segment."""
        # Ensure the Toplevel and canvas have settled to their correct geometry
        # before drawing.  The very first call (from __init__) runs before the
        # window has been shown, so without this the figure overflows its canvas
        # and the content appears duplicated on the right side.
        try:
            self.top.update()
        except tk.TclError:
            return   # window was already destroyed
        # ---------- shortcuts ------------------------------------------------
        emg = self.segments[self.cur_type][self.cur_idx]

        # auto‑repair an unexpected length mismatch --------------------------
        if len(emg) != len(self.t):
            self.t = np.linspace(self.t[0],
                                 self.t[-1] + (self.t[1] - self.t[0]),
                                 len(emg), endpoint=False)

        colour = self.color_map.get(self.cur_type, "tab:blue")
        lbl    = self.label_map .get(self.cur_type, self.cur_type)
        key    = (self.cur_type, self.cur_idx)

        # ---------- per‑segment metadata container --------------------------
        m = self.meta.setdefault(key, {})

        # ---------- sync silent-period checkbox for this segment ------------
        _type_wants_silent = self._silent_per_type.get(self.cur_type, False)
        _has_markers       = 'silent_start_idx' in m and 'silent_end_idx' in m
        _det_failed        = m.get('csp_detection_failed', False)
        _user_set          = key in self._silent_user_override

        if _user_set:
            # User explicitly toggled this segment — honour their choice.
            # Do not overwrite with auto-detection results or type defaults.
            self.enable_silent.set(self._silent_user_override[key])
        elif _has_markers:
            # Markers exist (auto-detected or manually placed) — show them.
            self.enable_silent.set(True)
        elif _det_failed:
            # Detection was attempted and failed — leave unticked so user
            # can decide to manually place markers by ticking it themselves.
            self.enable_silent.set(False)
        else:
            # Not yet attempted — checkbox will be set after detection below.
            self.enable_silent.set(False)

        # ---------- sync “exclude”, AUC & note widgets ----------------------
        self.exclude_var.set(m.get('exclude', False))
        self.enable_auc.set(m.get('auc_enabled', self._enable_auc_global))

        # note‑box follow‑through
        note_txt = m.get('note', '')
        if note_txt or self.note_enable_var.get():
            # show the widget and populate it (or clear if empty)
            if not self.note_box_is_shown:
                self.note_box.pack(fill="x", padx=10, pady=(4, 6))
                self.note_box_is_shown = True
            if not self._widget_alive(self.note_box):
                return
            self.note_box.delete("1.0", "end")
            self.note_box.insert("1.0", note_txt)
            self.note_enable_var.set(True)
        elif self.note_box_is_shown:
            self.note_box.pack_forget()
            self.note_box_is_shown = False
            self.note_enable_var.set(False)

        # ---------- automatic landmarks -------------------------------------
        # By default use the whole segment …
        p_max  = int(np.argmax(emg))
        p_min  = int(np.argmin(emg))

        # … but if the user defined a PTP window, constrain the search.
        #
        # Prefer the window the ANALYSIS used for this stimulus type. With
        # amplitude window anchoring the window is placed per type from that
        # type's median onset, so the file-wide pair is wrong for every type
        # that was anchored: the review would search a different interval and
        # place the peak markers somewhere the analysis never looked.
        _w0, _w1 = self._ptp_window_ms()
        if _w0 is not None and _w1 is not None:
            mask = (self.t >= _w0) & (self.t <= _w1)
            if np.any(mask):
                idxs = np.where(mask)[0]
                # local max/min *within* that window
                p_max = idxs[np.argmax(emg[idxs])]
                p_min = idxs[np.argmin(emg[idxs])]

        dt_ms  = self.t[1] - self.t[0]
        fs     = int(round(1000 / dt_ms))
        # Use the full analysis pre-stim window — inspector segments are now
        # extracted with prestim_ms pre-stim so the full baseline is available.
        _pre_ms = (self._analysis_pre_ms
                   if self._analysis_pre_ms is not None
                   else abs(int(self.t[0])))
        # A missing profile is bounded by the amplitude window and SAID, not
        # replaced by a constant.
        #
        # The fallback was a hardcoded 10-50 ms, and every detector bounds its
        # result by the minimum -- so a stimulus type absent from the map
        # reported an onset of about 10 ms whatever the trace did. It happens
        # most easily on a channel that was never set up: the maps are per
        # channel, so previewing EMG 2 with EMG 1 configured finds nothing.
        #
        # The pipeline warns in the same situation. This did not, which is why
        # the preview and the analysis disagreed with each other.
        _lat = self.latency_map.get(self.cur_type)
        if not _lat:
            _lat = (float(self.ptp_start_ms), float(self.ptp_end_ms))
            if not getattr(self, "_warned_no_latency", None):
                self._warned_no_latency = set()
            if self.cur_type not in self._warned_no_latency:
                self._warned_no_latency.add(self.cur_type)
                try:
                    self.status.config(
                        fg="#B03A2E",
                        text=(f"'{self.cur_type}' has no latency profile on "
                              f"this channel — onsets bounded by the amplitude "
                              f"window ({_lat[0]:.0f}-{_lat[1]:.0f} ms). Set "
                              f"its stimulus type and muscle group on tab 1a."))
                except Exception:
                    pass
        _min_lat, _max_lat = _lat

        # One shared dispatch with the pipeline (detection/dispatch.py). This
        # was previously a four-branch copy that did not know about methods
        # added later and forwarded no amplitude/peak-fraction parameters, so
        # a re-detection here could use a different algorithm, and different
        # settings, than the analysis it was reviewing.
        onset_ms = dispatch_onset(
            emg, fs, self.detection_params,
            pre_ms=_pre_ms,
            search_start_ms=self.ptp_start_ms or 10,
            search_end_ms=self.ptp_end_ms or 50,
            min_latency_ms=_min_lat,
            max_latency_ms=_max_lat,
            template=self._condition_template(),
        )
        stim_idx = np.argmin(np.abs(self.t))
        onset    = stim_idx if onset_ms is None else stim_idx + int(round(onset_ms / dt_ms))
        onset    = max(onset, stim_idx)

        # Why detection found nothing, when it found nothing.
        #
        # min_peak_amplitude gates all seven detectors: a trial whose response
        # is smaller than it is rejected before an onset is looked for. That is
        # the gate doing its job -- it is what stops an onset being fitted to
        # noise -- but a rejected trial and a trial whose search window was
        # wrong both read "not detected", and telling them apart took two
        # rounds of looking at waveforms.
        #
        # Measured here rather than returned by the detectors: they each apply
        # the gate against their own window and returning a reason from all
        # seven, through the median dispatcher, would be a much larger change
        # for the same sentence. Reported as "below the minimum" only when it
        # actually is, so the ordinary "no onset found" is not diluted.
        _fail_reason = None
        if onset_ms is None:
            try:
                _gate = float((self.detection_params or {})
                              .get("min_peak_amplitude", 0.0) or 0.0)
                _mask = (self.t >= _min_lat) & (self.t <= _max_lat)
                if _gate > 0 and np.any(_mask):
                    _p2p = float(np.ptp(emg[_mask]))
                    if _p2p < _gate:
                        _fail_reason = (
                            f"response {_p2p:.3f} mV is below the "
                            f"{_gate:.3f} mV minimum — lower 'Min peak "
                            f"amplitude' in Settings ▸ Preferences ▸ "
                            f"Detection, then re-run")
            except Exception:               # noqa: BLE001 — never block a draw
                _fail_reason = None

        # ---------- discard stale landmark indices --------------------------
        # Metadata persists across runs and is honoured by the setdefault calls
        # below, so an index stored when the analysis window was longer would be
        # reused against a shorter segment.  scatter() then raises IndexError
        # mid-draw and the inspector never opens at all.
        #
        # An index that cannot exist in this segment is stale, not a user edit
        # worth preserving, so drop it and let detection re-seed.  Clamping to
        # the segment edge instead would be worse: it silently plants a landmark
        # at a boundary the analyst never chose and reports it as a measurement.
        _seg_len = len(emg)
        _LANDMARKS = ('ptp_min_idx', 'ptp_max_idx', 'onset_idx',
                      'silent_start_idx', 'silent_end_idx',
                      'mep_offset_idx', 'auc_start_idx', 'auc_end_idx')
        _stale = [f for f in _LANDMARKS
                  if m.get(f) is not None
                  and not (0 <= int(m[f]) < _seg_len)]
        if _stale:
            for f in _stale:
                m.pop(f, None)
            if not getattr(self, '_stale_meta_warned', False):
                self._stale_meta_warned = True
                print(f"[inspector] Discarded landmark(s) {', '.join(_stale)} "
                      f"stored outside the current {_seg_len}-sample segment "
                      f"(analysis window changed since they were saved); "
                      f"re-detecting.")

        # ---------- discard landmarks the geometry has moved under ----------
        #
        # An index is a position in a segment, and it only means anything while
        # the segment is cut the same way. The check above catches an index that
        # no longer FITS. It does not catch one that still fits but no longer
        # points at what it was placed on -- and that is the commoner case.
        #
        # Applying a 17.5 ms event delay cuts the epoch 35 samples later, so the
        # response moves 17.5 ms earlier against the same axis while every
        # stored index stays put. On one real recording that put PTP min at
        # +0.013 mV and PTP max at -0.028 mV, and the status bar reported
        # "PTP: -0.04 mV": the labels had swapped over and a negative
        # peak-to-peak amplitude was presented as a measurement. Changing the
        # epoch window or the amplitude window moves things the same way.
        #
        # So the geometry that positions a landmark is recorded WITH it, and a
        # landmark whose geometry has changed is dropped rather than reused.
        # Manual edits go too, which is unavoidable: an edit is a position, and
        # the position no longer refers to what the analyst was looking at.
        _geom = self._segment_geometry()
        if m.get('_geometry') not in (None, _geom):
            _moved = [f for f in _LANDMARKS if f in m]
            for f in _moved:
                m.pop(f, None)
            m.pop('onset_auto_failed', None)
            if _moved and not getattr(self, '_geom_meta_warned', False):
                self._geom_meta_warned = True
                print(f"[inspector] Re-detecting landmarks for "
                      f"'{self.cur_type}': the event delay or the analysis "
                      f"window has changed since they were saved, so the "
                      f"stored positions no longer point at the same part of "
                      f"the response.")
        m['_geometry'] = _geom

        # ---------- seed metadata defaults ----------------------------------
        m.setdefault('ptp_min_idx', p_min)
        m.setdefault('ptp_max_idx', p_max)
        # Mark a NON-DETECTION as such rather than letting the stimulus index
        # pass for a measurement.
        #
        # When dispatch_onset returns None the marker falls back to the
        # stimulus, which reads as "Latency: 0.0 ms" -- and, worse, that index
        # was then written into the metadata and returned by "Save edits &
        # close". The analysis honours a stored onset_idx as a manual override,
        # so reviewing a trial whose onset could not be found silently turned a
        # blank latency into a measured 0.0 ms. The flag is cleared the moment
        # the analyst drags the marker, which is a real decision.
        if onset_ms is None and 'onset_idx' not in m:
            m['onset_auto_failed'] = True
            # Kept beside the flag so the status line can say WHY rather than
            # only that nothing was found.
            if _fail_reason:
                m['onset_fail_reason'] = _fail_reason
            else:
                m.pop('onset_fail_reason', None)
        elif onset_ms is not None:
            m.pop('onset_auto_failed', None)
            m.pop('onset_fail_reason', None)
        m.setdefault('onset_idx',   onset)

        # ---------- decide whether to run CSP detection ---------------------
        _user_set     = key in self._silent_user_override
        _user_on      = _user_set and self._silent_user_override[key]
        _user_off     = _user_set and not self._silent_user_override[key]


        if _user_on and not _has_markers:
            m.pop('csp_detection_failed', None)
            _det_failed = False
            _should_detect = True
        elif _user_off:
            _should_detect = False
        elif not _user_set and _type_wants_silent and not _has_markers and not _det_failed:
            _should_detect = True
        else:
            _should_detect = False


        if _should_detect:
            _csp_reason = []

            # ── cSP search anchor ─────────────────────────────────────────
            # The search starts at the 2nd PTP landmark, so the detector can
            # never place cSP onset inside the MEP. Everything else about the
            # window, including how csp_max_mep_offset_ms is applied, lives in
            # detect_csp_for_trial and is shared with the pipeline.
            #
            # This block used to compute the window itself and capped the
            # search END at second_peak + csp_max_mep_offset_ms. That field
            # means "the cSP must START within this many ms of the 2nd MEP
            # peak" (see its declaration in app.py), so capping the window
            # truncated the very quantity being measured: with the field at
            # its default of 100 ms, no silent period longer than ~100 ms
            # could be found here at all, while the pipeline -- which never
            # applied the cap -- reported the true duration for the same
            # trial. A comment here asserted the cap was not applied.
            _second_peak_idx = max(m['ptp_min_idx'], m['ptp_max_idx'])
            _second_peak_ms  = float(self.t[_second_peak_idx])

            if _second_peak_ms >= float(self.t[-1]):
                m['csp_detection_failed'] = True
                m['csp_reason'] = (
                    f"Search window collapsed: 2nd MEP peak at "
                    f"{_second_peak_ms:.0f} ms is at or past the end of the "
                    f"segment")
                self.enable_silent.set(False)
            else:
                csp = detect_csp_for_trial(
                    emg, fs, self.t,
                    self._csp_settings(),
                    second_peak_ms=_second_peak_ms,
                    pre_ms=(self._analysis_pre_ms
                            if self._analysis_pre_ms is not None
                            else abs(int(self.t[0]))),
                    reason_out=_csp_reason)
                m['csp_reason'] = _csp_reason[0] if _csp_reason else ''
                if csp is not None:
                    m['silent_start_idx'], m['silent_end_idx'] = csp
                    self.enable_silent.set(True)
                    # Clear user override — detection succeeded
                    self._silent_user_override.pop(key, None)
                else:
                    m['csp_detection_failed'] = True
                    m.pop('silent_start_idx', None)
                    m.pop('silent_end_idx',   None)
                    self.enable_silent.set(False)

        # ---------- auto AUC: onset → cSP start --------------------------------
        # Run silently for ALL event types to get the AUC endpoint, even when
        # the user has not selected this type for cSP measurement.
        # Results are stored in '_auc_csp_end_idx' (separate from visible cSP
        # markers) and only used to set the AUC window.
        if 'onset_idx' in m and 'auc_start_idx' not in m:
            # Use stored cSP if already detected (visible or hidden)
            _csp_end_for_auc = m.get('silent_start_idx', None)

            # The background cSP search below runs ONLY for stimulus types the
            # analyst has assigned to cSP measurement. It used to run for every
            # type, which meant a resting M-wave had its integration window
            # ended by a cortical-silent-period detector -- a quantity that does
            # not exist without voluntary contraction, and one the pipeline
            # never computes for a non-cSP type. The two therefore reported
            # different AUCs for the same trial. Everything else now ends the
            # window at the detected MEP offset, matching the analysis.
            if not self.enable_silent.get():
                _csp_end_for_auc = None
                m['_auc_csp_tried'] = True

            # If no visible cSP, try a silent background detection
            if _csp_end_for_auc is None and not m.get('_auc_csp_tried', False):
                m['_auc_csp_tried'] = True   # only attempt once per segment
                try:
                    _second_peak_idx = max(m['ptp_min_idx'], m['ptp_max_idx'])
                    _second_peak_ms  = float(self.t[_second_peak_idx])
                    if _second_peak_ms < float(self.t[-1]):
                        _bg_csp = detect_csp_for_trial(
                            emg, fs, self.t,
                            self._csp_settings(),
                            second_peak_ms=_second_peak_ms,
                            pre_ms=(self._analysis_pre_ms
                                    if self._analysis_pre_ms is not None
                                    else abs(int(self.t[0]))))
                        if _bg_csp is not None:
                            _csp_end_for_auc = _bg_csp[0]  # start of silence = end of MEP
                            m['_auc_csp_end_idx'] = _csp_end_for_auc
                except Exception:
                    pass

            # Set AUC from onset to cSP start (silent or visible)
            if _csp_end_for_auc is not None and m['onset_idx'] < _csp_end_for_auc:
                m['auc_start_idx'] = int(m['onset_idx'])
                m['auc_end_idx']   = int(_csp_end_for_auc)
        self.ax_raw.clear()
        # ---------- averaged-mode underlays (guarded) --------------------
        # Draw the individual clean traces faintly behind the mean so the
        # user can judge the average's quality. When no underlays were
        # supplied (the normal per-trial path) this is skipped entirely and
        # the plot is byte-for-byte unchanged.
        _underlay = self._underlays.get(self.cur_type) if self._underlays else None
        if _underlay is not None:
            for _utr in _underlay:
                if len(_utr) == len(self.t):
                    self.ax_raw.plot(self.t, _utr, color=colour,
                                     lw=0.4, alpha=0.20, zorder=1)
        # Event-type median, drawn BEHIND the trial so the trial stays the
        # figure's subject. This is the same waveform the derivative-ratio
        # detector compares each trial against, so what is shown is what the
        # algorithm saw rather than an approximation of it. Excluded trials are
        # left out, so removing a bad trial visibly updates the reference.
        self._median_line = None
        if self.show_median_var.get():
            _med = self._condition_template()
            if _med is not None and len(_med) == len(self.t):
                self._median_line = self.ax_raw.plot(
                    self.t, _med, color="0.35", lw=2.2, alpha=0.35, zorder=1,
                    label="event-type median")[0]

        self.ax_raw.plot(self.t, emg, color=colour, lw=1)
        self.ax_raw.axvline(0, color="k", ls="--")
        # Limit x-axis to the visible window even if segment has more pre-stim
        _xlim_left = (-self.visible_pre_ms
                      if self.visible_pre_ms is not None
                      else self.t[0])
        self.ax_raw.set_xlim(_xlim_left, self.t[-1])
        self.ax_raw.set(
            title=f"{lbl}  –  segment {self.cur_idx+1}/{len(self.segments[self.cur_type])}",
            ylabel=self._ylab()
        )

        # ---------- draggable markers ---------------------------------------
        self._dpts = []

        def _add(idx0, c, field, label=None):
            # A marker outside the segment is skipped rather than clamped: the
            # draw must not fail, but nor should a landmark appear at a sample
            # the detector never chose.  Stale values are normally cleared
            # above; this guards any path that reaches here with one anyway.
            if idx0 is None or not (0 <= int(idx0) < len(emg)):
                print(f"[inspector] Skipped marker '{field}' at index {idx0} "
                      f"(outside the {len(emg)}-sample segment).")
                return
            idx0 = int(idx0)
            mk   = 'x' if field.startswith('ptp_') else 'o'
            alp  = 0.6 if field == "onset_idx" else 1.0
            scat = self.ax_raw.scatter(self.t[idx0], emg[idx0],
                                       s=80, color=c, marker=mk, alpha=alp,
                                       zorder=3, label=label)
            self._dpts.append(
                DraggablePoint(
                    scat, self.t, emg, idx0,
                    lambda i, f=field: self._update_meta(f, i),
                    role=field, radius=self.snap_radius,
                    read_only=self.read_only
                )
            )

        _add(m['ptp_min_idx'], self.DOT_COLOURS["ptp_min_idx"], "ptp_min_idx", label="PTP min")
        _add(m['ptp_max_idx'], self.DOT_COLOURS["ptp_max_idx"], "ptp_max_idx", label="PTP max")
        _add(m['onset_idx'],   self.DOT_COLOURS["onset_idx"],   "onset_idx",   label="Onset")

        _has_csp = ('silent_start_idx' in m and 'silent_end_idx' in m)
        if self.enable_silent.get() and _has_csp:
            # This marker doubles as the MEP offset: during contraction the end
            # of the response and the start of the silent period are the same
            # event, so there is one marker for them and dragging it moves
            # both. offset_marker_field states that rule once, and
            # resolve_mep_offset applies the same precedence when quantifying.
            _add(m['silent_start_idx'], self.DOT_COLOURS["silent_start_idx"],
                 "silent_start_idx", label="cSP start / MEP offset")
            _add(m['silent_end_idx'],   self.DOT_COLOURS["silent_end_idx"],
                 "silent_end_idx",   label="cSP end")
        elif offset_marker_field(self.enable_silent.get(), _has_csp) \
                == "mep_offset_idx":
            # No silent period to end the response, so the offset gets a marker
            # of its own. Seeded from the envelope detector on first view; the
            # pipeline already gives a stored mep_offset_idx top precedence, so
            # a dragged value survives into the results without further work.
            if 'mep_offset_idx' not in m:
                _seed = self._seed_offset_idx(emg, m, stim_idx, dt_ms)
                if _seed is not None:
                    m['mep_offset_idx'] = _seed
            if 'mep_offset_idx' in m:
                _add(m['mep_offset_idx'], self.DOT_COLOURS["mep_offset_idx"],
                     "mep_offset_idx", label="MEP offset")

        self.ax_raw.legend(loc="upper right", fontsize=12, frameon=False)
        self.fig._draggables = self._dpts

        # ---------- AUC selector — two DraggableLines ----------------------
        # Remove previous fill and lines
        for _fc in self._auc_fill:
            try: _fc.remove()
            except Exception: pass
        self._auc_fill = []
        for _dl in self._auc_lines:
            try: _dl.remove()
            except Exception: pass
        self._auc_lines = []

        show_auc = self.enable_auc.get()
        if show_auc:
            # Ensure AUC window exists — default onset → cSP or onset+50ms
            if "auc_start_idx" not in m and "onset_idx" in m:
                a0 = int(m["onset_idx"])
                # End the integration where the response ends. The marker that
                # carries the offset is the cSP start during contraction and a
                # dedicated one otherwise, matching what the pipeline
                # integrates. The old onset + 50 ms rule was a fixed width
                # unrelated to the response, and it disagreed with the results
                # file for exactly that reason; it survives only as a last
                # resort when no offset could be established.
                _f = offset_marker_field(
                    self.enable_silent.get(),
                    'silent_start_idx' in m and 'silent_end_idx' in m)
                a1 = m.get(_f)
                if a1 is None or int(a1) <= a0:
                    a1 = min(len(self.t) - 1, a0 + int(50 * fs / 1000))
                m["auc_start_idx"] = a0
                m["auc_end_idx"]   = int(a1)

            # Enforce the link on LOAD, not only while dragging.
            #
            # The analysis stores auc_start_idx and auc_end_idx in the segment
            # metadata, so a reviewed file arrives with a window already set and
            # the seeding above -- guarded on the key being absent -- never runs.
            # The result was an AUC end inherited from the analysis sitting tens
            # of milliseconds away from the offset marker while the box said the
            # two were linked. "Linked" has to mean they agree whenever they are
            # shown, not merely that they move together once touched.
            if (getattr(self, "link_onset_auc", None)
                    and self.link_onset_auc.get()):
                _f = offset_marker_field(
                    self.enable_silent.get(),
                    'silent_start_idx' in m and 'silent_end_idx' in m)
                if "onset_idx" in m:
                    m["auc_start_idx"] = int(m["onset_idx"])
                if _f in m and int(m[_f]) > int(m.get("auc_start_idx", 0)):
                    m["auc_end_idx"] = int(m[_f])

            if "auc_start_idx" in m and "auc_end_idx" in m:
                # Draw initial fill
                self._redraw_auc_fill(m)

                def _on_start_moved(new_idx):
                    """AUC start line dragged."""
                    end_idx = m.get("auc_end_idx", new_idx + 1)
                    new_idx = max(0, min(new_idx, end_idx - 1))
                    m["auc_start_idx"] = new_idx
                    if self._auc_lines:
                        self._auc_lines[0].set_idx(new_idx)
                    # Sync onset if linked
                    if getattr(self, "link_onset_auc", None)                             and self.link_onset_auc.get():
                        m["onset_idx"] = new_idx
                        # A placed marker is a measurement, whatever the
                        # detector managed -- and this is a second way to place
                        # one. _update_meta clears the flag for the onset dot,
                        # but dragging the AUC START with linking on moves the
                        # onset without going through it, so the marker moved,
                        # the latency became computable, and the status still
                        # read "not detected".
                        m.pop("onset_auto_failed", None)
                        for dp in self._dpts:
                            if dp.role == "onset_idx":
                                dp.idx = new_idx
                                dp.point.set_offsets(
                                    [[self.t[new_idx], emg[new_idx]]])
                                break
                        self.canvas.draw()
                    self._redraw_auc_fill(m)
                    self._refresh_status()

                def _on_end_moved(new_idx):
                    """AUC end line dragged."""
                    start_idx = m.get("auc_start_idx", 0)
                    new_idx = max(start_idx + 1,
                                  min(new_idx, len(self.t) - 1))
                    m["auc_end_idx"] = new_idx
                    if len(self._auc_lines) >= 2:
                        self._auc_lines[1].set_idx(new_idx)
                    # Symmetric with the onset end: while linked, the AUC end
                    # and the offset are one quantity, so dragging either moves
                    # both. Without this the analyst can put them in two places
                    # and the file records two answers to the same question.
                    if getattr(self, "link_onset_auc", None) \
                            and self.link_onset_auc.get():
                        _f = offset_marker_field(
                            self.enable_silent.get(),
                            'silent_start_idx' in m and 'silent_end_idx' in m)
                        m[_f] = new_idx
                        for dp in self._dpts:
                            if dp.role == _f:
                                dp.idx = new_idx
                                dp.point.set_offsets(
                                    [[self.t[new_idx], emg[new_idx]]])
                                break
                    self._redraw_auc_fill(m)
                    self._refresh_status()

                self._auc_lines = [
                    DraggableLine(self.ax_raw, self.t,
                                  m["auc_start_idx"], _on_start_moved,
                                  color="tab:blue", lw=1.8, ls="--",
                                  read_only=self.read_only),
                    DraggableLine(self.ax_raw, self.t,
                                  m["auc_end_idx"],   _on_end_moved,
                                  color="tab:cyan",  lw=1.8, ls="--",
                                  read_only=self.read_only),
                ]

        # ---------- figure geometry ------------------------------------------
        # AUC selector now lives on the main plot — no second subplot.
        # Only adjust height for extra channel subplot if active.
        has_extra = bool(self._extra_axes)
        try:
            _canvas_h_px = self.canvas.get_tk_widget().winfo_height()
            _canvas_w_px = self.canvas.get_tk_widget().winfo_width()
            _dpi = self.fig.dpi
            if _canvas_h_px > 100 and _canvas_w_px > 100:
                _fig_h = _canvas_h_px / _dpi
                _fig_w = _canvas_w_px / _dpi
            else:
                _fig_h = self.FIG_H_RAW + (self.FIG_H_EXTRA if has_extra else 0)
                _fig_w = 12
        except Exception:
            _fig_h = self.FIG_H_RAW + (self.FIG_H_EXTRA if has_extra else 0)
            _fig_w = 12

        self.fig.set_size_inches(_fig_w, _fig_h)
        if not has_extra:
            self.ax_raw.set_position([0.07, 0.10, 0.90, 0.85])
        
        self._resize_window()
        self.canvas.draw_idle()

        # ---------- resize top‑level window (only when not maximised) --------
        _is_zoomed = False
        try:
            _is_zoomed = self.top.state() == "zoomed"
        except Exception:
            pass
        try:
            if not _is_zoomed:
                _is_zoomed = bool(self.top.attributes("-zoomed"))
        except Exception:
            pass

        if not _is_zoomed:
            self.top.update_idletasks()
            need_h = (self.hdr.winfo_reqheight() +
                      self.fig_frame.winfo_reqheight() +
                      self.status.winfo_reqheight() +
                      self.btn_bar.winfo_reqheight() +
                      self.btn_row.winfo_reqheight() + 40)
            if self.note_box_is_shown:
                need_h += self.note_box.winfo_reqheight()
            need_w = max(self.hdr.winfo_reqwidth(),
                         self.fig_frame.winfo_reqwidth(),
                         self.status.winfo_reqwidth(),
                         self.btn_bar.winfo_reqwidth(),
                         self.btn_row.winfo_reqwidth()) + 40
            if self.note_box_is_shown:
                need_w = max(need_w, self.note_box.winfo_reqwidth() + 40)
            self.top.geometry(f"{need_w}x{need_h}")

        # ---------- extra channel subplot -----------------------------------
        self._plot_extra_channel()

        # ---------- status line ----------------------------------------------
        self._refresh_status()


    # ──────────────────────────────────────────────────────────────────────────
    def _plot_extra_channel(self):
        """Draw (or clear) the extra channel subplot below ax_raw."""
        # Remove any previous extra axes
        for _ax in self._extra_axes:
            try: self.fig.delaxes(_ax)
            except Exception: pass
        self._extra_axes = []

        chan_name = self._extra_chan_var.get()
        if chan_name == "None" or chan_name not in self._extra_segs:
            # Only reset to full height if AUC panel is not visible
            self.ax_raw.set_position([0.07, 0.10, 0.90, 0.85])
            self.canvas.draw_idle()
            return

        chan_data = self._extra_segs[chan_name]
        # Expect new format: {"emg": array, "time": array, "fs": float,
        #                     "stim_times": {stim_type: [t_sec, ...]}}
        if not isinstance(chan_data, dict) or "emg" not in chan_data:
            self.ax_raw.set_position([0.07, 0.10, 0.90, 0.85])
            self.canvas.draw_idle()
            return

        emg_full   = chan_data["emg"]
        time_full  = chan_data["time"]   # seconds
        fs_x       = chan_data["fs"]
        stim_times = chan_data["stim_times"]

        t_list = stim_times.get(self.cur_type, [])
        if self.cur_idx >= len(t_list):
            self.ax_raw.set_position([0.10, 0.12, 0.87, 0.80])
            self.canvas.draw_idle()
            return

        # Slice on demand around the current stim time
        wide_s    = float(self._wide_var.get())
        t0_sec    = t_list[self.cur_idx]
        _wide_smp = int(wide_s * fs_x)
        _ix       = int(np.argmin(np.abs(time_full - t0_sec)))
        _s        = max(0, _ix - _wide_smp)
        _e        = min(len(emg_full), _ix + _wide_smp)
        wide_seg  = emg_full[_s:_e]
        # Time axis in ms relative to stim
        t_wide_ms = (time_full[_s:_e] - t0_sec) * 1000.0

        # ── Layout: ax_raw (top) + ax_ex (extra channel, bottom) ────────────
        try:
            _canvas_h_px = self.canvas.get_tk_widget().winfo_height()
            _canvas_w_px = self.canvas.get_tk_widget().winfo_width()
            _dpi = self.fig.dpi
            if _canvas_h_px > 100 and _canvas_w_px > 100:
                _fig_h = _canvas_h_px / _dpi
                _fig_w = _canvas_w_px / _dpi
            else:
                _fig_h = self.FIG_H_RAW + self.FIG_H_EXTRA
                _fig_w = 12
        except Exception:
            _fig_h = self.FIG_H_RAW + self.FIG_H_EXTRA
            _fig_w = 12

        self.fig.set_size_inches(_fig_w, _fig_h)
        self.ax_raw.set_position([0.07, 0.52, 0.90, 0.43])
        ax_ex = self.fig.add_axes([0.07, 0.10, 0.90, 0.35])

        self._extra_axes.append(ax_ex)
        self._resize_window()

        if self._flip_var.get():
            wide_seg = -wide_seg
        ax_ex.plot(t_wide_ms, wide_seg, color="0.35", lw=0.8)
        ax_ex.axvline(0, color="k", ls="--", lw=0.8)

        # Shaded rectangle showing the primary channel visible window
        _xleft  = -(self.visible_pre_ms if self.visible_pre_ms is not None
                    else wide_s * 1000)
        _xright = self.t[-1]
        ax_ex.axvspan(_xleft, _xright, alpha=0.12, color="steelblue", zorder=0)

        if len(wide_seg) > 0:
            _pad = (_np_ptp(wide_seg) * 0.1) if _np_ptp(wide_seg) > 0 else 0.1
            ax_ex.set_ylim(wide_seg.min() - _pad, wide_seg.max() + _pad)

        ax_ex.set_xlim(-wide_s * 1000, wide_s * 1000)
        ax_ex.set_xlabel("Time (ms)")
        ax_ex.set_ylabel(chan_name)
        ax_ex.grid(ls=":", lw=0.4)
        self.canvas.draw_idle()
  
    def _widget_alive(self, w):
        """Whether one Tk widget can still be used.

        Only for real Tk widgets. A matplotlib FigureCanvasTkAgg has no
        winfo_exists, so passing one here would raise, be caught, and report
        the window as dead -- permanently, on every redraw. Use
        canvas.get_tk_widget() if the drawing area itself ever needs checking.
        """
        try:
            return bool(w is not None and w.winfo_exists())
        except Exception:
            return False

    def _closed(self):
        """True once this window can no longer be drawn into.

        Tk delivers events that were already queued when a widget was
        destroyed. A keyboard binding -- Left or Right to step through trials --
        therefore fires once more after the window has gone, and every widget
        the handler touches has been torn down. In a multi-channel run two
        Inspectors open in succession, so this happens twice as often.

        Checking the Toplevel alone was not enough. It reported itself as
        existing while a child had already been destroyed, and the redraw then
        failed on the child:

            _tkinter.TclError: invalid command name
            ".!toplevel3.!frame4.!scrolledtext"

        Rather than reason about how a parent outlives its child in Tk's
        teardown -- which would need the exact answer to be right, and would
        break again if it changed -- this checks the widgets the redraw
        actually touches. A window missing any of them cannot be drawn into,
        whatever the reason.
        """
        if getattr(self, "_is_closing", False):
            return True
        for w in (getattr(self, "top", None), getattr(self, "note_box", None)):
            if w is None:
                continue
            if not self._widget_alive(w):
                return True
        return False

    def _close_preview(self):
        """Close a read-only window. Nothing is committed, by construction."""
        self._is_closing = True
        try:
            self.top.destroy()
        except tk.TclError:
            pass

    def _close_and_save(self):
        """Save all pending edits including note, then close."""
        self._is_closing = True
        # Always save the note box content regardless of whether it is
        # currently visible — the widget retains its text even when hidden.
        key = (self.cur_type, self.cur_idx)
        txt = self.note_box.get("1.0", "end").strip()
        if txt:
            self.meta.setdefault(key, {})['note'] = txt
        elif key in self.meta and 'note' in self.meta[key]:
            del self.meta[key]['note']

        # Strip landmarks that exist only because detection failed and the
        # marker had to be put somewhere. The analysis treats a stored
        # onset_idx as a manual override, so exporting the fallback would
        # convert "no onset found" into "onset at 0.0 ms" -- and everything
        # derived from it, offset, duration and the area window, with it.
        # Anything the analyst actually moved has already cleared the flag.
        for _k, _m in list(self.meta.items()):
            if _m.get('onset_auto_failed'):
                for _f in ('onset_idx', 'mep_offset_idx',
                           'auc_start_idx', 'auc_end_idx'):
                    _m.pop(_f, None)
            _m.pop('onset_auto_failed', None)
            # Explanatory text for the status line, not a measurement. It is
            # re-derived on every draw, so keeping it would only mean stale
            # prose in the session file.
            _m.pop('onset_fail_reason', None)

        self.top.destroy()

    # ---------------------------------------------------------------- status-bar
    def _refresh_status(self):
        k = (self.cur_type, self.cur_idx); m = self.meta[k]
        stim_idx = np.argmin(np.abs(self.t))
        dt_ms = self.t[1] - self.t[0]

        # ---------- status‑bar text -----------------------------------------
        seg     = self.segments[self.cur_type][self.cur_idx]
        ptp_amp = seg[m['ptp_max_idx']] - seg[m['ptp_min_idx']]
        # A negative peak-to-peak cannot exist, so it is reported as a fault.
        #
        # ptp_max below ptp_min means the two landmarks are not the maximum and
        # minimum of one response: they have swapped over, or they are sitting
        # on something that is not the response at all. The value has a sign
        # only because of the order they are subtracted in, and printing it as
        # "PTP: -0.04 mV" presents an impossibility as a measurement.
        #
        # Seen when stored landmarks outlived the geometry that positioned
        # them: an event delay moved the response 17.5 ms while the indices
        # stayed put, leaving PTP min at +0.013 mV and PTP max at -0.028 mV.
        # That is now prevented upstream, but this is the check that would have
        # caught it in one glance, and it is independent of the cause -- it
        # holds for any future reason the two markers end up the wrong way
        # round, including an analyst dragging them there.
        _ptp_inverted = bool(ptp_amp < 0)
        ptp_txt = (f"PTP:{ptp_amp:.2f} mV" if not _ptp_inverted
                   else f"PTP: max is BELOW min ({ptp_amp:.2f} mV) — "
                        f"the markers are the wrong way round")
        lat_ms  = (m['onset_idx'] - stim_idx) * dt_ms
        _no_onset = bool(m.get('onset_auto_failed'))
        # A non-detection is reported as such, not as 0.0 ms -- and with the
        # reason where there is one, because "not detected" alone does not
        # distinguish a response below the amplitude gate from a search window
        # that was looking in the wrong place, and those need opposite fixes.
        _why = m.get('onset_fail_reason') if _no_onset else None
        lat_txt = (f"Latency:{lat_ms:.1f} ms" if not _no_onset
                   else (f"Latency: not detected — {_why}" if _why
                         else "Latency: not detected"))


        # cSP duration and absolute EMG return time relative to stim
        csp_note = f"  ⓘ {m['csp_reason']}" if m.get('csp_reason') else ''
        silent_txt = ""
        if self.enable_silent.get() and \
        "silent_start_idx" in m and "silent_end_idx" in m:
            _csp_dur = (m["silent_end_idx"] - m["silent_start_idx"]) * dt_ms
            _csp_end = (m["silent_end_idx"] - stim_idx) * dt_ms
            silent_txt = f"    cSP:{_csp_dur:.1f} ms    cSP end:{_csp_end:.1f} ms"
        # MEP offset and duration. Which marker supplies the offset follows the
        # same rule the pipeline applies, so this read-out and the results file
        # cannot disagree: the cSP-start marker during contraction, a dedicated
        # marker otherwise.
        offset_txt = ""
        _fld = None if _no_onset else offset_marker_field(
            self.enable_silent.get(),
            'silent_start_idx' in m and 'silent_end_idx' in m)
        if _fld is not None and _fld in m:
            _off_ms = (m[_fld] - stim_idx) * dt_ms
            _dur_ms = _off_ms - lat_ms
            offset_txt = f"    Offset:{_off_ms:.1f} ms"
            if _dur_ms > 0:
                offset_txt += f"    Duration:{_dur_ms:.1f} ms"

        # existing AUC read‑out
        auc_txt = ""
        if "auc_start_idx" in m and "auc_end_idx" in m:
            auc_val = _np_trapz(np.abs(seg[m["auc_start_idx"]:m["auc_end_idx"]]),
                            dx=dt_ms / 1000)
            # In µV·s, because mV·s at three decimals reads 0.000 for
            # every MEP. A 0.1 mV response over 15 ms integrates to about
            # 3e-4 mV·s, so the whole plausible range for one MEP --
            # roughly 1e-4 to 2e-3 -- rounds to 0.000 or 0.001. The number
            # was on screen, told the analyst nothing, and looked like a
            # failed calculation. The same values in µV·s are 130
            # to 790, which read at a glance and compare between trials.
            #
            # Display only: AUC(mV*s) in the results file is unchanged, so
            # nothing downstream and no published value moves.
            auc_txt = f"    |AUC|:{auc_val * 1000.0:.1f} µV·s"

        self.status.config(
            text=(f"{ptp_txt}    "
                  f"{lat_txt}"
                  f"{offset_txt}{silent_txt}{auc_txt}{csp_note}"),
            # Coloured too: the status line is read at a glance and a
            # sentence among numbers is easy to slide past. Restored to
            # the normal colour when it is fine, so a fault on one trial
            # does not leave every later one looking wrong.
            fg=("#B03A2E" if _ptp_inverted else self._status_fg_default))


