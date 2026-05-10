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
from matplotlib.widgets import SpanSelector
import tkinter as tk
from tkinter import ttk, scrolledtext

from .compat import _np_trapz
from .detection import (detect_mep_onset_peak_fraction,
                        detect_mep_onset_bootstrap,
                        detect_csp_bootstrap)

class DraggablePoint:
    """
    A draggable scatter point.  When the user lets go,
    it snaps to the sample that best matches its *role*:
        • 'ptp_min_idx' → local minimum in a ±radius window  
        • 'ptp_max_idx' → local maximum in a ±radius window  
        • anything else → nearest sample (previous behaviour)
    """
    def __init__(self, point, time_axis, emg, idx0, update_cb, role='generic', radius=8):
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
        if event.inaxes is not self.point.axes:
            return
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
    }

    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, master, segments_dict, time_axis, metadata_dict,
                 label_map=None, color_map=None, emg_unit=None,
                 ptp_start_ms=10, ptp_end_ms=50,
                 visible_pre_ms=None,
                 onset_method="peak_fraction",
                 onset_bootstrap_crit=1.96, onset_bootstrap_n=500,
                 csp_search_start_ms=40, csp_search_end_ms=400,
                 csp_min_silence_ms=25, csp_min_return_ms=40,
                 csp_criterion=1.96, csp_significance=0.99,
                 csp_n_boot=1000, csp_rms_window_ms=10,
                 csp_types=None, analysis_pre_ms=None,
                 extra_segs=None, wide_window_s=3.0):

        # --------- book-keeping -----------------------------------------
        self.top = tk.Toplevel(master)
        self.top.title("Data Inspector – review")
        self.top.transient(master)
        self.top.grab_set()

        self.segments  = segments_dict
        self.t         = time_axis
        self.meta      = metadata_dict
        self.snap_radius = 8
        self.label_map = label_map or {}
        self.color_map = color_map or {}
        self.emg_unit  = emg_unit
        self.ptp_start_ms         = ptp_start_ms
        self.ptp_end_ms           = ptp_end_ms
        # visible_pre_ms: how much pre-stim to SHOW (xlim)
        # _analysis_pre_ms: full pre-stim used for detection (may be larger)
        self.visible_pre_ms       = visible_pre_ms
        self.onset_method         = onset_method
        self.onset_bootstrap_crit = onset_bootstrap_crit
        self.onset_bootstrap_n    = onset_bootstrap_n
        self.csp_search_start_ms = csp_search_start_ms
        self.csp_search_end_ms   = csp_search_end_ms
        self.csp_min_silence_ms  = csp_min_silence_ms
        self.csp_min_return_ms   = csp_min_return_ms
        self.csp_criterion       = csp_criterion
        self.csp_significance    = csp_significance
        self.csp_n_boot          = csp_n_boot
        self.csp_rms_window_ms   = csp_rms_window_ms
        self._analysis_pre_ms    = analysis_pre_ms
        # extra_segs: {chan_name: {stim_type: [wide_seg_array]}}
        self._extra_segs         = extra_segs or {}
        self._wide_window_s      = wide_window_s
        self._extra_axes         = []   # subplot axes for extra channels
        # Pre-populate silent period state from caller-specified csp_types.
        # Types in csp_types start ticked; all others start unticked.
        _csp_set = set(csp_types) if csp_types else set()
        self._silent_per_type = {
            k: (k in _csp_set) for k in segments_dict
        }

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

        # --------- matplotlib figure (2 rows) ---------------------------
        # Use plt.Figure (not plt.subplots) to avoid registering the figure
        # with the interactive TkAgg backend, which would create a ghost window.
        self.fig = plt.Figure(figsize=(8, 4))
        self.ax_raw = self.fig.add_subplot(111)
        self.ax_abs = None                                     # build on-demand
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --------- status bar -------------------------------------------
        self.status = tk.Label(self.top, anchor="w")
        self.status.pack(fill="x", padx=10, pady=4)

        # --------- toggles ----------------------------------------------
        self.btn_bar = tk.Frame(self.top)                # << keep a handle 👉 self.btn_bar
        self.btn_bar.pack(pady=(0, 6))

        self.enable_silent = tk.BooleanVar(value=True)
        self.enable_auc = tk.BooleanVar(value=False)
        self.exclude_var = tk.BooleanVar(value=False)
        self.note_enable_var = tk.BooleanVar(value=False)

        tk.Checkbutton(self.btn_bar, text="Silent period",
                    variable=self.enable_silent,
                    command=self._on_silent_toggle).pack(side="left", padx=10)

        tk.Checkbutton(self.btn_bar, text="AUC selector",
                    variable=self.enable_auc,
                    command=self._plot).pack(side="left")

        tk.Checkbutton(self.btn_bar, text="Exclude this segment",
                    variable=self.exclude_var,
                    command=lambda: self._set_exclude()).pack(side="left", padx=12) 

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
        
        # small note box (hidden until checkbox is ticked)     
        self.note_box = scrolledtext.ScrolledText(self.top, height=3)             
        self.note_box_is_shown = False 

        # --------- close ------------------------------------------------
        self.btn_row = tk.Button(self.top, text="Save edits & close",
                         width=20,
                         command=self._close_and_save)            
        self.btn_row.pack(side="bottom", pady=(0, 8))    # 👈 ALWAYS at the bottom
        # --------- internal state ---------------------------------------
        self.cur_type = self.dd_event["values"][0]
        self.cur_idx  = 0
        self.dd_event.current(0)

        # Span selector (created on demand)
        self._auc_span = None

        self._plot()      # first draw
    # ──────────────────────────────────────────────────────────────────────
    def _on_silent_toggle(self):
        self._silent_per_type[self.cur_type] = self.enable_silent.get()
        self._plot()

    def _first(self):
        self._save_note_from_widget()
        self._silent_per_type[self.cur_type] = self.enable_silent.get()
        self.cur_type, self.cur_idx = self.dd_event.get(), 0
        self._plot()

    def _next(self):
        self._save_note_from_widget()                    
        self.cur_idx = (self.cur_idx + 1) % len(self.segments[self.cur_type])
        self._plot()

    def _prev(self):
        self._save_note_from_widget()                       
        self.cur_idx = (self.cur_idx - 1) % len(self.segments[self.cur_type])
        self._plot()

    # ---------------------------------------------------------------- helper
    def _update_meta(self, field, new_idx):
        key = (self.cur_type, self.cur_idx)
        self.meta.setdefault(key, {})[field] = new_idx
        self._refresh_status()
    
    def _ylab(self, base="EMG"):
            return f"{base} ({self.emg_unit})" if self.emg_unit else base
    
    def _set_exclude(self):                                                
        key = (self.cur_type, self.cur_idx)
        self.meta.setdefault(key, {})
        self.meta[key]['exclude'] = self.exclude_var.get()
        self._refresh_status()

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
    
    def _resize_window(self):
        """Resize the Toplevel so every widget (note box included) is visible."""
        self.top.update_idletasks()       # make sure sizes are up‑to‑date

        pieces = [self.hdr,
                  self.canvas.get_tk_widget(),
                  self.status,
                  self.btn_bar,
                  self.btn_row]
        if self.note_box_is_shown:
            pieces.append(self.note_box)

        need_h = sum(p.winfo_reqheight() for p in pieces) + 40
        need_w = max(p.winfo_reqwidth()  for p in pieces) + 40
        self.top.geometry(f"{need_w}x{need_h}")

    # ---------------------------------------------------------------- plot
    def _plot(self):
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
        if _has_markers:
            # Markers exist (auto-detected or manually placed) — show them
            self.enable_silent.set(True)
        elif _det_failed:
            # Detection was attempted and failed for this segment —
            # leave checkbox unticked so user can decide whether to
            # manually place markers by ticking it themselves.
            self.enable_silent.set(False)
        else:
            # Not yet attempted — will auto-detect below if CSP type
            self.enable_silent.set(False)  # set after detection result

        # ---------- sync “exclude” & note widgets ---------------------------
        self.exclude_var.set(m.get('exclude', False))

        # note‑box follow‑through
        note_txt = m.get('note', '')
        if note_txt or self.note_enable_var.get():
            # show the widget and populate it (or clear if empty)
            if not self.note_box_is_shown:
                self.note_box.pack(fill="x", padx=10, pady=(4, 6))
                self.note_box_is_shown = True
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

        # … but if the user defined a PTP window, constrain the search
        if self.ptp_start_ms is not None and self.ptp_end_ms is not None:
            mask = (self.t >= self.ptp_start_ms) & (self.t <= self.ptp_end_ms)
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
        if self.onset_method == "bootstrap":
            onset_ms = detect_mep_onset_bootstrap(
                           emg, fs,
                           pre_ms=_pre_ms,
                           peak_search_start_ms=self.ptp_start_ms or 10,
                           peak_search_end_ms=self.ptp_end_ms or 50,
                           artefact_blank_ms=2,
                           criterion=self.onset_bootstrap_crit,
                           n_boot=self.onset_bootstrap_n)
        else:
            onset_ms = detect_mep_onset_peak_fraction(
                           emg, fs,
                           pre_ms=_pre_ms,
                           poststim_start_ms=self.ptp_start_ms or 10,
                           poststim_end_ms=self.ptp_end_ms   or 50)
        stim_idx = np.argmin(np.abs(self.t))
        onset    = stim_idx if onset_ms is None else stim_idx + int(round(onset_ms / dt_ms))
        onset    = max(onset, stim_idx)

        # ---------- seed metadata defaults ----------------------------------
        m.setdefault('ptp_min_idx', p_min)
        m.setdefault('ptp_max_idx', p_max)
        m.setdefault('onset_idx',   onset)
        # Auto-detect for CSP types on first display; also re-run if user
        # manually ticks the checkbox on a previously-failed segment.
        _user_manually_ticked = self.enable_silent.get() and _det_failed
        if _user_manually_ticked:
            m.pop('csp_detection_failed', None)   # allow fresh attempt
            _det_failed = False
        _should_detect = (_type_wants_silent or self.enable_silent.get()) \
                         and not _has_markers and not _det_failed
        if _should_detect:
            if 'silent_start_idx' not in m and \
                    not m.get('csp_detection_failed', False):
                _csp_reason = []
                csp = detect_csp_bootstrap(
                    emg, fs, self.t,
                    pre_ms=(self._analysis_pre_ms
                            if self._analysis_pre_ms is not None
                            else abs(int(self.t[0]))),
                    search_start_ms=self.csp_search_start_ms,
                    search_end_ms=min(self.csp_search_end_ms, float(self.t[-1])),
                    min_silence_ms=self.csp_min_silence_ms,
                    min_return_ms=self.csp_min_return_ms,
                    criterion=self.csp_criterion,
                    significance=self.csp_significance,
                    n_boot=self.csp_n_boot,
                    rms_window_ms=self.csp_rms_window_ms,
                    reason_out=_csp_reason)
                m['csp_reason'] = _csp_reason[0] if _csp_reason else ''
                if csp is not None:
                    m['silent_start_idx'], m['silent_end_idx'] = csp
                    self.enable_silent.set(True)   # detection succeeded
                else:
                    # Detection failed — untick so user sees it was attempted
                    # but found nothing. User can manually tick to place markers.
                    m['csp_detection_failed'] = True
                    m.pop('silent_start_idx', None)
                    m.pop('silent_end_idx',   None)
                    self.enable_silent.set(False)

        # ---------- clear axes & plot raw trace ------------------------------
        self.ax_raw.clear()
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
            mk   = 'x' if field.startswith('ptp_') else 'o'
            alp  = 0.6 if field == "onset_idx" else 1.0
            scat = self.ax_raw.scatter(self.t[idx0], emg[idx0],
                                       s=80, color=c, marker=mk, alpha=alp,
                                       zorder=3, label=label)
            self._dpts.append(
                DraggablePoint(
                    scat, self.t, emg, idx0,
                    lambda i, f=field: self._update_meta(f, i),
                    role=field, radius=self.snap_radius
                )
            )

        _add(m['ptp_min_idx'], self.DOT_COLOURS["ptp_min_idx"], "ptp_min_idx", label="PTP min")
        _add(m['ptp_max_idx'], self.DOT_COLOURS["ptp_max_idx"], "ptp_max_idx", label="PTP max")
        _add(m['onset_idx'],   self.DOT_COLOURS["onset_idx"],   "onset_idx",   label="Onset")

        if self.enable_silent.get() and \
                'silent_start_idx' in m and 'silent_end_idx' in m:
            _add(m['silent_start_idx'], self.DOT_COLOURS["silent_start_idx"],
                 "silent_start_idx", label="cSP start")
            _add(m['silent_end_idx'],   self.DOT_COLOURS["silent_end_idx"],
                 "silent_end_idx",   label="cSP end")

        self.ax_raw.legend(loc="upper right", fontsize=12, frameon=False)
        self.fig._draggables = self._dpts

        # ---------- AUC panel ------------------------------------------------
        show_auc = self.enable_auc.get()

        if show_auc and self.ax_abs is None:
            self.ax_abs = self.fig.add_axes([0.12, 0.10, 0.85, 0.25],
                                            sharex=self.ax_raw)
        elif not show_auc and self.ax_abs is not None:
            self.fig.delaxes(self.ax_abs)
            self.ax_abs = None
            if self._auc_span is not None:
                self._auc_span.set_visible(False)
                self._auc_span = None

        if show_auc:
            self.ax_abs.clear()
            # Show the unrectified waveform so the shape is easy to read;
            # AUC is still computed on the rectified signal in _refresh_status.
            self.ax_abs.plot(self.t, emg, color="0.4", lw=0.8)
            self.ax_abs.axhline(0, color="k", lw=0.5, ls=":")
            self.ax_abs.set_ylabel(self._ylab("EMG (AUC selector)"))
            # Shade any already-stored AUC window on the unrectified plot.
            if "auc_start_idx" in m and "auc_end_idx" in m:
                a0, a1 = m["auc_start_idx"], m["auc_end_idx"]
                self.ax_abs.axvspan(self.t[a0], self.t[a1],
                                    alpha=0.2, color="tab:blue", zorder=0)

            def _auc_cb(x0, x1):
                m["auc_start_idx"], m["auc_end_idx"] = sorted((
                    np.argmin(np.abs(self.t - x0)),
                    np.argmin(np.abs(self.t - x1))
                ))
                self._refresh_status()

            self._auc_span = SpanSelector(
                self.ax_abs, _auc_cb, "horizontal",
                useblit=True,
                props=dict(alpha=.30, facecolor="tab:blue"),
                interactive=True
            )

            if "auc_start_idx" in m and "auc_end_idx" in m:
                self._auc_span.extents = (self.t[m["auc_start_idx"]],
                                          self.t[m["auc_end_idx"]])

        # ---------- figure geometry ------------------------------------------
        if show_auc:
            self.fig.set_figheight(self.FIG_H_RAW + self.FIG_H_EXTRA)
            self.canvas.get_tk_widget().configure(
                height=int(self.fig.get_figheight() * self.fig.dpi))
            self.ax_raw.set_position([0.12, 0.42, 0.85, 0.53])
            self.ax_abs.set_position([0.12, 0.10, 0.85, 0.25])
        else:
            self.fig.set_figheight(self.FIG_H_RAW)
            self.canvas.get_tk_widget().configure(
                height=int(self.fig.get_figheight() * self.fig.dpi))
            self.ax_raw.set_position([0.12, 0.15, 0.85, 0.78])
        
        self._resize_window()
        self.canvas.draw_idle()

        # ---------- resize top‑level window so the note box fits -------------
        self.top.update_idletasks()

        need_h = (self.hdr.winfo_reqheight() +
                  self.canvas.get_tk_widget().winfo_reqheight() +
                  self.status.winfo_reqheight() +
                  self.btn_bar.winfo_reqheight() +
                  self.btn_row.winfo_reqheight() + 40)
        if self.note_box_is_shown:
            need_h += self.note_box.winfo_reqheight()

        need_w = max(self.hdr.winfo_reqwidth(),
                     self.canvas.get_tk_widget().winfo_reqwidth(),
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
            if self.ax_abs is None:
                self.ax_raw.set_position([0.10, 0.12, 0.87, 0.80])
            self.canvas.draw_idle()
            return

        chan_data = self._extra_segs[chan_name]
        # Expect new format: {"emg": array, "time": array, "fs": float,
        #                     "stim_times": {stim_type: [t_sec, ...]}}
        if not isinstance(chan_data, dict) or "emg" not in chan_data:
            if self.ax_abs is None:
                self.ax_raw.set_position([0.10, 0.12, 0.87, 0.80])
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

        # ── Layout: position ax_raw + ax_abs (AUC) + ax_ex (extra) ──────────
        show_auc = self.ax_abs is not None
        FIG_ROWS = self.FIG_H_RAW  # base height
        EXTRA_H  = self.FIG_H_EXTRA

        if show_auc:
            # AUC + extra channel: 3-panel layout
            total_h = self.FIG_H_RAW + self.FIG_H_EXTRA * 2
            self.fig.set_figheight(total_h)
            self.canvas.get_tk_widget().configure(height=int(total_h * self.fig.dpi))
            self.ax_raw.set_position([0.12, 0.62, 0.85, 0.33])
            self.ax_abs.set_position([0.12, 0.36, 0.85, 0.22])
            ax_ex = self.fig.add_axes([0.12, 0.07, 0.85, 0.22])
        else:
            # extra channel only: 2-panel layout
            total_h = self.FIG_H_RAW + self.FIG_H_EXTRA
            self.fig.set_figheight(total_h)
            self.canvas.get_tk_widget().configure(height=int(total_h * self.fig.dpi))
            self.ax_raw.set_position([0.12, 0.52, 0.85, 0.43])
            ax_ex = self.fig.add_axes([0.12, 0.10, 0.85, 0.35])

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
            _pad = (np.ptp(wide_seg) * 0.1) if np.ptp(wide_seg) > 0 else 0.1
            ax_ex.set_ylim(wide_seg.min() - _pad, wide_seg.max() + _pad)

        ax_ex.set_xlim(-wide_s * 1000, wide_s * 1000)
        ax_ex.set_xlabel("Time (ms)")
        ax_ex.set_ylabel(chan_name)
        ax_ex.grid(ls=":", lw=0.4)
        self.canvas.draw_idle()

    def _close_and_save(self):
        """Save all pending edits including note, then close."""
        # Save the current segment's note
        if self.note_box_is_shown:
            key = (self.cur_type, self.cur_idx)
            txt = self.note_box.get("1.0", "end").strip()
            if txt:
                self.meta.setdefault(key, {})['note'] = txt
            elif key in self.meta and 'note' in self.meta[key]:
                del self.meta[key]['note']
        self.top.destroy()

    # ---------------------------------------------------------------- status-bar
    def _refresh_status(self):
        k = (self.cur_type, self.cur_idx); m = self.meta[k]
        stim_idx = np.argmin(np.abs(self.t))
        dt_ms = self.t[1] - self.t[0]

        # ---------- status‑bar text -----------------------------------------
        seg     = self.segments[self.cur_type][self.cur_idx]
        ptp_amp = seg[m['ptp_max_idx']] - seg[m['ptp_min_idx']]
        lat_ms  = (m['onset_idx'] - stim_idx) * dt_ms


        # cSP duration and absolute EMG return time relative to stim
        csp_note = f"  ⓘ {m['csp_reason']}" if m.get('csp_reason') else ''
        silent_txt = ""
        if self.enable_silent.get() and \
        "silent_start_idx" in m and "silent_end_idx" in m:
            _csp_dur = (m["silent_end_idx"] - m["silent_start_idx"]) * dt_ms
            _csp_end = (m["silent_end_idx"] - stim_idx) * dt_ms
            silent_txt = f"    cSP:{_csp_dur:.1f} ms    cSP end:{_csp_end:.1f} ms"
        # existing AUC read‑out
        auc_txt = ""
        if "auc_start_idx" in m and "auc_end_idx" in m:
            auc_val = _np_trapz(np.abs(seg[m["auc_start_idx"]:m["auc_end_idx"]]),
                            dx=dt_ms / 1000)
            auc_txt = f"    |AUC|:{auc_val:.3f} mV·s"

        self.status.config(text=(
            f"PTP:{ptp_amp:.2f} mV    "
            f"Latency:{lat_ms:.1f} ms"
            f"{silent_txt}{auc_txt}"
        ))


