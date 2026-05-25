"""
mep_cmap.app
~~~~~~~~~~~~
TMSAnalysisApp — main GUI class.

Inherits from Stage2Mixin (group analysis tab) and FilterPreviewMixin
(filter preview popup). Core responsibilities: main window layout,
background analysis threading, session save/load, file browsing,
and wiring all modules together.

mep_cmap.app
~~~~~~~~~~~~
Main application class: TMSAnalysisApp.

Builds the Tkinter GUI, manages background analysis threads,
and wires together all the pipeline, inspector, and BIDS modules.
"""

import gc
import os
import re
import json
import time
import queue
import pathlib
from pathlib import Path
import datetime
import threading
import webbrowser
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter
from matplotlib.colors import Normalize
from matplotlib import gridspec as mgs
from scipy.signal import (
    butter, filtfilt, iirnotch,
    sosfiltfilt, sos2tf, freqz, group_delay,
    fftconvolve,
)
import pywt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext, font

from .compat import _np_trapz
from .bids import StudyMetadata, _sanitise_bids_label, TOOL_VERSION
from .dataset_session import (DatasetSession, FileEntry,
                               STATUS_NOT_STARTED, STATUS_IN_PROGRESS,
                               STATUS_NEEDS_REVIEW, STATUS_COMPLETE,
                               STATUS_STALE, STATUS_LABELS, STATUS_COLOURS)
from .io import (list_waveform_channels, extract_emg_waveform_and_fs,
                 extract_stim_times, detect_format, needs_wizard,
                 list_event_channels)
from .format_wizard import FormatWizard
from .filters import adaptive_mains_cancel
from .detection import detect_mep_onset_peak_fraction
from .inspector import DataInspectorWindow
from .pipeline       import run_pipeline
from .preferences    import prefs, apply_scaling
from .stage2         import Stage2Mixin
from .filter_preview import FilterPreviewMixin

class TMSAnalysisApp(Stage2Mixin, FilterPreviewMixin):
    def __init__(self, root):
        self.root = root
        # ── State that setup_gui() widgets depend on — must come first ────────
        self.crop_start        = None
        self.crop_end          = None
        self.crop_ranges       = None
        self.gap_ms_map        = {}
        self.reference_map     = {}
        self._reference_display = {}
        self.latency_map        = {}
        self.latency_stim_map   = {}
        self.latency_muscle_map = {}
        self.mmax_file             = tk.StringVar()
        self.plateau_tolerance     = tk.DoubleVar(value=10.0)
        self.extra_channel_indices = []
        self.wide_window_s         = tk.DoubleVar(value=3.0)
        self.emg_unit          = None
        # These must be initialised before _build_scrollable_container
        # because _build_session_tab references them directly
        self.file_path          = tk.StringVar()
        self.derivatives_path   = tk.StringVar()
        self._rawdata_path      = tk.StringVar()
        self._dataset           = None
        self._current_file_entry = None
        self._queue_progress_var = tk.StringVar(value="No files loaded")
        # ── Build GUI ─────────────────────────────────────────────────────────
        self._build_menu()
        self._build_scrollable_container()
        self.setup_gui()
        self.root.title(f"MEP-CMAP Analyser, Version {TOOL_VERSION} - May 2026")
        self.root.after(0, self._make_window_adaptive)
        # ─── BIDS / derivatives ──────────────────────────────────────────────
        self.study_metadata   = StudyMetadata()
        self._remembered_meta = None          # persists across files if user ticked "remember"
        # ─── background‑thread message queue ──────────────────────────────────
        self.msg_q = queue.Queue()
        self._last_outlier_result = None
        self._poll_queue()
        self.segments_metadata = {}
        
    # ───────────────────────────────────────────────────────────────────────────
    def _poll_queue(self):
        """Drain the worker‑thread queue and run GUI actions on the main thread."""
        # Run GC here (main thread only) — prevents BLAS threads triggering
        # Tcl_AsyncDelete by never letting automatic GC run in a worker thread.
        if not hasattr(self, '_gc_count'): self._gc_count = 0
        self._gc_count += 1
        if self._gc_count >= 20:
            self._gc_count = 0
            gc.collect()
        try:
            while True:                       # empty everything that’s waiting
                msg, *payload = self.msg_q.get_nowait()

                if msg == "log":
                    self._log_gui(payload[0])

                elif msg == "progress":
                    self.progress.set(payload[0])

                elif msg == "ask‑marker":
                    # run the picker; the result is stored on self.marker_choice
                    self._ask_marker_gui(payload[0])

                elif msg == "show‑outliers":
                    # ① run the dialog on the GUI thread
                    res = self._review_outliers_gui(*payload)
                    # ② hand the result back to the waiting worker
                    self._last_outlier_result = res

                elif msg == "show-inspector":
                    self._open_inspector_gui(*payload)

                elif msg == "done":
                    # Analysis finished — autosave regardless of whether the
                    # inspector was used
                    self._autosave_session()
                    # Mark file complete in dataset queue
                    if self._dataset is not None and hasattr(self, '_current_file_entry'):
                        fe = self._current_file_entry
                        if fe is not None:
                            fe.mark_complete()
                            self._dataset.save()
                            self._queue_refresh()

        except queue.Empty:
            pass

        # poll again in 75 ms
        self.root.after(75, self._poll_queue)

    # ───────────────────────────────────────────────────────────────────────────
    def _toggle_humbug_fields(self):
        """Enable/disable the harmonics entry in sync with the mains‑canceller."""
        state = 'normal' if self.apply_humbug.get() else 'disabled'
        self.harmonics_entry.config(state=state)

    def _review_outliers_gui(self, flagged_outliers, fs, pre_ms, post_ms, emg_unit=None):
        """
        Interactive review of outlier segments; returns a list containing only
        the outliers the user chose to KEEP.  Runs entirely on the Tk main thread.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        kept_segments = []        # what the user keeps

        # ───────────── helper to display one candidate ────────────────────────
        def show_next(i: int):
            if i >= len(flagged_outliers):          # no more → close dialog
                popup.destroy()
                return

            out = flagged_outliers[i]
            emg_seg = out["emg_segment"]
            t_axis  = np.linspace(-pre_ms, post_ms, len(emg_seg), endpoint=False)

            # ---- draw figure -------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t_axis, emg_seg)
            ax.axvline(0, color="black", linestyle="--")
            ax.set_xlim(-pre_ms, post_ms)
            ax.set_title(f'{out["file"]}  –  {out["stim_type"]}  –  seg {out["index"]+1}')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(f"EMG ({emg_unit})" if emg_unit else "EMG")
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            canvas.get_tk_widget().pack()

            # ---- update stats read‑out --------------------------------------
            stats_lbl.config(text=(
                f"Pre‑stim RMS: {out['rms']:.4f}  (z = {out['z_rms']:.2f})\n"
                f"MEP PTP:      {out['ptp']:.4f}  (z = {out['z_ptp']:.2f})"
            ))

            # ---- button callbacks -------------------------------------------
            def _keep():
                kept_segments.append(out)
                canvas.get_tk_widget().destroy()
                plt.close(fig)
                show_next(i + 1)

            def _remove():
                canvas.get_tk_widget().destroy()
                plt.close(fig)
                show_next(i + 1)

            keep_btn.config(command=_keep)
            remove_btn.config(command=_remove)

        # ───────────── Tk dialog scaffold ────────────────────────────────────
        popup = tk.Toplevel(self.root)
        popup.title("Review Outliers")

        stats_lbl = tk.Label(popup, text="", font=("Arial", 10))
        stats_lbl.pack(pady=5)

        btn_frame = tk.Frame(popup); btn_frame.pack(pady=8)
        keep_btn   = tk.Button(btn_frame, text="Keep",   width=15)
        keep_btn.pack(side="left",  padx=20)
        remove_btn = tk.Button(btn_frame, text="Remove", width=15)
        remove_btn.pack(side="right", padx=20)

        show_next(0)            # start with the first flagged segment
        popup.grab_set()        # make modal
        self.root.wait_window(popup)

        return kept_segments


    # ------------------------------------------------------------------
    def _log_gui(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")

    def _scale_fonts(self, target_width, reference=1280, min_size=12, max_size=18):
        """
        Resize every Tk named-font once, based on the final window width.

        Parameters
        ----------
        target_width : int   final window width in pixels
        reference    : int   width that corresponds to 100 % font size
        min_size     : int   never go smaller than this
        max_size     : int   never go larger than this
        """
        scale = max(0.75, min(target_width / reference, 1.20))
        for fname in font.names():
            f = font.nametofont(fname)
            new_size = max(min_size, min(int(f.cget("size") * scale), max_size))
            f.configure(size=new_size)
    
    def _ylab(self, base="EMG"):
        """Return 'EMG (mV)' if we know the unit, else just 'EMG'."""
        return f"{base} ({self.emg_unit})" if self.emg_unit else base

    @staticmethod
    def _get_monitor_origin(ref_widget):
        """
        Return (mon_x, mon_y, mon_w, mon_h) for the monitor that contains
        the mouse cursor.  Used by _cap_toplevel to centre dialogs on the
        correct physical screen in multi-monitor setups.
        """
        sw = ref_widget.winfo_screenwidth()
        sh = ref_widget.winfo_screenheight()
        try:
            px = ref_widget.winfo_pointerx()
            py = ref_widget.winfo_pointery()
        except Exception:
            return 0, 0, sw, sh
        mon_col = px // sw
        mon_row = py // sh
        return mon_col * sw, mon_row * sh, sw, sh

    @staticmethod
    def _cap_toplevel(win, frac_h=0.88, frac_w=0.92):
        """Cap a Toplevel to a fraction of the active monitor and centre it."""
        win.update_idletasks()
        mon_x, mon_y, sw, sh = TMSAnalysisApp._get_monitor_origin(win)
        max_w   = int(sw * frac_w)
        max_h   = int(sh * frac_h)
        req_w   = win.winfo_reqwidth()  + 40
        req_h   = win.winfo_reqheight() + 40
        final_w = min(req_w, max_w)
        final_h = min(req_h, max_h)
        x = mon_x + (sw - final_w) // 2
        y = mon_y + (sh - final_h) // 4
        win.geometry(f"{final_w}x{final_h}+{x}+{y}")

    def _make_window_adaptive(self):
        """Maximise on startup — eliminates font/size complaints across all screens.

        Professional analysis tools (MATLAB, Spike2, LabChart) open maximised.
        Falls back to 90%-of-screen geometry if the platform doesn't support
        the zoomed state.
        """
        import sys as _sys
        try:
            if _sys.platform in ("win32", "darwin"):
                self.root.state("zoomed")
            else:
                self.root.attributes("-zoomed", True)
        except Exception:
            # Fallback: 90% of active monitor, centred
            mon_x, mon_y, sw, sh = self._get_monitor_origin(self.root)
            h       = max(int(sh * 0.9), 600)
            final_w = min(max(self.root.winfo_reqwidth() + 36, 680), int(sw * 0.9))
            x       = mon_x + (sw - final_w) // 2
            y       = mon_y + (sh - h) // 4
            self.root.geometry(f"{final_w}x{h}+{x}+{y}")

        # Apply DPI-aware font scaling after window is settled
        apply_scaling(self.root)

    # ------------------------------------------------------------------
    def _build_scrollable_container(self):
        """
        Create the top-level Notebook with two tabs:
          • Tab 1 – Stage 1: single-file processing  (scrollable)
          • Tab 2 – Stage 2: group-level analysis
        """
        # ── Top-level notebook ────────────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # Centre the tab strip — ttk doesn't expose this directly,
        # so we use a custom style with anchor="center" on the tab area.
        _nb_style = ttk.Style()
        _nb_style.configure("Centered.TNotebook", tabposition="n")
        _nb_style.configure("Centered.TNotebook.Tab", anchor="center", padding=[20, 4])
        self.notebook.configure(style="Centered.TNotebook")

        # ── Derivatives status bar ─────────────────────────────────────────────
        # Persistent strip below tabs: red when unset, green when set.
        # Clicking it opens the folder browser directly.
        self._deriv_status_bar = tk.Label(
            self.root,
            text="⚠  Derivatives folder not set — File → Set Derivatives Folder",
            bg="#d9534f", fg="white",
            anchor="w", padx=10, pady=3,
            font=("TkDefaultFont", 9))
        self._deriv_status_bar.pack(fill="x")
        self._deriv_status_bar.bind(
            "<Button-1>", lambda e: self.browse_derivatives_folder())

        # ── Session tab (index 0) ──────────────────────────────────────────────
        self.tab_session = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_session, text="Dataset Setup")
        self._build_session_tab(self.tab_session)

        # ── Stage 1a: Labels & Analysis Setup (index 1) ───────────────────────
        self.tab1b_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1b_frame, text="Stage 1a – Labels & Analysis Setup")
        self._labels_tab_built = False
        self._labels_tab_confirmed = False

        # ── Stage 1b: Single File Processing (index 2) ────────────────────────
        tab1_outer = ttk.Frame(self.notebook)
        self.notebook.add(tab1_outer, text="Stage 1b – Single File Processing")

        # Fixed footer — packed FIRST (before canvas) so it stays pinned
        # at the bottom regardless of scroll position.
        self.footer_frame = tk.Frame(tab1_outer, bd=1, relief="raised")
        self.footer_frame.pack(side="bottom", fill="x")

        # Scrollable area fills the remaining space above the footer
        scroll_area = ttk.Frame(tab1_outer)
        scroll_area.pack(side="top", fill="both", expand=True)

        vscroll = ttk.Scrollbar(scroll_area, orient="vertical")
        vscroll.pack(side="right", fill="y")

        self.canvas = tk.Canvas(scroll_area, bd=0, highlightthickness=0,
                                yscrollcommand=vscroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vscroll.config(command=self.canvas.yview)

        self.main_frame = ttk.Frame(self.canvas)
        self._canvas_window = self.canvas.create_window(
            (0, 0), window=self.main_frame, anchor="nw")

        # Re-centre content whenever the canvas is resized.
        # Content is capped at 860px wide so it doesn't stretch awkwardly
        # on large monitors, then centred in the available space.
        MAX_CONTENT_W = 1100
        def _on_canvas_resize(event):
            cw = event.width
            content_w = min(cw, MAX_CONTENT_W)
            x = max(0, (cw - content_w) // 2)
            self.canvas.itemconfigure(self._canvas_window, width=content_w)
            self.canvas.coords(self._canvas_window, x, 0)
        self.canvas.bind("<Configure>", _on_canvas_resize)

        self.main_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        def _on_mousewheel(event):
            # Only scroll when Stage 1b processing tab is active (index 2)
            if self.notebook.index(self.notebook.select()) == 2:
                delta = event.delta if event.delta else (-120 if event.num == 5 else 120)
                self.canvas.yview_scroll(int(-delta / 120), "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(seq, _on_mousewheel)

        # ── Stage 1c: Normalisation — Optional (index 3) ─────────────────────
        self.tab1c_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1c_frame,
                          text="Stage 1c – Normalisation (optional)")
        self._build_normalisation_tab()

        # ── Stage 2: Group Analysis (index 4) ────────────────────────────────
        self.tab2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2_frame, text="Stage 2 – Group Analysis LME Setup")
        self._stage2_built = False
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ─── User Path & Data States ──────────────────────────────────────────
        self.label_map = {}
        self.color_map = {}
        self.marker_choice = tk.StringVar()
        self.plot_included = {}
        self.csp_types     = set()   # event types where silent period is detected
        self.available_markers = []
        self.channel_choice = tk.StringVar()
        self.channel_idx    = 0
    
    # ──────────────────────────────────────────────────────────────
    def run_analysis_start(self):
        """Called by the green *Run Analysis* button (GUI thread)."""


        # Guard: derivatives folder must be set before running
        if not self.derivatives_path.get():
            messagebox.showwarning(
                "Derivatives folder not set",
                "Please set a derivatives folder before running analysis.\n\n"
                "Use File → Set Derivatives Folder, or click the red bar at the top.",
                parent=self.root)
            self.browse_derivatives_folder()
            return

        # Guard: Tab 1b must be confirmed before running
        if getattr(self, "_labels_tab_built", False) and \
                not getattr(self, "_labels_tab_confirmed", False):
            messagebox.showwarning(
                "Setup not confirmed",
                "Please go to the 'Stage 1b – Labels & Analysis Setup' tab "
                "and click  ✔ Confirm Setup  before running analysis.",
                parent=self.root)
            self.notebook.select(self.tab1b_frame)
            return
        # Guard: prevent launching a second worker while one is already running.
        if getattr(self, '_analysis_running', False):
            messagebox.showwarning(
                "Analysis in progress",
                "An analysis is already running. Please wait for it to finish.",
                parent=self.root)
            return

        # Reset any stale result left by a previous failed/interrupted run.
        self._last_outlier_result = None

        # Reset the progress bar so the UI looks fresh for each run.
        self.progress.set(0)

        self._log_gui("🔍 Running analysis…")

        # ---- TAKE A SNAPSHOT OF ALL GUI VARIABLES ----
        params = dict(
            # file & marker
            input_path        = self.file_path.get(),
            marker_choice     = self.marker_choice.get(),

            # time windows & analysis settings
            pre_ms            = self.pre_time.get(),
            post_ms           = self.post_time.get(),
            ptp_start         = self.ptp_start.get(),
            ptp_end           = self.ptp_end.get(),
            prestim_ms        = self.prestim_ms.get(),

            # filter settings
            apply_filter      = self.apply_filter.get(),
            apply_bandpass    = self.apply_bandpass.get(),
            apply_notch       = self.apply_notch.get(),
            highpass          = self.highpass.get(),
            lowpass           = self.lowpass.get(),
            notch_freq        = self.notch_freq.get(),
            notch_q           = self.notch_q.get(),
            apply_humbug      = self.apply_humbug.get(),
            humbug_harmonics  = self.humbug_harmonics.get(),
            filter_order      = self.filter_order.get(),
            filter_family     = self.filter_family.get(),
            cheby_ripple      = self.cheby_ripple.get(),
            flexible_bandpass = self.use_advanced_bp.get(),
            hp_order          = self.hp_order_var.get(),
            lp_order          = self.lp_order_var.get(),
            filter_harmonics  = self.filter_harmonics.get(),

            # statistics & outliers
            enable_out_review = self.outlier_review.get(),
            outlier_threshold = self.outlier_threshold.get(),

            # onset detection
            peak_fraction         = self.onset_peak_fraction.get(),
            min_amp               = self.onset_min_amplitude.get(),
            slope_threshold       = self.onset_slope_threshold.get(),
            onset_method          = self.onset_method.get(),
            onset_bootstrap_crit  = self.onset_bootstrap_crit.get(),
            onset_bootstrap_n     = self.onset_bootstrap_n.get(),
            latency_map           = dict(self.latency_map),

            # misc
            enable_inspector  = self.enable_inspector.get(),
            channel_idx       = self.channel_idx,
            label_map         = self.label_map.copy(),
            color_map         = self.color_map.copy(),
            plot_included     = self.plot_included.copy(),
            crop_start        = self.crop_start,
            crop_end          = self.crop_end,
            crop_ranges       = getattr(self, "crop_ranges", None),
            gap_ms_map        = self.gap_ms_map,
            # BIDS
            study_metadata    = self.study_metadata,
            limb              = getattr(self.study_metadata, "limb", ""),
            measure           = getattr(self.study_metadata, "measure", ""),
            reference_map          = self.reference_map.copy(),
            mmax_file              = self.mmax_file.get(),
            plateau_tolerance      = self.plateau_tolerance.get() / 100.0,
            extra_channel_indices  = list(self.extra_channel_indices),
            wide_window_s          = self.wide_window_s.get(),
            derivatives_root  = self.derivatives_path.get().strip() or None,
        )

        # Close any stale pyplot figures on the main thread via after(),
        # so we never destroy Tk-embedded canvases mid-event which causes
        # Tcl_AsyncDelete crashes on Windows.
        def _safe_close_figs():
            import matplotlib.pyplot as _plt
            _plt.close('all')
        self.root.after(50, _safe_close_figs)

        # ---- START BACKGROUND THREAD ----
        t = threading.Thread(
            target=self._analysis_worker,
            args=(params,),
            daemon=True
        )
        t.start()

    # ──────────────────────────────────────────────────────────────
    def _open_inspector_gui(self, segments_dict, fs, pre_ms, post_ms,
                            unit, label_map, color_map, analysis_pre_ms=None,
                            extra_segs=None, wide_window_s=3.0):
        """GUI thread – open the Inspector, block until closed.
        pre_ms here is the analysis/extraction pre-stim (prestim_ms, e.g. 100ms).
        visible_pre_ms is the display window (pre_time, e.g. 20ms).
        """
        n = len(next(iter(segments_dict.values()))[0])
        time_axis = np.linspace(-pre_ms, post_ms, n, endpoint=False)
        _analysis_pre  = analysis_pre_ms if analysis_pre_ms is not None else pre_ms
        _visible_pre   = self.pre_time.get()  # display window only
        inspector = DataInspectorWindow(
            self.root, segments_dict, time_axis,
            # Seed with any previously stored metadata so notes, AUC windows
            # and manual marker positions survive re-runs within the same session.
            metadata_dict       = dict(getattr(self, 'segments_metadata', {})),
            label_map=label_map, color_map=color_map, emg_unit=unit,
            ptp_start_ms        = self.ptp_start.get(),
            ptp_end_ms          = self.ptp_end.get(),
            analysis_pre_ms     = _analysis_pre,
            visible_pre_ms      = _visible_pre,
            extra_segs          = extra_segs or {},
            wide_window_s       = wide_window_s,
            # Onset detection method
            onset_method        = self.onset_method.get(),
            onset_bootstrap_crit= self.onset_bootstrap_crit.get(),
            onset_bootstrap_n   = self.onset_bootstrap_n.get(),
            latency_map         = dict(self.latency_map),
            # CSP detection
            csp_search_start_ms = self.csp_search_start_ms.get(),
            csp_search_end_ms   = self.csp_search_end_ms.get(),
            csp_min_silence_ms  = self.csp_min_silence_ms.get(),
            csp_min_return_ms   = self.csp_min_return_ms.get(),
            csp_criterion       = self.csp_criterion.get(),
            csp_significance    = self.csp_significance.get(),
            csp_n_boot          = self.csp_n_boot.get(),
            csp_max_mep_offset_ms = self.csp_max_mep_offset_ms.get(),
            csp_types           = self.csp_types,
        )
        self.root.wait_window(inspector.top)
        self.segments_metadata = dict(inspector.meta)
        self._last_outlier_result = inspector.meta
        # Auto-save the session immediately so inspector edits
        # are never lost if the user forgets Save Session.
        self._autosave_session()


    def _show_inspector_cb(self, segments_dict, fs, pre_ms, post_ms,
                        unit, label_map, color_map, analysis_pre_ms=None,
                        extra_segs=None, wide_window_s=3.0):
        """
        Called by the worker thread.  Sends a message to the GUI thread and waits.
        Returns the inspector's metadata dict.
        """
        self.msg_q.put(("show-inspector",
                        segments_dict, fs, pre_ms, post_ms,
                        unit, label_map, color_map, analysis_pre_ms,
                        extra_segs, wide_window_s))
        while self._last_outlier_result is None:
            time.sleep(0.05)
        meta = self._last_outlier_result
        self._last_outlier_result = None
        return meta                                                     # <<< NEW

    # ──────────────────────────────────────────────────────────────
    def _analysis_worker(self, params):
        """Heavy number‑crunching (runs in a background thread).

        IMPORTANT: do NOT call matplotlib.use() from this thread.
        run_pipeline uses matplotlib.figure.Figure()+FigureCanvasAgg
        directly, so the global backend is irrelevant here.
        Calling matplotlib.use("Agg") changes global state and triggers
        Tcl async-handler cleanup from the wrong thread, causing the
        hard "Tcl_AsyncDelete" crash on Windows.
        """
        import time

        self._analysis_running = True
        try:
            # -------- marker selection (thread‑safe) ----------------
            marker = params["marker_choice"]
            if not marker:
                choices = ["Keyboard", "TTL", "DigMark"]
                self._marker_choice_result = None
                self.msg_q.put(("ask‑marker", choices))

                while self._marker_choice_result is None:
                    time.sleep(0.05)

                marker = self._marker_choice_result

            # -------- run the heavy pipeline ------------------------
            run_pipeline(
                input_path           = params["input_path"],
                marker_name          = marker,
                log_callback         = lambda txt: self.msg_q.put(("log", txt)),
                progress_callback    = lambda p: self.msg_q.put(("progress", p)),
                review_outliers_cb   = self._review_outliers_cb,
                show_inspector_cb    = self._show_inspector_cb,

                # every other option comes straight from params
                pre_ms               = params["pre_ms"],
                post_ms              = params["post_ms"],
                ptp_start            = params["ptp_start"],
                ptp_end              = params["ptp_end"],
                prestim_ms           = params["prestim_ms"],

                apply_humbug         = params["apply_humbug"],
                humbug_harmonics     = params['humbug_harmonics'],
                apply_filter         = params["apply_filter"],
                apply_bandpass       = params["apply_bandpass"],
                apply_notch          = params["apply_notch"],
                highpass             = params["highpass"],
                lowpass              = params["lowpass"],
                notch_freq           = params["notch_freq"],
                notch_q              = params["notch_q"],
                filter_order         = params["filter_order"],
                filter_family        = params["filter_family"],
                cheby_ripple         = params["cheby_ripple"],
                flexible_bandpass    = params["flexible_bandpass"],
                hp_order             = params["hp_order"],
                lp_order             = params["lp_order"],
                filter_harmonics     = params["filter_harmonics"],

                enable_outlier_review= params["enable_out_review"],
                outlier_threshold    = params["outlier_threshold"],
                peak_fraction        = params["peak_fraction"],
                min_peak_amplitude   = params["min_amp"],
                slope_threshold      = params["slope_threshold"],
                onset_method         = params["onset_method"],
                onset_bootstrap_crit = params["onset_bootstrap_crit"],
                onset_bootstrap_n    = params["onset_bootstrap_n"],
                latency_map          = params.get("latency_map", {}),
                csp_types            = params.get("csp_types", set()),
                csp_min_silence_ms   = params.get("csp_min_silence_ms", 25.0),
                csp_min_return_ms    = params.get("csp_min_return_ms", 40.0),
                csp_criterion        = params.get("csp_criterion", 1.96),
                csp_significance     = params.get("csp_significance", 0.99),
                csp_n_boot           = params.get("csp_n_boot", 1000),
                csp_search_end_ms    = params.get("csp_search_end_ms", 400.0),
                csp_max_mep_offset_ms= params.get("csp_max_mep_offset_ms", 100.0),
                existing_segments_metadata = dict(self.segments_metadata),

                enable_inspector     = params["enable_inspector"],
                channel_idx          = params["channel_idx"],
                custom_labels        = params["label_map"],
                color_map            = params["color_map"],
                plot_included        = params["plot_included"],
                crop_start           = params["crop_start"],
                crop_end             = params["crop_end"],
                crop_ranges          = params["crop_ranges"],
                gap_ms_map           = params["gap_ms_map"],
                # BIDS
                study_metadata       = params["study_metadata"],
                limb                 = params.get("limb", ""),
                measure              = params.get("measure", ""),
                reference_map         = params.get("reference_map", {}),
                mmax_file             = params.get("mmax_file", ""),
                plateau_tolerance     = params.get("plateau_tolerance", 0.10),
                extra_channel_indices = params.get("extra_channel_indices", []),
                wide_window_s         = params.get("wide_window_s", 3.0),
                derivatives_root     = params["derivatives_root"],
            )

            self.msg_q.put(("log", "✅ Analysis complete!"))
            self.msg_q.put(("progress", 100))
            self.msg_q.put(("done", None))   # triggers autosave on GUI thread

        except Exception as e:
            self.msg_q.put(("log", f"❌ Error: {e}"))

        finally:
            self._analysis_running = False
            
    # ──────────────────────────────────────────────────────────────
    def _review_outliers_cb(self, flagged, fs, pre_ms, post_ms, unit):
        """
        Called BY the worker thread, executes the outlier dialog ON the GUI
        thread, waits, and finally returns the user's decision.
        """
        # 1. send a message so the poller can open the dialog
        self.msg_q.put(("show‑outliers", flagged, fs, pre_ms, post_ms, unit))

        # 2. wait until the dialog sets the result
        while self._last_outlier_result is None:
            time.sleep(0.05)

        kept = self._last_outlier_result
        self._last_outlier_result = None
        return kept

    def _build_menu(self):
        """Build the application menu bar and attach it to root."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open File…",         command=lambda: self.browse_file())
        file_menu.add_command(label="Set Derivatives Folder…", command=lambda: self.browse_derivatives_folder())
        file_menu.add_separator()
        file_menu.add_command(label="Save Session",  command=lambda: self.save_session())
        file_menu.add_command(label="Load Session",  command=lambda: self.load_session())
        file_menu.add_separator()
        file_menu.add_command(label="Exit",          command=self.root.quit)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Preferences...", command=self._open_preferences)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation",
            command=lambda: webbrowser.open(
                "https://github.com/jandrushko/mep-cmap-analyser"))
        help_menu.add_command(label="Report an Issue",
            command=lambda: webbrowser.open(
                "https://github.com/jandrushko/mep-cmap-analyser/issues"))
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    def setup_gui(self):
        # ─── Input File + Channel (single compact row) ──────────────────────
        # ── Active file indicator ─────────────────────────────────────────────
        file_row = tk.Frame(self.main_frame)
        file_row.pack(fill='x', padx=10, pady=(10, 0))
        tk.Label(file_row, text="Active file:").pack(side='left')

        tk.Entry(file_row, textvariable=self.file_path, width=56,
                 state="readonly", fg="#555").pack(
            side='left', expand=True, fill='x', padx=(4, 4))
        tk.Label(file_row, text="  Channel:").pack(side='left')
        self.channel_var = tk.StringVar(value="—")
        self.channel_dd  = ttk.Combobox(file_row, textvariable=self.channel_var,
                                         state="disabled", width=14)
        self.channel_dd.pack(side='left', padx=(4, 0))
        self.channel_dd.bind("<<ComboboxSelected>>", self._on_channel_selected)

        # ─── Filter Parameter Setup (placeholders) ───────────────────────────
        self.apply_filter = tk.BooleanVar(value=True)
        self.apply_bandpass = tk.BooleanVar(value=True)
        self.apply_notch = tk.BooleanVar(value=False)
        self.filter_harmonics = tk.BooleanVar(value=False)
        # ─── Filter toggles (add these lines near the others) ─────────────────
        self.apply_humbug     = tk.BooleanVar(value=True)  # master on/off
        self.humbug_harmonics = tk.IntVar(value=6)         # cancel up to 6 over‑tones
        self.highpass = tk.IntVar(value=20)
        self.lowpass = tk.IntVar(value=450)
        self.notch_freq = tk.IntVar(value=50)
        self.notch_q = tk.IntVar(value=30)
        self.filter_order = tk.IntVar(value=2)
        self.filter_family = tk.StringVar(value="butter")
        self.cheby_ripple = tk.DoubleVar(value=1.0)
        self.use_advanced_bp = tk.BooleanVar(value=False)
        self.hp_order_var = tk.IntVar(value=2)
        self.lp_order_var = tk.IntVar(value=2)

        # ─── Other Settings (time windows, bootstrap, etc.) ──────────────────
        self.onset_peak_fraction      = tk.DoubleVar(value=0.15)
        self.onset_min_amplitude      = tk.DoubleVar(value=0.1)
        self.onset_slope_threshold    = tk.DoubleVar(value=0.08)
        self.onset_method             = tk.StringVar(value="bootstrap")
        self.onset_bootstrap_crit     = tk.DoubleVar(value=1.96)
        self.onset_bootstrap_n        = tk.IntVar(value=500)
        self.pre_time = tk.IntVar(value=20)
        self.post_time = tk.IntVar(value=400)
        self.ptp_start = tk.IntVar(value=10)
        self.ptp_end = tk.IntVar(value=50)
        self.prestim_ms = tk.IntVar(value=100)
        self.outlier_review = tk.BooleanVar(value=True)
        self.outlier_threshold = tk.DoubleVar(value=1.96)
        self.generate_individual_plots = tk.BooleanVar(value=True)
        self.apply_humbug = tk.BooleanVar(value=False)
        self.csp_search_start_ms    = tk.IntVar(value=40)
        self.csp_search_end_ms      = tk.IntVar(value=400)
        self.csp_min_silence_ms     = tk.IntVar(value=25)
        self.csp_min_return_ms      = tk.IntVar(value=40)
        self.csp_criterion          = tk.DoubleVar(value=1.96)
        self.csp_significance       = tk.DoubleVar(value=0.99)
        self.csp_n_boot             = tk.IntVar(value=1000)
        self.csp_max_mep_offset_ms  = tk.IntVar(value=100)  # cSP start must be within this many ms of 2nd MEP peak

        # ─── Log + Progress ─────────────────────────────────────────────────
        self.log_box = None
        self.progress = tk.DoubleVar(value=0)
        self.progress_bar = None

        # ─── Filter Settings Section ────────────────────────────────────────────
        filter_frame = tk.LabelFrame(self.main_frame, text="Filter Settings",
                                    padx=6, pady=10)
        filter_frame.pack(padx=6, pady=(10, 0), fill='x')

        # --- define toggle functions FIRST, then bind them to self ---
        def _toggle_bandpass_fields():
            state = 'normal' if self.apply_bandpass.get() else 'disabled'
            # These three are present in Row 1
            self.hp_entry.config(state=state)
            self.lp_entry.config(state=state)
            self.ord_entry.config(state=state)
            # If advanced BP is on, disable single order regardless
            if self.use_advanced_bp.get():
                self.ord_entry.config(state='disabled')

        def _toggle_bp_order_fields():
            adv = bool(self.use_advanced_bp.get())
            hp_lp_state = 'normal' if adv and self.apply_bandpass.get() else 'disabled'
            one_state   = 'disabled' if adv else ('normal' if self.apply_bandpass.get() else 'disabled')
            # These two entries live under the “Advanced bandpass” row
            self.hp_order_entry.config(state=hp_lp_state)
            self.lp_order_entry.config(state=hp_lp_state)
            # Single order mirrors advanced toggle
            self.ord_entry.config(state=one_state)

        def _toggle_notch_fields():
            state = 'normal' if self.apply_notch.get() else 'disabled'
            self.notch_freq_entry.config(state=state)
            self.notch_q_entry.config(state=state)
            self.filter_harmonics_chk.config(state=state)

        # expose as attributes (so commands can reference self.* safely)
        self.toggle_bandpass_fields = _toggle_bandpass_fields
        self.toggle_bp_order_fields = _toggle_bp_order_fields
        self.toggle_notch_fields    = _toggle_notch_fields

        # Row 0: Apply Filter (master switch)
        tk.Checkbutton(filter_frame, text="Apply Filter", variable=self.apply_filter)\
            .grid(row=0, column=0, sticky='w', pady=(0, 4))

        # Row 1: Bandpass Filter + HP/LP + Order
        tk.Checkbutton(
            filter_frame,
            text="Bandpass Filter",
            variable=self.apply_bandpass,
            command=self.toggle_bandpass_fields
        ).grid(row=1, column=0, sticky='w')

        tk.Label(filter_frame, text="HP (Hz):").grid(row=1, column=1, sticky='e', padx=(10, 2))
        self.hp_entry = tk.Entry(filter_frame, textvariable=self.highpass, width=6)
        self.hp_entry.grid(row=1, column=2, sticky='w')

        tk.Label(filter_frame, text="LP (Hz):").grid(row=1, column=3, sticky='e', padx=(10, 2))
        self.lp_entry = tk.Entry(filter_frame, textvariable=self.lowpass, width=6)
        self.lp_entry.grid(row=1, column=4, sticky='w')

        tk.Label(filter_frame, text="Order:").grid(row=1, column=5, sticky='e', padx=(10, 2))
        self.ord_entry = tk.Entry(filter_frame, textvariable=self.filter_order, width=4)
        self.ord_entry.grid(row=1, column=6, sticky='w')

        # Row 2–3: Advanced bandpass controls
        tk.Checkbutton(
            filter_frame,
            text="Advanced bandpass (Separate HP/LP orders)",
            variable=self.use_advanced_bp,
            command=lambda: (self.toggle_bandpass_fields(), self.toggle_bp_order_fields())
        ).grid(row=2, column=0, columnspan=5, sticky='w', pady=(6, 0))

        tk.Label(filter_frame, text="HP order:").grid(row=3, column=0, sticky='w', padx=6)
        self.hp_order_entry = tk.Entry(filter_frame, textvariable=self.hp_order_var, width=5)
        self.hp_order_entry.grid(row=3, column=1, sticky='w')

        tk.Label(filter_frame, text="LP order:").grid(row=3, column=2, sticky='e', padx=6)
        self.lp_order_entry = tk.Entry(filter_frame, textvariable=self.lp_order_var, width=5)
        self.lp_order_entry.grid(row=3, column=3, sticky='w')

        # ── row-4: Notch filter + harmonics ----------------------------------------
        tk.Checkbutton(
            filter_frame,
            text="Notch Filter",
            variable=self.apply_notch,
            command=self.toggle_notch_fields
        ).grid(row=4, column=0, sticky='w')

        tk.Label(filter_frame, text="Notch Freq (Hz):").grid(row=4, column=1, sticky='e', padx=(10, 2))
        self.notch_freq_entry = tk.Entry(filter_frame, textvariable=self.notch_freq, width=6)
        self.notch_freq_entry.grid(row=4, column=2, sticky='w')

        tk.Label(filter_frame, text="Q-factor:").grid(row=4, column=3, sticky='e', padx=(10, 2))
        self.notch_q_entry = tk.Entry(filter_frame, textvariable=self.notch_q, width=6)
        self.notch_q_entry.grid(row=4, column=4, sticky='w')

        # Filter Notch Harmonics on same row
        self.filter_harmonics_chk = tk.Checkbutton(
            filter_frame,
            text="Filter Harmonics",
            variable=self.filter_harmonics
        )
        self.filter_harmonics_chk.grid(row=4, column=5, sticky='w', padx=(10, 0))

        # ── row-5: mains noise canceller + harmonics + preview button ─────────────
        tk.Checkbutton(
            filter_frame,
            text="Mains Noise Canceller",
            variable=self.apply_humbug,
            command=self._toggle_humbug_fields
        ).grid(row=5, column=0, sticky='w')

        tk.Label(filter_frame, text="Mains Harmonics:").grid(row=5, column=1, sticky='e')
        self.harmonics_entry = tk.Entry(filter_frame, textvariable=self.humbug_harmonics, width=5)
        self.harmonics_entry.grid(row=5, column=2, sticky='w')
        self.harmonics_entry.config(state='disabled')

        tk.Button(
            filter_frame,
            text="🔍 Preview Filter",
            command=self.preview_filter_window
        ).grid(row=5, column=5, columnspan=2, sticky='w', padx=(10, 0))

        # initial states
        self.hp_order_entry.config(state='disabled')
        self.lp_order_entry.config(state='disabled')

        # now run toggles once (after widgets exist)
        self.toggle_bandpass_fields()
        self.toggle_bp_order_fields()
        self.toggle_notch_fields()

        # ─── Time + Onset Settings ─────────────────────────────────────────────────
        # ── Time Window + MEP Onset Detection ──────────────────────────────────
        # Redesigned as 4-column grid, split into two logical sub-sections:
        #   • Time Windows  (rows 0-2)
        #   • Onset Detection (rows 4+, separated by a horizontal rule)
        time_frame = tk.LabelFrame(
            self.main_frame,
            text="Time Window + MEP Onset Detection Settings (ms)",
            padx=6, pady=10)
        time_frame.pack(padx=6, pady=(10, 0), fill='x')

        # ── Sub-section: Time Windows ────────────────────────────────────────
        # Row 0
        tk.Label(time_frame, text="Pre-stim visible (ms):").grid(
            row=0, column=0, sticky='e', padx=6)
        tk.Entry(time_frame, textvariable=self.pre_time, width=6).grid(
            row=0, column=1, sticky='w')
        tk.Label(time_frame, text="Post-stim visible (ms):").grid(
            row=0, column=2, sticky='e', padx=6)
        tk.Entry(time_frame, textvariable=self.post_time, width=6).grid(
            row=0, column=3, sticky='w')
        # Row 1
        tk.Label(time_frame, text="PTP window start (ms):").grid(
            row=1, column=0, sticky='e', padx=6)
        tk.Entry(time_frame, textvariable=self.ptp_start, width=6).grid(
            row=1, column=1, sticky='w')
        tk.Label(time_frame, text="PTP window end (ms):").grid(
            row=1, column=2, sticky='e', padx=6)
        tk.Entry(time_frame, textvariable=self.ptp_end, width=6).grid(
            row=1, column=3, sticky='w')
        # Row 2
        tk.Label(time_frame, text="Pre-stim for analysis (ms):").grid(
            row=2, column=0, sticky='e', padx=6)
        tk.Entry(time_frame, textvariable=self.prestim_ms, width=6).grid(
            row=2, column=1, sticky='w')

        # ── Separator ────────────────────────────────────────────────────────
        ttk.Separator(time_frame, orient="horizontal").grid(
            row=3, column=0, columnspan=4, sticky='ew', pady=(8, 4))
        tk.Label(time_frame, text="MEP Onset Detection",
                 font=("TkDefaultFont", 9, "bold")).grid(
            row=3, column=0, columnspan=4, sticky='w', padx=6)

        # ── Sub-section: Onset Detection ─────────────────────────────────────
        # Row 4: method + min amplitude
        tk.Label(time_frame, text="Method:").grid(
            row=4, column=0, sticky='e', padx=6)
        onset_method_cb = ttk.Combobox(
            time_frame, textvariable=self.onset_method,
            values=["peak_fraction", "bootstrap"],
            state="readonly", width=14)
        onset_method_cb.grid(row=4, column=1, sticky='w')
        tk.Label(time_frame, text="Min amplitude (mV):").grid(
            row=4, column=2, sticky='e', padx=6)
        tk.Entry(time_frame, textvariable=self.onset_min_amplitude, width=6).grid(
            row=4, column=3, sticky='w')

        # Row 5: method-specific params (peak_fraction and bootstrap share this row)
        self._pf_lbl1 = tk.Label(time_frame, text="Peak fraction:")
        self._pf_lbl1.grid(row=5, column=0, sticky='e', padx=6)
        self._pf_ent1 = tk.Entry(time_frame, textvariable=self.onset_peak_fraction, width=6)
        self._pf_ent1.grid(row=5, column=1, sticky='w')
        self._pf_lbl2 = tk.Label(time_frame, text="Slope threshold (mV/ms):")
        self._pf_lbl2.grid(row=5, column=2, sticky='e', padx=6)
        self._pf_ent2 = tk.Entry(time_frame, textvariable=self.onset_slope_threshold, width=6)
        self._pf_ent2.grid(row=5, column=3, sticky='w')

        self._bs_lbl1 = tk.Label(time_frame, text="Z-score criterion:")
        self._bs_lbl1.grid(row=5, column=0, sticky='e', padx=6)
        self._bs_ent1 = tk.Entry(time_frame, textvariable=self.onset_bootstrap_crit, width=6)
        self._bs_ent1.grid(row=5, column=1, sticky='w')
        self._bs_lbl2 = tk.Label(time_frame, text="Bootstrap n:")
        self._bs_lbl2.grid(row=5, column=2, sticky='e', padx=6)
        self._bs_ent2 = tk.Entry(time_frame, textvariable=self.onset_bootstrap_n, width=6)
        self._bs_ent2.grid(row=5, column=3, sticky='w')

        def _on_onset_method_change(*_):
            is_pf = self.onset_method.get() == "peak_fraction"
            if is_pf:
                # Show peak-fraction widgets, hide bootstrap widgets
                for w in (self._pf_lbl1, self._pf_ent1,
                          self._pf_lbl2, self._pf_ent2):
                    w.grid()
                for w in (self._bs_lbl1, self._bs_ent1,
                          self._bs_lbl2, self._bs_ent2):
                    w.grid_remove()
            else:
                # Show bootstrap widgets, hide peak-fraction widgets
                for w in (self._pf_lbl1, self._pf_ent1,
                          self._pf_lbl2, self._pf_ent2):
                    w.grid_remove()
                for w in (self._bs_lbl1, self._bs_ent1,
                          self._bs_lbl2, self._bs_ent2):
                    w.grid()

        self.onset_method.trace_add("write", _on_onset_method_change)
        _on_onset_method_change()  # set initial visibility

        # ─── CSP Detection Settings ────────────────────────────────────────────────
        csp_frame = tk.LabelFrame(self.main_frame,
            text="CSP (Cortical Silent Period) Detection Settings", padx=6, pady=8)
        csp_frame.pack(padx=6, pady=(8,0), fill='x')
        tk.Label(csp_frame, text="Search start (ms post-stim):").grid(row=0,column=0,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_search_start_ms, width=5).grid(row=0,column=1,sticky='w')
        tk.Label(csp_frame, text="Search end (ms post-stim):").grid(row=0,column=2,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_search_end_ms, width=5).grid(row=0,column=3,sticky='w')
        tk.Label(csp_frame, text="Min silence (ms):").grid(row=1,column=0,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_min_silence_ms, width=5).grid(row=1,column=1,sticky='w')
        tk.Label(csp_frame, text="Min return (ms):").grid(row=1,column=2,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_min_return_ms, width=5).grid(row=1,column=3,sticky='w')
        tk.Label(csp_frame, text="Z-score criterion:").grid(row=2,column=0,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_criterion, width=5).grid(row=2,column=1,sticky='w')
        tk.Label(csp_frame, text="Bootstrap significance:").grid(row=2,column=2,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_significance, width=5).grid(row=2,column=3,sticky='w')
        tk.Label(csp_frame, text="Bootstrap iterations:").grid(row=3,column=0,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_n_boot, width=7).grid(row=3,column=1,sticky='w')
        tk.Label(csp_frame, text="Max offset from MEP 2nd peak (ms):").grid(row=3,column=2,sticky='e',padx=6)
        tk.Entry(csp_frame, textvariable=self.csp_max_mep_offset_ms, width=5).grid(row=3,column=3,sticky='w')
        tk.Label(csp_frame,
            text="Z-score: threshold multiplier (1.96 = 95% CI)  ·  Significance: bootstrap percentile for min duration (0.99 = 99th pct)  ·  "
                 "Max offset: cSP start must fall within this many ms after the 2nd MEP peak (prevents unrealistic late placements)",
            fg="grey",font=("TkDefaultFont",9,"italic")).grid(row=4,column=0,columnspan=4,sticky='w',padx=6,pady=(2,0))

        # ─── Outlier Detection ─────────────────────────────────────────────────
        out_frame = tk.LabelFrame(self.main_frame, text="Outlier Detection Settings",
                                  padx=6, pady=6)
        out_frame.pack(padx=6, pady=(10, 0), fill='x')
        tk.Checkbutton(out_frame, text="Enable Outlier Review",
                       variable=self.outlier_review).grid(row=0, column=0, sticky='w')
        tk.Label(out_frame, text="Z-score threshold:").grid(row=0, column=1, sticky='e', padx=(20,4))
        tk.Entry(out_frame, textvariable=self.outlier_threshold, width=6).grid(row=0, column=2, sticky='w')

        # ─── Analysis Options + Session + Run ─────────────────────────────────
        self.enable_inspector = tk.BooleanVar(value=True)
        run_frame = tk.LabelFrame(self.main_frame, text="Analysis Options",
                                  padx=6, pady=6)
        run_frame.pack(padx=6, pady=(10, 0), fill='x')
        tk.Checkbutton(run_frame, text="Generate individual plots per event type",
            variable=self.generate_individual_plots).grid(row=0, column=0, sticky='w', padx=4)
        tk.Checkbutton(run_frame, text="Enable Data Inspector",
            variable=self.enable_inspector).grid(row=0, column=1, sticky='w', padx=4)

        # Log stays in the scrollable area so it expands with content
        tk.Label(self.main_frame, text="Log:").pack(anchor='w', padx=10, pady=(10,0))
        self.log_box = scrolledtext.ScrolledText(self.main_frame, height=6, wrap=tk.WORD)
        self.log_box.pack(fill='both', expand=True, padx=10, pady=(0,5))

        # Small author label
        author_font = font.Font(size=12, slant="italic")
        tk.Label(self.main_frame, text="Author: Justin Andrushko PhD, Northumbria University",
                 font=author_font, anchor='center').pack(pady=(0,5))

        # ── Fixed footer: session buttons + run + progress bar ────────────────
        # Built here (after all tk vars exist) but packed into footer_frame
        # which was already placed at the bottom of tab1_outer.
        footer_inner = tk.Frame(self.footer_frame, padx=6, pady=4)
        footer_inner.pack(fill="x")

        tk.Button(footer_inner, text="💾 Save Session", width=14,
                  command=self.save_session).pack(side="left", padx=(6,4))
        tk.Button(footer_inner, text="📂 Load Session", width=14,
                  command=self.load_session).pack(side="left", padx=(0,4))
        tk.Button(footer_inner, text="▶  Run Analysis", width=14,
                  command=self.run_analysis_start).pack(side="left", padx=(12,4))
        self.progress_bar = ttk.Progressbar(footer_inner, variable=self.progress,
                                            maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(8,6))

        # --- auto-resize window to content on startup ---
        self.root.update_idletasks()
        self.root.geometry("")   # shrink-wrap to widgets

        # Optional: set a reasonable minimum so it doesn't get too small
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())




    def _autosave_session(self):
        """Silently save the session to the BIDS derivatives folder.

        Called automatically after the Data Inspector closes so that
        inspector edits (PTP markers, CSP boundaries, exclusions, notes)
        are never lost if the user forgets to click Save Session manually.

        The file is written as:
            <derivatives_root>/derivatives/<sub>/<ses>/<bids_prefix>_session.json
        or, if no derivatives root is configured:
            <source_file_dir>/derivatives/<sub>/<ses>/<bids_prefix>_session.json
        """
        try:
            import datetime, json
            from dataclasses import asdict as _ad

            fp   = self.file_path.get()
            meta = getattr(self, 'study_metadata', None)

            # ── Build save path ───────────────────────────────────────────────
            bids_prefix = (meta.bids_prefix() if meta and meta.bids_prefix()
                           else (pathlib.Path(fp).stem if fp else "mep_cmap"))

            source_dir  = os.path.dirname(fp) if fp else os.getcwd()
            deriv_root  = (self.derivatives_path.get()
                           if hasattr(self, 'derivatives_path') and
                              self.derivatives_path.get()
                           else source_dir)

            sub_ses     = meta.sub_ses_path() if meta else os.path.join("sub-unknown", "ses-01")
            # Avoid derivatives/derivatives/ — same fix as in pipeline.py
            _deriv_base = os.path.basename(os.path.normpath(deriv_root)).lower()
            if _deriv_base == "derivatives":
                save_dir = os.path.join(deriv_root, sub_ses)
            else:
                save_dir = os.path.join(deriv_root, "derivatives", sub_ses)
            os.makedirs(save_dir, exist_ok=True)
            save_path   = os.path.join(save_dir, f"{bids_prefix}_session.json")

            # ── Serialise (same logic as save_session, no dialog) ─────────────
            def _j(v):
                if isinstance(v, (np.integer,)):  return int(v)
                if isinstance(v, (np.floating,)): return float(v)
                if isinstance(v, bool):            return bool(v)
                return v

            meta_s = {
                f"{st}:{i}": {k: _j(v) for k, v in m.items()}
                for (st, i), m in self.segments_metadata.items()
            }
            sm = {}
            if meta:
                try: sm = _ad(meta)
                except Exception: pass

            s = {
                "pre_ms":                self.pre_time.get(),
                "post_ms":               self.post_time.get(),
                "ptp_start":             self.ptp_start.get(),
                "ptp_end":               self.ptp_end.get(),
                "prestim_ms":            self.prestim_ms.get(),
                "apply_filter":          self.apply_filter.get(),
                "apply_bandpass":        self.apply_bandpass.get(),
                "apply_notch":           self.apply_notch.get(),
                "highpass":              self.highpass.get(),
                "lowpass":               self.lowpass.get(),
                "notch_freq":            self.notch_freq.get(),
                "notch_q":               self.notch_q.get(),
                "filter_order":          self.filter_order.get(),
                "filter_family":         self.filter_family.get(),
                "cheby_ripple":          self.cheby_ripple.get(),
                "use_advanced_bp":       self.use_advanced_bp.get(),
                "hp_order":              self.hp_order_var.get(),
                "lp_order":              self.lp_order_var.get(),
                "filter_harmonics":      self.filter_harmonics.get(),
                "apply_humbug":          self.apply_humbug.get(),
                "humbug_harmonics":      self.humbug_harmonics.get(),
                "outlier_review":        self.outlier_review.get(),
                "outlier_threshold":     self.outlier_threshold.get(),
                "onset_peak_fraction":   self.onset_peak_fraction.get(),
                "onset_min_amplitude":   self.onset_min_amplitude.get(),
                "onset_slope_threshold": self.onset_slope_threshold.get(),
                "onset_method":          self.onset_method.get(),
                "onset_bootstrap_crit":  self.onset_bootstrap_crit.get(),
                "onset_bootstrap_n":     self.onset_bootstrap_n.get(),
                "enable_inspector":      self.enable_inspector.get(),
                "generate_individual_plots": self.generate_individual_plots.get(),
                "csp_search_start_ms":   self.csp_search_start_ms.get(),
                "csp_search_end_ms":     self.csp_search_end_ms.get(),
                "csp_min_silence_ms":    self.csp_min_silence_ms.get(),
                "csp_criterion":         self.csp_criterion.get(),
                "csp_significance":      self.csp_significance.get(),
                "csp_min_return_ms":     self.csp_min_return_ms.get(),
                "csp_n_boot":            self.csp_n_boot.get(),
                "csp_max_mep_offset_ms": self.csp_max_mep_offset_ms.get(),
                "csp_types":             list(self.csp_types),
                "wide_window_s":         self.wide_window_s.get(),
                "latency_map":           {k: list(v) for k, v in self.latency_map.items()},
                "latency_stim_map":      dict(self.latency_stim_map),
                "latency_muscle_map":    dict(self.latency_muscle_map),
            }
            # ── Compute study root for relative path storage ──────────────────
            # The JSON lives at:  <study_root>/derivatives/<sub>/<ses>/<name>.json
            # Walk up 3 levels from save_dir to get study_root.
            # Paths stored as relative to study_root so the file is portable
            # across computers (OneDrive, different user home folders, etc.).
            _json_deriv_dir = save_dir
            _study_root_for_json = os.path.dirname(
                os.path.dirname(os.path.dirname(_json_deriv_dir)))

            def _rel(p):
                """Store p relative to study root; fall back to basename if outside."""
                if not p:
                    return p
                try:
                    rel = os.path.relpath(p, _study_root_for_json)
                    return rel if not rel.startswith("..") else os.path.basename(p)
                except ValueError:
                    return os.path.basename(p)

            session = {
                "version":          "1.0",
                "saved_at":         datetime.datetime.now().isoformat(timespec="seconds"),
                "autosaved":        True,   # flag so user knows this wasn't a manual save
                "file_path":        _rel(fp),
                "marker_choice":    self.marker_choice.get(),
                "channel_idx":      self.channel_idx,
                "channel_choice":   self.channel_choice.get(),
                "crop_ranges":      self.crop_ranges,
                "crop_start":       self.crop_start,
                "crop_end":         self.crop_end,
                "label_map":        self.label_map,
                "color_map":        self.color_map,
                "plot_included":    self.plot_included,
                "gap_ms_map":       self.gap_ms_map,
                "reference_map":    self.reference_map,
                "reference_display": getattr(self, '_reference_display', {}),
                "latency_map":      {k: list(v) for k, v in self.latency_map.items()},
                "latency_stim_map":   dict(self.latency_stim_map),
                "latency_muscle_map": dict(self.latency_muscle_map),
                "mmax_file":        _rel(self.mmax_file.get()),
                "plateau_tolerance":self.plateau_tolerance.get(),
                "extra_channel_indices": self.extra_channel_indices,
                "wide_window_s":    self.wide_window_s.get(),
                "derivatives_path": _rel(self.derivatives_path.get()
                                         if hasattr(self, "derivatives_path") else ""),
                "study_metadata":   sm,
                "settings":         s,
                "segments_metadata": meta_s,
            }

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2)

            # Update the FileEntry in the dataset session
            if self._dataset is not None and hasattr(self, '_current_file_entry'):
                fe = self._current_file_entry
                if fe is not None:
                    fe.derivatives_json = save_path
                    fe.stim_letters = sorted(self.label_map.keys())
                    fe.stim_label_map = dict(self.label_map)
                    self._dataset.save()
                    self._queue_refresh()

            self._log_gui(
                f"💾 Session auto-saved → "
                f"{os.path.relpath(save_path, source_dir)}")

        except Exception as e:
            # Auto-save failures are non-fatal — just log, don't alert
            self._log_gui(f"⚠️  Auto-save failed: {e}")

    def save_session(self):
        """Serialise all GUI settings, file context, and inspector metadata."""
        fp = self.file_path.get()
        ddir = os.path.dirname(fp) if fp else os.getcwd()
        dname = (pathlib.Path(fp).stem+"_session.json") if fp else "mep_cmap_session.json"
        sp = filedialog.asksaveasfilename(title="Save session",initialdir=ddir,initialfile=dname,
            defaultextension=".json",filetypes=[("MEP-CMAP session","*.json"),("All files","*.*")],parent=self.root)
        if not sp: return
        def _j(v):
            if isinstance(v,(np.integer,)): return int(v)
            if isinstance(v,(np.floating,)): return float(v)
            if isinstance(v,bool): return bool(v)
            return v
        meta_s = {f"{st}:{i}":{k:_j(v) for k,v in m.items()} for (st,i),m in self.segments_metadata.items()}
        sm = {}
        if hasattr(self,'study_metadata') and self.study_metadata:
            try:
                from dataclasses import asdict as _ad; sm=_ad(self.study_metadata)
            except Exception: pass
        s = {"pre_ms":self.pre_time.get(),"post_ms":self.post_time.get(),
             "ptp_start":self.ptp_start.get(),"ptp_end":self.ptp_end.get(),
             "prestim_ms":self.prestim_ms.get(),"apply_filter":self.apply_filter.get(),
             "apply_bandpass":self.apply_bandpass.get(),"apply_notch":self.apply_notch.get(),
             "highpass":self.highpass.get(),"lowpass":self.lowpass.get(),
             "notch_freq":self.notch_freq.get(),"notch_q":self.notch_q.get(),
             "filter_order":self.filter_order.get(),"filter_family":self.filter_family.get(),
             "cheby_ripple":self.cheby_ripple.get(),"use_advanced_bp":self.use_advanced_bp.get(),
             "hp_order":self.hp_order_var.get(),"lp_order":self.lp_order_var.get(),
             "filter_harmonics":self.filter_harmonics.get(),"apply_humbug":self.apply_humbug.get(),
             "humbug_harmonics":self.humbug_harmonics.get(),"outlier_review":self.outlier_review.get(),
             "outlier_threshold":self.outlier_threshold.get(),
             "onset_peak_fraction":self.onset_peak_fraction.get(),
             "onset_min_amplitude":self.onset_min_amplitude.get(),
             "onset_slope_threshold":self.onset_slope_threshold.get(),
             "enable_inspector":self.enable_inspector.get(),"generate_individual_plots":self.generate_individual_plots.get(),
             "csp_search_start_ms":self.csp_search_start_ms.get(),
             "csp_search_end_ms":self.csp_search_end_ms.get(),
             "csp_min_silence_ms":self.csp_min_silence_ms.get(),
             "csp_criterion":self.csp_criterion.get(),
             "csp_significance":self.csp_significance.get(),
             "csp_min_return_ms":self.csp_min_return_ms.get(),
             "csp_n_boot":self.csp_n_boot.get(),
             "csp_max_mep_offset_ms":self.csp_max_mep_offset_ms.get(),
             "csp_types":list(self.csp_types)}
        session={"version":"1.0","saved_at":datetime.datetime.now().isoformat(timespec="seconds"),
                 "file_path":fp,"marker_choice":self.marker_choice.get(),
                 "channel_idx":self.channel_idx,"channel_choice":self.channel_choice.get(),
                 "crop_ranges":self.crop_ranges,"crop_start":self.crop_start,"crop_end":self.crop_end,
                 "label_map":self.label_map,"color_map":self.color_map,
                 "plot_included":self.plot_included,"gap_ms_map":self.gap_ms_map,
                 "derivatives_path":self.derivatives_path.get() if hasattr(self,"derivatives_path") else "",
                 "study_metadata":sm,"settings":s,"segments_metadata":meta_s}
        try:
            with open(sp,"w",encoding="utf-8") as f: json.dump(session,f,indent=2)
            self.log(f"\U0001f4be Session saved \u2192 {os.path.basename(sp)}")
        except Exception as e:
            messagebox.showerror("Save failed",str(e),parent=self.root)

    def _apply_loaded_session(self, sess: dict, json_path: str = ""):
        """
        Apply a loaded session dict to the current GUI state.
        Called by both load_session (user-initiated) and _load_file_entry
        (automatic restore when jumping to a previously processed file).

        json_path — the path of the JSON file being loaded.  When provided,
        relative paths stored in the session (file_path, mmax_file,
        derivatives_path) are resolved against the study root derived from
        that file's location.  This makes sessions portable across computers
        with different OneDrive / home-directory paths.
        """
        # ── Resolve helper ────────────────────────────────────────────────────
        def _abs(stored: str) -> str:
            """Resolve a stored (possibly relative) path to absolute."""
            if not stored:
                return stored
            if os.path.isabs(stored) and os.path.exists(stored):
                return stored          # absolute and valid on this machine
            if not json_path:
                return stored          # no anchor — return as-is
            # Derive study root from JSON location:
            # JSON lives at <study_root>/derivatives/<sub>/<ses>/<name>.json
            # Walk up 3 levels from the JSON's directory.
            json_dir    = os.path.dirname(os.path.abspath(json_path))
            study_root  = os.path.dirname(os.path.dirname(os.path.dirname(json_dir)))
            candidate   = os.path.normpath(
                os.path.join(study_root, stored.replace("\\", os.sep)))
            if os.path.exists(candidate):
                return candidate
            # Basename search under study root as last resort
            basename = os.path.basename(stored.replace("\\", os.sep))
            if basename:
                for dirpath, _dirs, files in os.walk(study_root):
                    if basename in files:
                        return os.path.join(dirpath, basename)
            return candidate   # best effort — caller handles missing file

        fp = _abs(sess.get("file_path", ""))
        self.file_path.set(fp)
        self.marker_choice.set(sess.get("marker_choice",""))
        self.channel_idx=sess.get("channel_idx",0); self.channel_choice.set(sess.get("channel_choice",""))
        cr=sess.get("crop_ranges"); self.crop_ranges=[tuple(r) for r in cr] if cr else None
        self.crop_start=sess.get("crop_start"); self.crop_end=sess.get("crop_end")
        self.label_map=sess.get("label_map",{}); self.color_map=sess.get("color_map",{})
        self.plot_included=sess.get("plot_included",{}); self.gap_ms_map=sess.get("gap_ms_map",{})
        self.reference_map=sess.get("reference_map",{})
        self._reference_display=sess.get("reference_display",{})
        _lm = sess.get("latency_map", {})
        self.latency_map = {k: tuple(v) for k, v in _lm.items()} if _lm else {}
        self.latency_stim_map   = sess.get("latency_stim_map", {})
        self.latency_muscle_map = sess.get("latency_muscle_map", {})
        if hasattr(self,"derivatives_path"):
            dp = _abs(sess.get("derivatives_path",""))
            if dp: self.derivatives_path.set(dp)
            self._update_deriv_status()
        sm=sess.get("study_metadata",{})
        if sm and hasattr(self,"study_metadata"):
            try:
                self.study_metadata=StudyMetadata(**{k:v for k,v in sm.items() if k in StudyMetadata.__dataclass_fields__})
            except Exception: pass
        if sess.get("mmax_file"): self.mmax_file.set(_abs(sess["mmax_file"]))
        if sess.get("plateau_tolerance"): self.plateau_tolerance.set(sess["plateau_tolerance"])
        # csp_types is stored in the settings sub-dict
        s=sess.get("settings",{})
        _b=lambda k,d:bool(s.get(k,d)); _i=lambda k,d:int(s.get(k,d))
        _f=lambda k,d:float(s.get(k,d)); _s=lambda k,d:str(s.get(k,d))
        self.csp_types = set(s.get("csp_types", sess.get("csp_types", [])))
        if s:
            try:
                self.apply_filter.set(_b("apply_filter",True))
                self.apply_bandpass.set(_b("apply_bandpass",True))
                self.apply_notch.set(_b("apply_notch",False))
                self.highpass.set(_i("highpass",20))
                self.lowpass.set(_i("lowpass",450))
                self.notch_freq.set(_i("notch_freq",50))
                self.notch_q.set(_i("notch_q",30))
                self.filter_order.set(_i("filter_order",2))
                self.filter_harmonics.set(_b("filter_harmonics",False))
                self.filter_family.set(_s("filter_family","butter"))
                self.cheby_ripple.set(_f("cheby_ripple",1.0))
                self.use_advanced_bp.set(_b("use_advanced_bp",False))
                self.hp_order_var.set(_i("hp_order",2))
                self.lp_order_var.set(_i("lp_order",2))
                self.apply_humbug.set(_b("apply_humbug",False))
                self.humbug_harmonics.set(_i("humbug_harmonics",6))
                self.pre_time.set(_i("pre_time",20))
                self.post_time.set(_i("post_time",400))
                self.ptp_start.set(_i("ptp_start",10))
                self.ptp_end.set(_i("ptp_end",50))
                self.prestim_ms.set(_i("prestim_ms",100))
                self.outlier_review.set(_b("outlier_review",True))
                self.outlier_threshold.set(_f("outlier_threshold",1.96))
                self.onset_method.set(_s("onset_method","bootstrap"))
                self.onset_bootstrap_crit.set(_f("onset_bootstrap_crit",1.96))
                self.onset_bootstrap_n.set(_i("onset_bootstrap_n",500))
                self.onset_peak_fraction.set(_f("onset_peak_fraction",0.15))
                self.onset_min_amplitude.set(_f("onset_min_amplitude",0.1))
                self.onset_slope_threshold.set(_f("onset_slope_threshold",0.08))
                self.enable_inspector.set(_b("enable_inspector",True))
                self.generate_individual_plots.set(_b("generate_individual_plots",True))
                self.wide_window_s.set(_f("wide_window_s",3.0))
                # onset_min_latency_ms / onset_max_latency_ms were removed in v0.8.4
                # (replaced by per-stim latency_map) — skip silently for old sessions
                self.csp_search_end_ms.set(_i("csp_search_end_ms",400))
                self.csp_min_silence_ms.set(_i("csp_min_silence_ms",25))
                self.csp_min_return_ms.set(_i("csp_min_return_ms",40))
                self.csp_criterion.set(_f("csp_criterion",1.96))
                self.csp_significance.set(_f("csp_significance",0.99))
                self.csp_n_boot.set(_i("csp_n_boot",1000))
                self.csp_max_mep_offset_ms.set(_i("csp_max_mep_offset_ms",100))
                self.csp_types = set(s.get("csp_types", []))
                _lm2 = s.get("latency_map", {})
                if _lm2:
                    self.latency_map = {k: tuple(v) for k, v in _lm2.items()}
                    self.latency_stim_map   = s.get("latency_stim_map", {})
                    self.latency_muscle_map = s.get("latency_muscle_map", {})
            except Exception:
                pass  # old session format — skip unrecognised settings
        restored={}
        for ks,m in sess.get("segments_metadata",{}).items():
            try:
                st,i_s=ks.rsplit(":",1); restored[(st,int(i_s))]=m
            except ValueError: continue
        self.segments_metadata=restored
        try: self.toggle_bandpass_fields(); self.toggle_bp_order_fields(); self.toggle_notch_fields(); self._toggle_humbug_fields()
        except Exception: pass

    def load_session(self):
        """Restore a previously saved JSON session."""
        lp = filedialog.askopenfilename(title="Load session",defaultextension=".json",
            filetypes=[("MEP-CMAP session","*.json"),("All files","*.*")],parent=self.root)
        if not lp: return
        try:
            with open(lp,"r",encoding="utf-8") as f: sess=json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed",str(e),parent=self.root); return
        self._reset_state_for_new_file()
        self._apply_loaded_session(sess, json_path=lp)
        fp = self.file_path.get()
        self.log(f"📂 Loaded from {os.path.basename(lp)}\n"
                 f"   File: {os.path.basename(fp) if fp else '(none)'}\n"
                 f"   Labels: {len(self.label_map)}  Inspector edits: {len(self.segments_metadata)}\n"
                 f"   ✅ Click Run Analysis to re-process.")
        if fp and not os.path.isfile(fp):
            messagebox.showwarning("File not found",
                f"Session references:\n  {fp}\n\nUse Browse to locate it.",
                parent=self.root)


    def _crop_selector(self, txt_file) -> bool:
        """
        Let the user pick **one or more** time‑ranges to analyse.
        Returns True if at least one range is confirmed.
        """

        # ── Load the data (unchanged) ───────────────────────────────────────────
        try:
            emg, fs, self.emg_unit = extract_emg_waveform_and_fs(
                txt_file, channel_idx=self.channel_idx)
            t = np.arange(emg.size) / fs
            stim_dict = extract_stim_times(
                txt_file, self.marker_choice.get() or "Keyboard")
        except Exception as e:
            messagebox.showerror("Could not preview file", str(e), parent=self.root)
            return False

        # ── Build the modal window ──────────────────────────────────────────────
        top = tk.Toplevel(self.root);  top.title("Select one or more ranges")
        top.grab_set()
        try:
            import sys as _sys
            if _sys.platform in ("win32", "darwin"):
                top.state("zoomed")
            else:
                top.attributes("-zoomed", True)
        except Exception:
            pass

        # ── Footer packed FIRST so canvas fills all remaining space ──────────
        list_lbl = tk.StringVar()
        footer = tk.Frame(top)
        footer.pack(side="bottom", fill="x")
        info = tk.Label(footer, textvariable=list_lbl, anchor="w")
        info.pack(fill="x", padx=10, pady=(4, 2))
        btn_frm = tk.Frame(footer)
        btn_frm.pack(pady=(0, 8))

        # ── Canvas fills all space above the footer ──────────────────────────────
        # Create a figure sized to the screen. Do NOT use expand=True on the
        # canvas widget — that makes the widget larger than the figure and
        # causes matplotlib to tile the rendered image into the blank space.
        _sw   = self.root.winfo_screenwidth()
        _sh   = self.root.winfo_screenheight()
        _dpi  = 96
        fig   = matplotlib.figure.Figure(
                    figsize=(_sw / _dpi, (_sh - 100) / _dpi), dpi=_dpi)
        fig.subplots_adjust(left=0.05, right=0.998, top=0.93, bottom=0.12)
        ax    = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=top)
        # expand=False prevents the Tk canvas widget from growing beyond the
        # figure size, which would cause the rendered image to be tiled.
        canvas.get_tk_widget().pack(fill="both", expand=False)

        # ── Plot the full trace + stim ticks ──────────────────────────────────────
        # Min-max envelope downsampling — preserves amplitude envelope of all
        # events while keeping point count low for fast blit interaction.
        _max_pts = 4000
        if len(t) > _max_pts:
            _chunk = len(t) // (_max_pts // 2)
            _n_chunks = len(t) // _chunk
            _t_ds, _emg_ds = [], []
            for _i in range(_n_chunks):
                _s = _i * _chunk
                _e = _s + _chunk
                _chunk_emg = emg[_s:_e]
                _chunk_t   = t[_s:_e]
                _imin = int(np.argmin(_chunk_emg))
                _imax = int(np.argmax(_chunk_emg))
                if _imin <= _imax:
                    _t_ds.extend([_chunk_t[_imin], _chunk_t[_imax]])
                    _emg_ds.extend([_chunk_emg[_imin], _chunk_emg[_imax]])
                else:
                    _t_ds.extend([_chunk_t[_imax], _chunk_t[_imin]])
                    _emg_ds.extend([_chunk_emg[_imax], _chunk_emg[_imin]])
            t_plot   = np.array(_t_ds)
            emg_plot = np.array(_emg_ds)
        else:
            t_plot, emg_plot = t, emg
        ax.plot(t_plot, emg_plot, lw=0.4, color="0.3")

        palette = plt.get_cmap("tab10").colors
        col_for = {k: palette[i % len(palette)]
                for i, k in enumerate(sorted(stim_dict))}

        y_min, y_max = emg.min(), emg.max()
        pad = 0.05 * (y_max - y_min) or 1
        ax.set_ylim(y_min, y_max + 3 * pad)

        for s_type, times in stim_dict.items():
            col = col_for[s_type]
            for x in times:
                ax.vlines(x, y_max + 0.2 * pad, y_max + 1.0 * pad,
                        color=col, lw=1.2, zorder=4)
                ax.text(x, y_max + 1.2 * pad, s_type,
                        ha="center", va="bottom",
                        fontsize=12, weight="bold",
                        color=col, zorder=5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self._ylab())
        canvas.draw_idle()

        # ── State holders ──────────────────────────────────────────────────────────
        spans: list[tuple[float, float]] = []
        patches = []

        def _update_list_label():
            if spans:
                txt = "Selected ranges (s):  " + ",  ".join(
                    f"[{s[0]:.2f} \u2013 {s[1]:.2f}]" for s in spans)
            else:
                txt = "No ranges yet \u2013 drag on the plot."
            list_lbl.set(txt)
        _update_list_label()

        # ── SpanSelector ───────────────────────────────────────────────────────────
        def _on_span(x0, x1):
            xmin, xmax = sorted((x0, x1))
            spans.append((xmin, xmax))
            p = ax.axvspan(xmin, xmax, alpha=.25, color="tab:blue")
            patches.append(p)
            _update_list_label()
            canvas.draw_idle()

        span_sel = SpanSelector(
            ax, _on_span, "horizontal",
            useblit=True,
            props=dict(alpha=.30, facecolor="tab:blue"),
            interactive=False)

        def _undo():
            if spans:
                spans.pop()
                p = patches.pop()
                p.remove()
                _update_list_label()
                canvas.draw_idle()

        def _clear():
            spans.clear()
            for p in patches:
                p.remove()
            patches[:] = []
            _update_list_label()
            canvas.draw_idle()

        tk.Button(btn_frm, text="Undo last",   width=10, command=_undo)\
            .pack(side="left", padx=4)
        tk.Button(btn_frm, text="Clear all",   width=10, command=_clear)\
            .pack(side="left", padx=4)

        def _accept():
            if not spans:
                messagebox.showwarning("Nothing selected",
                                    "Please drag one or more ranges first.",
                                    parent=top)
                return
            # save both the legacy single‑span vars *and* the new list
            self.crop_ranges = spans.copy()
            # for backward compatibility keep the 1st span in the old vars
            self.crop_start, self.crop_end = self.crop_ranges[0]
            top.destroy()

        tk.Button(btn_frm, text="Use these ranges", width=16,
                command=_accept)\
            .pack(side="left", padx=12)
        tk.Button(btn_frm, text="Cancel", width=10,
                command=lambda: (spans.clear(), top.destroy()))\
            .pack(side="left", padx=4)

        # ── Wait until the window closes ────────────────────────────────────────
        self.root.wait_window(top)
        return bool(spans)

    def _reset_state_for_new_file(self):
        """
        Forget everything that belongs to the *previous* file/run.

        Two categories of state exist:
        ┌─────────────────────────────────────────────────────────────────┐
        │ FILE-LEVEL  (reset here)    │ SESSION-LEVEL (preserved)         │
        ├─────────────────────────────────────────────────────────────────┤
        │ segments_metadata           │ label_map / color_map             │
        │  (all marker positions)     │ gap_ms_map / reference_map        │
        │ _last_outlier_result        │ latency_map                       │
        │ crop_ranges/start/end       │ csp_types                         │
        │ raw_emg cache               │ filter settings                   │
        │ channel selection           │ derivatives_path                  │
        │ _labels_tab_confirmed       │ mmax_file / plateau_tolerance     │
        │                             │ outlier settings                  │
        └─────────────────────────────────────────────────────────────────┘
        """
        # ── 1. Clear file-level raw data caches ────────────────────────────────
        for attr in ('raw_emg', 'prev_fs', 'last_times', 'last_stim'):
            if hasattr(self, attr):
                delattr(self, attr)

        # Clear the SMR segment cache so the previous file's data is freed
        try:
            from .formats.spike2_smr import clear_cache as _smr_clear
            _smr_clear()
        except Exception:
            pass

        # ── 2. Clear ALL marker metadata ──────────────────────────────────────
        # NOTE: segments_metadata is intentionally NOT cleared here.
        # _load_file_entry calls _apply_loaded_session which restores it from
        # the session JSON. For truly new files (not from queue), browse_file
        # clears it explicitly below.
        self._last_outlier_result = None

        # ── 3. Clear file-specific selections ──────────────────────────────────
        self.marker_choice.set('')   # force new marker scan on next file
        self.crop_start   = None
        self.crop_end     = None
        self.crop_ranges  = None
        self.extra_channel_indices = []

        # ── 4. Reset GUI widgets ───────────────────────────────────────────────
        self.progress.set(0)
        self.log_box.delete('1.0', tk.END)
        # Tab 1b must be re-confirmed for each new file (stim types may differ)
        self._labels_tab_confirmed = False
        # Reset channel dropdown — repopulated after file scan
        try:
            self.channel_dd["values"] = []
            self.channel_dd["state"]  = "disabled"
            self.channel_var.set("—")
        except Exception:
            pass

        # ── 5. Close any still-open matplotlib figures (saves RAM) ─────────────
        def _deferred_close():
            import matplotlib.pyplot as _plt
            _plt.close('all')
        self.root.after(100, _deferred_close)


    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 – Group Analysis
    # ══════════════════════════════════════════════════════════════════════════

    def _open_preferences(self):
        """Open the preferences dialog."""
        from .preferences import open_preferences_dialog
        open_preferences_dialog(self.root, on_apply=lambda r: None)

    def _show_about(self):
        """Show About dialog."""
        from .bids import TOOL_VERSION
        win = tk.Toplevel(self.root)
        win.title("About MEP-CMAP Analyser")
        win.resizable(False, False)
        win.transient(self.root)
        tk.Label(win, text="MEP-CMAP Analyser",
                 font=("TkDefaultFont", 13, "bold")).pack(pady=(16,2))
        tk.Label(win, text=f"Version {TOOL_VERSION}").pack()
        tk.Label(win, text="Author: Justin Andrushko PhD\nNorthumbria University",
                 justify="center", fg="grey").pack(pady=(6,4))
        tk.Label(win,
            text="BIDS-compliant TMS/EMG neurophysiology\n"
                 "analysis tool for MEP and cSP quantification.",
            justify="center").pack(padx=20, pady=(0,8))
        tk.Button(win, text="Close", width=10, command=win.destroy).pack(pady=(0,14))
        win.update_idletasks()
        _cx = self.root.winfo_rootx() + (self.root.winfo_width()  - win.winfo_width())  // 2
        _cy = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{_cx}+{_cy}")
        win.grab_set()

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1c — External normalisation (optional)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_normalisation_tab(self):
        """Build the Stage 1c normalisation tab."""
        f = self.tab1c_frame
        for w in f.winfo_children():
            w.destroy()

        tk.Label(f,
            text="Optional: normalise processed results using a reference file's PTP values.\n"
                 "Both files must be fully processed first (Stage 1a/1b).\n"
                 "Select the _trials.csv files from the derivatives/results folder.\n"
                 "The reference mean is computed using the same plateau detection as "
                 "internal normalisation: if the reference data has a reliable plateau "
                 "within the tolerance threshold, the plateau mean is used; "
                 "otherwise the peak value is used.\n"
                 "Results are written back into the main file's _trials.csv in-place.",
            justify="left", wraplength=950, fg="grey",
            font=("TkDefaultFont", 9, "italic")
        ).pack(anchor="w", padx=16, pady=(12, 6))

        # ── Plateau tolerance ─────────────────────────────────────────────────
        tol_row = tk.Frame(f)
        tol_row.pack(anchor="w", padx=16, pady=(0, 8))
        tk.Label(tol_row, text="Plateau tolerance (%):").pack(side="left")
        self._norm1c_plateau = tk.DoubleVar(value=self.plateau_tolerance.get())
        tk.Spinbox(tol_row, from_=1, to=50, increment=1, width=5,
                   textvariable=self._norm1c_plateau).pack(side="left", padx=6)
        tk.Label(tol_row,
            text="Trials within this % of the peak are averaged to form the "
                 "plateau mean. If fewer than 2 trials qualify, the peak is used.",
            fg="grey", font=("TkDefaultFont", 8)
        ).pack(side="left", padx=(4, 0))

        # ── Column headers ────────────────────────────────────────────────────
        hdr = tk.Frame(f)
        hdr.pack(fill="x", padx=16, pady=(4, 0))
        tk.Label(hdr, text="File to normalise  (_trials.csv)",
                 font=("TkDefaultFont", 9, "bold"), width=52, anchor="w")\
            .grid(row=0, column=0, padx=4, sticky="w")
        tk.Label(hdr, text="Reference file  (_trials.csv)",
                 font=("TkDefaultFont", 9, "bold"), width=52, anchor="w")\
            .grid(row=0, column=2, padx=4, sticky="w")

        # ── Pairing table ─────────────────────────────────────────────────────
        self._norm_pairs = []
        self._norm_pair_frame = tk.Frame(f)
        self._norm_pair_frame.pack(fill="x", padx=16, pady=4)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(f)
        btn_row.pack(anchor="w", padx=16, pady=8)
        tk.Button(btn_row, text="+ Add normalisation pair",
                  command=self._norm_add_pair).pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="▶ Apply normalisation",
                  bg="#5cb85c", fg="white",
                  command=self._norm_apply_all).pack(side="left")

        self._norm_log_var = tk.StringVar(value="")
        tk.Label(f, textvariable=self._norm_log_var,
                 fg="grey", justify="left")\
            .pack(anchor="w", padx=16, pady=4)

    def _norm_add_pair(self):
        """Add a new main→reference file pair row."""
        from tkinter import filedialog as _fd
        row = len(self._norm_pairs) + 1

        main_var = tk.StringVar()
        ref_var  = tk.StringVar()
        self._norm_pairs.append((main_var, ref_var))

        pf = self._norm_pair_frame

        tk.Entry(pf, textvariable=main_var, width=48, state="readonly")\
            .grid(row=row, column=0, padx=4, pady=2, sticky="w")
        tk.Button(pf, text="…", width=2,
                  command=lambda v=main_var: v.set(
                      _fd.askopenfilename(
                          title="Select trials CSV to normalise",
                          filetypes=[("CSV files", "*_trials.csv"),
                                     ("All CSV", "*.csv")]) or v.get()))\
            .grid(row=row, column=1, padx=2)

        tk.Entry(pf, textvariable=ref_var, width=48, state="readonly")\
            .grid(row=row, column=2, padx=4, pady=2, sticky="w")
        tk.Button(pf, text="…", width=2,
                  command=lambda v=ref_var: v.set(
                      _fd.askopenfilename(
                          title="Select reference trials CSV",
                          filetypes=[("CSV files", "*_trials.csv"),
                                     ("All CSV", "*.csv")]) or v.get()))\
            .grid(row=row, column=3, padx=2)

        tk.Button(pf, text="✕", fg="red", width=2,
                  command=lambda r=row, p=(main_var, ref_var):
                      self._norm_remove_pair(r, p))\
            .grid(row=row, column=4, padx=4)

    def _norm_remove_pair(self, row, pair):
        if pair in self._norm_pairs:
            self._norm_pairs.remove(pair)
        for w in self._norm_pair_frame.grid_slaves(row=row):
            w.destroy()

    def _norm_apply_all(self):
        """Apply normalisation for all configured pairs."""
        import pandas as _pd
        import numpy as _np

        results = []
        for main_path, ref_path in [(m.get(), r.get())
                                     for m, r in self._norm_pairs]:
            if not main_path or not ref_path:
                results.append("⚠️  Skipped — missing file path")
                continue
            if not os.path.isfile(main_path):
                results.append(f"⚠️  Not found: {os.path.basename(main_path)}")
                continue
            if not os.path.isfile(ref_path):
                results.append(f"⚠️  Not found: {os.path.basename(ref_path)}")
                continue
            try:
                msg = self._apply_normalisation_pair(main_path, ref_path)
                results.append(msg)
            except Exception as e:
                results.append(f"❌ {os.path.basename(main_path)}: {e}")

        self._norm_log_var.set("\n".join(results))

    def _apply_normalisation_pair(self, main_csv: str, ref_csv: str) -> str:
        """Apply normalisation from ref_csv to main_csv using the same
        plateau-detection logic as internal normalisation.
        Updates Normalised_PTP, Reference_Type, Reference_Mean(mV),
        Reference_N in the main CSV in-place.
        """
        import pandas as _pd
        import numpy as _np

        df_main = _pd.read_csv(main_csv)
        df_ref  = _pd.read_csv(ref_csv)

        # Get clean reference PTPs (exclude outliers)
        _ref_ptps = _pd.to_numeric(
            df_ref.loc[df_ref["Outlier_Decision"] != "Outlier", "PTP(mV)"],
            errors='coerce').dropna().tolist()

        if not _ref_ptps:
            return (f"⚠️  {os.path.basename(ref_csv)}: "
                    f"no clean trials found in reference file")

        # Run the same plateau detection as internal normalisation
        from .normalisation import compute_mmax as _cmmax
        plateau_tol = self._norm1c_plateau.get() / 100.0
        _result = _cmmax(_ref_ptps, plateau_tolerance=plateau_tol)
        ref_mean = _result["mmax"]
        ref_n    = _result["n_plateau"]
        ref_type = _result["method"]

        if ref_mean is None or ref_mean <= 0:
            return (f"⚠️  {os.path.basename(ref_csv)}: "
                    f"could not compute reference mean")

        # Apply to all clean trials in main file
        _col_idx = {c: i for i, c in enumerate(df_main.columns)}
        _ptp_col = "PTP(mV)"
        _norm_col = "Normalised_PTP"
        _rtype_col = "Reference_Type"
        _rmean_col = "Reference_Mean(mV)"
        _rn_col    = "Reference_N"

        for col in [_norm_col, _rtype_col, _rmean_col, _rn_col]:
            if col not in df_main.columns:
                df_main[col] = ""
            df_main[col] = df_main[col].astype(object)

        mask = df_main["Outlier_Decision"] != "Outlier"
        ptps = _pd.to_numeric(df_main.loc[mask, _ptp_col], errors='coerce')
        df_main.loc[mask, _norm_col]  = (ptps / ref_mean).round(4)
        df_main.loc[mask, _rtype_col] = ref_type
        df_main.loc[mask, _rmean_col] = round(ref_mean, 4)
        df_main.loc[mask, _rn_col]    = ref_n

        df_main.to_csv(main_csv, index=False)

        # Also update the _trials_with_outliers.csv if it exists
        _with_out = main_csv.replace("_trials.csv", "_trials_with_outliers.csv")
        if os.path.isfile(_with_out):
            df_all = _pd.read_csv(_with_out)
            for col in [_norm_col, _rtype_col, _rmean_col, _rn_col]:
                if col not in df_all.columns:
                    df_all[col] = ""
                df_all[col] = df_all[col].astype(object)
            _mask_all = df_all["Outlier_Decision"] != "Outlier"
            _ptps_all = _pd.to_numeric(
                df_all.loc[_mask_all, _ptp_col], errors='coerce')
            df_all.loc[_mask_all, _norm_col]  = (_ptps_all / ref_mean).round(4)
            df_all.loc[_mask_all, _rtype_col] = ref_type
            df_all.loc[_mask_all, _rmean_col] = round(ref_mean, 4)
            df_all.loc[_mask_all, _rn_col]    = ref_n
            df_all.to_csv(_with_out, index=False)

        # ── Update summary files ──────────────────────────────────────────────
        # Recompute Mean/SD Normalised_PTP from the updated trials data,
        # grouped by StimType — same logic as pipeline_write_outputs.
        def _update_summary(summary_csv, trials_df):
            if not os.path.isfile(summary_csv):
                return
            df_sum = _pd.read_csv(summary_csv)
            # Ensure columns exist
            for col in ["Mean_Normalised_PTP", "SD_Normalised_PTP",
                        "Reference_Type", "Reference_Mean(mV)", "Reference_N"]:
                if col not in df_sum.columns:
                    df_sum[col] = _np.nan
                df_sum[col] = df_sum[col].astype(object)

            clean = trials_df[trials_df["Outlier_Decision"] != "Outlier"].copy()
            clean[_norm_col] = _pd.to_numeric(clean[_norm_col], errors='coerce')

            for idx, row in df_sum.iterrows():
                st = row.get("StimType", row.get("Stim_Type", ""))
                grp = clean[clean["StimType"] == st][_norm_col].dropna()
                if len(grp) > 0:
                    df_sum.at[idx, "Mean_Normalised_PTP"] = round(float(grp.mean()), 4)
                    df_sum.at[idx, "SD_Normalised_PTP"]   = round(float(grp.std(ddof=1)), 4) \
                                                             if len(grp) > 1 else _np.nan
                    df_sum.at[idx, "Reference_Type"]      = ref_type
                    df_sum.at[idx, "Reference_Mean(mV)"]  = round(ref_mean, 4)
                    df_sum.at[idx, "Reference_N"]         = ref_n
            df_sum.to_csv(summary_csv, index=False)

        _summary_csv      = main_csv.replace("_trials.csv", "_summary.csv")
        _summary_with_out = main_csv.replace("_trials.csv", "_summary_with_outliers.csv")

        # Use the updated trials data for summary recomputation
        _update_summary(_summary_csv, df_main)
        if os.path.isfile(_with_out):
            _update_summary(_summary_with_out,
                            _pd.read_csv(_with_out))

        return (f"✅ {os.path.basename(main_csv)}: "
                f"normalised to {ref_type} = {ref_mean:.4f} mV "
                f"(N={ref_n}, from {os.path.basename(ref_csv)})")

    def _browse_mmax_file(self):
        """Browse for an external M-wave reference file."""
        path = filedialog.askopenfilename(
            title="Select M-wave reference file",
            filetypes=[("Spike2 export", "*.txt")],
            parent=self.root)
        if path:
            self.mmax_file.set(path)
            self.log(f"📐 Mmax reference file: {os.path.basename(path)}")

    def _build_session_tab(self, parent: tk.Frame):
        """Build the Dataset Setup tab."""

        # ── Step 1: Dataset Setup ─────────────────────────────────────────────
        setup_frame = tk.LabelFrame(parent, text="Step 1 — Open Dataset",
                                    padx=8, pady=6)
        setup_frame.pack(fill='x', padx=10, pady=(10, 4))

        study_row = tk.Frame(setup_frame)
        study_row.pack(fill='x', pady=(0, 6))
        tk.Button(study_row, text="📂  Open study folder",
                  font=("TkDefaultFont", 9, "bold"),
                  command=self._open_study_folder).pack(side='left', padx=(0, 8))
        tk.Label(study_row,
                 text="Auto-detects rawdata/ and derivatives/ subfolders",
                 fg="grey").pack(side='left')

        ttk.Separator(setup_frame, orient='horizontal').pack(fill='x', pady=4)
        tk.Label(setup_frame, text="Or set manually:", fg="#555").pack(anchor='w')

        raw_row = tk.Frame(setup_frame)
        raw_row.pack(fill='x', pady=(4, 2))
        tk.Label(raw_row, text="Raw data folder:", width=18, anchor='w').pack(side='left')
        self._rawdata_path = tk.StringVar()
        self._raw_status_lbl = tk.Label(raw_row, text="Not set", fg="#888", width=6)
        self._raw_status_lbl.pack(side='right')
        tk.Button(raw_row, text="Browse…",
                  command=self._browse_raw_folder).pack(side='right', padx=(4, 0))
        tk.Entry(raw_row, textvariable=self._rawdata_path,
                 state="readonly", fg="#555").pack(side='left', fill='x', expand=True, padx=(4, 4))

        deriv_row2 = tk.Frame(setup_frame)
        deriv_row2.pack(fill='x', pady=(2, 4))
        tk.Label(deriv_row2, text="Derivatives folder:", width=18, anchor='w').pack(side='left')
        self._deriv_status_lbl2 = tk.Label(deriv_row2, text="Not set", fg="#888", width=6)
        self._deriv_status_lbl2.pack(side='right')
        tk.Button(deriv_row2, text="Browse…",
                  command=self.browse_derivatives_folder).pack(side='right', padx=(4, 0))
        tk.Entry(deriv_row2, textvariable=self.derivatives_path,
                 state="readonly", fg="#555").pack(side='left', fill='x', expand=True, padx=(4, 4))

        def _update_status(*_):
            self._raw_status_lbl.config(
                **({"text": "✅", "fg": "#5cb85c"} if self._rawdata_path.get()
                   else {"text": "Not set", "fg": "#888"}))
            self._deriv_status_lbl2.config(
                **({"text": "✅", "fg": "#5cb85c"} if self.derivatives_path.get()
                   else {"text": "Not set", "fg": "#888"}))
        self._rawdata_path.trace_add("write", _update_status)
        self.derivatives_path.trace_add("write", _update_status)

        # ── Step 2: File Queue ────────────────────────────────────────────────
        queue_frame = tk.LabelFrame(parent,
            text="Step 2 — File Queue  (double-click a file to load it)",
            padx=6, pady=4)
        queue_frame.pack(fill='both', expand=True, padx=10, pady=(0, 6))

        q_toolbar = tk.Frame(queue_frame)
        q_toolbar.pack(fill='x', pady=(0, 4))
        tk.Button(q_toolbar, text="+ Add file(s)",
                  command=self.browse_file).pack(side='left', padx=(0, 4))
        tk.Button(q_toolbar, text="+ Add folder",
                  command=self.browse_folder).pack(side='left', padx=(0, 4))
        tk.Button(q_toolbar, text="🔄 Refresh",
                  command=self._queue_refresh_from_raw).pack(side='left', padx=(0, 4))
        tk.Button(q_toolbar, text="💾 Save queue",
                  command=self._queue_save).pack(side='left', padx=(0, 8))
        tk.Button(q_toolbar, text="Remove selected",
                  command=self._queue_remove_selected).pack(side='left', padx=(0, 4))
        tk.Button(q_toolbar, text="▲",
                  command=self._queue_move_up, width=2).pack(side='left')
        tk.Button(q_toolbar, text="▼",
                  command=self._queue_move_down, width=2).pack(side='left', padx=(2, 0))
        tk.Button(q_toolbar, text="▶  Run all unprocessed",
                  command=self._queue_run_all,
                  bg="#5cb85c", fg="white",
                  font=("TkDefaultFont", 9, "bold")).pack(side='right', padx=(0, 4))
        tk.Button(q_toolbar, text="▶  Run selected",
                  command=self._queue_run_selected).pack(side='right', padx=(0, 4))

        # ── File-load progress bar (hidden until a file is loading) ─────────
        _pb_style = ttk.Style()
        _pb_style.configure("FileLoad.Horizontal.TProgressbar",
                            thickness=8, troughcolor="#ddd", background="#2196F3")
        self._load_prog_frame = tk.Frame(queue_frame)
        self._load_prog_label = tk.Label(
            self._load_prog_frame, text="", fg="#555",
            font=("TkDefaultFont", 8), anchor="w")
        self._load_prog_label.pack(side="left", padx=(0, 8))
        self._load_prog_bar = ttk.Progressbar(
            self._load_prog_frame,
            style="FileLoad.Horizontal.TProgressbar",
            orient="horizontal", mode="determinate", length=300)
        self._load_prog_bar.pack(side="left", fill="x", expand=True)
        self._load_prog_frame.pack(fill="x", pady=(0, 4))
        self._load_prog_frame.pack_forget()   # hidden until loading begins

        q_cols = ("status", "sub", "ses", "limb", "label", "filetype",
                  "stim_types", "last_processed", "size", "date", "path")
        tree_frame = tk.Frame(queue_frame)
        tree_frame.pack(fill='both', expand=True)
        self._queue_tree = ttk.Treeview(tree_frame, columns=q_cols,
            show="headings", height=14, selectmode="extended")

        _sort_state = {}
        def _sort_by(col):
            reverse = _sort_state.get(col, False)
            if col == "size":
                # Sort numerically using the raw byte count stored as a tag
                # value on each row, not the human-readable "1.2 MB" string.
                def _size_key(iid):
                    try:
                        return float(self._queue_tree.item(iid, "tags")[1])
                    except Exception:
                        return 0.0
                items = sorted(self._queue_tree.get_children(),
                               key=_size_key, reverse=reverse)
                for i, iid in enumerate(items):
                    self._queue_tree.move(iid, "", i)
            else:
                items = [(self._queue_tree.set(iid, col), iid)
                         for iid in self._queue_tree.get_children()]
                items.sort(reverse=reverse)
                for i, (_, iid) in enumerate(items):
                    self._queue_tree.move(iid, "", i)
            _sort_state[col] = not reverse
            arrow = " ▲" if not reverse else " ▼"
            for c in q_cols:
                self._queue_tree.heading(c,
                    text=self._queue_tree.heading(c)["text"].rstrip(" ▲▼"))
            self._queue_tree.heading(col,
                text=self._queue_tree.heading(col)["text"] + arrow)

        for col, text, width in [
            ("status",         "Status",        120),
            ("sub",            "Subject",         80),
            ("ses",            "Session",         60),
            ("limb",           "Limb",            70),
            ("label",          "File",           260),
            ("filetype",       "Type",            55),
            ("stim_types",     "Stim types",     150),
            ("last_processed", "Last processed", 130),
            ("size",           "Size",            70),
            ("date",           "Modified",       130),
            ("path",           "Path",           500),
        ]:
            self._queue_tree.heading(col, text=text, command=lambda c=col: _sort_by(c))
            self._queue_tree.column(col, width=width, stretch=False, minwidth=30)
        self._queue_tree.column("path", width=500, stretch=True, minwidth=200)

        q_vs = ttk.Scrollbar(tree_frame, orient="vertical",   command=self._queue_tree.yview)
        q_hs = ttk.Scrollbar(tree_frame, orient="horizontal",  command=self._queue_tree.xview)
        self._queue_tree.configure(yscrollcommand=q_vs.set, xscrollcommand=q_hs.set)
        self._queue_tree.grid(row=0, column=0, sticky="nsew")
        q_vs.grid(row=0, column=1, sticky="ns")
        q_hs.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        for status, colour in {
            "not_started": "#888888", "in_progress": "#f0a500",
            "needs_review": "#d9534f", "complete": "#5cb85c",
            "stale": "#8b6914", "skipped": "#aaaaaa",
        }.items():
            self._queue_tree.tag_configure(status, foreground=colour)

        self._queue_tree.bind("<Double-1>", self._queue_on_double_click)

        # Right-click context menu
        _ctx = tk.Menu(self._queue_tree, tearoff=0)
        _ctx.add_command(label="Load & process", command=lambda: self._queue_on_double_click(None))
        _ctx.add_command(label="Mark as rerun", command=self._queue_mark_rerun)
        _ctx.add_separator()
        _ctx.add_command(label="✏️  Rename / audit filename…", command=self._queue_rename_selected)
        _ctx.add_separator()
        _ctx.add_command(label="Remove selected", command=self._queue_remove_selected)
        _ctx.add_separator()
        _ctx.add_command(label="Show excluded files…", command=self._queue_show_excluded)

        def _show_ctx(event):
            iid = self._queue_tree.identify_row(event.y)
            if iid:
                self._queue_tree.selection_set(iid)
            try:
                _ctx.tk_popup(event.x_root, event.y_root)
            finally:
                _ctx.grab_release()
        self._queue_tree.bind("<Button-3>", _show_ctx)

        self._queue_progress_var = tk.StringVar(value="No files loaded")
        tk.Label(parent, textvariable=self._queue_progress_var,
                 fg="grey", anchor="w").pack(fill='x', padx=10, pady=(0, 4))

    def _get_or_create_dataset(self) -> DatasetSession:
        """Return current dataset session, creating one if needed."""
        if self._dataset is None:
            deriv = self.derivatives_path.get()
            root  = deriv if deriv else os.path.expanduser("~")
            self._dataset = DatasetSession.load_or_create(root)
        return self._dataset

    def _queue_refresh(self):
        """Redraw the queue treeview from current dataset state."""
        if not hasattr(self, '_queue_tree'):
            return
        tree = self._queue_tree
        tree.delete(*tree.get_children())

        ds = self._dataset
        if ds is None or not ds.files:
            self._queue_progress_var.set("No files loaded")
            return

        for fe in ds.files:
            status_label = STATUS_LABELS.get(fe.status, fe.status)
            stim_str     = ", ".join(
                f"{v}({k})" for k, v in fe.stim_label_map.items()
            ) if fe.stim_label_map else (
                ", ".join(fe.stim_letters) if fe.stim_letters else "—"
            )
            last = fe.last_processed[:16].replace("T", " ") if fe.last_processed else "—"

            # Parse BIDS fields from path
            bn = os.path.basename(fe.path)
            import re as _re
            _sub  = next((_re.sub(r'^sub-','',p) for p in bn.split('_') if p.startswith('sub-')), "—")
            _ses  = next((_re.sub(r'^ses-','',p) for p in bn.split('_') if p.startswith('ses-')), "—")
            _limb = next((p.split('-',1)[1] for p in bn.split('_') if p.startswith('limb-')), "—")

            # File size, modification date, and format type
            try:
                _stat  = os.stat(fe.path)
                _bytes = _stat.st_size
                if _bytes >= 1_073_741_824:
                    _size = f"{_bytes/1_073_741_824:.1f} GB"
                elif _bytes >= 1_048_576:
                    _size = f"{_bytes/1_048_576:.1f} MB"
                else:
                    _size = f"{_bytes/1024:.0f} KB"
                from datetime import datetime as _dt
                _date = _dt.fromtimestamp(_stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                _bytes = 0
                _size  = "—"
                _date  = "—"

            _ext = os.path.splitext(fe.path)[1].lower()
            _ftype = {".txt": "TXT", ".smr": "SMR", ".adibin": "ADIBIN"}.get(_ext, _ext.lstrip(".").upper() or "—")

            # Tags: (status_tag, raw_bytes_str) — raw bytes used for numeric size sort
            tree.insert("", "end", iid=fe.id,
                        values=(status_label, _sub, _ses, _limb,
                                fe.label or fe.basename,
                                _ftype,
                                stim_str, last, _size, _date, fe.path),
                        tags=(fe.status, str(_bytes)))

        n_done  = ds.n_complete
        n_total = ds.n_total
        self._queue_progress_var.set(
            f"{n_done} / {n_total} files complete"
            + (" — ✅ All done!" if ds.all_complete else ""))

    def _queue_selected_ids(self) -> list:
        return list(self._queue_tree.selection())

    def _queue_selected_id(self) -> str | None:
        sel = self._queue_selected_ids()
        return sel[0] if sel else None

    def _queue_remove_selected(self):
        ids = self._queue_selected_ids()
        if not ids or self._dataset is None:
            return
        for fid in ids:
            fe = self._dataset.get_file(fid)
            if fe:
                # Remember this path was explicitly excluded so refresh doesn't re-add it
                if not hasattr(self._dataset, 'excluded_paths'):
                    self._dataset.excluded_paths = set()
                self._dataset.excluded_paths.add(os.path.normpath(fe.path))
            self._dataset.remove_file(fid)
        self._dataset.save()
        self._queue_refresh()

    def _queue_move_up(self):
        fid = self._queue_selected_id()
        if fid and self._dataset:
            self._dataset.move_up(fid)
            self._queue_refresh()
            self._queue_tree.selection_set(fid)

    def _queue_move_down(self):
        fid = self._queue_selected_id()
        if fid and self._dataset:
            self._dataset.move_down(fid)
            self._queue_refresh()
            self._queue_tree.selection_set(fid)

    def _queue_on_double_click(self, event):
        """Load the double-clicked file and switch to Labels tab."""
        fid = self._queue_selected_id()
        if not fid or self._dataset is None:
            return
        fe = self._dataset.get_file(fid)
        if fe:
            self._load_file_entry(fe)

    def _queue_show_excluded(self):
        """Show excluded files and allow user to re-include any of them."""
        if self._dataset is None:
            messagebox.showinfo("No dataset", "No dataset loaded.", parent=self.root)
            return
        excluded = getattr(self._dataset, 'excluded_paths', set())
        if not excluded:
            messagebox.showinfo("No excluded files",
                "No files have been excluded from this dataset.",
                parent=self.root)
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Excluded files")
        dlg.transient(self.root)
        dlg.resizable(True, True)
        dlg.grab_set()

        tk.Label(dlg,
                 text="These files were previously removed from the queue.\n"
                      "Tick any you want to re-include, then click Restore.",
                 padx=12, pady=8, justify="left").pack(anchor="w")

        # Scrollable frame for checkboxes
        frame_outer = tk.Frame(dlg)
        frame_outer.pack(fill="both", expand=True, padx=12, pady=4)
        canvas = tk.Canvas(frame_outer, height=300)
        vsb = ttk.Scrollbar(frame_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        vars_ = {}
        for path in sorted(excluded):
            v = tk.BooleanVar(value=False)
            vars_[path] = v
            row = tk.Frame(inner)
            row.pack(fill="x", pady=1)
            tk.Checkbutton(row, variable=v).pack(side="left")
            tk.Label(row, text=os.path.basename(path),
                     anchor="w", width=40).pack(side="left")
            tk.Label(row, text=path, fg="grey",
                     anchor="w").pack(side="left", padx=(4, 0))

        def _restore():
            to_restore = [p for p, v in vars_.items() if v.get()]
            if not to_restore:
                messagebox.showinfo("Nothing selected",
                    "Tick at least one file to restore.", parent=dlg)
                return
            for path in to_restore:
                self._dataset.excluded_paths.discard(path)
                # Re-add to queue
                if self._dataset.get_by_path(path) is None:
                    label = self._dataset.label_from_bids(path)
                    self._dataset.add_file(path, label=label)
            self._dataset.save()
            self._queue_refresh()
            self.log(f"↩️  Restored {len(to_restore)} file(s) to the queue")
            dlg.destroy()

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=8)
        tk.Button(btn_row, text="Restore selected", command=_restore,
                  bg="#5cb85c", fg="white").pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", command=dlg.destroy,
                  width=10).pack(side="left", padx=6)

        # Centre over main window
        self.root.update_idletasks()
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width()  - dlg.winfo_width())  // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _queue_mark_rerun(self):
        """Reset selected complete files to not_started so they can be reprocessed."""
        ids = self._queue_selected_ids()
        if not ids or self._dataset is None:
            return
        for fid in ids:
            fe = self._dataset.get_file(fid)
            if fe and fe.status == STATUS_COMPLETE:
                fe.status = STATUS_NOT_STARTED
        self._dataset.save()
        self._queue_refresh()

    # ── Filename rename / BIDS audit ──────────────────────────────────────────

    # Expected BIDS filename entity pattern:
    #   sub-<label>[_ses-<label>][_limb-<label>][_task-<label>][_run-<index>]
    #   followed by an optional suffix, ending in a supported extension
    _BIDS_ENTITIES = re.compile(
        r'^'
        r'(?P<sub>sub-[A-Za-z0-9]+)'
        r'(?:_(?P<ses>ses-[A-Za-z0-9]+))?'
        r'(?:_(?P<limb>limb-[A-Za-z0-9]+))?'
        r'(?:_(?P<task>task-[A-Za-z0-9]+))?'
        r'(?:_(?P<run>run-[0-9]+))?'
        r'(?:_(?P<suffix>[^.]+))?'
        r'\.(txt|smr|adibin)$',
        re.IGNORECASE,
    )

    _SUPPORTED_EXTENSIONS = {".txt", ".smr", ".adibin"}

    def _audit_filename(self, basename: str) -> list:
        """Return a list of human-readable issue strings for *basename*.
        Empty list means no issues found.
        """
        issues = []
        name, ext = os.path.splitext(basename)

        if ext.lower() not in self._SUPPORTED_EXTENSIONS:
            issues.append(f"Extension '{ext}' — expected .txt, .smr, or .adibin")

        parts = name.split("_")

        # sub- entity
        sub_parts = [p for p in parts if p.startswith("sub-")]
        if not sub_parts:
            issues.append("Missing 'sub-<label>' entity  (e.g. sub-001)")
        elif len(sub_parts) > 1:
            issues.append(f"Duplicate 'sub-' entity: {sub_parts}")
        else:
            lbl = sub_parts[0][4:]
            if not lbl:
                issues.append("Empty sub- label")
            if not re.match(r'^[A-Za-z0-9]+$', lbl):
                issues.append(f"sub- label '{lbl}' contains non-alphanumeric characters")

        # ses- entity (optional but check if malformed)
        ses_parts = [p for p in parts if p.startswith("ses-")]
        if len(ses_parts) > 1:
            issues.append(f"Duplicate 'ses-' entity: {ses_parts}")
        elif ses_parts:
            lbl = ses_parts[0][4:]
            if not re.match(r'^[A-Za-z0-9]+$', lbl):
                issues.append(f"ses- label '{lbl}' contains non-alphanumeric characters")

        # limb- entity (optional)
        limb_parts = [p for p in parts if p.startswith("limb-")]
        if len(limb_parts) > 1:
            issues.append(f"Duplicate 'limb-' entity: {limb_parts}")

        # spaces / special characters
        if " " in name:
            issues.append("Filename contains spaces (use hyphens or underscores)")
        for ch in r'\/:*?"<>|':
            if ch in name:
                issues.append(f"Filename contains forbidden character '{ch}'")

        # Inconsistent capitalisation of known entities
        for entity in ("Sub-", "SES-", "Ses-", "LIMB-", "Limb-",
                       "TASK-", "Task-", "RUN-", "Run-"):
            if entity in basename:
                issues.append(
                    f"Entity '{entity}' should be lower-case  "
                    f"(e.g. '{entity.lower()}')")

        # Trailing / leading underscores
        if name.startswith("_") or name.endswith("_"):
            issues.append("Filename starts or ends with an underscore")

        # Double underscores
        if "__" in name:
            issues.append("Filename contains consecutive underscores '__'")

        return issues

    def _queue_rename_selected(self):
        """Open the rename / BIDS-audit dialog for the selected file.

        Shows any BIDS naming issues, lets the user type a corrected name with
        a live preview, then renames the file on disk and updates the queue.
        """
        fid = self._queue_selected_id()
        if not fid or self._dataset is None:
            return
        fe = self._dataset.get_file(fid)
        if fe is None:
            return

        old_path = fe.path
        old_name = fe.basename
        issues   = self._audit_filename(old_name)

        # ── Build dialog ──────────────────────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title("Rename / audit filename")
        win.transient(self.root)
        win.grab_set()
        win.resizable(True, False)
        win.minsize(680, 10)

        # Current path
        tk.Label(win, text="Current path:", font=("TkDefaultFont", 9, "bold"),
                 anchor="w").grid(row=0, column=0, sticky="w", padx=10, pady=(12, 2))
        tk.Label(win, text=old_path, fg="#555", wraplength=640, justify="left",
                 anchor="w").grid(row=1, column=0, columnspan=2, sticky="w",
                                  padx=10, pady=(0, 8))

        ttk.Separator(win, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=4)

        # Audit results
        tk.Label(win, text="BIDS naming audit:",
                 font=("TkDefaultFont", 9, "bold"), anchor="w"
                 ).grid(row=3, column=0, sticky="w", padx=10, pady=(4, 2))

        audit_frm = tk.Frame(win, bd=1, relief="sunken", bg="#fffde7")
        audit_frm.grid(row=4, column=0, columnspan=2, sticky="ew",
                       padx=10, pady=(0, 8))
        if issues:
            for issue in issues:
                tk.Label(audit_frm, text=f"  \u26a0  {issue}",
                         fg="#b26a00", bg="#fffde7",
                         font=("TkDefaultFont", 9), anchor="w"
                         ).pack(fill="x", padx=4, pady=1)
        else:
            tk.Label(audit_frm,
                     text="  \u2705  No issues found \u2014 filename looks BIDS-compliant",
                     fg="#2e7d32", bg="#fffde7",
                     font=("TkDefaultFont", 9), anchor="w"
                     ).pack(fill="x", padx=4, pady=4)

        ttk.Separator(win, orient="horizontal").grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=4)

        # New name entry
        tk.Label(win, text="New filename:", font=("TkDefaultFont", 9, "bold"),
                 anchor="w").grid(row=6, column=0, sticky="w", padx=10, pady=(4, 2))

        name_var = tk.StringVar(value=old_name)
        entry = ttk.Entry(win, textvariable=name_var, width=60)
        entry.grid(row=7, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4))
        entry.focus_set()
        entry.select_range(0, len(os.path.splitext(old_name)[0]))

        # Live preview with re-audit
        preview_var = tk.StringVar()
        def _update_preview(*_):
            nn = name_var.get().strip()
            new_path = os.path.join(os.path.dirname(old_path), nn)
            live_issues = self._audit_filename(nn)
            if live_issues:
                col  = "#b26a00"
                text = "\u26a0  " + "   |   ".join(live_issues[:3])
            else:
                col  = "#2e7d32"
                text = "\u2705  Looks good \u2192 " + new_path
            preview_var.set(text)
            preview_lbl.config(fg=col)
        name_var.trace_add("write", _update_preview)

        preview_lbl = tk.Label(win, textvariable=preview_var,
                               wraplength=640, justify="left",
                               font=("TkDefaultFont", 8), anchor="w")
        preview_lbl.grid(row=8, column=0, columnspan=2, sticky="w",
                         padx=10, pady=(0, 8))
        _update_preview()

        # BIDS template hint
        tk.Label(win,
                 text="Suggested pattern:  sub-<label>_ses-<label>_limb-<left|right>_<date>.txt",
                 fg="#888", font=("TkDefaultFont", 8)
                 ).grid(row=9, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        ttk.Separator(win, orient="horizontal").grid(
            row=10, column=0, columnspan=2, sticky="ew", padx=10, pady=4)

        warn_lbl = tk.Label(win, text="", fg="#d9534f",
                            font=("TkDefaultFont", 9))
        warn_lbl.grid(row=11, column=0, columnspan=2, sticky="w", padx=10)

        def _do_rename(_e=None):
            new_name = name_var.get().strip()
            if not new_name:
                warn_lbl.config(text="Name cannot be empty.")
                return
            if new_name == old_name:
                win.destroy()
                return
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            if os.path.exists(new_path):
                warn_lbl.config(
                    text=f"A file named '{new_name}' already exists in that folder.")
                return
            if not os.path.isfile(old_path):
                warn_lbl.config(
                    text="Original file not found on disk \u2014 cannot rename.")
                return
            try:
                os.rename(old_path, new_path)
            except OSError as exc:
                warn_lbl.config(text=f"Rename failed: {exc}")
                return
            # Update FileEntry
            fe.path  = new_path
            fe.label = new_name
            # Update derivatives_json if it embeds the old stem
            if fe.derivatives_json:
                old_stem = os.path.splitext(old_name)[0]
                new_stem = os.path.splitext(new_name)[0]
                fe.derivatives_json = fe.derivatives_json.replace(
                    old_stem, new_stem)
            # Update excluded_paths if old path was tracked there
            if self._dataset:
                if old_path in self._dataset.excluded_paths:
                    self._dataset.excluded_paths.discard(old_path)
                    self._dataset.excluded_paths.add(new_path)
                self._dataset.save()
            # Keep active file path in sync
            if self.file_path.get() == old_path:
                self.file_path.set(new_path)
            self._queue_refresh()
            self._log_gui(f"\u270f\ufe0f  Renamed:  {old_name}  \u2192  {new_name}")
            win.destroy()

        btn_bar = tk.Frame(win)
        btn_bar.grid(row=12, column=0, columnspan=2, pady=(4, 12))
        tk.Button(btn_bar, text="\u270f\ufe0f  Rename file",
                  bg="#2196F3", fg="white", width=16,
                  command=_do_rename).pack(side="left", padx=8)
        tk.Button(btn_bar, text="Cancel", width=10,
                  command=win.destroy).pack(side="left", padx=4)
        entry.bind("<Return>", _do_rename)

        win.columnconfigure(0, weight=1)
        win.update_idletasks()
        # Centre over main window
        cx = self.root.winfo_rootx() + (self.root.winfo_width()  - win.winfo_width())  // 2
        cy = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{cx}+{cy}")

    def _queue_save(self):
        """Explicitly save the current queue state to mep_cmap_dataset.json."""
        if self._dataset is None:
            messagebox.showinfo("No dataset",
                "No dataset loaded — add files first.", parent=self.root)
            return
        if self._dataset.save():
            self.log(f"💾 Queue saved → {self._dataset.json_path}")
        else:
            messagebox.showerror("Save failed",
                "Could not save the queue. Check that the derivatives folder is set.",
                parent=self.root)

    def _queue_run_all(self):
        """Process all unprocessed files sequentially."""
        if self._dataset is None:
            messagebox.showwarning("No dataset",
                "Add files to the queue first.", parent=self.root)
            return
        nxt = self._dataset.next_unprocessed()
        if nxt is None:
            # All complete — offer to rerun from start
            resp = messagebox.askyesno(
                "All done",
                "All files in the queue have been processed.\n\n"
                "Would you like to rerun from the beginning?",
                parent=self.root)
            if resp:
                self._load_file_entry(self._dataset.files[0], auto_run=True)
            return
        self._load_file_entry(nxt, auto_run=True)

    def _queue_run_selected(self):
        """Process only the selected files."""
        ids = self._queue_selected_ids()
        if not ids or self._dataset is None:
            return
        entries = [self._dataset.get_file(fid) for fid in ids
                   if self._dataset.get_file(fid) is not None]
        unprocessed = [fe for fe in entries
                       if fe.status not in ("complete", "skipped")]

        if not unprocessed:
            # All selected files are complete — offer to rerun them
            resp = messagebox.askyesno(
                "Already done",
                f"All {len(entries)} selected file(s) are already complete.\n\n"
                "Would you like to rerun them anyway?\n"
                "(Useful for changing settings or reviewing results)",
                parent=self.root)
            if resp:
                # Reset status to allow reprocessing
                for fe in entries:
                    fe.status = STATUS_NOT_STARTED
                self._dataset.save()
                self._queue_refresh()
                self._load_file_entry(entries[0], auto_run=True)
            return
        self._load_file_entry(unprocessed[0], auto_run=True)

    def _load_file_entry(self, fe: FileEntry, auto_run: bool = False):
        """Load a FileEntry into the Stage 1 processing pipeline."""
        self._reset_state_for_new_file()
        self.segments_metadata = {}   # clear before restore
        self.file_path.set(fe.path)
        self._current_file_entry = fe
        self.log(f"📄 Loading: {fe.basename}")

        # Restore per-file session if available
        if fe.derivatives_json and os.path.isfile(fe.derivatives_json):
            try:
                import json as _json
                with open(fe.derivatives_json, encoding="utf-8") as fh:
                    sess = _json.load(fh)
                self._apply_loaded_session(sess, json_path=fe.derivatives_json)
                self.log(f"💾 Restored session — {len(self.segments_metadata)} segment(s) with saved edits")
            except Exception as e:
                self.log(f"⚠️  Could not restore session: {e}")

        # Update status
        if fe.status == STATUS_NOT_STARTED:
            fe.mark_in_progress()
            if self._dataset:
                self._dataset.save()
            self._queue_refresh()

        # ── Show load progress bar above the file tree ────────────────────────
        try:
            _bytes = os.path.getsize(fe.path)
            _size_str = (f"{_bytes/1_048_576:.1f} MB" if _bytes >= 1_048_576
                         else f"{_bytes/1024:.0f} KB")
        except OSError:
            _size_str = ""

        self._load_prog_bar["value"] = 0
        self._load_prog_label.config(
            text=f"Reading…  {_size_str}" if _size_str else "Reading…")
        self._load_prog_frame.pack(fill="x", pady=(0, 4),
                                   before=self._queue_tree.master)
        self.root.update_idletasks()

        # Parse on a background thread so the UI stays responsive
        _result:   list = []
        _progress: list = [5]

        def _worker():
            try:
                _progress[0] = 10
                from .io import list_waveform_channels as _lwc
                _lwc(fe.path)           # warm up; result used in _browse_file_path
                _progress[0] = 100
                _result.append(("ok",))
            except Exception as exc:
                _result.append(("err", exc))
                _progress[0] = 0

        _ready = [False]

        def _poll():
            pct = _progress[0]
            self._load_prog_bar["value"] = pct
            if pct == 100:
                self._load_prog_label.config(text=f"✅ Loaded  {_size_str}")
            if not _result:
                self.root.after(80, _poll)
                return
            _ready[0] = True

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(80, _poll)
        while not _ready[0]:
            self.root.update()

        self._load_prog_frame.pack_forget()

        if _result[0][0] == "err":
            messagebox.showerror("Load error", str(_result[0][1]), parent=self.root)
            return

        # Trigger the normal file loading flow
        self._browse_file_path(fe.path, auto_run=auto_run)

    def _open_study_folder(self):
        """Open a BIDS-style study folder — auto-detects rawdata/ and derivatives/."""
        folder = filedialog.askdirectory(title="Select study root folder")
        if not folder:
            return
        # Auto-detect rawdata/ subfolder
        raw_candidate = os.path.join(folder, "rawdata")
        if os.path.isdir(raw_candidate):
            self._rawdata_path.set(Path(raw_candidate).as_posix())
            study_root = folder
            self.log(f"📂 Raw data: {raw_candidate}")
        else:
            # No rawdata/ subfolder — treat the folder itself as raw data
            self._rawdata_path.set(Path(folder).as_posix())
            study_root = str(Path(folder).parent)
            self.log(f"📂 Raw data: {folder}")

        # Derivatives always sits beside rawdata/ at the same level
        deriv_candidate = str(Path(study_root) / "derivatives")
        self.derivatives_path.set(Path(deriv_candidate).as_posix())
        os.makedirs(deriv_candidate, exist_ok=True)
        self.log(f"📁 Derivatives: {deriv_candidate}")
        self._update_deriv_status()
        self._dataset = DatasetSession.load_or_create(deriv_candidate)
        self._queue_refresh_from_raw()

    def _browse_raw_folder(self):
        """Manually set the raw data folder — derivatives defaults to sibling folder."""
        folder = filedialog.askdirectory(title="Select raw data folder")
        if not folder:
            return
        self._rawdata_path.set(Path(folder).as_posix())
        self.log(f"📂 Raw data folder: {Path(folder).as_posix()}")

        # Default derivatives to ../derivatives (beside rawdata, not inside it)
        parent = str(Path(folder).parent)
        deriv_default = str(Path(parent) / "derivatives")

        # Only auto-set if derivatives not already configured
        if not self.derivatives_path.get():
            self.derivatives_path.set(Path(deriv_default).as_posix())
            os.makedirs(deriv_default, exist_ok=True)
            self.log(f"📁 Derivatives auto-set: {deriv_default}")
            self._update_deriv_status()
            self._dataset = DatasetSession.load_or_create(deriv_default)

        self._queue_refresh_from_raw()

    def _queue_refresh_from_raw(self):
        """Scan the raw data folder and add any new data files to the queue.
        Files previously removed by the user are not re-added."""
        raw = self._rawdata_path.get()
        if not raw:
            messagebox.showinfo("No raw data folder",
                "Set a raw data folder first using Step 1.", parent=self.root)
            return
        EXCLUDE = ("metric_definitions", "metrics_definitions",
                   "channel_info", "_readme")
        import glob as _glob
        _EXTS = ("*.txt", "*.smr", "*.adibin")
        all_files = []
        for _pat in _EXTS:
            all_files.extend(_glob.glob(os.path.join(raw, "**", _pat), recursive=True))
        files = sorted(
            f for f in all_files
            if not any(p in os.path.basename(f).lower() for p in EXCLUDE)
        )
        if not files:
            messagebox.showinfo("No files found",
                "No .txt data files found in the raw data folder.",
                parent=self.root)
            return
        ds = self._get_or_create_dataset()
        excluded = getattr(ds, 'excluded_paths', set())
        added = 0
        skipped_excluded = 0
        for fpath in files:
            norm = os.path.normpath(fpath)
            if norm in excluded:
                skipped_excluded += 1
                continue
            if ds.get_by_path(fpath) is None:
                label = ds.label_from_bids(fpath)
                ds.add_file(fpath, label=label)
                added += 1
        ds.save()
        self._queue_refresh()
        msg = f"🔄 Refreshed: {added} new file(s) added ({len(files)} found)"
        if skipped_excluded:
            msg += f", {skipped_excluded} previously excluded skipped"
        self.log(msg)

    def browse_folder(self):
        """Add all valid data files from a selected folder (recursive)."""
        folder = filedialog.askdirectory(
            title="Select folder or BIDS rawdata root")
        if not folder:
            return

        EXCLUDE_PATTERNS = (
            "metric_definitions",
            "metrics_definitions",
            "channel_info",
            "events",
            "_readme",
        )
        import glob as _glob
        _EXTS = ("*.txt", "*.smr", "*.adibin")
        all_files = []
        for _pat in _EXTS:
            all_files.extend(_glob.glob(os.path.join(folder, "**", _pat), recursive=True))
        files = sorted(
            f for f in all_files
            if not any(p in os.path.basename(f).lower() for p in EXCLUDE_PATTERNS)
        )

        if not files:
            messagebox.showinfo("No files found",
                "No data files (.txt, .smr, .adibin) found in that folder or its subfolders.\n\n"
                "If your files are in a different format, use '+ Add file(s)' instead.",
                parent=self.root)
            return

        ds = self._get_or_create_dataset()
        added = 0
        for fpath in files:
            if ds.get_by_path(fpath) is None:
                label = ds.label_from_bids(fpath)
                ds.add_file(fpath, label=label)
                added += 1

        ds.save()
        self._queue_refresh()
        self.log(f"📂 Added {added} file(s) from {os.path.basename(folder)}"
                 + (f" ({len(files)-added} already in queue)" if len(files) > added else ""))

    def browse_file(self):
        """Add one or more files to the queue and load the first one."""
        fpaths = filedialog.askopenfilenames(
            title="Select data file(s)",
            filetypes=[
                ("All supported formats", "*.txt *.smr *.adibin"),
                ("Spike2 / LabChart text export", "*.txt"),
                ("Spike2 native", "*.smr"),
                ("ADInstruments binary", "*.adibin"),
                ("All files", "*.*"),
            ]
        )
        if not fpaths:
            return

        ds = self._get_or_create_dataset()
        first_new = None
        for fpath in fpaths:
            fe = ds.get_by_path(fpath)
            if fe is None:
                label = ds.label_from_bids(fpath)
                fe = ds.add_file(fpath, label=label)
                if first_new is None:
                    first_new = fe
        ds.save()
        self._queue_refresh()

        # Load the first newly added file (or re-load if already in queue)
        target = first_new or ds.get_by_path(fpaths[0])
        if target:
            self.segments_metadata = {}   # fresh file — clear any stale edits
            self._load_file_entry(target)

    def _browse_file_path(self, fpath: str, auto_run: bool = False):
        # Guard: skip non-data files that may have been added to the queue
        EXCLUDE = ("metric_definitions", "metrics_definitions",
                   "channel_info", "_readme")
        if any(p in os.path.basename(fpath).lower() for p in EXCLUDE):
            self.log(f"⏭  Skipping non-data file: {os.path.basename(fpath)}")
            if self._dataset and hasattr(self, '_current_file_entry')                     and self._current_file_entry:
                self._current_file_entry.status = "skipped"
                self._dataset.save()
                self._queue_refresh()
            return

        # Auto-detect M-wave reference file in same folder
        if not self.mmax_file.get():
            import glob as _glob
            _folder = os.path.dirname(fpath)
            _candidates = [
                f for f in _glob.glob(os.path.join(_folder, "*.txt"))
                if any(kw in os.path.basename(f).lower()
                       for kw in ("mwave","mmax","m-wave","m_wave"))
            ]
            if _candidates:
                self.mmax_file.set(_candidates[0])
                self.log(f"📐 Auto-detected Mmax file: "
                         f"{os.path.basename(_candidates[0])}")

        marker_set = set()
        stim_events: dict[str, list[float]] = {}

        # ── Detect file format and scan accordingly ───────────────────────────
        _fmt = detect_format(fpath)

        # ── Generic TSV: launch Format Wizard if no sidecar config yet ────────
        if _fmt == 'generic_tsv' and needs_wizard(fpath):
            self.log("🔧 Generic TSV detected — launching Format Wizard…")

            def _on_wizard_complete(cfg, _fpath=fpath, _auto=auto_run):
                if cfg is None:
                    self.log("⚠️  Format Wizard cancelled — file not loaded.")
                    return
                self.log(
                    f"✅ Format Wizard complete — "
                    f"{len([c for c in cfg['channels'] if c['role'] != 'ignore'])} "
                    f"signal(s) defined, fs={cfg['fs']} Hz"
                )
                self._browse_file_path(_fpath, auto_run=_auto)

            FormatWizard(self.root, fpath, on_complete=_on_wizard_complete)
            return

        if _fmt == 'labchart':
            self.marker_choice.set('A')
            self.log("📋 LabChart format detected — stim times from analogue trigger channel")
            # stim_events populated later via extract_stim_times in pipeline

        elif _fmt == 'generic_tsv':
            self.marker_choice.set('A')
            self.log("📋 Generic TSV format — stim times from Stim/Trigger channel")
            # stim_events populated later via extract_stim_times in pipeline

        elif _fmt == 'cfwb':
            self.marker_choice.set('A')
            self.log("📋 ADInstruments binary (CFWB) format — stim times from trigger channel")

        elif _fmt == 'spike2_smr':
            self.log("📋 Spike2 SMR format detected — reading via Neo")
            try:
                from .formats.spike2_smr import (
                    has_config    as _smr_has_cfg,
                    save_config   as _smr_save_cfg,
                    load_config   as _smr_load_cfg,
                    get_channel_info as _smr_info,
                )
                if not _smr_has_cfg(fpath):
                    # First open — show channel assignment dialog
                    info = _smr_info(fpath)
                    analogue = info.get("analogue", [])
                    events   = info.get("events",   [])
                    epochs   = info.get("epochs",   [])
                    spikes   = info.get("spikes",   [])

                    if not analogue:
                        self.log("❌ No analogue channels found in SMR file.")
                        return

                    # Build a flat stim options list with type labels
                    stim_options = []
                    for n in events:
                        stim_options.append(f"[Event] {n}")
                    for n in epochs:
                        stim_options.append(f"[DigMark/Epoch] {n}")
                    for n in spikes:
                        stim_options.append(f"[Spike] {n}")
                    _STIM_KW = ("stim", "trig", "ttl", "digmark")
                    for n in analogue:
                        if any(kw in n.lower() for kw in _STIM_KW):
                            stim_options.append(f"[Analogue] {n}")
                    if not stim_options:
                        stim_options = [f"[Analogue] {n}" for n in analogue]

                    _chosen = {}

                    def _show_smr_dialog(
                        _analogue=analogue,
                        _stim_options=stim_options,
                    ):
                        import tkinter as tk
                        from tkinter import ttk
                        dlg = tk.Toplevel(self.root)
                        dlg.title("SMR Channel Assignment")
                        dlg.transient(self.root)
                        dlg.resizable(False, False)
                        dlg.grab_set()

                        tk.Label(
                            dlg,
                            text=(
                                f"File: {os.path.basename(fpath)}\n\n"
                                "Choose the EMG channel and the stim/trigger source.\n"
                                "Your choices are saved and will not be asked again."
                            ),
                            justify="left", padx=16, pady=10,
                        ).pack(anchor="w")

                        frm = tk.Frame(dlg, padx=16)
                        frm.pack(fill="x", pady=4)

                        # EMG channel
                        tk.Label(frm, text="EMG channel:", anchor="w", width=22) \
                            .grid(row=0, column=0, sticky="w", pady=6)
                        emg_var = tk.StringVar(value=_analogue[0])
                        ttk.Combobox(
                            frm, textvariable=emg_var,
                            values=_analogue, state="readonly", width=30,
                        ).grid(row=0, column=1, sticky="w")

                        # Stim/trigger channel
                        tk.Label(
                            frm,
                            text="Stim/trigger channel:",
                            anchor="w", width=22,
                        ).grid(row=1, column=0, sticky="w", pady=6)
                        stim_var = tk.StringVar(value=_stim_options[0])
                        ttk.Combobox(
                            frm, textvariable=stim_var,
                            values=_stim_options, state="readonly", width=30,
                        ).grid(row=1, column=1, sticky="w")

                        note = (
                            "Tip: Event channels (DigMark, Keyboard) use\n"
                            "     timestamps directly.  Analogue channels\n"
                            "     use threshold-crossing detection."
                        )
                        tk.Label(dlg, text=note, justify="left",
                                 fg="grey", padx=16).pack(anchor="w", pady=(0, 4))

                        def _ok():
                            raw_stim = stim_var.get()
                            # Strip the [Type] prefix to get the bare channel name
                            if "] " in raw_stim:
                                raw_stim = raw_stim.split("] ", 1)[1]
                            _chosen["emg"]  = emg_var.get()
                            _chosen["stim"] = raw_stim
                            dlg.destroy()

                        def _cancel():
                            dlg.destroy()

                        btn = tk.Frame(dlg)
                        btn.pack(pady=(4, 12))
                        tk.Button(btn, text="Save & continue",
                                  width=16, command=_ok).pack(side="left", padx=6)
                        tk.Button(btn, text="Cancel",
                                  width=10, command=_cancel).pack(side="left", padx=6)

                        dlg.update_idletasks()
                        x = (self.root.winfo_x()
                             + (self.root.winfo_width() - dlg.winfo_width()) // 2)
                        y = (self.root.winfo_y()
                             + (self.root.winfo_height() - dlg.winfo_height()) // 2)
                        dlg.geometry(f"+{x}+{y}")
                        self.root.wait_window(dlg)

                    _show_smr_dialog()

                    if not _chosen:
                        self.log("⚠️  SMR channel assignment cancelled.")
                        return

                    _smr_save_cfg(fpath, _chosen["emg"], _chosen["stim"])
                    self.log(
                        f"   EMG: {_chosen['emg']} | "
                        f"Stim: {_chosen['stim']} — saved to sidecar"
                    )
                    self.marker_choice.set(_chosen["stim"])

                else:
                    cfg = _smr_load_cfg(fpath)
                    stim_ch = cfg.get("stim_channel", "A")
                    self.log(
                        f"   EMG: {cfg.get('emg_channel')} | "
                        f"Stim channel: {stim_ch} — loaded from sidecar"
                    )
                    # Scan the stim channel for available marker codes,
                    # same as the text-format scan builds marker_set
                    from .formats.spike2_smr import (
                        get_event_codes_for_channel as _smr_codes,
                        extract_stim_times as _smr_stim,
                    )
                    event_codes = _smr_codes(fpath, stim_ch)
                    if len(event_codes) > 1:
                        self.log(
                            f"   Found {len(event_codes)} marker codes: "
                            + ", ".join(event_codes)
                        )
                        self._ask_marker_gui(event_codes)
                    elif event_codes:
                        self.marker_choice.set(event_codes[0])
                        self.log(f"   Marker code: {event_codes[0]}")
                    else:
                        # No discrete codes — use the channel name itself
                        # (analogue threshold fallback)
                        self.marker_choice.set(stim_ch)

            except ImportError:
                self.log(
                    "❌ Neo is not installed. Install it with:  pip install neo\n"
                    "   Native .smr files cannot be read without Neo."
                )
                return
            except Exception as e:
                self.log(f"❌ Error reading SMR file: {e}")
                return

        else:
            # Spike2 text export: scan for DigMark channels and stim timestamps
            stim_pattern = re.compile(r'^([\d.]+)\s+"(.{1})\?\?\?"')
            try:
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                for i in range(len(lines)):
                    if lines[i].strip().startswith('"Marker"') and i + 2 < len(lines):
                        m = lines[i + 2].strip().strip('"')
                        if m:
                            marker_set.add(m)
                for line in lines:
                    m = stim_pattern.match(line.strip())
                    if m:
                        t_s = float(m.group(1))
                        stype = m.group(2)
                        stim_events.setdefault(stype, []).append(t_s)
            except Exception as e:
                self.log(f"❌ Error reading {os.path.basename(fpath)}: {e}")
                return

            if len(marker_set) > 1:
                self._ask_marker_gui(sorted(marker_set))
            elif marker_set:
                self.marker_choice.set(next(iter(marker_set)))
        
        # ── populate inline channel dropdown
        chan_list = list_waveform_channels(fpath)
        self._populate_channel_dropdown(chan_list)

        # ── prompt channel selection if more than one channel available
        if len(chan_list) > 1:
            # Show a dropdown dialog rather than a text entry
            dlg = tk.Toplevel(self.root)
            dlg.title("Select channel")
            dlg.transient(self.root)
            dlg.resizable(False, False)
            dlg.grab_set()

            tk.Label(dlg, text="Select the EMG channel to analyse:",
                     padx=16, pady=(10)).pack()

            chosen_var = tk.StringVar(value=chan_list[0])
            cb = ttk.Combobox(dlg, textvariable=chosen_var,
                              values=chan_list, state="readonly", width=28)
            cb.pack(padx=16, pady=6)
            cb.current(0)

            def _ok():
                dlg.destroy()
            def _cancel():
                chosen_var.set("")
                dlg.destroy()

            btn_row = tk.Frame(dlg)
            btn_row.pack(pady=(0, 10))
            tk.Button(btn_row, text="OK", width=10,
                      command=_ok).pack(side='left', padx=6)
            tk.Button(btn_row, text="Cancel", width=10,
                      command=_cancel).pack(side='left', padx=6)

            # Centre over main window
            self.root.update_idletasks()
            dlg.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - dlg.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
            dlg.geometry(f"+{x}+{y}")
            self.root.wait_window(dlg)

            chosen = chosen_var.get()
            if chosen and chosen in chan_list:
                self.channel_var.set(chosen)
                self.channel_idx = chan_list.index(chosen)

        # All channels available in inspector extra channel dropdown
        self.extra_channel_indices = list(range(len(chan_list)))

        # ── Data range selection ───────────────────────────────────────────────
        _saved_crop_ranges = getattr(self, "crop_ranges", None)
        _saved_crop_start  = getattr(self, "crop_start", None)
        _saved_crop_end    = getattr(self, "crop_end", None)
        _has_saved_range   = bool(_saved_crop_ranges or
                                  (_saved_crop_start is not None
                                   and _saved_crop_end is not None))

        if _has_saved_range:
            if _saved_crop_ranges:
                _range_desc = f"{len(_saved_crop_ranges)} range(s) previously selected"
            else:
                _range_desc = f"{_saved_crop_start:.1f}s – {_saved_crop_end:.1f}s"
            dlg = tk.Toplevel(self.root)
            dlg.title("Data range")
            dlg.transient(self.root)
            dlg.resizable(False, False)
            dlg.grab_set()
            tk.Label(dlg,
                     text=f"A data range was previously saved:\n  {_range_desc}\n\nHow would you like to proceed?",
                     padx=16, pady=10, justify="left").pack()
            _choice = tk.StringVar(value="reuse")
            btn_frame = tk.Frame(dlg)
            btn_frame.pack(pady=(0, 12))
            def _pick(val):
                _choice.set(val)
                dlg.destroy()
            tk.Button(btn_frame, text="Use saved range", width=18,
                      command=lambda: _pick("reuse")).pack(side="left", padx=6)
            tk.Button(btn_frame, text="Select new range", width=18,
                      command=lambda: _pick("new")).pack(side="left", padx=6)
            tk.Button(btn_frame, text="Use whole file", width=18,
                      command=lambda: _pick("whole")).pack(side="left", padx=6)
            self.root.update_idletasks()
            dlg.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width()  - dlg.winfo_width())  // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
            dlg.geometry(f"+{x}+{y}")
            self.root.wait_window(dlg)
            choice = _choice.get()
            if choice == "reuse":
                pass
            elif choice == "new":
                self.crop_ranges = None; self.crop_start = None; self.crop_end = None
                if not self._crop_selector(fpath):
                    return
            else:
                self.crop_ranges = None; self.crop_start = None; self.crop_end = None
        else:
            whole = messagebox.askyesno(
                "Analyse whole recording?",
                "Analyse the entire file?\nChoose 'No' to pick a range interactively.",
                parent=self.root)
            if not whole:
                if not self._crop_selector(fpath):
                    return

        # ── Filter stim types to those with at least one event in the selected range
        if _fmt in ('labchart', 'generic_tsv'):
            # For LabChart, stim types come from the pipeline (stim channel detection).
            # Pre-populate with a single type 'A' — user can relabel in Stage 1a.
            stim_types_found = {'A'}
        elif self.crop_ranges:
            stim_types_found = {
                stype for stype, times in stim_events.items()
                if any(start <= t <= end
                       for t in times
                       for start, end in self.crop_ranges)
            }
        elif self.crop_start is not None and self.crop_end is not None:
            stim_types_found = {
                stype for stype, times in stim_events.items()
                if any(self.crop_start <= t <= self.crop_end for t in times)
            }
        else:
            stim_types_found = set(stim_events.keys())

        # ── prompt for study metadata (BIDS)
        self.prompt_study_metadata()

        # ── build / rebuild Tab 1b with per-stim config
        if stim_types_found:
            self._build_labels_tab(sorted(stim_types_found))

    @staticmethod
    def _parse_bids_from_filename(fpath):
        """Extract sub-ID, session, task, timepoint from a BIDS-style filename.
        e.g. sub-015_ses-2_task-limb_tp-pre_... → {participant_id:'sub-015', ...}
        Returns dict; any unparsed field is empty string.
        """
        name   = pathlib.Path(fpath).stem
        result = {'participant_id': '', 'session': '', 'task': '',
                  'timepoint': '', 'limb': '', 'measure': ''}
        for part in name.split('_'):
            pl = part.lower()
            if   pl.startswith('sub-'):     result['participant_id'] = part
            elif pl.startswith('ses-'):     result['session']        = part
            elif pl.startswith('task-'):    result['task']           = part[5:]
            elif pl.startswith('tp-'):      result['timepoint']      = part[3:]
            elif pl.startswith('limb-'):    result['limb']           = part[5:]
            elif pl.startswith('measure-'): result['measure']        = part[8:]
        return result

    def prompt_study_metadata(self, context: str = ""):
        """
        Modal dialog to collect BIDS-style metadata.
        context: optional filename shown at top to clarify which file this is for.
        """
        parsed = self._parse_bids_from_filename(self.file_path.get())
        carry = self._remembered_meta or self.study_metadata
        v_sub     = tk.StringVar(value=parsed['participant_id'] or carry.participant_id)
        v_ses     = tk.StringVar(value=parsed['session']        or carry.session or 'ses-01')
        v_task    = tk.StringVar(value=parsed['task']           or carry.task)
        v_tp      = tk.StringVar(value=parsed['timepoint']      or carry.timepoint)
        v_limb    = tk.StringVar(value=parsed['limb']           or getattr(carry, 'limb', ''))
        v_measure = tk.StringVar(value=parsed['measure']        or getattr(carry, 'measure', ''))
        v_rem     = tk.BooleanVar(value=self._remembered_meta is not None)

        win = tk.Toplevel(self.root)
        win.title("Study Metadata (BIDS)" + (f" — {context}" if context else ""))
        win.resizable(False, False)
        win.transient(self.root)

        pad = dict(padx=10, pady=4)

        if context:
            tk.Label(win, text=f"📋 External normalisation file: {context}",
                     fg="#d9534f", font=("TkDefaultFont", 9, "bold")).grid(
                     row=0, column=0, columnspan=3, **pad, sticky="w")
            tk.Label(win, text="Enter metadata for BIDS-style output naming.",
                     font=("TkDefaultFont", 9, "italic")).grid(
                     row=1, column=0, columnspan=3, **pad, sticky="w")
            _row_offset = 2
        else:
            tk.Label(win, text="Enter study metadata for BIDS-style output naming.",
                     font=("TkDefaultFont", 9, "italic")).grid(
                     row=0, column=0, columnspan=3, **pad, sticky="w")
            _row_offset = 1

        # Helper to add a labelled row
        def _row(r, label, var, example):
            tk.Label(win, text=label).grid(row=r+_row_offset, column=0, sticky="e", **pad)
            tk.Entry(win, textvariable=var, width=22).grid(row=r+_row_offset, column=1, sticky="w", **pad)
            tk.Label(win, text=example, fg="grey", font=("TkDefaultFont", 8))\
                .grid(row=r+_row_offset, column=2, sticky="w", padx=(0, 10))

        _row(1, "Participant ID *",  v_sub,  "e.g.  sub-JD001  or  JD001")
        _row(2, "Session",           v_ses,  "e.g.  ses-01  (default: ses-01)")
        _row(3, "Limb",              v_limb, "e.g.  left / right  (auto-detected)")
        _row(4, "Task label",        v_task, "e.g.  fatigue  (optional)")
        _row(5, "Timepoint",         v_tp,   "e.g.  pre / post  (optional)")

        # Measure type — dropdown of common TMS paradigms
        tk.Label(win, text="Measure type").grid(row=6+_row_offset, column=0, sticky="e", **pad)
        measure_frame = tk.Frame(win)
        measure_frame.grid(row=6+_row_offset, column=1, columnspan=2, sticky="w")
        _measure_choices = ['CSE', 'SICI', 'ICF', 'LICI', 'SAI', 'LAI', 'M-wave', 'CMEP', 'Other']
        measure_cb = ttk.Combobox(measure_frame, textvariable=v_measure,
                                  values=_measure_choices, width=10)
        measure_cb.pack(side="left")
        tk.Label(measure_frame, text="or type your own",
                 fg="grey", font=("TkDefaultFont", 8)).pack(side="left", padx=(6,0))

        tk.Checkbutton(win, text="Remember these settings for the next file",
                       variable=v_rem)          .grid(row=7+_row_offset, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 2))

        err_lbl = tk.Label(win, text="", fg="red")
        err_lbl.grid(row=8+_row_offset, column=0, columnspan=3, sticky="w", padx=10)

        def _save(_e=None):
            raw_sub = v_sub.get().strip()
            if not raw_sub:
                err_lbl.config(text="Participant ID is required.")
                return
            # Ensure sub- prefix
            if not raw_sub.lower().startswith("sub-"):
                raw_sub = "sub-" + raw_sub
            # Sanitise each field
            sub  = "sub-" + _sanitise_bids_label(raw_sub[4:])
            ses  = "ses-" + _sanitise_bids_label(v_ses.get().lstrip("ses-").strip() or "01")
            task = _sanitise_bids_label(v_task.get()) if v_task.get().strip() else ""
            tp   = _sanitise_bids_label(v_tp.get())   if v_tp.get().strip()   else ""

            limb    = _sanitise_bids_label(v_limb.get()).lower()    if v_limb.get().strip()    else ""
            measure = _sanitise_bids_label(v_measure.get())         if v_measure.get().strip() else ""
            self.study_metadata = StudyMetadata(
                participant_id = sub,
                session        = ses,
                task           = task,
                timepoint      = tp,
                limb           = limb,
                measure        = measure,
            )
            self._remembered_meta = self.study_metadata if v_rem.get() else None
            win.destroy()

        btn_row = tk.Frame(win)
        btn_row.grid(row=9+_row_offset, column=0, columnspan=3, pady=10)
        tk.Button(btn_row, text="OK", width=10, command=_save).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=win.destroy).pack(side="left", padx=6)

        win.bind("<Return>", _save)
        win.bind("<Escape>", lambda _e: win.destroy())
        win.update_idletasks()
        # Centre over main window
        px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
        pw, ph = self.root.winfo_width(),  self.root.winfo_height()
        w,  h  = win.winfo_width(),        win.winfo_height()
        win.geometry(f"+{px+(pw-w)//2}+{py+(ph-h)//2}")
        win.grab_set()
        self.root.wait_window(win)

    def browse_derivatives_folder(self):
        """Let the user choose where the derivatives/ root lives."""
        folder = filedialog.askdirectory(
            title="Select derivatives root folder",
            mustexist=False,
        )
        if not folder:
            return
        # Safeguard: warn if derivatives would be inside rawdata
        raw = self._rawdata_path.get() if hasattr(self, '_rawdata_path') else ""
        if raw and os.path.normpath(folder).startswith(os.path.normpath(raw)):
            if not messagebox.askyesno(
                "Derivatives inside raw data?",
                f"The selected folder is inside your raw data folder:\n\n"
                f"  Raw:         {raw}\n"
                f"  Derivatives: {folder}\n\n"
                f"It is strongly recommended to keep derivatives beside rawdata/, "
                f"not inside it.\n\nUse this folder anyway?",
                parent=self.root):
                return
        folder = str(Path(folder))
        self.derivatives_path.set(Path(folder).as_posix())
        os.makedirs(folder, exist_ok=True)
        self.log(f"📁 Derivatives folder: {Path(folder).as_posix()}")
        self._update_deriv_status()
        self._dataset = DatasetSession.load_or_create(folder)
        self._queue_refresh()

    def _update_deriv_status(self):
        """Update the derivatives status bar colour and text."""
        try:
            path = self.derivatives_path.get()
        except Exception:
            path = ""
        if path:
            display = path if len(path) <= 70 else "…" + path[-67:]
            self._deriv_status_bar.config(
                text=f"✔  Derivatives: {display}",
                bg="#5cb85c", fg="white")
        else:
            self._deriv_status_bar.config(
                text="⚠  Derivatives folder not set — click here or use File → Set Derivatives Folder",
                bg="#d9534f", fg="white")

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def update_progress(self, value):
        self.progress.set(value)
        self.root.update_idletasks()
    
    # ──────────────────────────────────────────────────────────────
    def _ask_marker_gui(self, choices):
        """Modal dialog → chooses the marker source (GUI thread)."""
        win = tk.Toplevel(self.root)
        win.title("Select marker source")

        v = tk.StringVar(value=choices[0])

        tk.Label(win, text="Multiple marker sources found.\nChoose one:")\
            .pack(padx=10, pady=(10, 4))
        ttk.OptionMenu(win, v, choices[0], *choices).pack(padx=10, pady=6)

        def _ok():
            self._marker_choice_result = v.get()   # <-- plain string
            self.marker_choice.set(v.get())        # keep GUI field in sync
            win.destroy()

        tk.Button(win, text="OK", command=_ok).pack(pady=10)
        win.grab_set()
        self.root.wait_window(win)
    
    def _review_outliers_gui(self, flagged_outliers, fs, pre_ms, post_ms, emg_unit=None):
        """
        Interactive review of outlier segments; returns a list with only the
        outliers the user chooses to KEEP.  Runs entirely on the Tk main thread.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        reviewed_segments = []          # what the user decides to keep

        # ---------------------------------------------------------------- helper
        def show_next(index: int):
            """Draw the dialog for the outlier at <index> (or close when done)."""
            if index >= len(flagged_outliers):
                popup.destroy()
                return

            outlier = flagged_outliers[index]
            emg      = outlier["emg_segment"]
            time_ax  = np.linspace(-pre_ms, post_ms, len(emg), endpoint=False)

            # ---- Matplotlib figure --------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(time_ax, emg)
            ax.axvline(0, color='black', linestyle='--')
            ax.set_xlim(-pre_ms, post_ms)
            ax.set_title(f'{outlier["file"]} – {outlier["stim_type"]} – seg {outlier["index"]+1}')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(f"EMG ({emg_unit})" if emg_unit else "EMG")
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            canvas.get_tk_widget().pack()

            # ---- Stats read‑out -----------------------------------------------
            stats_label.config(text=(
                f"Pre‑stim RMS: {outlier['rms']:.4f}  (z = {outlier['z_rms']:.2f})\n"
                f"MEP PTP:      {outlier['ptp']:.4f}  (z = {outlier['z_ptp']:.2f})"
            ))

            # ---- Button callbacks ---------------------------------------------
            def keep():
                reviewed_segments.append(outlier)
                canvas.get_tk_widget().destroy()
                plt.close(fig)           # fully dispose the Tk figure
                show_next(index + 1)

            def remove():
                canvas.get_tk_widget().destroy()
                plt.close(fig)
                show_next(index + 1)

            keep_btn.config(command=keep)
            remove_btn.config(command=remove)

        # ── Tk dialog scaffold ─────────────────────────────────────────────────--
        popup = tk.Toplevel(self.root)
        popup.title("Review Outliers")

        stats_label = tk.Label(popup, text="", font=("Arial", 10))
        stats_label.pack(pady=5)

        keep_btn   = tk.Button(popup, text="Keep",   width=15)
        keep_btn.pack(side="left",  padx=20, pady=10)
        remove_btn = tk.Button(popup, text="Remove", width=15)
        remove_btn.pack(side="right", padx=20, pady=10)

        show_next(0)          # display the first outlier
        popup.grab_set()      # modal
        self.root.wait_window(popup)

        return reviewed_segments


    def _prompt_extra_channels(self, all_channels, other_channels):
        """
        Ask the user which additional channels to show in the Data Inspector
        for visual reference (no quantification).
        """
        win = tk.Toplevel(self.root)
        win.title("Additional channels for Data Inspector")
        win.transient(self.root)
        win.resizable(False, False)

        tk.Label(win,
            text="Select channels to show alongside the primary EMG\n"
                 "in the Data Inspector (visual reference only, no quantification):",
            justify="left").pack(padx=12, pady=(10, 6))

        # Checkboxes — one per non-primary channel
        _vars = {}
        for cname in other_channels:
            v = tk.BooleanVar(value=False)
            tk.Checkbutton(win, text=cname, variable=v,
                           anchor="w").pack(fill="x", padx=20, pady=1)
            _vars[cname] = v

        # Wide window spinbox
        w_frame = tk.Frame(win)
        w_frame.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(w_frame, text="Wide window (±s):").pack(side="left")
        tk.Spinbox(w_frame, from_=0.5, to=30.0, increment=0.5, width=6,
                   textvariable=self.wide_window_s).pack(side="left", padx=6)
        tk.Label(w_frame, text="seconds either side of stim",
                 fg="grey").pack(side="left")

        def _ok():
            self.extra_channel_indices = [
                all_channels.index(cname)
                for cname, v in _vars.items() if v.get()
            ]
            win.destroy()

        def _skip():
            self.extra_channel_indices = []
            win.destroy()

        btn = tk.Frame(win)
        btn.pack(pady=(6, 10))
        tk.Button(btn, text="OK", width=10, command=_ok).pack(side="left", padx=6)
        tk.Button(btn, text="Skip", width=10, command=_skip).pack(side="left", padx=6)

        win.bind("<Return>", lambda _: _ok())
        win.bind("<Escape>", lambda _: _skip())
        win.update_idletasks()
        win.grab_set()
        self.root.wait_window(win)

    def _populate_channel_dropdown(self, channel_names):
        """Populate the inline channel combobox after file load."""
        self.channel_dd["values"] = channel_names
        self.channel_dd["state"]  = "readonly"
        self.channel_var.set(channel_names[0])
        self.channel_idx   = 0
        self.channel_choice.set(channel_names[0])

    def _on_channel_selected(self, _event=None):
        """Called when the user changes the channel combobox."""
        name = self.channel_var.get()
        names = list(self.channel_dd["values"])
        if name in names:
            self.channel_idx   = names.index(name)
            self.channel_choice.set(name)

    # ──────────────────────────────────────────────────────────────────────
    def _build_labels_tab(self, stim_types):
        """
        Build (or rebuild) the Stage 1b tab with per-stim configuration:
          • label, colour, include in combined plot, gap (ms)
          • detect CSP checkbox
          • internal normalisation reference (ratio to another stim type)
          • external Mmax file + plateau tolerance
        Preserves existing settings for stim types that appear in both
        the previous and new file (session-level persistence without restart).
        Called from browse_file after stim types are discovered.
        """
        import importlib

        # Clear existing tab content
        for w in self.tab1b_frame.winfo_children():
            w.destroy()

        # ── store stim types for validation ──────────────────────────────────
        self._current_stim_types = list(sorted(stim_types))

        colour_choices = [
            "darkgreen","deeppink","brown","black","deepskyblue","maroon",
            "springgreen","mediumvioletred","seagreen","hotpink","turquoise",
            "navy","orange","indigo","darkorange","midnightblue","saddlebrown",
            "blue","darkred","royalblue","firebrick","darkslategray","brown",
            "slateblue","purple",
        ]

        # ── outer scroll area ─────────────────────────────────────────────────
        outer = tk.Frame(self.tab1b_frame)
        outer.pack(fill="both", expand=True)

        vscroll = ttk.Scrollbar(outer, orient="vertical")
        vscroll.pack(side="right", fill="y")
        cv = tk.Canvas(outer, bd=0, highlightthickness=0,
                       yscrollcommand=vscroll.set)
        cv.pack(side="left", fill="both", expand=True)
        vscroll.config(command=cv.yview)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: cv.configure(scrollregion=cv.bbox("all")))

        # ── header hint ───────────────────────────────────────────────────────
        tk.Label(inner,
            text="Configure labels, colours, and analysis options for each "
                 "stimulus type found in the loaded file.\n"
                 "Click  ✔  Confirm Setup  when ready — Run Analysis will "
                 "not proceed until this is confirmed.\n"
                 "Gap (ms): time blanked immediately before each stimulus pulse. "
                 "Use this to exclude signal that would contaminate the pre-stimulus baseline. "
                 "Example: in paired-pulse protocols (e.g., SICI, ICF) a conditioning pulse precedes "
                 "the test pulse by a fixed interstimulus interval — setting the gap to just "
                 "longer than that interval prevents the conditioning artefact from being included "
                 "in the background EMG measurement. Leave at 0 if unused.",
            fg="grey", justify="left", wraplength=900
        ).grid(row=0, column=0, columnspan=9, sticky="w", padx=10, pady=(10,6))

        # ── Latency lookup table — read from user preferences ─────────────────
        # Users can edit these in Settings → Preferences → Latency Profiles.
        LATENCY_PROFILES = prefs.latency_profiles_as_dict()
        MUSCLE_OPTIONS   = prefs.muscle_options()
        self._LATENCY_PROFILES = LATENCY_PROFILES
        self._MUSCLE_OPTIONS   = MUSCLE_OPTIONS

        # ── column headers ────────────────────────────────────────────────────
        headers = ["Stim", "Label", "Colour", "In combined",
                   "Gap (ms)", "Detect CSP", "Normalise to (internal)",
                   "Plateau (%)",
                   "Stim type", "Muscle group", "Min lat (ms)", "Max lat (ms)"]
        for c, h in enumerate(headers):
            tk.Label(inner, text=h,
                     font=("TkDefaultFont", 9, "bold"))\
                .grid(row=1, column=c, padx=6, pady=(0,4), sticky="w")

        # ── per-stim rows ─────────────────────────────────────────────────────
        self._lab_entry_label   = {}
        self._lab_entry_colour  = {}
        self._lab_entry_include = {}
        self._lab_entry_gap     = {}
        self._lab_entry_csp     = {}
        self._lab_entry_ref     = {}
        self._lab_entry_plateau = {}
        self._lat_min_vars      = {}
        self._lat_max_vars      = {}
        self._lat_stype_vars    = {}
        self._lat_muscle_vars   = {}

        for r, stim in enumerate(sorted(stim_types), start=2):
            tk.Label(inner, text=f"{stim}:")\
                .grid(row=r, column=0, sticky="e", padx=(8,2))

            # Label
            v_lbl = tk.StringVar(value=self.label_map.get(stim, stim))
            tk.Entry(inner, textvariable=v_lbl, width=18)\
                .grid(row=r, column=1, padx=4, sticky="w")
            self._lab_entry_label[stim] = v_lbl

            # Colour
            v_col = tk.StringVar(
                value=self.color_map.get(stim,
                    colour_choices[(r-2) % len(colour_choices)]))
            tk.OptionMenu(inner, v_col, *colour_choices)\
                .grid(row=r, column=2, padx=4, sticky="w")
            self._lab_entry_colour[stim] = v_col

            # Include in combined plot
            v_inc = tk.BooleanVar(value=self.plot_included.get(stim, True))
            tk.Checkbutton(inner, variable=v_inc)\
                .grid(row=r, column=3, padx=10, sticky="w")
            self._lab_entry_include[stim] = v_inc

            # Gap ms
            v_gap = tk.DoubleVar(value=self.gap_ms_map.get(stim, 0.0))
            tk.Entry(inner, textvariable=v_gap, width=6)\
                .grid(row=r, column=4, padx=4, sticky="w")
            self._lab_entry_gap[stim] = v_gap

            # Detect CSP
            v_csp = tk.BooleanVar(value=(stim in self.csp_types))
            tk.Checkbutton(inner, variable=v_csp)\
                .grid(row=r, column=5, padx=10, sticky="w")
            self._lab_entry_csp[stim] = v_csp

            # Internal normalisation reference
            _ref_display = getattr(self, '_reference_display', {}).get(stim, "None")
            v_ref = tk.StringVar(value=_ref_display)
            ref_cb = ttk.Combobox(inner, textvariable=v_ref,
                                   width=26, state="readonly")
            ref_cb.grid(row=r, column=6, padx=6, sticky="w")
            self._lab_entry_ref[stim] = (v_ref, ref_cb)

            # Plateau tolerance (per-stim, default from global)
            v_plat = tk.DoubleVar(value=self.plateau_tolerance.get())
            tk.Spinbox(inner, from_=1, to=30, increment=1, width=5,
                       textvariable=v_plat)\
                .grid(row=r, column=7, padx=4, sticky="w")
            self._lab_entry_plateau[stim] = v_plat

            # Stim type dropdown
            _def_stype, _def_muscle = prefs.default_latency_key
            _prev_stype  = self.latency_stim_map.get(stim, _def_stype)
            _prev_muscle = self.latency_muscle_map.get(stim, _def_muscle)
            _prev_lat    = self.latency_map.get(stim)
            v_stype = tk.StringVar(value=_prev_stype)
            stype_cb = ttk.Combobox(inner, textvariable=v_stype,
                                    values=list(MUSCLE_OPTIONS.keys()),
                                    state="readonly", width=14)
            stype_cb.grid(row=r, column=8, padx=4, sticky="w")

            # Muscle group — restore saved value, ensuring it's valid for stim type
            _muscle_opts = MUSCLE_OPTIONS.get(_prev_stype, ["Hand / FDI"])
            if _prev_muscle not in _muscle_opts:
                _prev_muscle = _muscle_opts[0]
            v_muscle = tk.StringVar(value=_prev_muscle)
            muscle_cb = ttk.Combobox(inner, textvariable=v_muscle,
                                     values=_muscle_opts,
                                     state="readonly", width=22)
            muscle_cb.grid(row=r, column=9, padx=4, sticky="w")

            self._lat_stype_vars[stim]  = v_stype
            self._lat_muscle_vars[stim] = v_muscle

            # Pre-fill min/max from saved latency_map if available;
            # otherwise fall back to the profile for the currently selected muscle
            if _prev_lat:
                _def_min, _def_max = _prev_lat
            else:
                _def_min, _def_max = LATENCY_PROFILES.get(
                    (_prev_stype, _prev_muscle),
                    LATENCY_PROFILES.get(prefs.default_latency_key, (18.0, 28.0))
                )
            v_min = tk.DoubleVar(value=_def_min)
            v_max = tk.DoubleVar(value=_def_max)
            tk.Entry(inner, textvariable=v_min, width=5)\
                .grid(row=r, column=10, padx=4, sticky="w")
            tk.Entry(inner, textvariable=v_max, width=5)\
                .grid(row=r, column=11, padx=4, sticky="w")

            self._lat_min_vars[stim] = v_min
            self._lat_max_vars[stim] = v_max

            # Wire stim type → muscle options → auto-fill latency
            def _make_lat_callbacks(vs, vm, vmin, vmax, mcb, has_saved):
                def _on_stype(*_):
                    opts = MUSCLE_OPTIONS.get(vs.get(), ["Custom"])
                    mcb["values"] = opts
                    if vm.get() not in opts:
                        vm.set(opts[0])
                    _on_muscle()
                def _on_muscle(*_):
                    profile = LATENCY_PROFILES.get((vs.get(), vm.get()))
                    if profile:
                        vmin.set(profile[0])
                        vmax.set(profile[1])
                vs.trace_add("write", _on_stype)
                vm.trace_add("write", _on_muscle)
                if not has_saved:
                    _on_muscle()  # set defaults only if no saved value
            _make_lat_callbacks(v_stype, v_muscle, v_min, v_max, muscle_cb,
                                has_saved=bool(_prev_lat))

        # ── populate reference dropdowns ──────────────────────────────────────
        def _build_ref_options():
            for stim, (v_ref, ref_cb) in self._lab_entry_ref.items():
                others  = [s for s in sorted(stim_types) if s != stim]
                options = ["None"] + [
                    f"Normalise to {s}  ({self._lab_entry_label[s].get() or s})"
                    for s in others
                ]
                ref_cb["values"] = options
                cur = v_ref.get()
                if cur not in options:
                    v_ref.set("None")
        _build_ref_options()
        for v_lbl in self._lab_entry_label.values():
            v_lbl.trace_add("write", lambda *_: _build_ref_options())

        # ── global Mmax file row (shared fallback) ────────────────────────────

        # ── confirm button ────────────────────────────────────────────────────
        footer = tk.Frame(self.tab1b_frame, bd=1, relief="raised")
        footer.pack(side="bottom", fill="x")
        self._confirm_btn_var = tk.StringVar(value="⚠  Setup not confirmed")
        confirm_btn = tk.Button(
            footer,
            textvariable=self._confirm_btn_var,
            bg="#d9534f", fg="white",
            font=("TkDefaultFont", 10, "bold"),
            command=self._confirm_labels_tab)
        confirm_btn.pack(side="left", padx=12, pady=6, ipadx=10)
        self._confirm_btn_widget = confirm_btn
        tk.Label(footer,
            text="Confirm when you have finished configuring each stimulus type.",
            fg="grey").pack(side="left", padx=6)

        self._labels_tab_built     = True
        self._labels_tab_confirmed = False
        self._confirm_btn_var.set("⚠  Setup not confirmed — click to confirm")

        # Switch to Stage 1a Labels tab so user can configure stim types
        self.root.update_idletasks()
        self.notebook.select(self.tab1b_frame)

    def _browse_mmax_for_var(self, string_var):
        """Interactively configure an external normalisation reference file.
        Collects: file path, EMG channel, stim label, crop range, BIDS metadata.
        Stores the result as a JSON config string in string_var.
        """
        from tkinter import filedialog as _fd
        import json as _json

        path = _fd.askopenfilename(
            title="Select external normalisation reference file",
            filetypes=[("Data files", "*.txt"), ("All files", "*.*")])
        if not path:
            return

        # ── Step 1: Channel selection ─────────────────────────────────────────
        try:
            chan_list = list_waveform_channels(path)
        except Exception as e:
            messagebox.showerror("File error",
                f"Could not read channels:\n{e}", parent=self.root)
            return

        chosen_channel = 0
        if len(chan_list) > 1:
            dlg = tk.Toplevel(self.root)
            dlg.title(f"External file — Select EMG channel")
            dlg.transient(self.root)
            dlg.resizable(False, False)
            dlg.grab_set()
            tk.Label(dlg,
                text=f"File: {os.path.basename(path)}\n\nSelect the EMG channel:",
                padx=16, pady=8, justify="left").pack(anchor="w")
            _ch_var = tk.StringVar(value=chan_list[0])
            ttk.Combobox(dlg, textvariable=_ch_var, values=chan_list,
                         state="readonly", width=30).pack(padx=16, pady=4)
            tk.Button(dlg, text="OK", width=10,
                      command=dlg.destroy).pack(pady=(0, 10))
            self.root.update_idletasks(); dlg.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width()  - dlg.winfo_width())  // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
            dlg.geometry(f"+{x}+{y}")
            self.root.wait_window(dlg)
            chosen_channel = (chan_list.index(_ch_var.get())
                              if _ch_var.get() in chan_list else 0)

        # ── Step 2: Stim label ────────────────────────────────────────────────
        used = set(self.label_map.keys()) if self.label_map else {'A'}
        avail = [c for c in 'BCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in used]
        default_lbl = avail[0] if avail else 'Z'

        fmt = detect_format(path)
        stim_label = default_lbl
        if fmt == 'labchart':
            result = simpledialog.askstring(
                "External file — Stim label",
                f"Assign a single-letter label for:\n{os.path.basename(path)}\n\n"
                f"Must differ from main file labels: {', '.join(sorted(used))}",
                initialvalue=default_lbl, parent=self.root)
            if result:
                stim_label = result.strip().upper()[:1] or default_lbl

        # ── Step 3: Data range ────────────────────────────────────────────────
        crop_start, crop_end = None, None
        whole = messagebox.askyesno(
            "External file — Data range",
            f"Analyse the entire file?\n{os.path.basename(path)}\n\n"
            "Choose 'No' to select a specific range.",
            parent=self.root)
        if not whole:
            # Temporarily swap state so _crop_selector works on ext file
            _orig_path  = self.file_path.get()
            _orig_ch    = self.channel_idx
            _orig_cs    = self.crop_start
            _orig_ce    = self.crop_end
            _orig_cr    = getattr(self, 'crop_ranges', None)
            _orig_mc    = self.marker_choice.get()
            self.file_path.set(path)
            self.channel_idx = chosen_channel
            self.crop_start  = None
            self.crop_end    = None
            self.crop_ranges = None
            self.marker_choice.set(stim_label)
            self._crop_selector(path)
            crop_start = self.crop_start
            crop_end   = self.crop_end
            self.file_path.set(_orig_path)
            self.channel_idx = _orig_ch
            self.crop_start  = _orig_cs
            self.crop_end    = _orig_ce
            self.crop_ranges = _orig_cr
            self.marker_choice.set(_orig_mc)

        # ── Step 4: BIDS metadata ─────────────────────────────────────────────
        _orig_path = self.file_path.get()
        _orig_meta = getattr(self, 'study_metadata', None)
        self.file_path.set(path)
        self.prompt_study_metadata(context=os.path.basename(path))
        bids_participant_id = self.study_metadata.participant_id
        bids_session        = self.study_metadata.session
        bids_task           = self.study_metadata.task
        bids_timepoint      = self.study_metadata.timepoint
        bids_measure        = self.study_metadata.measure
        self.file_path.set(_orig_path)
        if _orig_meta is not None:
            self.study_metadata = _orig_meta

        # ── Store config ──────────────────────────────────────────────────────
        config = {
            "path":                path,
            "channel_idx":         chosen_channel,
            "stim_label":          stim_label,
            "crop_start":          crop_start,
            "crop_end":            crop_end,
            "all_channels":        chan_list,
            "bids_participant_id": bids_participant_id,
            "bids_session":        bids_session,
            "bids_task":           bids_task,
            "bids_timepoint":      bids_timepoint,
            "bids_measure":        bids_measure,
        }
        string_var.set(_json.dumps(config))
        self.log(f"📋 External ref: {os.path.basename(path)} "
                 f"| Ch {chosen_channel} ({chan_list[chosen_channel]}) "
                 f"| Label '{stim_label}'"
                 + (f" | t=[{crop_start:.1f},{crop_end:.1f}]s"
                    if crop_start is not None else " | full file"))

    def _confirm_labels_tab(self):
        """Read Tab 1b widgets into self.* dicts and mark setup as confirmed."""
        self.label_map     = {k: (v.get().strip() or k)
                              for k, v in self._lab_entry_label.items()}
        self.color_map     = {k: v.get()
                              for k, v in self._lab_entry_colour.items()}
        self.plot_included = {k: v.get()
                              for k, v in self._lab_entry_include.items()}
        self.gap_ms_map    = {k: float(v.get() or 0.)
                              for k, v in self._lab_entry_gap.items()}
        self.csp_types     = {k for k, v in self._lab_entry_csp.items()
                              if v.get()}
        self.reference_map = {}
        for k, (v_ref, _) in self._lab_entry_ref.items():
            sel = v_ref.get()
            if sel and sel != "None" and sel.startswith("Normalise to "):
                ref_letter = sel.split("to ")[1].strip().split(" ")[0]
                self.reference_map[k] = ref_letter
                self._reference_display = getattr(self, '_reference_display', {})
                self._reference_display[k] = sel

        # Per-stim latency bounds + stim type/muscle selections
        self.latency_map = {
            k: (float(self._lat_min_vars[k].get()),
                float(self._lat_max_vars[k].get()))
            for k in self._lat_min_vars
        }
        self.latency_stim_map = {
            k: v.get() for k, v in self._lat_stype_vars.items()
        }
        self.latency_muscle_map = {
            k: v.get() for k, v in self._lat_muscle_vars.items()
        }

        self._labels_tab_confirmed = True
        self._confirm_btn_var.set("✔  Setup confirmed")
        self._confirm_btn_widget.config(bg="#5cb85c")
        self.log("✔ Label & analysis setup confirmed — ready to run.\n")
        # Switch back to Stage 1a so user can hit Run Analysis
        self.notebook.select(2)
