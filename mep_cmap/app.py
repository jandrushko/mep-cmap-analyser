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
from .io import (list_waveform_channels, extract_emg_waveform_and_fs,
                 extract_stim_times)
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
        # ── Detect DPI and apply initial scaling ──────────────────────────────
        prefs.detect_dpi(root)
        apply_scaling(root)
        # ── State that setup_gui() widgets depend on — must come first ────────
        self.crop_start        = None
        self.crop_end          = None
        self.crop_ranges       = None
        self.gap_ms_map        = {}
        self.reference_map         = {}
        self.mmax_file             = tk.StringVar()
        self.plateau_tolerance     = tk.DoubleVar(value=10.0)
        self.extra_channel_indices = []    # additional channels for inspector
        self.wide_window_s         = tk.DoubleVar(value=3.0)
        self.emg_unit          = None
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
    def _cap_toplevel(win, frac_h=0.88, frac_w=0.92):
        """Cap a Toplevel to a fraction of screen size and centre it."""
        win.update_idletasks()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        max_w = int(sw * frac_w)
        max_h = int(sh * frac_h)
        req_w = win.winfo_reqwidth()  + 40
        req_h = win.winfo_reqheight() + 40
        final_w = min(req_w, max_w)
        final_h = min(req_h, max_h)
        x = (sw - final_w) // 2
        y = (sh - final_h) // 4
        win.geometry(f"{final_w}x{final_h}+{x}+{y}")

    def _make_window_adaptive(self):
        """Size the window to fit content, up to 90% of the screen in each dimension."""

        # 1) Screen dimensions
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        # 2) Height: 90% of screen height, with a 600px floor
        h = int(sh * 0.9)
        h = max(600, h)

        # 3) Width: ask Tk for the window's natural (shrink-wrapped) required width.
        #    setup_gui() already called geometry("") + update_idletasks(), so by the
        #    time this after(0) callback runs the layout has settled and
        #    winfo_reqwidth() returns the true content width.
        #    We must NOT force the window to 1px first — that collapses every
        #    fill='x' frame and produces a falsely small measurement.
        self.root.update_idletasks()
        natural_w = self.root.winfo_reqwidth()

        # 4) Add scrollbar width + a little breathing room
        padding = 36          # 17px scrollbar + ~19px padding
        desired_w = natural_w + padding

        # 5) Clamp to [min:680px, max:90% of screen width]
        min_w   = 680
        max_w   = int(sw * 0.9)
        final_w = min(max(desired_w, min_w), max_w)

        # 6) Center and apply geometry
        x = (sw - final_w) // 2
        y = (sh - h)       // 4
        self.root.geometry(f"{final_w}x{h}+{x}+{y}")

        # 7) Apply DPI-aware font scaling
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

        # ── Tab 1: Stage 1 (scrollable content + fixed footer) ───────────────
        tab1_outer = ttk.Frame(self.notebook)
        self.notebook.add(tab1_outer, text="Stage 1 – Single File Processing")

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
        self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        self.main_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        def _on_mousewheel(event):
            # Only scroll when Stage 1 tab is active
            if self.notebook.index(self.notebook.select()) == 0:
                delta = event.delta if event.delta else (-120 if event.num == 5 else 120)
                self.canvas.yview_scroll(int(-delta / 120), "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(seq, _on_mousewheel)

        # ── Tab 2: Stage 2 (group analysis) ──────────────────────────────────
        self.tab2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2_frame, text="Stage 2 – Group Analysis")
        # Stage 2 content is built lazily on first tab switch
        self._stage2_built = False
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ─── User Path & Data States ──────────────────────────────────────────
        self.file_path = tk.StringVar()
        self.derivatives_path = tk.StringVar()
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
            # CSP detection
            csp_search_start_ms = self.csp_search_start_ms.get(),
            csp_search_end_ms   = self.csp_search_end_ms.get(),
            csp_min_silence_ms  = self.csp_min_silence_ms.get(),
            csp_min_return_ms   = self.csp_min_return_ms.get(),
            csp_criterion       = self.csp_criterion.get(),
            csp_significance    = self.csp_significance.get(),
            csp_n_boot          = self.csp_n_boot.get(),
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
        file_menu.add_command(label="Open File...",  command=lambda: self.browse_file())
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
        # ─── Input File Selection ────────────────────────────────────────────
        tk.Label(self.main_frame, text="Select Input File:").pack(anchor='w', padx=10, pady=(10, 0))
        path_frame = tk.Frame(self.main_frame)
        path_frame.pack(fill='x', padx=10)
        tk.Entry(path_frame, textvariable=self.file_path, width=50).pack(side='left', expand=True, fill='x')
        tk.Button(path_frame, text="Browse", command=self.browse_file).pack(side='right')

        # ─── Derivatives Folder Selection ────────────────────────────────────
        tk.Label(self.main_frame,
                 text="Derivatives Folder  (outputs saved here as: <folder>/derivatives/sub-XX/ses-XX/):")            .pack(anchor='w', padx=10, pady=(6, 0))
        deriv_frame = tk.Frame(self.main_frame)
        deriv_frame.pack(fill='x', padx=10)
        tk.Entry(deriv_frame, textvariable=self.derivatives_path, width=50)            .pack(side='left', expand=True, fill='x')
        tk.Button(deriv_frame, text="Browse", command=self.browse_derivatives_folder)            .pack(side='right')
        tk.Label(self.main_frame,
                 text="Leave blank to save beside the source .txt file.",
                 fg="grey", font=("TkDefaultFont", 8)).pack(anchor='w', padx=10)

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
        self.csp_search_start_ms = tk.IntVar(value=40)
        self.csp_search_end_ms   = tk.IntVar(value=400)
        self.csp_min_silence_ms  = tk.IntVar(value=25)
        self.csp_min_return_ms   = tk.IntVar(value=40)
        self.csp_criterion       = tk.DoubleVar(value=1.96)
        self.csp_significance    = tk.DoubleVar(value=0.99)
        self.csp_n_boot          = tk.IntVar(value=1000)

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

        # ── row-5: mains noise canceller + harmonics -------------------------------
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

        # ── row-6: preview filter button -------------------------------------------
        pf_row = tk.Frame(filter_frame)
        pf_row.grid(row=6, column=0, columnspan=6, sticky='ew', pady=(8, 4))

        tk.Button(
            pf_row,
            text="Preview Filter",
            command=self.preview_filter_window
        ).pack()  # pack centers by default

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
        tk.Label(csp_frame,
            text="Z-score: threshold multiplier (1.96 = 95% CI)  ·  Significance: bootstrap percentile for min duration (0.99 = 99th pct)  ·  Min silence / Min return in ms",
            fg="grey",font=("TkDefaultFont",9,"italic")).grid(row=4,column=0,columnspan=4,sticky='w',padx=6,pady=(2,0))

        # ─── Normalisation Settings ────────────────────────────────────────────
        norm_frame = tk.LabelFrame(
            self.main_frame, text="Normalisation Settings (optional)",
            padx=6, pady=8)
        norm_frame.pack(padx=6, pady=(10,0), fill='x')

        tk.Label(norm_frame,
            text="Mmax normalisation and paired-pulse ratios are configured\n"
                 "per-condition in the label setup dialog (shown after file selection).",
            fg="grey", justify="left").grid(row=0, column=0, columnspan=4,
                                             sticky="w", padx=4, pady=(0,4))

        tk.Label(norm_frame, text="Mmax reference file:").grid(
            row=1, column=0, sticky="e", padx=6)
        tk.Entry(norm_frame, textvariable=self.mmax_file, width=40).grid(
            row=1, column=1, sticky="ew", padx=4)
        tk.Button(norm_frame, text="Browse",
            command=self._browse_mmax_file).grid(row=1, column=2, padx=4)
        tk.Button(norm_frame, text="Clear",
            command=lambda: self.mmax_file.set("")).grid(row=1, column=3, padx=4)

        tk.Label(norm_frame, text="Plateau tolerance (%):").grid(
            row=2, column=0, sticky="e", padx=6)
        tk.Spinbox(norm_frame, from_=1, to=30, increment=1, width=5,
            textvariable=self.plateau_tolerance).grid(
            row=2, column=1, sticky="w", pady=(4,0))
        tk.Label(norm_frame,
            text="Trials within this % of peak PTP count toward Mmax average.",
            fg="grey", font=("TkDefaultFont",8)).grid(
            row=2, column=1, columnspan=3, sticky="w", padx=(60,0))

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
            save_dir    = os.path.join(deriv_root, "derivatives", sub_ses)
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
                "csp_search_start_ms":   self.csp_search_start_ms.get(),
                "csp_search_end_ms":     self.csp_search_end_ms.get(),
                "csp_min_silence_ms":    self.csp_min_silence_ms.get(),
                "csp_criterion":         self.csp_criterion.get(),
                "csp_significance":      self.csp_significance.get(),
                "csp_min_return_ms":     self.csp_min_return_ms.get(),
                "csp_n_boot":            self.csp_n_boot.get(),
                "csp_types":             list(self.csp_types),
            }
            session = {
                "version":          "1.0",
                "saved_at":         datetime.datetime.now().isoformat(timespec="seconds"),
                "autosaved":        True,   # flag so user knows this wasn't a manual save
                "file_path":        fp,
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
                "mmax_file":        self.mmax_file.get(),
                "plateau_tolerance":self.plateau_tolerance.get(),
                "extra_channel_indices": self.extra_channel_indices,
                "wide_window_s":    self.wide_window_s.get(),
                "derivatives_path": (self.derivatives_path.get()
                                     if hasattr(self, "derivatives_path") else ""),
                "study_metadata":   sm,
                "settings":         s,
                "segments_metadata": meta_s,
            }

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2)

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
             "enable_inspector":self.enable_inspector.get(),
             "csp_search_start_ms":self.csp_search_start_ms.get(),
             "csp_search_end_ms":self.csp_search_end_ms.get(),
             "csp_min_silence_ms":self.csp_min_silence_ms.get(),
             "csp_criterion":self.csp_criterion.get(),
             "csp_significance":self.csp_significance.get(),
             "csp_min_return_ms":self.csp_min_return_ms.get(),
             "csp_n_boot":self.csp_n_boot.get(),
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
        fp=sess.get("file_path",""); self.file_path.set(fp)
        self.marker_choice.set(sess.get("marker_choice",""))
        self.channel_idx=sess.get("channel_idx",0); self.channel_choice.set(sess.get("channel_choice",""))
        cr=sess.get("crop_ranges"); self.crop_ranges=[tuple(r) for r in cr] if cr else None
        self.crop_start=sess.get("crop_start"); self.crop_end=sess.get("crop_end")
        self.label_map=sess.get("label_map",{}); self.color_map=sess.get("color_map",{})
        self.plot_included=sess.get("plot_included",{}); self.gap_ms_map=sess.get("gap_ms_map",{})
        if hasattr(self,"derivatives_path"):
            dp=sess.get("derivatives_path","")
            if dp: self.derivatives_path.set(dp)
        sm=sess.get("study_metadata",{})
        if sm and hasattr(self,"study_metadata"):
            try:
                self.study_metadata=StudyMetadata(**{k:v for k,v in sm.items() if k in StudyMetadata.__dataclass_fields__})
            except Exception: pass
        s=sess.get("settings",{})
        _b=lambda k,d:bool(s.get(k,d)); _i=lambda k,d:int(s.get(k,d))
        _f=lambda k,d:float(s.get(k,d)); _s=lambda k,d:str(s.get(k,d))
        self.pre_time.set(_i("pre_ms",20)); self.post_time.set(_i("post_ms",400))
        self.ptp_start.set(_i("ptp_start",10)); self.ptp_end.set(_i("ptp_end",50))
        self.prestim_ms.set(_i("prestim_ms",100)); self.apply_filter.set(_b("apply_filter",True))
        self.apply_bandpass.set(_b("apply_bandpass",True)); self.apply_notch.set(_b("apply_notch",False))
        self.highpass.set(_i("highpass",20)); self.lowpass.set(_i("lowpass",450))
        self.notch_freq.set(_i("notch_freq",50)); self.notch_q.set(_i("notch_q",30))
        self.filter_order.set(_i("filter_order",2)); self.filter_family.set(_s("filter_family","butter"))
        self.cheby_ripple.set(_f("cheby_ripple",1.0)); self.use_advanced_bp.set(_b("use_advanced_bp",False))
        self.hp_order_var.set(_i("hp_order",2)); self.lp_order_var.set(_i("lp_order",2))
        self.filter_harmonics.set(_b("filter_harmonics",False)); self.apply_humbug.set(_b("apply_humbug",False))
        self.humbug_harmonics.set(_i("humbug_harmonics",6)); self.outlier_review.set(_b("outlier_review",True))
        self.outlier_threshold.set(_f("outlier_threshold",1.96))
        self.onset_peak_fraction.set(_f("onset_peak_fraction",0.15))
        self.onset_min_amplitude.set(_f("onset_min_amplitude",0.1))
        self.onset_slope_threshold.set(_f("onset_slope_threshold",0.08))
        self.onset_method.set(sess.get("settings",{}).get("onset_method","bootstrap"))
        self.onset_bootstrap_crit.set(_f("onset_bootstrap_crit",1.96))
        self.onset_bootstrap_n.set(int(_f("onset_bootstrap_n",500)))
        self.enable_inspector.set(_b("enable_inspector",True))
        self.csp_search_start_ms.set(_i("csp_search_start_ms",40))
        self.csp_search_end_ms.set(_i("csp_search_end_ms",400))
        self.csp_min_silence_ms.set(_i("csp_min_silence_ms",25))
        self.csp_criterion.set(_f("csp_criterion",1.96))
        self.csp_significance.set(_f("csp_significance",0.99))
        self.csp_min_return_ms.set(_i("csp_min_return_ms",40))
        self.csp_n_boot.set(_i("csp_n_boot",1000))
        self.csp_types = set(sess.get("csp_types", []))
        try: self.toggle_bandpass_fields(); self.toggle_bp_order_fields(); self.toggle_notch_fields(); self._toggle_humbug_fields()
        except Exception: pass
        restored={}
        for ks,m in sess.get("segments_metadata",{}).items():
            try:
                st,i_s=ks.rsplit(":",1); restored[(st,int(i_s))]=m
            except ValueError: continue
        self.segments_metadata=restored
        self.log(f"\U0001f4c2 Loaded from {os.path.basename(lp)}\n"
                 f"   File: {os.path.basename(fp) if fp else '(none)'}\n"
                 f"   Labels: {len(self.label_map)}  Inspector edits: {len(self.segments_metadata)}\n"
                 f"   \u2705 Click Run Analysis to re-process.")
        if fp and not os.path.isfile(fp):
            messagebox.showwarning("File not found",f"Session references:\n  {fp}\n\nUse Browse to locate it.",parent=self.root)


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

        fig, ax = plt.subplots(figsize=(9, 3))
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Plot the full trace + stim ticks (identical to old code) ────────────
        ax.plot(t, emg, lw=0.4, color="0.3")

        # 2️⃣  ★★ NEW – draw DigMark ticks + labels ★★
        palette = plt.get_cmap("tab10").colors                 # 10 nice colours
        col_for = {k: palette[i % len(palette)]
                for i, k in enumerate(sorted(stim_dict))}

        y_min, y_max = emg.min(), emg.max()
        pad = 0.05 * (y_max - y_min) or 1                      # ≥1 mV head‑room
        ax.set_ylim(y_min, y_max + 3 * pad)                    # space for labels

        for s_type, times in stim_dict.items():
            col = col_for[s_type]
            for x in times:
                # coloured tick
                ax.vlines(x, y_max + 0.2 * pad, y_max + 1.0 * pad,
                        color=col, lw=1.2, zorder=4)
                # one‑letter label
                ax.text(x, y_max + 1.2 * pad, s_type,
                        ha="center", va="bottom",
                        fontsize=12, weight="bold",
                        color=col, zorder=5)

        # 3️⃣  Axis labels (your original lines)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self._ylab())
        fig.tight_layout()

        # ... (leave your existing stim‑tick code unchanged) ...
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self._ylab())
        fig.tight_layout()

        # ── State holders ───────────────────────────────────────────────────────
        spans: list[tuple[float, float]] = []   # final list of (xmin,xmax)
        patches = []                            # the Rectangle artists we draw
        list_lbl = tk.StringVar()

        def _update_list_label():
            if spans:
                txt = "Selected ranges (s):  " + ",  ".join(
                    f"[{s[0]:.2f} – {s[1]:.2f}]" for s in spans)
            else:
                txt = "No ranges yet – drag on the plot."
            list_lbl.set(txt)

        # ── SpanSelector callback ───────────────────────────────────────────────
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

        # ── Control buttons & info line ─────────────────────────────────────────
        info = tk.Label(top, textvariable=list_lbl, anchor="w")
        info.pack(fill="x", padx=10, pady=(6, 2))
        _update_list_label()

        btn_frm = tk.Frame(top);  btn_frm.pack(pady=8)

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
        """Forget everything that belongs to the *previous* file/run."""
        # ── 1. Remove attributes that cache raw data or preview info ───────────
        for attr in ('raw_emg', 'prev_fs', 'last_times', 'last_stim'):
            if hasattr(self, attr):
                delattr(self, attr)

        # ── 2. Clear user-editable maps & selections ───────────────────────────
        self.label_map.clear()
        self.color_map.clear()
        self.plot_included.clear()
        self.reference_map.clear()
        self.mmax_file.set("")
        self.extra_channel_indices = []
        self.csp_types = set()   # event types where CSP detection is ON
        self.marker_choice.set('')          # force new marker scan
        self.crop_start = None
        self.crop_end = None

        # ── 3. Reset GUI widgets ───────────────────────────────────────────────
        self.progress.set(0)
        self.log_box.delete('1.0', tk.END)

        # ── 4. Close any still-open matplotlib figures (saves RAM) ─────────────
        # Deferred via after() so Tk-embedded canvases are not destroyed
        # mid-event, which causes Tcl_AsyncDelete crashes on Windows.
        def _deferred_close():
            import matplotlib.pyplot as _plt
            _plt.close('all')
        self.root.after(100, _deferred_close)


    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 – Group Analysis
    # ══════════════════════════════════════════════════════════════════════════

    def _open_preferences(self):
        """Open the preferences dialog."""
        import tkinter.ttk as _ttk
        win = tk.Toplevel(self.root)
        win.title("Preferences")
        win.resizable(False, False)
        win.transient(self.root)

        # Current value
        _scale_var = tk.DoubleVar(value=prefs.font_scale)

        tk.Label(win, text="UI & Font Scale", font=("TkDefaultFont", 10, "bold"))\
            .grid(row=0, column=0, columnspan=3, padx=16, pady=(14,4), sticky="w")
        tk.Label(win, text="Smaller", fg="grey").grid(row=1, column=0, padx=(16,4))
        _slider = _ttk.Scale(win, from_=0.7, to=1.5, variable=_scale_var,
                             orient="horizontal", length=220)
        _slider.grid(row=1, column=1, padx=4)
        tk.Label(win, text="Larger", fg="grey").grid(row=1, column=2, padx=(4,16))

        _pct_lbl = tk.Label(win, text=f"{int(prefs.font_scale*100)}%")
        _pct_lbl.grid(row=2, column=1, pady=(2,8))

        def _on_slide(*_):
            _pct_lbl.config(text=f"{int(_scale_var.get()*100)}%")
        _scale_var.trace_add("write", _on_slide)

        tk.Label(win, text="Affects fonts, buttons, padding and window sizes.",
                 fg="grey", font=("TkDefaultFont", 8))\
            .grid(row=3, column=0, columnspan=3, padx=16, pady=(0,10))

        btn_row = tk.Frame(win); btn_row.grid(row=4, column=0, columnspan=3, pady=(0,12))

        def _apply():
            prefs.set_font_scale(_scale_var.get())
            apply_scaling(self.root)
            self.root.after(0, self._make_window_adaptive)
            win.destroy()

        def _reset():
            _scale_var.set(1.0)
            _pct_lbl.config(text="100%")

        tk.Button(btn_row, text="Apply",          width=10, command=_apply)\
            .pack(side="left", padx=6)
        tk.Button(btn_row, text="Reset to 100%",  width=12, command=_reset)\
            .pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel",         width=10, command=win.destroy)\
            .pack(side="left", padx=6)

        win.update_idletasks()
        _cx = self.root.winfo_rootx() + (self.root.winfo_width()  - win.winfo_width())  // 2
        _cy = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{_cx}+{_cy}")
        win.grab_set()

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

    def _browse_mmax_file(self):
        """Browse for an external M-wave reference file."""
        path = filedialog.askopenfilename(
            title="Select M-wave reference file",
            filetypes=[("Spike2 export", "*.txt")],
            parent=self.root)
        if path:
            self.mmax_file.set(path)
            self.log(f"📐 Mmax reference file: {os.path.basename(path)}")

    def browse_file(self):
        fpath = filedialog.askopenfilename(
            title="Select a Spike2 .txt file",
            filetypes=[("Spike2 export", "*.txt")]
        )
        if not fpath:
            return
        self._reset_state_for_new_file()
        # Remember the chosen file
        self.file_path.set(fpath)
        self.log(f"📄 Selected file: {os.path.basename(fpath)}")
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
        # Map each stim type to all its timestamps (seconds) across the whole file.
        # We filter to the crop range later, after the user has selected it.
        stim_events: dict[str, list[float]] = {}
        stim_pattern = re.compile(r'^([\d.]+)\s+"(.{1})\?\?\?"')

        # ── scan the single file
        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()
                # Detect marker sources
                for i in range(len(lines)):
                    if lines[i].strip().startswith('"Marker"') and i + 2 < len(lines):
                        m = lines[i + 2].strip().strip('"')
                        if m:
                            marker_set.add(m)
                # Detect stim types + timestamps
                for line in lines:
                    m = stim_pattern.match(line.strip())
                    if m:
                        t_s = float(m.group(1))
                        stype = m.group(2)
                        stim_events.setdefault(stype, []).append(t_s)

        except Exception as e:
            self.log(f"❌ Error reading {os.path.basename(fpath)}: {e}")
            return
        
        # ── prompt for marker choice (if >1)
        if len(marker_set) > 1:
            self.prompt_marker_choice(sorted(marker_set))
        elif marker_set:
            self.marker_choice.set(next(iter(marker_set)))
        
        # ── prompt for channel choice (if >1)
        chan_list = list_waveform_channels(fpath)
        if len(chan_list) > 1:
            self.prompt_channel_choice(chan_list)
        else:
            self.channel_choice.set(chan_list[0])
            self.channel_idx = 0

        # All channels loaded automatically — user selects in inspector dropdown
        self.extra_channel_indices = list(range(len(chan_list)))

         # --------------------------------------------------------------------------
        # Ask whether to analyse the whole file
        whole = messagebox.askyesno(
                    "Analyse whole recording?",
                    "Analyse the entire file?\n"
                    "Choose ‘No’ to pick a start- and end-point interactively.",
                    parent=self.root)
        if not whole:
            # The helper below shows a plot where the user drags over
            # the region of interest.
            if not self._crop_selector(fpath):
                # user cancelled → abort further set-up
                return
        # --------------------------------------------------------------------------

        # ── Filter stim types to those with at least one event in the selected range
        if self.crop_ranges:
            # Multi-range crop: keep types that appear in ANY of the selected spans
            stim_types_found = {
                stype for stype, times in stim_events.items()
                if any(start <= t <= end
                       for t in times
                       for start, end in self.crop_ranges)
            }
        elif self.crop_start is not None and self.crop_end is not None:
            # Single-range crop
            stim_types_found = {
                stype for stype, times in stim_events.items()
                if any(self.crop_start <= t <= self.crop_end for t in times)
            }
        else:
            # Whole file: include everything
            stim_types_found = set(stim_events.keys())

        # ── prompt for study metadata (BIDS)
        self.prompt_study_metadata()

        # ── prompt for custom labels / colours
        if stim_types_found:
            self.prompt_for_custom_labels(sorted(stim_types_found))

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

    def prompt_study_metadata(self):
        """
        Modal dialog to collect BIDS-style metadata.
        Fields are auto-populated from the filename, then overlaid with
        remembered settings if the user ticked "Remember these settings".
        """
        parsed = self._parse_bids_from_filename(self.file_path.get())
        # participant_id and session are FILE-SPECIFIC — always use the
        # parsed filename values so loading a new file auto-updates them.
        # task and timepoint are STUDY-WIDE — carry them over from
        # remembered / previous metadata as a convenience default.
        carry = self._remembered_meta or self.study_metadata
        v_sub     = tk.StringVar(value=parsed['participant_id'] or carry.participant_id)
        v_ses     = tk.StringVar(value=parsed['session']        or carry.session or 'ses-01')
        v_task    = tk.StringVar(value=parsed['task']           or carry.task)
        v_tp      = tk.StringVar(value=parsed['timepoint']      or carry.timepoint)
        v_limb    = tk.StringVar(value=parsed['limb']           or getattr(carry, 'limb', ''))
        v_measure = tk.StringVar(value=parsed['measure']        or getattr(carry, 'measure', ''))
        v_rem     = tk.BooleanVar(value=self._remembered_meta is not None)

        win = tk.Toplevel(self.root)
        win.title("Study Metadata (BIDS)")
        win.resizable(False, False)
        win.transient(self.root)

        pad = dict(padx=10, pady=4)

        tk.Label(win, text="Enter study metadata for BIDS-style output naming.",
                 font=("TkDefaultFont", 9, "italic")).grid(
                 row=0, column=0, columnspan=3, **pad, sticky="w")

        # Helper to add a labelled row
        def _row(r, label, var, example):
            tk.Label(win, text=label).grid(row=r, column=0, sticky="e", **pad)
            tk.Entry(win, textvariable=var, width=22).grid(row=r, column=1, sticky="w", **pad)
            tk.Label(win, text=example, fg="grey", font=("TkDefaultFont", 8))              .grid(row=r, column=2, sticky="w", padx=(0, 10))

        _row(1, "Participant ID *",  v_sub,  "e.g.  sub-JD001  or  JD001")
        _row(2, "Session",           v_ses,  "e.g.  ses-01  (default: ses-01)")
        _row(3, "Limb",              v_limb, "e.g.  left / right  (auto-detected)")
        _row(4, "Task label",        v_task, "e.g.  fatigue  (optional)")
        _row(5, "Timepoint",         v_tp,   "e.g.  pre / post  (optional)")

        # Measure type — dropdown of common TMS paradigms
        tk.Label(win, text="Measure type").grid(row=6, column=0, sticky="e", **pad)
        measure_frame = tk.Frame(win)
        measure_frame.grid(row=6, column=1, columnspan=2, sticky="w")
        _measure_choices = ['CSE', 'SICI', 'ICF', 'LICI', 'SAI', 'LAI', 'M-wave', 'CMEP', 'Other']
        measure_cb = ttk.Combobox(measure_frame, textvariable=v_measure,
                                  values=_measure_choices, width=10)
        measure_cb.pack(side="left")
        tk.Label(measure_frame, text="or type your own",
                 fg="grey", font=("TkDefaultFont", 8)).pack(side="left", padx=(6,0))

        tk.Checkbutton(win, text="Remember these settings for the next file",
                       variable=v_rem)          .grid(row=7, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 2))

        err_lbl = tk.Label(win, text="", fg="red")
        err_lbl.grid(row=8, column=0, columnspan=3, sticky="w", padx=10)

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
        btn_row.grid(row=9, column=0, columnspan=3, pady=10)
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
        if folder:
            self.derivatives_path.set(folder)
            self.log(f"📁 Derivatives folder: {folder}")

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

    def prompt_channel_choice(self, channel_names):
        win = tk.Toplevel(self.root)
        win.title("Select EMG channel")
        tk.Label(win, text="Choose waveform channel to analyse:")\
           .pack(padx=10, pady=(10, 4))

        self.channel_choice.set(channel_names[0])
        tk.OptionMenu(win, self.channel_choice, *channel_names)\
           .pack(padx=10, pady=4)

        tk.Button(win, text="OK", command=win.destroy)\
           .pack(pady=(0, 10))

        self._cap_toplevel(win, frac_h=0.5, frac_w=0.5)
        self.root.wait_window(win)   # modal
        # remember the index so we don’t have to rescan later
        self.channel_idx = channel_names.index(self.channel_choice.get())

    def prompt_for_custom_labels(self, stim_types):
        """
        Ask the user for:
        • a pretty label                    (free-text)
        • a colour                          (dropdown)
        • include / exclude from combo plot (checkbox)
        • gap (ms) to omit before stim      (float, 0 = no gap)
        • detect CSP for this type          (checkbox)

        Saves:
            self.label_map     {letter: str}
            self.color_map     {letter: str}
            self.plot_included {letter: bool}
            self.gap_ms_map    {letter: float}
            self.csp_types     {letter}  (set of types with CSP enabled)
        """
        # ───────────────────────── window / layout ──────────────────────────
        win = tk.Toplevel(self.root)
        win.title("Custom Stim Labels & Pre-Stim Gap for Baseline Analysis")
        win.transient(self.root)
        win.resizable(True, True)

        tk.Label(win, text="Edit labels, colours and analysis gaps (ms):")\
            .grid(row=0, column=0, columnspan=6, pady=(10, 6))

        hdr = ["Stim", "Label", "Colour", "Include in combined", "Gap (ms)", "Detect CSP", "Normalisation"]
        for c, txt in enumerate(hdr):
            tk.Label(win, text=txt, font=("TkDefaultFont", 9, "bold"))\
                .grid(row=1, column=c, padx=6, sticky="w")

        # ───────────────────────── widgets per stimulus ─────────────────────
        colour_choices = [
                # Top 5: very high neighbor contrast
                "darkgreen",     # #006400
                "deeppink",      # #FF1493
                "gold",          # #FFD700
                "black",         # #000000
                "deepskyblue",   # #00BFFF
                "maroon",        # #800000
                "springgreen",   # #00FF7F
                "mediumvioletred", # #C71585
                "seagreen",      # #2E8B57
                "hotpink",       # #FF69B4
                "turquoise",     # #40E0D0
                "navy",          # #000080
                "orange",        # #FFA500
                "indigo",        # #4B0082
                "darkorange",    # #FF8C00
                "midnightblue",  # #191970
                "saddlebrown",   # #8B4513
                "blue",          # #0000FF
                "darkred",       # #8B0000
                "royalblue",     # #4169E1
                "firebrick",     # #B22222
                "darkslategray", # #2F4F4F
                "brown",         # #A52A2A
                "slateblue",     # #6A5ACD
                "purple",        # #800080
            ]

        entry_label, entry_colour, entry_include, entry_gap, entry_csp, entry_ref = {}, {}, {}, {}, {}, {}

        for r, stim in enumerate(sorted(stim_types), start=2):
            tk.Label(win, text=f"{stim}:").grid(row=r, column=0, sticky="e", padx=(8,2))

            v_lbl = tk.StringVar(value=stim)
            tk.Entry(win, textvariable=v_lbl, width=18)\
                .grid(row=r, column=1, padx=4, sticky="w")
            entry_label[stim] = v_lbl

            v_col = tk.StringVar(value=colour_choices[(r-2) % len(colour_choices)])
            tk.OptionMenu(win, v_col, *colour_choices)\
                .grid(row=r, column=2, padx=4, sticky="w")
            entry_colour[stim] = v_col

            v_inc = tk.BooleanVar(value=True)
            tk.Checkbutton(win, variable=v_inc)\
                .grid(row=r, column=3, padx=10, sticky="w")
            entry_include[stim] = v_inc

            v_gap = tk.DoubleVar(value=0.0)
            tk.Entry(win, textvariable=v_gap, width=6)\
                .grid(row=r, column=4, padx=4, sticky="w")
            entry_gap[stim] = v_gap

            v_csp = tk.BooleanVar(value=False)
            tk.Checkbutton(win, variable=v_csp)\
                .grid(row=r, column=5, padx=10, sticky="w")
            entry_csp[stim] = v_csp

            # Reference dropdown — populated after all rows are created
            v_ref = tk.StringVar(value=self.reference_map.get(stim, "None"))
            ref_cb = ttk.Combobox(win, textvariable=v_ref, width=28, state="readonly")
            ref_cb.grid(row=r, column=6, padx=6, sticky="w")
            entry_ref[stim] = (v_ref, ref_cb)

        # Populate reference dropdowns now that all stim labels are known
        def _build_ref_options():
            for stim, (v_ref, ref_cb) in entry_ref.items():
                others = [s for s in sorted(stim_types) if s != stim]
                options = ["None"] + [
                    f"Normalise to {s}  ({entry_label[s].get() or s})"
                    for s in others
                ]
                ref_cb["values"] = options
                # Restore previous selection if valid
                cur = v_ref.get()
                if cur not in options:
                    v_ref.set("None")
        _build_ref_options()
        # Re-build when any label changes (so SP ref names stay current)
        for v_lbl in entry_label.values():
            v_lbl.trace_add("write", lambda *_: _build_ref_options())

        # ───────────────────────── save & close ─────────────────────────────
        def _save(_e=None):
            self.label_map     = {k: (v.get().strip() or k) for k, v in entry_label.items()}
            self.color_map     = {k: v.get()               for k, v in entry_colour.items()}
            self.plot_included = {k: v.get()               for k, v in entry_include.items()}
            self.gap_ms_map    = {k: float(v.get() or 0.)  for k, v in entry_gap.items()}
            self.csp_types     = {k for k, v in entry_csp.items() if v.get()}
            # Build reference_map — {stim: ref_stim_letter}
            self.reference_map = {}
            for k, (v_ref, _) in entry_ref.items():
                sel = v_ref.get()
                if sel and sel != "None" and sel.startswith("Normalise to "):
                    # "Normalise to X  (label)" → extract X
                    ref_letter = sel.split("to ")[1].strip().split(" ")[0]
                    self.reference_map[k] = ref_letter
            win.destroy()
            self.log("🔍 Data loaded – ready to apply filter settings and start analysis…\n")

        row_end = len(stim_types) + 2
        tk.Button(win, text="OK", width=10, command=_save)\
            .grid(row=row_end, column=0, columnspan=5, pady=12)

        # Convenience bindings
        win.bind("<Return>", _save)
        win.bind("<Escape>", lambda _e: win.destroy())

        # ───────────────────────── center the dialog ─────────────────────────
        # Do this AFTER widgets are laid out so sizes are accurate.
        win.update_idletasks()
        # Center relative to the main window (fallback to screen if needed)
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            w, h   = win.winfo_width(), win.winfo_height()
            x = px + (pw - w) // 2
            y = py + (ph - h) // 2
        except Exception:
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            w, h   = win.winfo_width(), win.winfo_height()
            x = (sw - w) // 2
            y = (sh - h) // 2
        win.geometry(f"+{x}+{y}")
        win.lift(self.root)
        self._cap_toplevel(win)
        win.grab_set()                     # modal
        self.root.wait_window(win)