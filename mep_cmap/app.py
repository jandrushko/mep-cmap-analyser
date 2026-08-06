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
import copy
import pathlib
import re as _re
import copy
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
from .bidsify_tab import BidsifyTabMixin
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
from .normalisation import EXCLUDED_DECISIONS
from .preferences    import prefs, apply_scaling, accent_button_kw
from .stage2         import Stage2Mixin
from .filter_preview import FilterPreviewMixin

def _make_bids_prefix(meta_prefix: str, file_stem: str) -> str:
    """Build a unique, clean BIDS prefix from metadata and source file stem.

    Strategy
    --------
    1. No metadata → use file stem as-is.
    2. File stem is a substring of the metadata prefix → use prefix as-is.
    3. Strip universally redundant tokens from the stem:
         - sub-XX / ses-XX  (always encoded in the directory path)
         - bare noise words: "session", "data", "raw"
         - "emg" when the stem is BIDS-originated (starts with "sub-")
    4. Keep only tokens not already present in the metadata prefix
       (token-level exact match — avoids false positives like "01" matching
       inside "ses-01").
    5. No novel tokens → return prefix as-is.
       Novel tokens exist → append them.
    """
    if not meta_prefix:
        return file_stem
    if file_stem in meta_prefix:
        return meta_prefix

    _is_bids_stem = bool(_re.match(r"^sub-", file_stem, _re.I))
    _NOISE = {"session", "data", "raw"}
    if _is_bids_stem:
        _NOISE.add("emg")

    meta_tokens = set(meta_prefix.split("_"))
    stem_tokens = [t for t in file_stem.split("_")
                   if not _re.match(r"^(sub|ses)-", t, _re.I)
                   and t.lower() not in _NOISE]

    novel = [t for t in stem_tokens if t not in meta_tokens]

    if not novel:
        return meta_prefix
    return f"{meta_prefix}_{'_'.join(novel)}"


class TMSAnalysisApp(Stage2Mixin, FilterPreviewMixin, BidsifyTabMixin):
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
        self._bidsify_state      = None
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

                elif msg == "bidsify-convert-done":
                    self._bidsify_convert_done(payload[0])

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
        _nb_style.configure("Centered.TNotebook.Tab", anchor="center", padding=[22, 7])
        self.notebook.configure(style="Centered.TNotebook")
        # Sub-level notebooks (Setup ▸ …, Stage 1 ▸ …): visually secondary —
        # tighter tabs and a tinted strip so the two tab rows read as different
        # levels. (Tab background colours honour the theme; the size contrast is
        # the reliable cue on native Windows themes.)
        _nb_style.configure("Sub.TNotebook", background="#c9d3de")
        _nb_style.configure("Sub.TNotebook.Tab", padding=[12, 2])
        _nb_style.map("Sub.TNotebook.Tab",
                      background=[("selected", "#f4f7fa"), ("!selected", "#c9d3de")],
                      foreground=[("selected", "#123a5e"), ("!selected", "#333333")])

        # ── Derivatives status bar ─────────────────────────────────────────────
        # Persistent strip below tabs: red when unset, green when set.
        # Clicking it opens the folder browser directly.
        self._deriv_status_bar = tk.Label(
            self.root,
            text="⚠  Derivatives folder not set — File → Set Derivatives Folder",
            **accent_button_kw("red"),
            anchor="w", padx=10, pady=3,
            font="TkDefaultFont")
        self._deriv_status_bar.pack(fill="x")
        self._deriv_status_bar.bind(
            "<Button-1>", lambda e: self.browse_derivatives_folder())

        # ══ Top-level notebook: Setup | Stage 1: Single File | Stage 2: Group Level
        # self.notebook (created above) is now the TOP notebook holding three
        # groups. "Setup" and "Stage 1" each contain a sub-notebook; "Stage 2"
        # holds its content directly.

        # ── "Setup" group ──────────────────────────────────────
        self.setup_outer = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_outer, text="Setup")
        self.nb_setup = ttk.Notebook(self.setup_outer, style="Sub.TNotebook")
        self.nb_setup.pack(fill="both", expand=True)

        self.tab_session = ttk.Frame(self.nb_setup)
        self.nb_setup.add(self.tab_session, text="Dataset")
        self._build_session_tab(self.tab_session)

        self.tab_bidsify = ttk.Frame(self.nb_setup)
        self.nb_setup.add(self.tab_bidsify, text="BIDS-ify")
        self._build_bidsify_tab(self.tab_bidsify)

        # ── "Stage 1: Single File" group ──────────────────────────
        self.stage1_outer = ttk.Frame(self.notebook)
        self.notebook.add(self.stage1_outer, text="First Level: Single File")
        # Persistent header — active file / channel / marker, visible on every
        # Stage 1 sub-tab (populated by setup_gui()).
        self._stage1_header = ttk.Frame(self.stage1_outer)
        self._stage1_header.pack(side="top", fill="x")
        ttk.Separator(self.stage1_outer, orient="horizontal").pack(side="top", fill="x")
        self.nb_stage1 = ttk.Notebook(self.stage1_outer, style="Sub.TNotebook")
        self.nb_stage1.pack(fill="both", expand=True)

        # Stage 1a — Labels & Analysis Setup
        self.tab1b_frame = ttk.Frame(self.nb_stage1)
        self.nb_stage1.add(self.tab1b_frame, text="1a – Labels & Analysis Setup")
        self._labels_tab_built = False
        self._labels_tab_confirmed = False

        # Stage 1b — Data Filtering  (Filter Settings; body populated by setup_gui)
        self.tab_filter = ttk.Frame(self.nb_stage1)
        self.nb_stage1.add(self.tab_filter, text="1b – Data Filtering")
        self._filter_body, self.canvas_filter = self._make_scroll_body(self.tab_filter)

        # Stage 1c — Feature Detection Setup  (detection/analysis + Run footer)
        self.tab_detect = ttk.Frame(self.nb_stage1)
        self.nb_stage1.add(self.tab_detect, text="1c – Feature Detection Setup")
        # Fixed footer (▶ Run Analysis) pinned at the bottom of 1c
        self.footer_frame = tk.Frame(self.tab_detect, bd=1, relief="raised")
        self.footer_frame.pack(side="bottom", fill="x")
        self._detect_body, self.canvas_detect = self._make_scroll_body(self.tab_detect)

        # setup_gui() builds the Filter section into the 1b body, then swaps
        # self.main_frame to the 1c body for the detection/analysis sections + Run.
        self.main_frame = self._filter_body

        def _on_mousewheel(event):
            # Scroll whichever processing body (1b or 1c) is currently on screen.
            for _cv in (self.canvas_filter, self.canvas_detect):
                if _cv.winfo_ismapped():
                    delta = event.delta if event.delta else (-120 if event.num == 5 else 120)
                    _cv.yview_scroll(int(-delta / 120), "units")
                    break
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.tab_detect.bind_all(seq, _on_mousewheel)

        # Stage 1c — Normalisation (optional)
        self.tab1c_frame = ttk.Frame(self.nb_stage1)
        self.nb_stage1.add(self.tab1c_frame, text="1d – Normalisation (optional)")
        self._build_normalisation_tab()

        # Add-ons
        self.tab_addons = ttk.Frame(self.nb_stage1)
        self.nb_stage1.add(self.tab_addons, text="Add-ons")
        self._build_addons_tab(self.tab_addons)

        # ── "Stage 2: Group Level" group (no sub-tabs) ─────────────────
        self.stage2_outer = ttk.Frame(self.notebook)
        self.notebook.add(self.stage2_outer, text="Second Level: Group")
        self.nb_stage2 = ttk.Notebook(self.stage2_outer, style="Sub.TNotebook")
        self.nb_stage2.pack(fill="both", expand=True)
        # Group Analysis (LME) sub-tab — content built lazily on first visit
        self.tab2_frame = ttk.Frame(self.nb_stage2)
        self.nb_stage2.add(self.tab2_frame, text="Group Analysis (LME)")
        self._stage2_built = False
        # Group-level (second-level) Add-ons sub-tab
        self.tab_group_addons = ttk.Frame(self.nb_stage2)
        self.nb_stage2.add(self.tab_group_addons, text="Add-ons")
        self._build_group_addons_tab(self.tab_group_addons)

        # ── Tab-change dispatch (widget-identity; bound to every notebook) ──
        # One handler fires on any tab change in any notebook and refreshes
        # whatever just became visible (Stage 2 lazy build, BIDS-ify worklist,
        # Add-ons rescan) using winfo_ismapped() — no fragile tab indices.
        for _nb in (self.notebook, self.nb_setup, self.nb_stage1, self.nb_stage2):
            _nb.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")

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
                "Please go to the '1a – Labels & Analysis Setup' tab "
                "and click  ✔ Confirm Setup  before running analysis.",
                parent=self.root)
            self.notebook.select(self.stage1_outer)
            self.nb_stage1.select(self.tab1b_frame)
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
            onset_bigoni_smooth_ms   = self.onset_bigoni_smooth_ms.get(),
            onset_bigoni_min_run_ms  = self.onset_bigoni_min_run_ms.get(),
            onset_bigoni_walkback_sd = self.onset_bigoni_walkback_sd.get(),
            latency_map           = dict(self.latency_map),
            onset_anchor          = self.onset_anchor.get(),
            onset_anchor_halfwidth_ms = self.onset_anchor_halfwidth.get(),

            # misc
            enable_inspector  = self.enable_inspector.get(),
            average_mode      = self.average_mode.get(),
            channel_idx       = self.channel_idx,
            label_map         = self.label_map.copy(),
            color_map         = self.color_map.copy(),
            plot_included     = self.plot_included.copy(),
            crop_start        = self.crop_start,
            crop_end          = self.crop_end,
            crop_ranges       = getattr(self, "crop_ranges", None),
            gap_ms_map        = self.gap_ms_map,
            # BIDS
            study_metadata    = copy.deepcopy(self.study_metadata),
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
                            extra_segs=None, wide_window_s=3.0, auto_meta=None, underlays=None):
        """GUI thread – open the Inspector, block until closed.
        pre_ms here is the analysis/extraction pre-stim (prestim_ms, e.g. 100ms).
        visible_pre_ms is the display window (pre_time, e.g. 20ms).
        """
        n = len(next(iter(segments_dict.values()))[0])
        time_axis = np.linspace(-pre_ms, post_ms, n, endpoint=False)
        _analysis_pre  = analysis_pre_ms if analysis_pre_ms is not None else pre_ms
        _visible_pre   = self.pre_time.get()  # display window only
        # Single source of truth: start from the anchored auto-onset seed, then
        # layer any previously saved edits on top so a resumed session shows the
        # final result (manual edits win per field). A fresh start (empty saved
        # metadata) therefore shows the pure auto onsets.
        _seed = {k: dict(v) for k, v in (auto_meta or {}).items()}
        for _k, _m in getattr(self, 'segments_metadata', {}).items():
            _seed.setdefault(_k, {}).update(_m)
        inspector = DataInspectorWindow(
            self.root, segments_dict, time_axis,
            # Seed = anchored auto onsets, overlaid with saved manual edits.
            metadata_dict       = _seed,
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
            onset_bigoni_smooth_ms   = self.onset_bigoni_smooth_ms.get(),
            onset_bigoni_min_run_ms  = self.onset_bigoni_min_run_ms.get(),
            onset_bigoni_walkback_sd = self.onset_bigoni_walkback_sd.get(),
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
            enable_auc          = self.enable_auc_global.get(),
            underlays           = underlays or {},
        )
        self.root.wait_window(inspector.top)
        self.segments_metadata = dict(inspector.meta)
        self._last_outlier_result = inspector.meta
        # Auto-save the session immediately so inspector edits
        # are never lost if the user forgets Save Session.
        self._autosave_session()


    def _show_inspector_cb(self, segments_dict, fs, pre_ms, post_ms,
                        unit, label_map, color_map, analysis_pre_ms=None,
                        extra_segs=None, wide_window_s=3.0, auto_meta=None, underlays=None):
        """
        Called by the worker thread.  Sends a message to the GUI thread and waits.
        Returns the inspector's metadata dict.
        """
        self.msg_q.put(("show-inspector",
                        segments_dict, fs, pre_ms, post_ms,
                        unit, label_map, color_map, analysis_pre_ms,
                        extra_segs, wide_window_s, auto_meta, underlays))
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
                onset_bigoni_smooth_ms   = params.get("onset_bigoni_smooth_ms",   0.5),
                onset_bigoni_min_run_ms  = params.get("onset_bigoni_min_run_ms",  0.5),
                onset_bigoni_walkback_sd = params.get("onset_bigoni_walkback_sd", 1.0),
                latency_map          = params.get("latency_map", {}),
                onset_anchor         = params.get("onset_anchor", False),
                onset_anchor_halfwidth_ms = params.get("onset_anchor_halfwidth_ms", 8.0),
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
                average_mode         = params["average_mode"],
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
        settings_menu.add_separator()
        settings_menu.add_command(label="Check for updates\u2026",
                                  command=self._check_for_updates)

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

    def _make_scroll_body(self, parent, max_content_w=1100):
        """Create a vertically-scrollable, width-capped, centred body frame inside
        `parent`. Returns (body_frame, canvas). Used for the 1b/1c processing bodies."""
        scroll_area = ttk.Frame(parent)
        scroll_area.pack(side="top", fill="both", expand=True)
        vscroll = ttk.Scrollbar(scroll_area, orient="vertical")
        vscroll.pack(side="right", fill="y")
        canvas = tk.Canvas(scroll_area, bd=0, highlightthickness=0,
                           yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.config(command=canvas.yview)
        body = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        def _resize(event):
            content_w = min(event.width, max_content_w)
            x = max(0, (event.width - content_w) // 2)
            canvas.itemconfigure(win, width=content_w)
            canvas.coords(win, x, 0)
        canvas.bind("<Configure>", _resize)
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        return body, canvas

    def setup_gui(self):
        # ─── Input File + Channel (single compact row) ──────────────────────
        # ── Active file indicator ─────────────────────────────────────────────
        # Active-file row lives in the persistent Stage 1 header so it stays
        # visible across all Stage 1 sub-tabs (not inside a scrolled body).
        file_row = tk.Frame(self._stage1_header)
        file_row.pack(fill='x', padx=10, pady=(8, 8))
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

        # Event marker dropdown — only enabled when >1 marker type available
        tk.Label(file_row, text="  Event marker:").pack(side='left')
        self._marker_dd = ttk.Combobox(file_row, textvariable=self.marker_choice,
                                        state="disabled", width=16)
        self._marker_dd.pack(side='left', padx=(4, 0))
        self._marker_dd.bind("<<ComboboxSelected>>", lambda e: None)

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
        self.onset_peak_fraction      = tk.DoubleVar(value=prefs.onset_peak_frac)
        self.onset_min_amplitude      = tk.DoubleVar(value=prefs.onset_min_peak_amplitude)
        self.onset_slope_threshold    = tk.DoubleVar(value=prefs.onset_slope_threshold)
        self.onset_method             = tk.StringVar(value=prefs.onset_method)
        self.onset_bootstrap_crit     = tk.DoubleVar(value=prefs.onset_bootstrap_crit)
        self.onset_bootstrap_n        = tk.IntVar(value=prefs.onset_bootstrap_n)
        self.onset_bigoni_smooth_ms   = tk.DoubleVar(value=prefs.onset_bigoni_smooth_ms)
        self.onset_bigoni_min_run_ms  = tk.DoubleVar(value=prefs.onset_bigoni_min_run_ms)
        self.onset_bigoni_walkback_sd = tk.DoubleVar(value=prefs.onset_bigoni_walkback_sd)
        # Onset search-window anchoring (per-run detection config; not a global pref)
        self.onset_anchor             = tk.BooleanVar(value=False)
        self.onset_anchor_halfwidth   = tk.DoubleVar(value=8.0)
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

        # ── Confirm-and-advance button on the Data Filtering (1b) tab ─────────
        _filt_bar = tk.Frame(self.main_frame)
        _filt_bar.pack(fill="x", padx=6, pady=(12, 6))
        tk.Button(_filt_bar,
                  text="\u2714  Confirm filter settings  \u2192  Feature Detection",
                  **accent_button_kw("green"),
                  command=lambda: self.nb_stage1.select(self.tab_detect)
                  ).pack(anchor="w")

        # ── Split point: the Filter section above lives on 1b (Data Filtering);
        # everything below builds into the 1c (Feature Detection Setup) body. ──
        self.main_frame = self._detect_body

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
        tk.Label(time_frame, text="MEP Onset Detection").grid(
            row=3, column=0, columnspan=4, sticky='w', padx=6)

        # ── Sub-section: Onset Detection ─────────────────────────────────────
        # Method label (read-only, reflects current preference)
        _METHOD_LABELS = {
            "peak_fraction":  "Peak Fraction",
            "bootstrap":      "Bootstrap Threshold",
            "bigoni":         "Derivative-based (Bigoni et al. 2022)",
            "bigoni_walkback":"Derivative-based + Walkback (Modified Bigoni)",
        }
        _method_display = _METHOD_LABELS.get(self.onset_method.get(), self.onset_method.get())
        self._onset_method_lbl = tk.Label(
            time_frame, text=f"Method: {_method_display}", anchor="w", fg="#444")
        self._onset_method_lbl.grid(row=4, column=0, columnspan=3, sticky='w', padx=6, pady=(2, 0))

        def _update_onset_label(*_):
            m = self.onset_method.get()
            self._onset_method_lbl.config(
                text=f"Method: {_METHOD_LABELS.get(m, m)}")

        self.onset_method.trace_add("write", _update_onset_label)

        def _open_detection_prefs():
            from .preferences import open_preferences_dialog, prefs as _prefs
            def _on_prefs_apply(r):
                self.onset_method.set(_prefs.onset_method)
                self.onset_bigoni_smooth_ms.set(_prefs.onset_bigoni_smooth_ms)
                self.onset_bigoni_min_run_ms.set(_prefs.onset_bigoni_min_run_ms)
                self.onset_bigoni_walkback_sd.set(_prefs.onset_bigoni_walkback_sd)
                self.onset_bootstrap_crit.set(_prefs.onset_bootstrap_crit)
                self.onset_bootstrap_n.set(_prefs.onset_bootstrap_n)
                self.onset_peak_fraction.set(_prefs.onset_peak_frac)
                self.onset_slope_threshold.set(_prefs.onset_slope_threshold)
                _update_onset_label()
            open_preferences_dialog(self.root, on_apply=_on_prefs_apply)

        tk.Button(
            time_frame, text="⚙ Configure in Preferences → Detection",
            command=_open_detection_prefs, cursor="hand2"
        ).grid(row=4, column=3, sticky='w', padx=6, pady=(2, 4))

        # Onset search-window anchoring (median-waveform seed)
        tk.Checkbutton(
            time_frame,
            text="Anchor onset search window to sample-median onset",
            variable=self.onset_anchor
        ).grid(row=5, column=0, columnspan=3, sticky='w', padx=6, pady=(0, 4))
        tk.Label(time_frame, text="Window ± (ms):").grid(
            row=5, column=3, sticky='e', padx=(6, 2), pady=(0, 4))
        tk.Entry(time_frame, textvariable=self.onset_anchor_halfwidth, width=5).grid(
            row=5, column=3, sticky='w', padx=(96, 6), pady=(0, 4))

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
        _csp_help = tk.Label(csp_frame,
            text="Z-score: threshold multiplier (1.96 = 95% CI)  ·  Significance: bootstrap percentile for min duration (0.99 = 99th pct)  ·  "
                 "Max offset: cSP start must fall within this many ms after the 2nd MEP peak (prevents unrealistic late placements)",
            fg="grey", justify="left", wraplength=1000)
        _csp_help.grid(row=4,column=0,columnspan=4,sticky='w',padx=6,pady=(2,0))
        # Wrap the footnote to the frame's width so it never forces the grid wider
        # than the visible area (which was clipping the right-hand column).
        csp_frame.bind("<Configure>",
                       lambda e, _l=_csp_help: _l.configure(wraplength=max(300, e.width - 24)))

        # ─── Outlier Detection ─────────────────────────────────────────────────
        out_frame = tk.LabelFrame(self.main_frame, text="Outlier Detection Settings",
                                  padx=6, pady=6)
        out_frame.pack(padx=6, pady=(10, 0), fill='x')
        tk.Checkbutton(out_frame, text="Enable Outlier Review",
                       variable=self.outlier_review).grid(row=0, column=0, sticky='w')
        tk.Label(out_frame, text="Z-score threshold:").grid(row=0, column=1, sticky='e', padx=(20,4))
        tk.Entry(out_frame, textvariable=self.outlier_threshold, width=6).grid(row=0, column=2, sticky='w')

        # ─── Analysis Options + Session + Run ─────────────────────────────────
        self.enable_inspector    = tk.BooleanVar(value=True)
        self.enable_auc_global   = tk.BooleanVar(value=True)
        self.average_mode        = tk.BooleanVar(value=False)
        run_frame = tk.LabelFrame(self.main_frame, text="Analysis Options",
                                  padx=6, pady=6)
        run_frame.pack(padx=6, pady=(10, 0), fill='x')
        tk.Checkbutton(run_frame, text="Generate individual plots per event type",
            variable=self.generate_individual_plots).grid(row=0, column=0, sticky='w', padx=4)
        tk.Checkbutton(run_frame, text="Enable Data Inspector",
            variable=self.enable_inspector).grid(row=0, column=1, sticky='w', padx=4)
        tk.Checkbutton(run_frame, text="Compute AUC",
            variable=self.enable_auc_global).grid(row=1, column=0, sticky='w', padx=4)
        tk.Checkbutton(run_frame, text="Analyse average waveform per event type",
            variable=self.average_mode).grid(row=1, column=1, sticky='w', padx=4)

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
            _file_stem   = pathlib.Path(fp).stem if fp else "mep_cmap"
            _meta_prefix = meta.bids_prefix() if meta else ""
            # Delegate to the module-level helper so BIDS-named source files
            # never produce redundant prefix tokens in the session JSON name.
            bids_prefix = _make_bids_prefix(_meta_prefix, _file_stem)

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
                "onset_bigoni_smooth_ms":   self.onset_bigoni_smooth_ms.get(),
                "onset_bigoni_min_run_ms":  self.onset_bigoni_min_run_ms.get(),
                "onset_bigoni_walkback_sd": self.onset_bigoni_walkback_sd.get(),
                "onset_anchor":          self.onset_anchor.get(),
                "onset_anchor_halfwidth": self.onset_anchor_halfwidth.get(),
                "enable_inspector":      self.enable_inspector.get(),
                "average_mode":          self.average_mode.get(),
                "generate_individual_plots": self.generate_individual_plots.get(),
                "enable_auc_global":     self.enable_auc_global.get(),
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
             "enable_inspector":self.enable_inspector.get(),"average_mode":self.average_mode.get(),"generate_individual_plots":self.generate_individual_plots.get(),
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

    def _apply_loaded_session(self, sess: dict, json_path: str = "", preserve_file_path: bool = False):
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
        if not preserve_file_path:
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
                self.onset_bigoni_smooth_ms.set(_f("onset_bigoni_smooth_ms",0.5))
                self.onset_bigoni_min_run_ms.set(_f("onset_bigoni_min_run_ms",0.5))
                self.onset_bigoni_walkback_sd.set(_f("onset_bigoni_walkback_sd",1.0))
                self.onset_anchor.set(_b("onset_anchor",False))
                self.onset_anchor_halfwidth.set(_f("onset_anchor_halfwidth",8.0))
                self.onset_peak_fraction.set(_f("onset_peak_fraction",0.15))
                self.onset_min_amplitude.set(_f("onset_min_amplitude",0.1))
                self.onset_slope_threshold.set(_f("onset_slope_threshold",0.08))
                self.enable_inspector.set(_b("enable_inspector",True))
                self.average_mode.set(_b("average_mode",False))
                self.generate_individual_plots.set(_b("generate_individual_plots",True))
                self.enable_auc_global.set(_b("enable_auc_global",True))
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

    # ──────────────────────────────────────────────────────────────────────────
    # Check for updates (GitHub Releases; manual; assisted install)
    # ──────────────────────────────────────────────────────────────────────────
    _GH_REPO = "jandrushko/mep-cmap-analyser"

    def _check_for_updates(self):
        """Query GitHub Releases in the background and report the result."""
        import threading
        threading.Thread(target=self._check_for_updates_worker, daemon=True).start()

    def _check_for_updates_worker(self):
        import json, urllib.request, urllib.error
        base = f"https://api.github.com/repos/{self._GH_REPO}"
        hdrs = {"Accept": "application/vnd.github+json",
                "User-Agent": "MEP-CMAP-Analyser"}

        def _get(path):
            req = urllib.request.Request(base + path, headers=hdrs)
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.load(r)

        # 1) Latest published Release (has notes + download page)
        try:
            data = _get("/releases/latest")
            latest = str(data.get("tag_name", "")).lstrip("vV").strip()
            notes  = (data.get("body") or "").strip()
            page   = data.get("html_url") or f"https://github.com/{self._GH_REPO}/releases"
            self.root.after(0, lambda: self._show_update_result(latest, notes, page))
            return
        except urllib.error.HTTPError as he:
            if he.code != 404:
                self.root.after(0, lambda he=he: messagebox.showinfo(
                    "Check for updates",
                    f"GitHub returned an error (HTTP {he.code}).\n\nPlease try again later.",
                    parent=self.root))
                return
            # 404 → no published Releases; fall through to git tags.
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showinfo(
                "Check for updates",
                "Couldn't reach GitHub to check for updates.\n\n"
                f"({e})\n\nCheck your internet connection and try again.",
                parent=self.root))
            return

        # 2) Fall back to version tags (repo tags releases but hasn't published one)
        try:
            tags = _get("/tags")
        except Exception:
            tags = None
        if not tags:
            self.root.after(0, lambda: messagebox.showinfo(
                "Check for updates",
                "No published releases or version tags were found for this project "
                f"on GitHub yet.\n\nYou're running version {TOOL_VERSION}.",
                parent=self.root))
            return
        best = max((str(t.get("name", "")).lstrip("vV").strip() for t in tags),
                   key=self._version_tuple, default="")
        page = f"https://github.com/{self._GH_REPO}/releases"
        self.root.after(0, lambda: self._show_update_result(best, "", page))
        return

    @staticmethod
    def _version_tuple(v):
        import re
        nums = tuple(int(x) for x in re.findall(r"\d+", v or ""))
        return nums or (0,)

    def _show_update_result(self, latest, notes, page):
        import sys, webbrowser
        cur = TOOL_VERSION
        if not latest:
            messagebox.showinfo("Check for updates",
                f"You're running version {cur}.\n\nCouldn't read the latest release "
                "version from GitHub.", parent=self.root)
            return
        if self._version_tuple(latest) <= self._version_tuple(cur):
            messagebox.showinfo("Check for updates",
                f"You're up to date.\n\nInstalled:  {cur}\nLatest:      {latest}",
                parent=self.root)
            return
        trimmed = notes if len(notes) <= 700 else notes[:700].rstrip() + "\u2026"
        head = (f"A new version is available.\n\nInstalled:  {cur}\nLatest:      {latest}\n")
        if trimmed:
            head += f"\nWhat's new:\n{trimmed}\n"
        if getattr(sys, "frozen", False):
            # Compiled .exe / .app — can't safely self-replace; open the download page.
            if messagebox.askyesno("Update available",
                    head + "\nOpen the download page to get the new build?",
                    parent=self.root):
                webbrowser.open(page)
        else:
            # pip / source install — offer an assisted pip upgrade.
            if messagebox.askyesno("Update available",
                    head + "\nUpdate now with pip? (the app must be restarted afterwards)",
                    parent=self.root):
                self._run_pip_upgrade(page)

    def _run_pip_upgrade(self, page):
        import sys, subprocess, threading, webbrowser
        pkg = "mep-cmap-analyser"
        def _worker():
            try:
                r = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", pkg],
                                   capture_output=True, text=True)
                ok = (r.returncode == 0)
                tail = (r.stderr or r.stdout or "")[-500:]
                self.root.after(0, lambda: messagebox.showinfo(
                    "Update",
                    "Update installed. Please close and reopen the app to use the new version."
                    if ok else f"pip couldn't complete the update.\n\n{tail}",
                    parent=self.root))
                if not ok:
                    self.root.after(0, lambda: webbrowser.open(page))
            except Exception as e:
                self.root.after(0, lambda e=e: messagebox.showerror(
                    "Update", f"Update failed: {e}", parent=self.root))
        threading.Thread(target=_worker, daemon=True).start()

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
    # Stage 1d — External normalisation (optional)
    # ──────────────────────────────────────────────────────────────────────────

    # ───────────────────────────────────────────────────────────────
    # Add-ons tab  (modular post-hoc analyses; see mep_cmap/addons.py)
    # ───────────────────────────────────────────────────────────────
    def _build_addons_tab(self, parent):
        """Build the Add-ons tab. Content is (re)built on selection."""
        self._addons_entries = []
        self._addons_setting_vars = {}
        for w in parent.winfo_children():
            w.destroy()
        tk.Label(parent, text="Add-ons",
                 font=("TkDefaultFont", 13, "bold")).pack(pady=(12, 2))
        tk.Label(parent,
            text="Run optional analysis add-ons on your processed results.\n"
                 "Add-ons read the saved waveform bundle (<prefix>_segments.npz)\n"
                 "and write their own new files — they never change core outputs.\n"
                 "Set your own add-ons folder in Preferences → Add-ons.",
            justify="left", fg="grey").pack(anchor="w", padx=16, pady=(0, 8))
        # Add-ons read the BIDS derivatives folder configured for the session
        # (File \u2192 Set Derivatives Folder) \u2014 not a per-run option.
        self._addons_status = tk.Label(parent, anchor="w", justify="left", fg="grey")
        self._addons_status.pack(anchor="w", padx=16, pady=(0, 4))
        _br = tk.Frame(parent); _br.pack(fill="x", padx=16, pady=(0, 4))
        tk.Button(_br, text="Rescan add-ons",
                  command=self._addons_discover).pack(side="left")
        self._addons_list_frame = tk.Frame(parent)
        self._addons_list_frame.pack(fill="both", expand=True, padx=16, pady=(8, 4))
        tk.Label(parent, text="Log:", anchor="w").pack(anchor="w", padx=16)
        self._addons_log_text = tk.Text(parent, height=8, wrap="word")
        self._addons_log_text.pack(fill="both", expand=False, padx=16, pady=(0, 12))
        self._addons_refresh_status()
        self._addons_discover()

    def _addons_refresh_status(self):
        """Show which derivatives folder add-ons will read (or prompt to set it)."""
        d = ""
        if hasattr(self, "derivatives_path"):
            d = (self.derivatives_path.get() or "").strip()
        if d:
            self._addons_status.config(
                fg="grey",
                text=f"Reading results from your derivatives folder:\n{d}")
        else:
            self._addons_status.config(
                fg="#b00020",
                text="\u26a0 No derivatives folder set \u2014 use File \u2192 Set "
                     "Derivatives Folder before running add-ons.")

    def _addons_log(self, msg):
        try:
            self._addons_log_text.insert("end", str(msg) + "\n")
            self._addons_log_text.see("end")
        except Exception:
            print(msg)

    def _addons_discover(self):
        """Scan built-in + user folders and rebuild the add-on list."""
        from . import addons as _addons
        try:
            self._addons_entries = _addons.discover_all("single_file", prefs.addons_path,
                                                        log=self._addons_log)
        except Exception as e:
            self._addons_entries = []
            self._addons_log(f"add-on discovery failed: {e}")
        for w in self._addons_list_frame.winfo_children():
            w.destroy()
        if not self._addons_entries:
            tk.Label(self._addons_list_frame,
                     text="No add-ons found. Built-in add-ons ship with the app; "
                          "add your own folder in Preferences → Add-ons.",
                     fg="grey", justify="left").pack(anchor="w")
            return
        for entry in self._addons_entries:
            row = tk.Frame(self._addons_list_frame, relief="groove", bd=1)
            row.pack(fill="x", pady=3)
            txt = tk.Frame(row); txt.pack(side="left", fill="x", expand=True, padx=6, pady=4)
            tk.Label(txt, text=entry["name"],
                     font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            _desc = entry["description"] or "(no description)"
            if entry["version"] or entry["author"]:
                _desc += f"   —  v{entry['version']}  {entry['author']}"
            tk.Label(txt, text=_desc, fg="grey", wraplength=650,
                     justify="left").pack(anchor="w")
            # Per-add-on settings (declared via ADDON_SETTINGS) render here and
            # are merged into the run config. Values persist across rescans.
            if entry.get("settings"):
                svars = self._addons_setting_vars.setdefault(entry["name"], {})
                sframe = tk.Frame(txt); sframe.pack(anchor="w", fill="x", pady=(4, 0))
                for sp in entry["settings"]:
                    key = sp.get("key")
                    if not key:
                        continue
                    if key not in svars:
                        svars[key] = tk.StringVar(value=str(sp.get("default", "")))
                    _rr = tk.Frame(sframe); _rr.pack(anchor="w", fill="x")
                    tk.Label(_rr, text=f"{sp.get('label', key)}:").pack(side="left")
                    tk.Entry(_rr, textvariable=svars[key], width=8).pack(side="left", padx=(4, 0))
                    if sp.get("help"):
                        tk.Label(sframe, text=sp["help"], fg="grey", justify="left",
                                 wraplength=600, font="TkSmallCaptionFont").pack(anchor="w")
            tk.Button(row, text="Run",
                      command=lambda e=entry: self._addons_run(e)).pack(
                          side="right", padx=8, pady=4)

    def _addons_build_config(self):
        """Analysis settings handed to add-ons as context.config."""
        def _g(attr, default=None):
            v = getattr(self, attr, None)
            try:
                return v.get() if v is not None else default
            except Exception:
                return default
        return {
            "ptp_start":  _g("ptp_start", 10),
            "ptp_end":    _g("ptp_end", 50),
            "prestim_ms": _g("prestim_ms", 100),
            "pre_ms":     _g("pre_time", 20),
            "post_ms":    _g("post_time", 400),
            # Per-condition onset (latency) windows {stim_type: (min_ms, max_ms)}.
            # MEPFeatX-style add-ons use this as t_onset; without it they fall back
            # to the wide PTP window, which makes onset/latency unreliable.
            "latency_map": dict(getattr(self, "latency_map", {}) or {}),
        }

    def _addons_run(self, entry):
        """Run one add-on on every *_segments.npz bundle under the results folder."""
        import os, glob
        from . import addons as _addons
        folder = ""
        if hasattr(self, "derivatives_path"):
            folder = (self.derivatives_path.get() or "").strip()
        if not folder or not os.path.isdir(folder):
            self._addons_log("No derivatives folder set. Use File \u2192 Set "
                             "Derivatives Folder, then run an analysis so the "
                             "waveform bundle is saved.")
            return
        bundles = sorted(glob.glob(os.path.join(folder, "**", "*_segments.npz"),
                                   recursive=True))
        if not bundles:
            self._addons_log(f"No *_segments.npz bundles under {folder}. "
                             f"Run an analysis first (the bundle is saved with results).")
            return
        cfg = self._addons_build_config()
        # merge this add-on's own GUI settings (ADDON_SETTINGS) into the config
        for sp in entry.get("settings", []):
            var = self._addons_setting_vars.get(entry["name"], {}).get(sp.get("key"))
            if var is None:
                continue
            raw = var.get()
            try:
                typ = sp.get("type", "str")
                if typ == "int":
                    cfg[sp["key"]] = int(float(raw))
                elif typ == "float":
                    cfg[sp["key"]] = float(raw)
                elif typ == "bool":
                    cfg[sp["key"]] = str(raw).strip().lower() in ("1", "true", "yes", "on")
                else:
                    cfg[sp["key"]] = raw
            except Exception:
                self._addons_log(f"add-ons: '{sp.get('key')}' value '{raw}' invalid; using add-on default.")
        self._addons_log(f"— Running '{entry['name']}' on {len(bundles)} bundle(s) —")
        n_written = 0
        for bpath in bundles:
            try:
                contexts = _addons.load_contexts(bpath, config=cfg, log=self._addons_log)
            except Exception as ex:
                self._addons_log(f"  {os.path.basename(bpath)}: could not load ({ex})")
                continue
            for ctx in contexts:
                res = _addons.run_addon(entry, ctx)
                if res["ok"]:
                    for pth in res["paths"]:
                        n_written += 1
                        self._addons_log(f"  ✓ {os.path.basename(pth)}")
                else:
                    self._addons_log(f"  ✗ {entry['name']} failed on {ctx.bids_prefix}:")
                    _err = (res["error"] or "").strip().splitlines()
                    self._addons_log(_err[-1] if _err else "unknown error")
        self._addons_log(f"Done. {n_written} file(s) written.")

    def _on_addons_tab_selected(self, event=None):
        """Deprecated: tab-change handling is now unified in _on_tab_changed
        (visibility-based, nested-notebook aware). Kept as a thin delegate."""
        self._on_tab_changed(event)

    # ──────────────────────────────────────────────────────────────────────────
    # Second-level (group) Add-ons tab
    # ──────────────────────────────────────────────────────────────────────────
    def _build_group_addons_tab(self, parent):
        """Build the second-level (group-level) Add-ons tab."""
        self._gaddons_entries = []
        self._gaddons_setting_vars = {}
        for w in parent.winfo_children():
            w.destroy()
        tk.Label(parent, text="Add-ons",
                 font=("TkDefaultFont", 13, "bold")).pack(pady=(12, 2))
        tk.Label(parent,
            text="Run optional group-level add-ons on your built group table.\n"
                 "Add-ons read the group file (group_level_LME_ready.csv) that\n"
                 "Group Analysis (LME) builds, and write their own new files \u2014\n"
                 "they never change it. Put your own group add-ons in the\n"
                 "'group_level' subfolder of your Preferences \u2192 Add-ons folder.",
            justify="left", fg="grey").pack(anchor="w", padx=16, pady=(0, 8))
        self._gaddons_status = tk.Label(parent, anchor="w", justify="left", fg="grey")
        self._gaddons_status.pack(anchor="w", padx=16, pady=(0, 4))
        _br = tk.Frame(parent); _br.pack(fill="x", padx=16, pady=(0, 4))
        tk.Button(_br, text="Rescan add-ons",
                  command=self._group_addons_discover).pack(side="left")
        self._gaddons_list_frame = tk.Frame(parent)
        self._gaddons_list_frame.pack(fill="both", expand=True, padx=16, pady=(8, 4))
        tk.Label(parent, text="Log:", anchor="w").pack(anchor="w", padx=16)
        self._gaddons_log_text = tk.Text(parent, height=8, wrap="word")
        self._gaddons_log_text.pack(fill="both", expand=False, padx=16, pady=(0, 12))
        self._group_addons_refresh_status()
        self._group_addons_discover()

    def _group_addons_refresh_status(self):
        d = ""
        if hasattr(self, "derivatives_path"):
            d = (self.derivatives_path.get() or "").strip()
        if d:
            self._gaddons_status.config(
                fg="grey",
                text=f"Reading the group table from your derivatives folder:\n{d}")
        else:
            self._gaddons_status.config(
                fg="#b00020",
                text="\u26a0 No derivatives folder set \u2014 set it (File \u2192 Set "
                     "Derivatives Folder), then build the group file in Group Analysis (LME).")

    def _group_addons_log(self, msg):
        try:
            self._gaddons_log_text.insert("end", str(msg) + "\n")
            self._gaddons_log_text.see("end")
        except Exception:
            print(msg)

    def _group_addons_discover(self):
        """Scan built-in + user 'group_level' folders and rebuild the list."""
        from . import addons as _addons
        try:
            self._gaddons_entries = _addons.discover_all("group_level", prefs.addons_path,
                                                         log=self._group_addons_log)
        except Exception as e:
            self._gaddons_entries = []
            self._group_addons_log(f"add-on discovery failed: {e}")
        for w in self._gaddons_list_frame.winfo_children():
            w.destroy()
        if not self._gaddons_entries:
            tk.Label(self._gaddons_list_frame,
                     text="No group-level add-ons found. Built-in ones ship with the app; "
                          "add your own in the 'group_level' subfolder of your add-ons folder.",
                     fg="grey", justify="left").pack(anchor="w")
            return
        for entry in self._gaddons_entries:
            row = tk.Frame(self._gaddons_list_frame, relief="groove", bd=1)
            row.pack(fill="x", pady=3)
            txt = tk.Frame(row); txt.pack(side="left", fill="x", expand=True, padx=6, pady=4)
            tk.Label(txt, text=entry["name"],
                     font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            _desc = entry["description"] or "(no description)"
            if entry["version"] or entry["author"]:
                _desc += f"   \u2014  v{entry['version']}  {entry['author']}"
            tk.Label(txt, text=_desc, fg="grey", wraplength=650,
                     justify="left").pack(anchor="w")
            if entry.get("settings"):
                svars = self._gaddons_setting_vars.setdefault(entry["name"], {})
                sframe = tk.Frame(txt); sframe.pack(anchor="w", fill="x", pady=(4, 0))
                for sp in entry["settings"]:
                    key = sp.get("key")
                    if not key:
                        continue
                    if key not in svars:
                        svars[key] = tk.StringVar(value=str(sp.get("default", "")))
                    _rr = tk.Frame(sframe); _rr.pack(anchor="w", fill="x")
                    tk.Label(_rr, text=f"{sp.get('label', key)}:").pack(side="left")
                    tk.Entry(_rr, textvariable=svars[key], width=8).pack(side="left", padx=(4, 0))
                    if sp.get("help"):
                        tk.Label(sframe, text=sp["help"], fg="grey", justify="left",
                                 wraplength=600, font="TkSmallCaptionFont").pack(anchor="w")
            tk.Button(row, text="Run",
                      command=lambda e=entry: self._group_addons_run(e)).pack(
                          side="right", padx=8, pady=4)

    def _group_addons_run(self, entry):
        """Run one group-level add-on on every group_level_LME_ready.csv found."""
        import os, glob
        from . import addons as _addons
        folder = ""
        if hasattr(self, "derivatives_path"):
            folder = (self.derivatives_path.get() or "").strip()
        if not folder or not os.path.isdir(folder):
            self._group_addons_log("No derivatives folder set. Use File \u2192 Set "
                                   "Derivatives Folder, then build the group file.")
            return
        targets = sorted(glob.glob(
            os.path.join(folder, "**", "group_level_LME_ready.csv"), recursive=True))
        if not targets:
            self._group_addons_log(
                "No group_level_LME_ready.csv found. Build it first in "
                "Second Level \u2192 Group Analysis (LME).")
            return
        # base config = merged ADDON_SETTINGS only (group add-ons need no analysis params)
        cfg = {}
        for sp in entry.get("settings", []):
            var = self._gaddons_setting_vars.get(entry["name"], {}).get(sp.get("key"))
            if var is None:
                continue
            raw = var.get()
            try:
                typ = sp.get("type", "str")
                if typ == "int":
                    cfg[sp["key"]] = int(float(raw))
                elif typ == "float":
                    cfg[sp["key"]] = float(raw)
                elif typ == "bool":
                    cfg[sp["key"]] = str(raw).strip().lower() in ("1", "true", "yes", "on")
                else:
                    cfg[sp["key"]] = raw
            except Exception:
                self._group_addons_log(f"add-ons: '{sp.get('key')}' value '{raw}' invalid; using default.")
        self._group_addons_log(f"\u2014 Running '{entry['name']}' on {len(targets)} group file(s) \u2014")
        n_written = 0
        for tpath in targets:
            try:
                contexts = _addons.load_group_contexts(tpath, config=cfg,
                                                       log=self._group_addons_log)
            except Exception as ex:
                self._group_addons_log(f"  {os.path.basename(tpath)}: could not load ({ex})")
                continue
            for ctx in contexts:
                res = _addons.run_addon(entry, ctx)
                if res["ok"]:
                    for pth in res["paths"]:
                        n_written += 1
                        self._group_addons_log(f"  \u2713 {os.path.basename(pth)}")
                else:
                    self._group_addons_log(f"  \u2717 {entry['name']} failed:")
                    _err = (res["error"] or "").strip().splitlines()
                    self._group_addons_log(_err[-1] if _err else "unknown error")
        self._group_addons_log(f"Done. {n_written} file(s) written.")

    def _build_normalisation_tab(self):
        """Build the Stage 1d normalisation tab."""
        f = self.tab1c_frame
        for w in f.winfo_children():
            w.destroy()

        tk.Label(f,
            text="Optional: normalise processed results using a reference file's PTP values.\n"
                 "Both files must be fully processed first (First Level).\n"
                 "Select the _trials.csv files from the derivatives/results folder.\n"
                 "The reference mean is computed using the same plateau detection as "
                 "internal normalisation: if the reference data has a reliable plateau "
                 "within the tolerance threshold, the plateau mean is used; "
                 "otherwise the peak value is used.\n"
                 "Results are written back into the main file's _trials.csv in-place.",
            justify="left", wraplength=950, fg="grey"
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
            fg="grey", font="TkSmallCaptionFont"
        ).pack(side="left", padx=(4, 0))

        # ── Column headers ────────────────────────────────────────────────────
        hdr = tk.Frame(f)
        hdr.pack(fill="x", padx=16, pady=(4, 0))
        tk.Label(hdr, text="File to normalise  (_trials.csv)", width=52, anchor="w")\
            .grid(row=0, column=0, padx=4, sticky="w")
        tk.Label(hdr, text="Reference file  (_trials.csv)", width=52, anchor="w")\
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
                  **accent_button_kw("green"),
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
        """Apply normalisation for all configured pairs.

        If either file has multiple stim types, shows a mapping dialog
        before applying so the user can specify which ref stim type
        normalises which main stim type.
        """
        import pandas as _pd

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
                df_main = _pd.read_csv(main_path)
                df_ref  = _pd.read_csv(ref_path)

                main_stims = sorted(df_main["StimType"].dropna().unique().tolist())                              if "StimType" in df_main.columns else []
                ref_stims  = sorted(df_ref["StimType"].dropna().unique().tolist())                              if "StimType" in df_ref.columns else []

                # Show mapping dialog when either file has multiple stim types
                if len(main_stims) > 1 or len(ref_stims) > 1:
                    stim_map = self._norm_stim_mapping_dialog(
                        main_stims or ["(all)"],
                        ref_stims  or ["(all)"],
                        os.path.basename(main_path),
                        os.path.basename(ref_path),
                    )
                    if stim_map is None:
                        results.append(f"⏭  {os.path.basename(main_path)}: cancelled")
                        continue
                else:
                    # Single stim type in both — map all to all
                    stim_map = {(main_stims[0] if main_stims else "(all)"):
                                (ref_stims[0]  if ref_stims  else "(all)")}

                msg = self._apply_normalisation_pair(
                    main_path, ref_path, stim_map=stim_map)
                results.append(msg)
            except Exception as e:
                results.append(f"❌ {os.path.basename(main_path)}: {e}")

        self._norm_log_var.set("\n".join(results))

    def _norm_stim_mapping_dialog(self, main_stims: list, ref_stims: list,
                                   main_name: str, ref_name: str) -> dict | None:
        """Show a dialog for mapping main stim types to reference stim types.

        Returns a dict {main_stim: ref_stim} or None if cancelled.
        ref_stim of None means skip normalisation for that main stim type.
        """
        result = {}
        cancelled = [False]

        dlg = tk.Toplevel(self.root)
        dlg.title("Normalisation — Stim Type Mapping")
        dlg.transient(self.root)
        dlg.resizable(True, False)
        dlg.grab_set()

        tk.Label(dlg,
            text=f"Multiple stimulus types detected. Specify which reference stim type\n"
                 f"normalises each main stim type. Select 'None' to skip a stim type.\n\n"
                 f"Main file:      {main_name}\n"
                 f"Reference file: {ref_name}",
            justify="left",
        ).pack(anchor="w", padx=16, pady=(12, 6))

        tbl = tk.Frame(dlg)
        tbl.pack(fill="x", padx=16, pady=8)

        tk.Label(tbl, text="Main stim type", width=22, anchor="w")            .grid(row=0, column=0, padx=4, pady=2, sticky="w")
        tk.Label(tbl, text="→", width=3).grid(row=0, column=1)
        tk.Label(tbl, text="Reference stim type", width=22, anchor="w")            .grid(row=0, column=2, padx=4, pady=2, sticky="w")

        ttk.Separator(tbl, orient="horizontal")            .grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)

        ref_options = ["None"] + ref_stims
        row_vars = {}
        for i, ms in enumerate(main_stims):
            tk.Label(tbl, text=ms, anchor="w", width=22)                .grid(row=i+2, column=0, padx=4, pady=3, sticky="w")
            tk.Label(tbl, text="→", width=3).grid(row=i+2, column=1)
            # Default: match by name if possible, else first ref stim
            default = ms if ms in ref_stims else (ref_stims[0] if ref_stims else "None")
            v = tk.StringVar(value=default)
            ttk.Combobox(tbl, textvariable=v, values=ref_options,
                         state="readonly", width=20)                .grid(row=i+2, column=2, padx=4, pady=3, sticky="w")
            row_vars[ms] = v

        def _apply():
            for ms, v in row_vars.items():
                chosen = v.get()
                result[ms] = None if chosen == "None" else chosen
            dlg.destroy()

        def _cancel():
            cancelled[0] = True
            dlg.destroy()

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=(4, 12))
        tk.Button(btn_row, text="Apply mapping", width=14,
                  **accent_button_kw("green"), command=_apply).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=_cancel).pack(side="left", padx=6)

        dlg.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width()  - dlg.winfo_width())  // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")
        self.root.wait_window(dlg)

        return None if cancelled[0] else result

    def _apply_normalisation_pair(self, main_csv: str, ref_csv: str,
                                   stim_map: dict | None = None) -> str:
        """Apply normalisation from ref_csv to main_csv.

        stim_map : {main_stim_type: ref_stim_type} or None.
            When None, all trials in ref_csv are used as the reference pool.
            When provided, each main stim type is normalised using only the
            corresponding ref stim type trials from ref_csv.
            A ref_stim_type of None means skip that main stim type.
        """
        import pandas as _pd
        import numpy as _np
        from .normalisation import compute_mmax as _cmmax

        df_main = _pd.read_csv(main_csv)
        df_ref  = _pd.read_csv(ref_csv)

        plateau_tol = self._norm1c_plateau.get() / 100.0
        _ptp_col   = "PTP(mV)"
        _norm_col  = "Normalised_PTP"
        _rtype_col = "Reference_Type"
        _rmean_col = "Reference_Mean(mV)"
        _rn_col    = "Reference_N"

        for col in [_norm_col, _rtype_col, _rmean_col, _rn_col,
                    "Normalised_PTP_per_PreStimRMS",
                    "Normalised_Adjusted_PTP_QR"]:
            if col not in df_main.columns:
                df_main[col] = ""
            df_main[col] = df_main[col].astype(object)

        has_stim_col = "StimType" in df_main.columns and "StimType" in df_ref.columns

        def _update_summary(summary_csv, trials_df):
            """Recompute normalisation columns in a summary CSV from updated trials data."""
            if not os.path.isfile(summary_csv):
                return
            df_sum = _pd.read_csv(summary_csv)
            for col in ["Mean_Normalised_PTP", "SD_Normalised_PTP",
                        "Reference_Type", "Reference_Mean(mV)", "Reference_N",
                        "Mean_PTP_per_PreStimRMS", "SD_PTP_per_PreStimRMS",
                        "Mean_Normalised_PTP_per_PreStimRMS", "SD_Normalised_PTP_per_PreStimRMS",
                        "Mean_Normalised_Adjusted_PTP_QR", "SD_Normalised_Adjusted_PTP_QR"]:
                if col not in df_sum.columns:
                    df_sum[col] = _np.nan
                df_sum[col] = df_sum[col].astype(object)

            clean = trials_df[~trials_df["Outlier_Decision"].isin(EXCLUDED_DECISIONS)].copy()
            for col in ["Normalised_PTP", "PTP_per_PreStimRMS",
                        "Normalised_PTP_per_PreStimRMS",
                        "Normalised_Adjusted_PTP_QR",
                        "Reference_Mean(mV)", "Reference_N"]:
                if col in clean.columns:
                    clean[col] = _pd.to_numeric(clean[col], errors='coerce')

            for idx, row in df_sum.iterrows():
                st   = row.get("StimType", "")
                grp  = clean[clean["StimType"] == st] if "StimType" in clean.columns                        else clean
                if len(grp) == 0:
                    continue
                # Normalised_PTP
                _g = grp["Normalised_PTP"].dropna() if "Normalised_PTP" in grp.columns else _pd.Series(dtype=float)
                if len(_g) > 0:
                    df_sum.at[idx, "Mean_Normalised_PTP"] = round(float(_g.mean()), 4)
                    df_sum.at[idx, "SD_Normalised_PTP"]   = round(float(_g.std(ddof=1)), 4) if len(_g) > 1 else _np.nan
                # Reference info — take first non-null value
                for _rc in ["Reference_Type", "Reference_Mean(mV)", "Reference_N"]:
                    if _rc in grp.columns:
                        _rv = grp[_rc].dropna()
                        if len(_rv) > 0:
                            df_sum.at[idx, _rc] = _rv.iloc[0]
                # PTP_per_PreStimRMS
                _g2 = grp["PTP_per_PreStimRMS"].dropna() if "PTP_per_PreStimRMS" in grp.columns else _pd.Series(dtype=float)
                if len(_g2) > 0:
                    df_sum.at[idx, "Mean_PTP_per_PreStimRMS"] = round(float(_g2.mean()), 4)
                    df_sum.at[idx, "SD_PTP_per_PreStimRMS"]   = round(float(_g2.std(ddof=1)), 4) if len(_g2) > 1 else _np.nan
                # Normalised_PTP_per_PreStimRMS
                _g3 = grp["Normalised_PTP_per_PreStimRMS"].dropna() if "Normalised_PTP_per_PreStimRMS" in grp.columns else _pd.Series(dtype=float)
                if len(_g3) > 0:
                    df_sum.at[idx, "Mean_Normalised_PTP_per_PreStimRMS"] = round(float(_g3.mean()), 4)
                    df_sum.at[idx, "SD_Normalised_PTP_per_PreStimRMS"]   = round(float(_g3.std(ddof=1)), 4) if len(_g3) > 1 else _np.nan
                # Normalised_Adjusted_PTP_QR (re-derived against the new reference)
                _g4 = grp["Normalised_Adjusted_PTP_QR"].dropna() if "Normalised_Adjusted_PTP_QR" in grp.columns else _pd.Series(dtype=float)
                if len(_g4) > 0:
                    df_sum.at[idx, "Mean_Normalised_Adjusted_PTP_QR"] = round(float(_g4.mean()), 4)
                    df_sum.at[idx, "SD_Normalised_Adjusted_PTP_QR"]   = round(float(_g4.std(ddof=1)), 4) if len(_g4) > 1 else _np.nan

            df_sum.to_csv(summary_csv, index=False)

        if stim_map and has_stim_col:
            # Per-stim-type normalisation
            msgs = []
            for main_st, ref_st in stim_map.items():
                if ref_st is None:
                    continue  # user chose to skip this stim type

                # Reference pool: clean trials of ref_st in df_ref
                ref_mask = (df_ref["StimType"] == ref_st) &                            (~df_ref["Outlier_Decision"].isin(EXCLUDED_DECISIONS))
                ref_ptps = _pd.to_numeric(
                    df_ref.loc[ref_mask, _ptp_col], errors='coerce').dropna().tolist()

                if not ref_ptps:
                    msgs.append(f"⚠️  No clean {ref_st} trials in reference")
                    continue

                result  = _cmmax(ref_ptps, plateau_tolerance=plateau_tol)
                ref_mean = result["mmax"]
                ref_n    = result["n_plateau"]
                ref_type = result["method"]

                if not ref_mean or ref_mean <= 0:
                    msgs.append(f"⚠️  Could not compute mean for {ref_st}")
                    continue

                # Apply to matching main stim type
                main_mask = (df_main["StimType"] == main_st) &                             (~df_main["Outlier_Decision"].isin(EXCLUDED_DECISIONS))
                ptps = _pd.to_numeric(df_main.loc[main_mask, _ptp_col], errors='coerce')
                df_main.loc[main_mask, _norm_col]  = (ptps / ref_mean).round(4)
                df_main.loc[main_mask, _rtype_col] = ref_type
                df_main.loc[main_mask, _rmean_col] = round(ref_mean, 4)
                df_main.loc[main_mask, _rn_col]    = ref_n
                # Normalised_PTP_per_PreStimRMS
                if "PreStimRMS" in df_main.columns:
                    _rms_vals = _pd.to_numeric(df_main.loc[main_mask, "PreStimRMS"], errors="coerce")
                    _norm_ptp_vals = _pd.to_numeric(df_main.loc[main_mask, _norm_col], errors='coerce')
                    df_main.loc[main_mask, "Normalised_PTP_per_PreStimRMS"] = \
                        (_norm_ptp_vals / _rms_vals).round(4)
                # Normalised_Adjusted_PTP_QR = Adjusted_PTP_QR / raw reference mean
                if "Adjusted_PTP_QR(mV)" in df_main.columns:
                    _adj_vals = _pd.to_numeric(df_main.loc[main_mask, "Adjusted_PTP_QR(mV)"], errors="coerce")
                    df_main.loc[main_mask, "Normalised_Adjusted_PTP_QR"] = \
                        (_adj_vals / ref_mean).round(4)
                msgs.append(f"✓  {main_st} → {ref_st} (mean {ref_mean:.3f} mV, n={ref_n})")

            df_main.to_csv(main_csv, index=False)
            # Update summary files
            _summary_csv_ps      = main_csv.replace("_trials.csv", "_summary.csv")
            _summary_with_out_ps = main_csv.replace("_trials.csv", "_summary_with_outliers.csv")
            _update_summary(_summary_csv_ps, df_main)
            # summary_with_outliers built from same trials df (all rows, clean filtering inside)
            _update_summary(_summary_with_out_ps, df_main)
            return f"✅  {os.path.basename(main_csv)}:\n" + "\n".join(f"    {m}" for m in msgs)

        else:
            # Single pool — use all clean ref trials regardless of stim type
            ref_mask = ~df_ref["Outlier_Decision"].isin(EXCLUDED_DECISIONS)
            ref_ptps = _pd.to_numeric(
                df_ref.loc[ref_mask, _ptp_col], errors='coerce').dropna().tolist()

            if not ref_ptps:
                return (f"⚠️  {os.path.basename(ref_csv)}: "
                        f"no clean trials found in reference file")

            result   = _cmmax(ref_ptps, plateau_tolerance=plateau_tol)
            ref_mean = result["mmax"]
            ref_n    = result["n_plateau"]
            ref_type = result["method"]

            if ref_mean is None or ref_mean <= 0:
                return (f"⚠️  {os.path.basename(ref_csv)}: "
                        f"could not compute reference mean")

            mask = ~df_main["Outlier_Decision"].isin(EXCLUDED_DECISIONS)
            ptps = _pd.to_numeric(df_main.loc[mask, _ptp_col], errors='coerce')
            df_main.loc[mask, _norm_col]  = (ptps / ref_mean).round(4)
            df_main.loc[mask, _rtype_col] = ref_type
            df_main.loc[mask, _rmean_col] = round(ref_mean, 4)
            df_main.loc[mask, _rn_col]    = ref_n
            # Normalised_PTP_per_PreStimRMS
            if "PreStimRMS" in df_main.columns:
                _rms_vals = _pd.to_numeric(df_main.loc[mask, "PreStimRMS"], errors="coerce")
                df_main.loc[mask, "Normalised_PTP_per_PreStimRMS"] = \
                    ((ptps / ref_mean) / _rms_vals).round(4)
            # Normalised_Adjusted_PTP_QR = Adjusted_PTP_QR / raw reference mean
            if "Adjusted_PTP_QR(mV)" in df_main.columns:
                _adj_vals = _pd.to_numeric(df_main.loc[mask, "Adjusted_PTP_QR(mV)"], errors="coerce")
                df_main.loc[mask, "Normalised_Adjusted_PTP_QR"] = \
                    (_adj_vals / ref_mean).round(4)

            df_main.to_csv(main_csv, index=False)

        # _trials_with_outliers.csv removed — _trials.csv contains all trials

        # ── Update summary files ─────────────────────────────────────────────
        _summary_csv      = main_csv.replace("_trials.csv", "_summary.csv")
        _summary_with_out = main_csv.replace("_trials.csv", "_summary_with_outliers.csv")
        _update_summary(_summary_csv, df_main)
        # summary_with_outliers uses same df (all rows, clean filtering inside _update_summary)
        _update_summary(_summary_with_out, df_main)

        return (f"✅ {os.path.basename(main_csv)}: "
                f"normalised to {ref_type} = {ref_mean:.4f} mV "
                f"(N={ref_n}, from {os.path.basename(ref_csv)})")

    def _browse_mmax_file(self):
        """Browse for an external M-wave reference file."""
        path = filedialog.askopenfilename(
            title="Select M-wave reference file",
            filetypes=[("Data files", "*.txt *.smr *.adibin *.edf *.bdf"),
                       ("All files", "*.*")],
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
        # No custom font: use the live default font exactly like the Browse
        # and Run selected buttons, so the height matches regardless of the
        # app's DPI/font scaling.
        tk.Button(study_row, text="📂  Open study folder",
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
        tk.Button(q_toolbar, text="▶  Run selected",
                  command=self._queue_run_selected).pack(side='right', padx=(0, 4))

        # ── File-load progress bar (hidden until a file is loading) ─────────
        _pb_style = ttk.Style()
        _pb_style.configure("FileLoad.Horizontal.TProgressbar",
                            thickness=8, troughcolor="#ddd", background="#2196F3")
        self._load_prog_frame = tk.Frame(queue_frame)
        self._load_prog_label = tk.Label(
            self._load_prog_frame, text="", fg="#555",
            font="TkSmallCaptionFont", anchor="w")
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
        _ctx.add_command(label="🔄  Reset & reprocess from scratch…", command=self._queue_reset_file)
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
            _ftype = {".txt": "TXT", ".smr": "SMR", ".adibin": "ADIBIN", ".edf": "EDF", ".bdf": "BDF"}.get(_ext, _ext.lstrip(".").upper() or "—")

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
                  **accent_button_kw("green")).pack(side="left", padx=6)
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

    def _queue_reset_file(self):
        """Fully reset a file — delete derivatives, sidecars, and all state.

        After reset the file is treated as if it has never been processed.
        The user can then reconfigure in Stage 1a/1b and re-run from scratch.
        """
        import pathlib
        ids = self._queue_selected_ids()
        if not ids or self._dataset is None:
            return

        fids_to_reset = []
        for fid in ids:
            fe = self._dataset.get_file(fid)
            if fe:
                fids_to_reset.append((fid, fe))

        if not fids_to_reset:
            return

        names = "\n".join(f"  • {os.path.basename(fe.path)}"
                           for _, fe in fids_to_reset)
        confirmed = messagebox.askyesno(
            "Reset & reprocess",
            f"This will permanently delete all processed results and sidecar "
            f"config files for:\n\n{names}\n\n"
            f"The file(s) will be treated as new. This cannot be undone.\n\n"
            f"Continue?",
            icon="warning",
            parent=self.root,
        )
        if not confirmed:
            return

        deleted, skipped = [], []
        for fid, fe in fids_to_reset:
            p = pathlib.Path(fe.path)

            # 1. Delete derivatives JSON (pipeline output)
            if fe.derivatives_json and os.path.isfile(fe.derivatives_json):
                try:
                    os.remove(fe.derivatives_json)
                    deleted.append(os.path.basename(fe.derivatives_json))
                except Exception as exc:
                    skipped.append(f"{os.path.basename(fe.derivatives_json)}: {exc}")

            # 2. Delete any figures associated with this file
            if fe.derivatives_json:
                _deriv_dir = os.path.dirname(fe.derivatives_json)
                _fig_dir   = os.path.join(os.path.dirname(_deriv_dir), "figures")
                if os.path.isdir(_fig_dir):
                    import glob as _glob
                    for _f in _glob.glob(os.path.join(_fig_dir, "*_traces.png")):
                        try:
                            os.remove(_f)
                            deleted.append(os.path.basename(_f))
                        except Exception:
                            pass

            # 3. Delete SMR channel assignment sidecar
            smr_cfg = p.with_suffix(".smr_config.json")
            if smr_cfg.exists():
                try:
                    smr_cfg.unlink()
                    deleted.append(smr_cfg.name)
                except Exception as exc:
                    skipped.append(f"{smr_cfg.name}: {exc}")

            # 4. Delete generic TSV / LabChart TXT wizard sidecar
            tsv_cfg = p.with_suffix(".tsv_config.json")
            if tsv_cfg.exists():
                try:
                    tsv_cfg.unlink()
                    deleted.append(tsv_cfg.name)
                except Exception as exc:
                    skipped.append(f"{tsv_cfg.name}: {exc}")

            # 5. Reset FileEntry state completely
            fe.derivatives_json = ""
            fe.status           = STATUS_NOT_STARTED
            fe.last_processed   = ""
            fe.stim_letters     = []
            fe.stim_label_map   = {}
            fe.review_flags     = {}

        self._dataset.save()
        self._queue_refresh()

        # Log summary
        if deleted:
            self.log(f"🔄 Reset complete. Deleted: {', '.join(deleted)}")
        if skipped:
            self.log(f"⚠️  Could not delete: {'; '.join(skipped)}")

    def _update_marker_dropdown(self):
        """Refresh the event marker combobox in the active file row."""
        if not hasattr(self, "_marker_dd"):
            return
        markers = self.available_markers or []
        self._marker_dd["values"] = markers
        if len(markers) > 1:
            self._marker_dd.config(state="readonly")
            # Pre-select current marker_choice if valid, else first
            if self.marker_choice.get() not in markers:
                self.marker_choice.set(markers[0])
        elif len(markers) == 1:
            self._marker_dd.config(state="disabled")
            self.marker_choice.set(markers[0])
        else:
            self._marker_dd.config(state="disabled")

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

    _SUPPORTED_EXTENSIONS = {".txt", ".smr", ".adibin", ".edf", ".bdf"}

    def _audit_filename(self, basename: str) -> list:
        """Return a list of human-readable issue strings for *basename*.
        Empty list means no issues found.
        """
        issues = []
        name, ext = os.path.splitext(basename)

        if ext.lower() not in self._SUPPORTED_EXTENSIONS:
            issues.append(f"Extension '{ext}' — expected .txt, .smr, .adibin, .edf, or .bdf")

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
        tk.Label(win, text="Current path:",
                 anchor="w").grid(row=0, column=0, sticky="w", padx=10, pady=(12, 2))
        tk.Label(win, text=old_path, fg="#555", wraplength=640, justify="left",
                 anchor="w").grid(row=1, column=0, columnspan=2, sticky="w",
                                  padx=10, pady=(0, 8))

        ttk.Separator(win, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=4)

        # Audit results
        tk.Label(win, text="BIDS naming audit:", anchor="w"
                 ).grid(row=3, column=0, sticky="w", padx=10, pady=(4, 2))

        audit_frm = tk.Frame(win, bd=1, relief="sunken", bg="#fffde7")
        audit_frm.grid(row=4, column=0, columnspan=2, sticky="ew",
                       padx=10, pady=(0, 8))
        if issues:
            for issue in issues:
                tk.Label(audit_frm, text=f"  \u26a0  {issue}",
                         fg="#b26a00", bg="#fffde7",
                         font="TkDefaultFont", anchor="w"
                         ).pack(fill="x", padx=4, pady=1)
        else:
            tk.Label(audit_frm,
                     text="  \u2705  No issues found \u2014 filename looks BIDS-compliant",
                     fg="#2e7d32", bg="#fffde7",
                     font="TkDefaultFont", anchor="w"
                     ).pack(fill="x", padx=4, pady=4)

        ttk.Separator(win, orient="horizontal").grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=4)

        # New name entry
        tk.Label(win, text="New filename:",
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
                               font="TkSmallCaptionFont", anchor="w")
        preview_lbl.grid(row=8, column=0, columnspan=2, sticky="w",
                         padx=10, pady=(0, 8))
        _update_preview()

        # BIDS template hint
        tk.Label(win,
                 text="Suggested pattern:  sub-<label>_ses-<label>_limb-<left|right>_<date>.txt",
                 fg="#888", font="TkSmallCaptionFont"
                 ).grid(row=9, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        ttk.Separator(win, orient="horizontal").grid(
            row=10, column=0, columnspan=2, sticky="ew", padx=10, pady=4)

        warn_lbl = tk.Label(win, text="", fg="#d9534f",
                            font="TkDefaultFont")
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
                  **accent_button_kw("blue"), width=16,
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
                self._apply_loaded_session(sess, json_path=fe.derivatives_json, preserve_file_path=True)
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
        # Save restored channel before _browse_file_path resets it
        self._restored_channel_choice = self.channel_choice.get()
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
        # Only the BrainVision .vhdr header is globbed — never .eeg/.vmrk —
        # so each recording enters the queue once, not three times.
        _EXTS = ("*.txt", "*.smr", "*.adibin", "*.edf", "*.bdf",
                 "*.vhdr", "*.acq")
        all_files = []
        for _pat in _EXTS:
            all_files.extend(_glob.glob(os.path.join(raw, "**", _pat), recursive=True))
        files = sorted(
            f for f in all_files
            if not any(p in os.path.basename(f).lower() for p in EXCLUDE)
            and "sourcedata" not in os.path.normpath(f).lower().split(os.sep)
        )
        if not files:
            messagebox.showinfo("No files found",
                "No data files (.txt, .smr, .adibin, .edf, .bdf) found in the raw data folder.",
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
        # Only the BrainVision .vhdr header is globbed — never .eeg/.vmrk —
        # so each recording enters the queue once, not three times.
        _EXTS = ("*.txt", "*.smr", "*.adibin", "*.edf", "*.bdf",
                 "*.vhdr", "*.acq")
        all_files = []
        for _pat in _EXTS:
            all_files.extend(_glob.glob(os.path.join(folder, "**", _pat), recursive=True))
        files = sorted(
            f for f in all_files
            if not any(p in os.path.basename(f).lower() for p in EXCLUDE_PATTERNS)
            and "sourcedata" not in os.path.normpath(f).lower().split(os.sep)
        )

        if not files:
            messagebox.showinfo("No files found",
                "No data files (.txt, .smr, .adibin, .edf, .bdf) found in that folder or its subfolders.\n\n"
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
                ("All supported formats",
                 "*.txt *.smr *.adibin *.edf *.bdf *.vhdr *.acq *.mat *.csv"),
                ("Spike2 / LabChart text export", "*.txt"),
                ("Spike2 native", "*.smr"),
                ("ADInstruments binary", "*.adibin"),
                ("BrainVision header", "*.vhdr"),
                ("BIOPAC AcqKnowledge", "*.acq"),
                ("LabChart / AcqKnowledge MATLAB export", "*.mat"),
                ("KinEMG / NI-DAQ CSV", "*.csv"),
                ("BIDS EDF/BDF", "*.edf *.bdf"),
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
                f for _mp in ("*.txt", "*.edf", "*.bdf")
                for f in _glob.glob(os.path.join(_folder, _mp))
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

        elif _fmt == 'edf':
            self.marker_choice.set('A')
            self.log("📋 BIDS EDF/BDF format — stim times from sidecar _events.tsv "
                     "(or EDF+ annotations)")
            # stim_events populated later via extract_stim_times in pipeline

        elif _fmt in ('brainvision', 'labchart_mat', 'mne'):
            # Marker-based formats: the file's own event labels define the stim
            # types, so they must be read at load time.  Without this branch the
            # format falls through to the Spike2 text scanner below, stim_events
            # stays empty, stim_types_found is empty, and _build_labels_tab() is
            # never called — the workflow silently stalls after the crop step.
            _disp = {'brainvision': 'BrainVision',
                     'labchart_mat': 'LabChart MATLAB export',
                     'mne': 'MNE-supported'}.get(_fmt, _fmt)
            self.marker_choice.set('A')
            self.log(f"📋 {_disp} format detected — stim times from embedded "
                     f"event markers")
            try:
                # '' = return every label; the marker argument is a filter for
                # these readers, and at discovery time we want all stim types.
                stim_events = extract_stim_times(fpath, '')
            except Exception as _e:
                self.log(f"   ⚠️  Could not read event markers: {_e}")
                stim_events = {}
            if stim_events:
                self.available_markers = sorted(stim_events)
                self.log("   Event labels found: " + ", ".join(
                    f"{k} ({len(v)})" for k, v in sorted(stim_events.items())))
            else:
                self.log("   ⚠️  No event markers found in this recording")

        elif _fmt in ('acqknowledge_acq', 'acqknowledge_mat', 'brainsight'):
            # Trigger/marker-channel formats: one stim type, labelled by
            # marker_choice — same contract as labchart / cfwb.
            _disp = {'acqknowledge_acq': 'BIOPAC AcqKnowledge',
                     'acqknowledge_mat': 'BIOPAC AcqKnowledge MATLAB export',
                     'brainsight': 'Brainsight neuronavigation'}.get(_fmt, _fmt)
            self.marker_choice.set('A')
            self.log(f"📋 {_disp} format detected — stim times from event "
                     f"markers / trigger channel")

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
                    self.available_markers = stim_options

                    _chosen = {}

                    def _show_smr_dialog(
                        _analogue=analogue,
                        _stim_options=stim_options,
                    ):
                        import tkinter as tk
                        from tkinter import ttk
                        dlg = tk.Toplevel(self.root)
                        dlg.title("Channel Assignment")
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
                        _digmark_default = next(
                            (o for o in _stim_options if "DigMark" in o),
                            _stim_options[0]
                        )
                        stim_var = tk.StringVar(value=_digmark_default)
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
                    # Set marker_choice to the full prefixed option so
                    # _update_marker_dropdown finds it in the list
                    _bare = _chosen["stim"]
                    _full = next(
                        (o for o in stim_options
                         if o.endswith(f"] {_bare}") or o == _bare),
                        _bare
                    )
                    self.marker_choice.set(_full)
                    self.available_markers = stim_options

                else:
                    cfg = _smr_load_cfg(fpath)
                    stim_ch = cfg.get("stim_channel", "A")
                    # Rebuild stim_options from file info for the marker dropdown
                    info = _smr_info(fpath)
                    _events  = info.get("events",  [])
                    _epochs  = info.get("epochs",  [])
                    _spikes  = info.get("spikes",  [])
                    _analogue = info.get("analogue", [])
                    stim_options = []
                    for n in _events:  stim_options.append(f"[Event] {n}")
                    for n in _epochs:  stim_options.append(f"[DigMark/Epoch] {n}")
                    for n in _spikes:  stim_options.append(f"[Spike] {n}")
                    _STIM_KW = ("stim", "trig", "ttl", "digmark")
                    for n in _analogue:
                        if any(kw in n.lower() for kw in _STIM_KW):
                            stim_options.append(f"[Analogue] {n}")
                    if not stim_options:
                        stim_options = [f"[Analogue] {n}" for n in _analogue]
                    # Find the full prefixed option matching the saved stim channel
                    _matched = next(
                        (o for o in stim_options if o.endswith(f"] {stim_ch}") or o == stim_ch),
                        stim_ch
                    )
                    self.marker_choice.set(_matched)
                    self.available_markers = stim_options
                    self.log(
                        f"   EMG: {cfg.get('emg_channel')} | "
                        f"Stim channel: {stim_ch} — loaded from sidecar"
                    )

            except ImportError:
                self.log(
                    "❌ Neo is not installed. Install it with:  pip install neo\n"
                    "   Native .smr files cannot be read without Neo."
                )
                return
            except Exception as e:
                self.log(f"❌ Error reading SMR file: {e}")
                return

        elif _fmt == 'spike2':
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
                self.available_markers = sorted(marker_set)
                self._ask_marker_gui(sorted(marker_set))
            elif marker_set:
                self.available_markers = sorted(marker_set)
                self.marker_choice.set(next(iter(marker_set)))

        else:
            # No load-time handler for this format.
            #
            # Every value detect_format() can return must be represented in the
            # chain above.  When one is not, stim_events stays empty, so
            # stim_types_found is empty, so _build_labels_tab() is never called
            # and the workflow stalls silently after the crop step with no error
            # shown to the user.  This branch previously held the Spike2 text
            # scanner, which meant any unhandled format was silently scanned for
            # DigMark lines and found nothing.  Fail loudly instead.
            #
            # tests/test_format_coverage.py asserts this branch is unreachable.
            self.log(f"⚠️  Format '{_fmt}' can be read but has no load-time "
                     f"handler in _browse_file_path(), so its stimulus types "
                     f"cannot be determined. Please report this bug.")
            messagebox.showwarning(
                "Format not wired into the workflow",
                f"{os.path.basename(fpath)} was recognised as format "
                f"'{_fmt}' and its data can be read, but the analysis "
                f"workflow has no handler for it yet, so no stimulus types "
                f"could be identified.\n\nPlease report this at:\n"
                f"https://github.com/jandrushko/mep-cmap-analyser/issues",
                parent=self.root)
        
        # ── populate inline channel dropdown
        chan_list = list_waveform_channels(fpath)
        self._populate_channel_dropdown(chan_list)
        # Restore previously saved channel — use _restored_channel_choice
        # which is set by _load_file_entry before calling _browse_file_path,
        # since _populate_channel_dropdown resets to index 0.
        _saved_ch = getattr(self, '_restored_channel_choice', None)                     or self.channel_choice.get()
        if _saved_ch and _saved_ch in chan_list:
            self.channel_var.set(_saved_ch)
            self.channel_idx = chan_list.index(_saved_ch)
            self.channel_choice.set(_saved_ch)
        self._restored_channel_choice = None  # clear after use
        self._update_marker_dropdown()

        # ── Unified channel + event marker assignment dialog (Spike2 text exports only)
        # SMR files have their own dialog. Generic TSV/LabChart/CFWB use the
        # Format Wizard or have no event marker concept — exclude them here.
        markers = self.available_markers or []
        _needs_assign_dlg = (
            _fmt not in ('spike2_smr', 'generic_tsv', 'labchart', 'cfwb')
            and (len(chan_list) > 1 or len(markers) > 1)
        )
        if _needs_assign_dlg:
            _chosen = {}

            def _show_assign_dlg(
                _chan_list=chan_list,
                _markers=markers,
            ):
                import tkinter as _tk
                from tkinter import ttk as _ttk
                dlg = _tk.Toplevel(self.root)
                dlg.title("Channel Assignment")
                dlg.transient(self.root)
                dlg.resizable(False, False)
                dlg.grab_set()

                _tk.Label(
                    dlg,
                    text=(
                        f"File: {os.path.basename(fpath)}\n\n"
                        "Choose the EMG channel and the event/marker source.\n"
                        "Your choices are saved and will not be asked again."
                    ),
                    justify="left",
                ).pack(anchor="w", padx=16, pady=(12, 6))

                frm = _tk.Frame(dlg, padx=16, pady=8)
                frm.pack(fill="x")

                # EMG channel row (only show if >1 channel)
                emg_var = _tk.StringVar(value=chan_list[0] if chan_list else "")
                if len(_chan_list) > 1:
                    _tk.Label(frm, text="EMG channel:", anchor="w", width=22)                        .grid(row=0, column=0, sticky="w", pady=6)
                    _ttk.Combobox(frm, textvariable=emg_var,
                                  values=_chan_list, state="readonly", width=28)                        .grid(row=0, column=1, sticky="w")

                # Event marker row (only show if >1 marker)
                _cur_marker = self.marker_choice.get()
                stim_var = _tk.StringVar(
                    value=_cur_marker if _cur_marker in _markers else
                    (next((m for m in _markers if "DigMark" in m), _markers[0])
                     if _markers else ""))
                if len(_markers) > 1:
                    _tk.Label(frm, text="Event/marker source:", anchor="w", width=22)                        .grid(row=1, column=0, sticky="w", pady=6)
                    _ttk.Combobox(frm, textvariable=stim_var,
                                  values=_markers, state="readonly", width=28)                        .grid(row=1, column=1, sticky="w")

                _tk.Label(
                    frm,
                    text="Tip: Event channels use timestamps directly.\n"
                         "Analogue channels use threshold-crossing detection.",
                    fg="grey", justify="left",
                ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

                def _save():
                    _chosen["emg"]  = emg_var.get()
                    _chosen["stim"] = stim_var.get()
                    dlg.destroy()

                def _cancel_dlg():
                    dlg.destroy()

                btn_r = _tk.Frame(dlg)
                btn_r.pack(pady=(4, 12))
                _tk.Button(btn_r, text="Save & continue", width=14,
                           command=_save).pack(side="left", padx=6)
                _tk.Button(btn_r, text="Cancel", width=10,
                           command=_cancel_dlg).pack(side="left", padx=6)

                self.root.update_idletasks()
                dlg.update_idletasks()
                x = self.root.winfo_x() + (self.root.winfo_width()  - dlg.winfo_width())  // 2
                y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
                dlg.geometry(f"+{x}+{y}")
                self.root.wait_window(dlg)

            _show_assign_dlg()

            if _chosen.get("emg") and _chosen["emg"] in chan_list:
                self.channel_var.set(_chosen["emg"])
                self.channel_idx = chan_list.index(_chosen["emg"])
            if _chosen.get("stim"):
                self.marker_choice.set(_chosen["stim"])
                self.available_markers = markers

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
        if _fmt in ('labchart', 'generic_tsv', 'cfwb',
                    'acqknowledge_acq', 'acqknowledge_mat', 'brainsight'):
            _mc = self.marker_choice.get()
            stim_types_found = {_mc} if _mc else {'A'}

        elif _fmt == 'spike2_smr':
            # Use get_event_codes_for_channel (cached segment) to get all
            # marker codes, then filter by the selected crop range.
            try:
                from .formats.spike2_smr import (
                    get_event_codes_for_channel as _smr_codes,
                    load_config as _smr_lcfg,
                    has_config  as _smr_hcfg,
                )
                _stim_ch = (
                    _smr_lcfg(fpath).get("stim_channel", self.marker_choice.get())
                    if _smr_hcfg(fpath) else self.marker_choice.get()
                )
                _codes = _smr_codes(fpath, _stim_ch)
                if _codes:
                    # We have discrete event codes — get their timestamps and
                    # filter to only those present in the selected crop range
                    from .io import extract_stim_times as _io_est
                    try:
                        _all_stim = _io_est(fpath, _stim_ch)
                    except Exception:
                        _all_stim = {c: [] for c in _codes}

                    if self.crop_ranges:
                        stim_types_found = {
                            stype for stype, times in _all_stim.items()
                            if any(start <= t <= end
                                   for t in times
                                   for start, end in self.crop_ranges)
                        } or set(_codes)
                    elif self.crop_start is not None and self.crop_end is not None:
                        stim_types_found = {
                            stype for stype, times in _all_stim.items()
                            if any(self.crop_start <= t <= self.crop_end for t in times)
                        } or set(_codes)
                    else:
                        stim_types_found = set(_all_stim.keys()) or set(_codes)
                else:
                    # No discrete codes (analogue threshold fallback)
                    stim_types_found = {_stim_ch} if _stim_ch else {'A'}
            except Exception as _e:
                self.log(f"   ⚠️  Could not scan SMR codes: {_e} — using fallback")
                _mc = self.marker_choice.get()
                stim_types_found = {_mc} if _mc else {'A'}

            if stim_types_found:
                self.log(f"   Marker codes in range: {', '.join(sorted(stim_types_found))}")

        elif _fmt in ('edf', 'brainvision', 'labchart_mat', 'mne'):
            # Marker-based formats (BIDS EDF/BDF sidecar _events.tsv or EDF+
            # annotations; BrainVision .vmrk; LabChart .mat comments; MNE
            # annotations) — read the actual labels present so the labels tab
            # matches what the pipeline will produce.
            try:
                _all_stim = extract_stim_times(fpath, '')
            except Exception as _e:
                self.log(f"   ⚠️  Could not read events: {_e}")
                _all_stim = {}
            if self.crop_ranges:
                stim_types_found = {
                    stype for stype, times in _all_stim.items()
                    if any(start <= t <= end
                           for t in times for start, end in self.crop_ranges)
                } or set(_all_stim.keys())
            elif self.crop_start is not None and self.crop_end is not None:
                stim_types_found = {
                    stype for stype, times in _all_stim.items()
                    if any(self.crop_start <= t <= self.crop_end for t in times)
                } or set(_all_stim.keys())
            else:
                stim_types_found = set(_all_stim.keys())
            if not stim_types_found:
                stim_types_found = {self.marker_choice.get() or 'A'}

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
                  'timepoint': '', 'limb': '', 'measure': '', 'acq': ''}
        for part in name.split('_'):
            pl = part.lower()
            if   pl.startswith('sub-'):     result['participant_id'] = part
            elif pl.startswith('ses-'):     result['session']        = part
            elif pl.startswith('task-'):    result['task']           = part[5:]
            elif pl.startswith('tp-'):      result['timepoint']      = part[3:]
            elif pl.startswith('limb-'):    result['limb']           = part[5:]
            elif pl.startswith('measure-'): result['measure']        = part[8:]
            elif pl.startswith('acq-'):     result['acq']            = part[4:]
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
        v_acq     = tk.StringVar(value=parsed['acq']            or getattr(carry, 'acq',     ''))
        v_rem     = tk.BooleanVar(value=self._remembered_meta is not None)

        win = tk.Toplevel(self.root)
        win.title("Study Metadata (BIDS)" + (f" — {context}" if context else ""))
        win.resizable(False, False)
        win.transient(self.root)

        pad = dict(padx=10, pady=4)

        if context:
            tk.Label(win, text=f"📋 External normalisation file: {context}",
                     fg="#d9534f").grid(
                     row=0, column=0, columnspan=3, **pad, sticky="w")
            tk.Label(win, text="Enter metadata for BIDS-style output naming.").grid(
                     row=1, column=0, columnspan=3, **pad, sticky="w")
            _row_offset = 2
        else:
            tk.Label(win, text="Enter study metadata for BIDS-style output naming.").grid(
                     row=0, column=0, columnspan=3, **pad, sticky="w")
            _row_offset = 1

        # Helper to add a labelled row
        def _row(r, label, var, example):
            tk.Label(win, text=label).grid(row=r+_row_offset, column=0, sticky="e", **pad)
            tk.Entry(win, textvariable=var, width=22).grid(row=r+_row_offset, column=1, sticky="w", **pad)
            tk.Label(win, text=example, fg="grey", font="TkSmallCaptionFont")\
                .grid(row=r+_row_offset, column=2, sticky="w", padx=(0, 10))

        _row(1, "Participant ID *",  v_sub,  "e.g.  sub-JD001  or  JD001")
        _row(2, "Session",           v_ses,  "e.g.  ses-01  (default: ses-01)")
        _row(3, "Limb",              v_limb, "e.g.  left / right  (auto-detected)")
        _row(4, "Task label",        v_task, "e.g.  fatigue  (optional)")
        _row(5, "Timepoint",         v_tp,   "e.g.  pre / post  (optional)")
        _row(6, "Acquisition / Cond.", v_acq,  "e.g.  cond-rest  or  cond-30  (optional)")

        # Measure type — dropdown of common TMS paradigms
        tk.Label(win, text="Measure type").grid(row=7+_row_offset, column=0, sticky="e", **pad)
        measure_frame = tk.Frame(win)
        measure_frame.grid(row=7+_row_offset, column=1, columnspan=2, sticky="w")
        _measure_choices = ['CSE', 'SICI', 'ICF', 'LICI', 'SAI', 'LAI', 'M-wave', 'CMEP', 'Other']
        measure_cb = ttk.Combobox(measure_frame, textvariable=v_measure,
                                  values=_measure_choices, width=10)
        measure_cb.pack(side="left")
        tk.Label(measure_frame, text="or type your own",
                 fg="grey", font="TkSmallCaptionFont").pack(side="left", padx=(6,0))

        tk.Checkbutton(win, text="Remember these settings for the next file",
                       variable=v_rem)          .grid(row=8+_row_offset, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 2))

        err_lbl = tk.Label(win, text="", fg="red")
        err_lbl.grid(row=9+_row_offset, column=0, columnspan=3, sticky="w", padx=10)

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
            acq     = _sanitise_bids_label(v_acq.get())             if v_acq.get().strip()     else ""
            self.study_metadata = StudyMetadata(
                participant_id = sub,
                session        = ses,
                task           = task,
                timepoint      = tp,
                limb           = limb,
                measure        = measure,
                acq            = acq,
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

    # ──────────────────────────────────────────────────────────────────────────
    # BIDS-ify — shared helpers used by the BIDS-ify tab (bidsify_tab.py)
    # ──────────────────────────────────────────────────────────────────────────
    def _show_bidsify_preview(self, plan) -> bool:
        """Modal dry-run preview. Returns True if the user clicks Proceed."""
        win = tk.Toplevel(self.root)
        win.title("BIDS-ify — preview (dry run)")
        win.transient(self.root)
        tk.Label(win,
                 text="Review the plan below. Nothing is written until you click Proceed.",
                 fg="#d9534f").pack(
                 anchor="w", padx=10, pady=(10, 4))
        txt = scrolledtext.ScrolledText(win, width=104, height=24, wrap="none")
        txt.pack(fill="both", expand=True, padx=10, pady=4)
        txt.insert("1.0", plan.preview_text())
        txt.config(state="disabled")

        result = {"go": False}
        btns = tk.Frame(win)
        btns.pack(fill="x", pady=8)

        def _go():
            result["go"] = True
            win.destroy()

        tk.Button(btns, text="Proceed", **accent_button_kw("green"),
                  command=_go).pack(side="right", padx=(0, 10))
        tk.Button(btns, text="Cancel",
                  command=win.destroy).pack(side="right", padx=(0, 6))

        win.update_idletasks()
        px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
        pw, ph = self.root.winfo_width(), self.root.winfo_height()
        w, h = win.winfo_width(), win.winfo_height()
        win.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        win.grab_set()
        self.root.wait_window(win)
        return result["go"]

    def _bidsify_done_gui(self, results):
        """Summarise a finished BIDS-ify run (runs on the main thread)."""
        ok = sum(1 for r in results if getattr(r, "ok", False))
        fail = len(results) - ok
        msg = f"BIDS-ify complete: {ok} succeeded, {fail} failed."
        self.log(msg)

        problems = []
        for r in results:
            if not getattr(r, "ok", False):
                why = (getattr(r, "error", "")
                       or "; ".join(getattr(r, "discrepancies", []))
                       or "unknown error")
                problems.append(f"• {os.path.basename(r.source_path)}: {why}")
        if problems:
            messagebox.showwarning(
                "BIDS-ify",
                msg + "\n\nProblems:\n" + "\n".join(problems), parent=self.root)
        else:
            messagebox.showinfo("BIDS-ify", msg, parent=self.root)

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
                **accent_button_kw("green"))
        else:
            self._deriv_status_bar.config(
                text="⚠  Derivatives folder not set — click here or use File → Set Derivatives Folder",
                **accent_button_kw("red"))

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
        Build (or rebuild) the Stage 1a labels tab with per-stim configuration:
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
            tk.Label(inner, text=h)\
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
            **accent_button_kw("red"),
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
        self.notebook.select(self.stage1_outer)
        self.nb_stage1.select(self.tab1b_frame)

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
        self._confirm_btn_widget.config(**accent_button_kw("green"))
        self.log("✔ Label & analysis setup confirmed — ready to run.\n")
        # Switch back to Stage 1a so user can hit Run Analysis
        self.notebook.select(self.stage1_outer)
        self.nb_stage1.select(self.tab_filter)
