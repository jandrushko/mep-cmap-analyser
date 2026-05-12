"""
mep_cmap.filter_preview
~~~~~~~~~~~~~~~~~~~~~~~
Filter preview window mixin.

Contains preview_filter_window — the interactive frequency-domain and
time-domain filter inspection popup.  Mixed into TMSAnalysisApp via
FilterPreviewMixin.
"""

import os
import glob
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from matplotlib.colors import Normalize
from matplotlib import gridspec as mgs
from scipy.signal import (
    butter, freqz, group_delay, filtfilt,
    sosfiltfilt, sos2tf, iirnotch, fftconvolve,
)

from .io import extract_emg_waveform_and_fs, extract_stim_times
from .filters import adaptive_mains_cancel


class FilterPreviewMixin:
    """
    Mixin providing the Filter Preview popup.
    All methods are intended to be used as part of TMSAnalysisApp.
    """

    def preview_filter_window(self):
        import os, glob
        import numpy as np
        import tkinter as tk
        from tkinter import ttk, simpledialog, messagebox
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
        from matplotlib.colors import Normalize
        from matplotlib import gridspec as mgs
        from scipy.signal import (
            butter, freqz, group_delay, filtfilt, sosfiltfilt, sos2tf, iirnotch, fftconvolve
        )

        # Compact fonts
        matplotlib.rcParams.update({
            "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8
        })

        # ───────────────────────────── Load / crop ─────────────────────────────
        if not hasattr(self, "raw_emg") or not hasattr(self, "prev_fs"):
            sel = self.file_path.get()
            txts = []
            if sel.lower().endswith(".txt") and os.path.isfile(sel):
                txts = [sel]
            elif os.path.isdir(sel):
                txts = sorted(glob.glob(os.path.join(sel, "*.txt")))
            if txts:
                try:
                    self.raw_emg, self.prev_fs, self.emg_unit = extract_emg_waveform_and_fs(
                        txts[0], channel_idx=self.channel_idx
                    )
                    self.last_times = np.arange(len(self.raw_emg)) / self.prev_fs
                    self.last_stim  = extract_stim_times(txts[0], self.marker_choice.get())

                    # honour crop
                    if getattr(self, "crop_ranges", None):
                        keep = np.zeros_like(self.last_times, dtype=bool)
                        for a, b in self.crop_ranges:
                            keep |= ((self.last_times >= a) & (self.last_times <= b))
                    elif (self.crop_start is not None and self.crop_end is not None):
                        keep = ((self.last_times >= self.crop_start) & (self.last_times <= self.crop_end))
                    else:
                        keep = slice(None)
                    self.raw_emg    = self.raw_emg[keep]
                    self.last_times = self.last_times[keep]

                    # drop markers outside kept span(s)
                    for k in list(self.last_stim):
                        ts = [t for t in self.last_stim[k] if self.last_times[0] <= t <= self.last_times[-1]]
                        if ts: self.last_stim[k] = ts
                        else:  self.last_stim.pop(k)
                except Exception as e:
                    messagebox.showerror("Preview error", str(e), parent=self.root); return

        if not getattr(self, "prev_fs", None):
            rate = simpledialog.askinteger("Sampling rate required",
                "Enter EMG sampling rate (Hz):", parent=self.root, minvalue=200, maxvalue=50000)
            if not rate: return
            self.prev_fs = rate

        fs  = float(self.prev_fs)
        nyq = fs / 2.0
        eps = 1e-12

        # Persist per-event trial selection across refreshes
        if not hasattr(self, "_preview_sel_trial"):
            self._preview_sel_trial = {}  # dict: event_type -> selected index (int)

        # ─────────────────────────── Filter helpers ────────────────────────────
        def design_notch_pairs(fs, f0, Q, include_harmonics=False):
            f0 = float(f0); Q = float(Q); pairs = []; hs=[f0]
            if include_harmonics:
                k = 2
                while k*f0 < (fs/2.0) - 1.0:
                    hs.append(k*f0); k += 1
            for f in hs:
                pairs.append(iirnotch(w0=float(f), Q=Q, fs=fs))
            return pairs

        def adaptive_mains_cancel(x, fs, mains_freq=50.0, n_harmonics=1):
            y = np.asarray(x, float)
            for k in range(1, int(n_harmonics)+1):
                b,a = iirnotch(w0=float(mains_freq)*k, Q=30.0, fs=fs)
                y = filtfilt(b,a,y)
            return y

        # ── replace _apply_pipeline with this ----------------------------
        def _apply_pipeline(x):
            y = np.asarray(x, float)

            # mains canceller & notch – unchanged
            if self.apply_humbug.get():
                y = adaptive_mains_cancel(
                    y, fs, mains_freq=float(self.notch_freq.get()),
                    n_harmonics=int(self.humbug_harmonics.get())
                )
            if self.apply_notch.get():
                for b,a in design_notch_pairs(
                    fs, float(self.notch_freq.get()), float(self.notch_q.get()),
                    include_harmonics=bool(self.filter_harmonics.get())
                ):
                    y = filtfilt(b, a, y)

            # bandpass – honour advanced toggle
            if self.apply_bandpass.get():
                hp_hz = max(float(self.highpass.get()), 0.1)
                lp_hz = min(float(self.lowpass.get()), nyq - 1e-6)
                if hp_hz >= lp_hz:
                    lp_hz = hp_hz + 1.0

                if self.use_advanced_bp.get():
                    # Advanced: separate orders, butter HP then LP
                    hp_ord = max(int(self.hp_order_var.get()), 1)
                    lp_ord = max(int(self.lp_order_var.get()), 1)
                    sos_hp = butter(hp_ord, hp_hz, btype='highpass', fs=fs, output='sos')
                    sos_lp = butter(lp_ord, lp_hz, btype='lowpass',  fs=fs, output='sos')
                    y = sosfiltfilt(sos_hp, y)
                    y = sosfiltfilt(sos_lp, y)
                else:
                    # Regular: single order, Butterworth bandpass
                    order = int(self.filter_order.get())
                    sos = butter(order, [hp_hz, lp_hz], btype='band', fs=fs, output='sos')
                    y = sosfiltfilt(sos, y)
            return y

        # ── replace _freq_response_curves with this ----------------------
        def _freq_response_curves():
            w = np.linspace(0, nyq, 4096)
            H = np.ones_like(w, dtype=np.complex128)

            # cascade any notches
            if self.apply_notch.get():
                for b,a in design_notch_pairs(
                    fs, float(self.notch_freq.get()), float(self.notch_q.get()),
                    include_harmonics=bool(self.filter_harmonics.get())
                ):
                    _, Hn = freqz(b, a, worN=w, fs=fs)
                    H *= Hn

            hp_hz = max(float(self.highpass.get()), 0.1)
            lp_hz = min(float(self.lowpass.get()), nyq - 1e-6)
            if hp_hz >= lp_hz:
                lp_hz = hp_hz + 1.0

            if self.apply_bandpass.get():
                if self.use_advanced_bp.get():
                    # Advanced: butter HP + LP with separate orders
                    hp_ord = max(int(self.hp_order_var.get()), 1)
                    lp_ord = max(int(self.lp_order_var.get()), 1)
                    b1, a1 = sos2tf(butter(hp_ord, hp_hz, btype='highpass', fs=fs, output='sos'))
                    b2, a2 = sos2tf(butter(lp_ord, lp_hz, btype='lowpass',  fs=fs, output='sos'))
                    _, Hb = freqz(b1, a1, worN=w, fs=fs)
                    _, Hl = freqz(b2, a2, worN=w, fs=fs)
                    H *= Hb * Hl
                    try:
                        w_gd, gd = group_delay((np.convolve(b1, b2), np.convolve(a1, a2)), w=w, fs=fs)
                    except Exception:
                        w_gd, gd = w, np.zeros_like(w)
                else:
                    # Regular: single order, Butterworth bandpass
                    order = int(self.filter_order.get())
                    b, a = butter(order, [hp_hz, lp_hz], btype='band', fs=fs)
                    _, Hb = freqz(b, a, worN=w, fs=fs)
                    H *= Hb
                    try:
                        w_gd, gd = group_delay((b, a), w=w, fs=fs)
                    except Exception:
                        w_gd, gd = w, np.zeros_like(w)
            else:
                w_gd, gd = w, np.zeros_like(w)

            return w, H, w_gd, gd * 1000.0

        # ───────────────────── Window & fixed figure sizes ─────────────────────
        popup = tk.Toplevel(self.root); popup.title("Filter Preview")

        self.root.update_idletasks()
        main_w = self.root.winfo_width()
        main_h = self.root.winfo_height()
        base_w = max(900, int(main_w * 0.95))
        base_h = max(600, int(main_h * 0.99))
        popup.geometry(f"{base_w}x{base_h}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        popup.resizable(True, True)

        # ── Safe cleanup on close ──────────────────────────────────────────────
        # Explicitly close all three figures BEFORE Tk destroys the canvas
        # widgets.  Without this, garbage-collecting a FigureCanvasTkAgg after
        # its Tk window is gone causes Tcl_AsyncDelete crashes on Windows.
        def _on_preview_close():
            try:
                import matplotlib.pyplot as _plt
                for _fig in (fig1, fig2, fig3):
                    try:
                        _plt.close(_fig)
                    except Exception:
                        pass
            except NameError:
                pass   # figures not yet created – nothing to close
            try:
                popup.destroy()
            except Exception:
                pass

        popup.protocol("WM_DELETE_WINDOW", _on_preview_close)

        outer   = ttk.Frame(popup); outer.pack(fill="both", expand=True)
        vscroll = tk.Scrollbar(outer, orient="vertical");   vscroll.pack(side="right", fill="y")
        hscroll = tk.Scrollbar(outer, orient="horizontal"); hscroll.pack(side="bottom", fill="x")
        canvas  = tk.Canvas(outer, yscrollcommand=vscroll.set, xscrollcommand=hscroll.set, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.config(command=canvas.yview); hscroll.config(command=canvas.xview)
        content = ttk.Frame(canvas); canvas.create_window((0,0), window=content, anchor="nw")
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _prune_axes(fig, keep_axes):
            for ax in list(fig.axes):
                if ax not in keep_axes:
                    try: fig.delaxes(ax)
                    except Exception: pass
            fig.canvas.draw_idle()

       # ────────────────────────── Controls (refresh) ─────────────────────────
        ctrl = tk.Frame(content); ctrl.pack(fill="x", padx=10, pady=5)

        # ── Row 0: Apply Filter (for parity with main GUI)
        tk.Checkbutton(ctrl, text="Apply Filter", variable=self.apply_filter)\
            .grid(row=0, column=0, sticky='w', pady=(0, 4))

        # ── Row 1: Bandpass + HP/LP + single Order (columns match main GUI)
        cb_bp = tk.Checkbutton(ctrl, text="Bandpass Filter", variable=self.apply_bandpass)
        cb_bp.grid(row=1, column=0, sticky='w')

        tk.Label(ctrl, text="HP (Hz):").grid(row=1, column=1, sticky='e', padx=(10, 2))
        hp_freq_entry = tk.Entry(ctrl, textvariable=self.highpass, width=6)
        hp_freq_entry.grid(row=1, column=2, sticky='w')

        tk.Label(ctrl, text="LP (Hz):").grid(row=1, column=3, sticky='e', padx=(10, 2))
        lp_freq_entry = tk.Entry(ctrl, textvariable=self.lowpass, width=6)
        lp_freq_entry.grid(row=1, column=4, sticky='w')

        tk.Label(ctrl, text="Order:").grid(row=1, column=5, sticky='e', padx=(10, 2))
        ord_entry = tk.Entry(ctrl, textvariable=self.filter_order, width=4)
        ord_entry.grid(row=1, column=6, sticky='w')

        # ── Row 2: Advanced bandpass toggle (span like main)
        adv_chk = tk.Checkbutton(
            ctrl,
            text="Advanced bandpass (Separate HP/LP orders)",
            variable=self.use_advanced_bp
        )
        adv_chk.grid(row=2, column=0, columnspan=5, sticky='w', pady=(6, 0))

        # ── Row 3: HP/LP order entries (compact, same columns)
        tk.Label(ctrl, text="HP order:").grid(row=3, column=0, sticky='w', padx=6)
        hp_ord_entry = tk.Entry(ctrl, textvariable=self.hp_order_var, width=5)
        hp_ord_entry.grid(row=3, column=1, sticky='w')

        tk.Label(ctrl, text="LP order:").grid(row=3, column=2, sticky='e', padx=6)
        lp_ord_entry = tk.Entry(ctrl, textvariable=self.lp_order_var, width=5)
        lp_ord_entry.grid(row=3, column=3, sticky='w')

        # ── Row 4: Notch + Freq + Q + “Filter Harmonics” (same row)
        cb_not = tk.Checkbutton(ctrl, text="Notch Filter", variable=self.apply_notch)
        cb_not.grid(row=4, column=0, sticky='w')

        tk.Label(ctrl, text="Notch (Hz):").grid(row=4, column=1, sticky='e', padx=(10, 2))
        notch_freq_entry = tk.Entry(ctrl, textvariable=self.notch_freq, width=6)
        notch_freq_entry.grid(row=4, column=2, sticky='w')

        tk.Label(ctrl, text="Q:").grid(row=4, column=3, sticky='e', padx=(10, 2))
        notch_q_entry = tk.Entry(ctrl, textvariable=self.notch_q, width=6)
        notch_q_entry.grid(row=4, column=4, sticky='w')

        cb_fhar = tk.Checkbutton(ctrl, text="Filter Harmonics", variable=self.filter_harmonics)
        cb_fhar.grid(row=4, column=5, sticky='w', padx=(10, 0))

        # ── Row 5: Mains canceller + harmonics entry (like main)
        cb_hum = tk.Checkbutton(ctrl, text="Mains Noise Cancel", variable=self.apply_humbug)
        cb_hum.grid(row=5, column=0, sticky='w')

        tk.Label(ctrl, text="Harmonics:").grid(row=5, column=1, sticky='e')
        mains_harm_entry = tk.Entry(ctrl, textvariable=self.humbug_harmonics, width=4)
        mains_harm_entry.grid(row=5, column=2, sticky='w')

        # ── Row 0 (right): Refresh button
        def _on_refresh():
            _recompute_everything()
            _redraw_plots()
        tk.Button(ctrl, text="Refresh filter ↺", command=_on_refresh)\
        .grid(row=0, column=7, padx=(20, 0), sticky='w')

        # ── Live enable/disable: keep preview AND MAIN GUI in sync ─────────────
        def _sync_enable_states_preview_only():
            bp  = bool(self.apply_bandpass.get())
            adv = bool(self.use_advanced_bp.get())
            nt  = bool(self.apply_notch.get())
            hum = bool(self.apply_humbug.get())

            # Bandpass-dependent fields (in the PREVIEW window)
            hp_freq_entry.config(state='normal' if bp else 'disabled')
            lp_freq_entry.config(state='normal' if bp else 'disabled')
            # Single order disabled when Advanced ON; else follows BP
            ord_entry.config(state=('disabled' if adv else ('normal' if bp else 'disabled')))
            # Advanced orders only when BOTH BP and Advanced are ON
            hp_lp_state = 'normal' if (bp and adv) else 'disabled'
            hp_ord_entry.config(state=hp_lp_state)
            lp_ord_entry.config(state=hp_lp_state)

            # Notch-dependent fields (preview)
            notch_freq_entry.config(state='normal' if nt else 'disabled')
            notch_q_entry.config(state='normal' if nt else 'disabled')
            cb_fhar.config(state='normal' if nt else 'disabled')

            # Mains canceller-dependent harmonics (preview)
            mains_harm_entry.config(state='normal' if hum else 'disabled')

        def _sync_all(*_):
            # 1) Update the PREVIEW widgets
            _sync_enable_states_preview_only()
            # 2) Update the MAIN GUI widgets
            try:
                self.toggle_bandpass_fields()
                self.toggle_bp_order_fields()
                self.toggle_notch_fields()
                self._toggle_humbug_fields()
            except Exception:
                # If preview is opened very early, main widgets might not be built yet
                pass

        # Re-bind traces (avoid stacking when reopening)
        for var in (self.apply_bandpass, self.use_advanced_bp, self.apply_notch, self.apply_humbug):
            try:
                var.trace_remove('write', var._preview_link_id)
            except Exception:
                pass
            var._preview_link_id = var.trace_add('write', _sync_all)

        # Initial sync for both preview and main GUI
        _sync_all()

        # ───────────────────────────────── Tabs ─────────────────────────────────
        tabs = ttk.Notebook(content); tabs.pack(fill="both", expand=True, padx=10, pady=5)

        # Fixed figure sizes (scaled from popup)
        dpi = 100
        fig_w_in  = max(6.8, (base_w - 140) / dpi)
        avail_h_px = base_h - 210
        fig_h1_in  = max(5, 0.2 * avail_h_px / dpi)   # Freq & FFT
        fig_h2_in  = max(4.8, 0.54 * avail_h_px / dpi)   # Wavelet
        fig_h3_in  = max(4.2, 0.50 * avail_h_px / dpi)   # Time-domain

        # Tab 1: Freq & FFT
        tab1 = tk.Frame(tabs); tabs.add(tab1, text="Freq & FFT")
        fig1 = plt.Figure(figsize=(fig_w_in, fig_h1_in), dpi=dpi, constrained_layout=True)
        gs1  = mgs.GridSpec(3, 1, figure=fig1, height_ratios=[0.25, 0.25, 0.5])
        ax1  = fig1.add_subplot(gs1[0,0])
        ax2  = fig1.add_subplot(gs1[1,0], sharex=ax1)
        ax3  = fig1.add_subplot(gs1[2,0], sharex=ax1)

        # fixed-height wrapper so the canvas can't expand
        tab1_fig_wrap = tk.Frame(tab1, height=int(0.9 * avail_h_px))  # tweak 0.32 as you like
        tab1_fig_wrap.pack(fill="x", padx=0, pady=0)
        tab1_fig_wrap.pack_propagate(False)  # critical: don't let children force taller height

        can1 = FigureCanvasTkAgg(fig1, master=tab1_fig_wrap)
        w1 = can1.get_tk_widget()
        w1.pack(fill="both", expand=True)

        tb1  = NavigationToolbar2Tk(can1, tab1); tb1.update(); tb1.pack(side="top", fill="x")

        freq_ctrl = tk.Frame(tab1); freq_ctrl.pack(fill="x", padx=10, pady=(4,8))
        tk.Label(freq_ctrl, text="X min (Hz):").pack(side="left")
        xmin_e = tk.Entry(freq_ctrl, width=6); xmin_e.insert(0, "1"); xmin_e.pack(side="left", padx=(0,8))
        tk.Label(freq_ctrl, text="X max (Hz):").pack(side="left")
        xmax_e = tk.Entry(freq_ctrl, width=6); xmax_e.insert(0, str(int(nyq))); xmax_e.pack(side="left", padx=(0,12))
        tk.Button(freq_ctrl, text="Update axis", command=lambda: _redraw_plots(update_xlims=True)).pack(side="right", padx=6)

        # Tab 2: Wavelet TFR
        tab2 = tk.Frame(tabs); tabs.add(tab2, text="Wavelet TFR")

        # Defaults
        tfr_log_freq      = tk.BooleanVar(value=False)
        tfr_norm_scale    = tk.BooleanVar(value=False)
        tfr_baseline_norm = tk.BooleanVar(value=True)
        tfr_method        = tk.StringVar(value="eeglab")   # "eeglab" or "cycles"
        tfr_cycles_var    = tk.DoubleVar(value=6.0)
        tfr_fixed_ms      = tk.DoubleVar(value=40.0)       # default 40 ms
        tfr_use_raw       = tk.BooleanVar(value=True)
        tfr_colors_mode   = tk.StringVar(value="robust")
        marg_start_ms_var = tk.DoubleVar(value=2.0)
        marg_end_ms_var   = tk.DoubleVar(value=80.0)
        marg_stat         = tk.StringVar(value="mean")
        tfr_avg_trials    = tk.BooleanVar(value=True)      # average vs single

        # User-adjustable frequency range (min/max) with clamp to Nyquist
        tfr_fmin_var = tk.DoubleVar(value=10.0)
        tfr_fmax_var = tk.DoubleVar(value=min(1000.0, nyq))

        # Row A
        rowA = tk.Frame(tab2); rowA.pack(fill="x", padx=10, pady=(6,0))
        tk.Label(rowA, text="Method:").pack(side="left")
        ttk.Radiobutton(rowA, text="Fixed window",   variable=tfr_method, value="eeglab",
                        command=lambda: _redraw_plots()).pack(side="left", padx=(4,12))
        ttk.Radiobutton(rowA, text="Constant cycles", variable=tfr_method, value="cycles",
                        command=lambda: _redraw_plots()).pack(side="left", padx=(0,12))
        tk.Checkbutton(rowA, text="Log frequency axis", variable=tfr_log_freq,
                    command=lambda: _redraw_plots()).pack(side="left", padx=(0,12))

        # Row A2: Frequency range controls
        rowA2 = tk.Frame(tab2); rowA2.pack(fill="x", padx=10, pady=(4,0))
        tk.Label(rowA2, text="Freq min (Hz):").pack(side="left")
        tk.Spinbox(rowA2, from_=1.0, to=max(1.0, nyq-1.0), increment=1.0, width=8,
                textvariable=tfr_fmin_var, command=lambda: _redraw_plots()).pack(side="left", padx=(4,12))
        tk.Label(rowA2, text=f"Freq max (Hz) (≤ {int(nyq)}):").pack(side="left")
        tk.Spinbox(rowA2, from_=5.0, to=max(5.0, nyq), increment=5.0, width=8,
                textvariable=tfr_fmax_var, command=lambda: _redraw_plots()).pack(side="left", padx=(4,12))

        # Row B1
        rowB1 = tk.Frame(tab2); rowB1.pack(fill="x", padx=10, pady=(4,0))
        tk.Checkbutton(rowB1, text="Baseline-normalize (pre-stim) → dB", variable=tfr_baseline_norm,
                    command=lambda: _redraw_plots()).pack(side="left", padx=(0,12))
        tk.Checkbutton(rowB1, text="Normalize by √scale (cycles)", variable=tfr_norm_scale,
                    command=lambda: _redraw_plots()).pack(side="left", padx=(0,12))
        tk.Checkbutton(rowB1, text="Use RAW signal (no BP)", variable=tfr_use_raw,
                    command=lambda: (_rebuild_segments_only(), _redraw_plots())).pack(side="left", padx=(0,12))

        # Row B2
        rowB2 = tk.Frame(tab2); rowB2.pack(fill="x", padx=10, pady=(2,0))
        tk.Label(rowB2, text="Cycles:").pack(side="left")
        tk.Spinbox(rowB2, from_=3.0, to=12.0, increment=0.5, width=5,
                textvariable=tfr_cycles_var, command=lambda: _redraw_plots()).pack(side="left", padx=(4,12))
        tk.Label(rowB2, text="Fixed window (ms):").pack(side="left")
        tk.Spinbox(rowB2, from_=10.0, to=120.0, increment=2.0, width=6,
                textvariable=tfr_fixed_ms, command=lambda: _redraw_plots()).pack(side="left", padx=(4,12))
        tk.Label(rowB2, text="Color scale:").pack(side="left")
        cb_scale = ttk.Combobox(rowB2, width=8, state="readonly",
                                textvariable=tfr_colors_mode, values=["robust","full"])
        cb_scale.pack(side="left"); cb_scale.bind("<<ComboboxSelected>>", lambda e: _redraw_plots())

        # Row C
        rowC = tk.Frame(tab2); rowC.pack(fill="x", padx=10, pady=(4,0))
        tk.Label(rowC, text="Marginal window (ms):").pack(side="left")
        tk.Label(rowC, text="Start").pack(side="left", padx=(6,2))
        tk.Spinbox(rowC, from_=0.0, to=500.0, increment=5.0, width=6,
                textvariable=marg_start_ms_var, command=lambda: _redraw_plots()).pack(side="left")
        tk.Label(rowC, text="End").pack(side="left", padx=(6,2))
        tk.Spinbox(rowC, from_=0.0, to=500.0, increment=5.0, width=6,
                textvariable=marg_end_ms_var, command=lambda: _redraw_plots()).pack(side="left")
        tk.Label(rowC, text="Marginal statistic:").pack(side="left", padx=(12,2))
        ms_cb = ttk.Combobox(rowC, width=6, state="readonly", textvariable=marg_stat, values=["mean","p95","max"])
        ms_cb.pack(side="left"); ms_cb.bind("<<ComboboxSelected>>", lambda e: _redraw_plots())
        tk.Label(rowC, text="Event type:").pack(side="left", padx=(16,4))
        dd_tfr = ttk.Combobox(rowC, values=sorted(self.last_stim), state="readonly"); dd_tfr.pack(side="left")

        # Row B3: Average vs Single-trial + live label + Prev/Next
        rowB3 = tk.Frame(tab2); rowB3.pack(fill="x", padx=10, pady=(2,0))
        tk.Checkbutton(rowB3, text="Average trials (TFR)", variable=tfr_avg_trials,
                    command=lambda: _on_tfr_event_change()).pack(side="left", padx=(0,12))
        _tfr_idx_label = tk.Label(rowB3, text=""); _tfr_idx_label.pack(side="left")

        tfr_nav = tk.Frame(rowB3); tfr_nav.pack(side="right")
        tk.Button(tfr_nav, text="◀ Prev", width=7, command=lambda: _bump_tfr(-1)).pack(side="left", padx=(8,6))
        tk.Button(tfr_nav, text="Next ▶", width=7, command=lambda: _bump_tfr(+1)).pack(side="left")

        # Wavelet figure
        # 1) Fixed-height wrapper for the CANVAS ONLY (toolbar stays outside)
        tab2_fig_wrap = tk.Frame(tab2)
        tab2_fig_wrap.pack(fill="x", expand=False)
        tab2_fig_wrap.pack_propagate(False)  # canvas cannot grow this frame

        # 2-column GridSpec: col-0 = heatmap, col-1 = colorbar.
        # ax_m (row 1) spans BOTH columns so there is no empty bottom-right cell.
        # Explicit subplot margins prevent make_axes_locatable from shifting
        # ax_m beyond ax_w and creating a phantom ghost strip on the right.
        fig2 = plt.Figure(figsize=(fig_w_in, fig_h2_in), dpi=dpi)
        gs2  = mgs.GridSpec(2, 2, figure=fig2,
                            width_ratios=[12, 1],
                            height_ratios=[3, 1],
                            left=0.08, right=0.82,
                            top=0.95, bottom=0.08,
                            hspace=0.45, wspace=0.10)

        ax_w = fig2.add_subplot(gs2[0, 0])   # heatmap
        ax_c = fig2.add_subplot(gs2[0, 1])   # colorbar
        ax_m = fig2.add_subplot(gs2[1, :])   # marginal – spans both columns

        can2 = FigureCanvasTkAgg(fig2, master=tab2_fig_wrap)
        w2 = can2.get_tk_widget()
        w2.pack(fill="both", expand=True)

        # 2) Place the toolbar OUTSIDE the fixed-height wrapper (so it never gets clipped)
        tb2 = NavigationToolbar2Tk(can2, tab2)   # <— note 'tab2' here, not 'tab2_fig_wrap'
        tb2.update()
        tb2.pack(side="top", fill="x")

        # Create a placeholder image/artist; will be replaced by QuadMesh
        im = ax_w.imshow(np.zeros((2, 2)), aspect="auto", origin="lower",
                        extent=[-self.pre_time.get(), self.post_time.get(), 10, 1000])
        cbar = fig2.colorbar(im, cax=ax_c)

        ax_w.set_ylabel("Frequency (Hz)")
        ax_w.set_xlabel("Time (ms)")
        ax_m.set_xlabel("Frequency (Hz)", labelpad=4)
        ax_m.set_ylabel("Mean power")

        ax_w.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax_w.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax_m.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax_m.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # 3) After controls are laid out, set the canvas wrapper height to your budget
        def _resize_wavelet_canvas():
            tab2.update_idletasks()
            # Total height budget you want for the WHOLE wavelet tab area
            budget_px = int(0.90 * avail_h_px)   # tweak: 0.55–0.90; larger = taller
            # Subtract ONLY the controls above the figure (toolbar is outside now!)
            ctrl_h = sum(fr.winfo_reqheight() for fr in (rowA, rowA2, rowB1, rowB2, rowC, rowB3))
            target = max(260, budget_px - ctrl_h)
            tab2_fig_wrap.config(height=target)

        tab2.after_idle(_resize_wavelet_canvas)

        def _prune_fig2():
            _prune_axes(fig2, {ax_w, ax_m, cbar.ax})
        _prune_fig2()
        
        # Tab 3: Time-domain
        tab3 = tk.Frame(tabs); tabs.add(tab3, text="Time-domain")
        td_use_raw = tk.BooleanVar(value=False)
        rowTD = tk.Frame(tab3); rowTD.pack(fill="x", padx=10, pady=(6,0))
        tk.Checkbutton(rowTD, text="Use RAW signal (no BP)", variable=td_use_raw,
                    command=lambda: (_rebuild_segments_only(), _redraw_plots())).pack(side="left", padx=(0,12))
        tk.Label(rowTD, text="Event type:").pack(side="left")
        dd_td = ttk.Combobox(rowTD, values=sorted(self.last_stim), state="readonly"); dd_td.pack(side="left", padx=(6,0))

        # TD navigation
        sel_trial_idx = tk.IntVar(value=0)
        td_nav = tk.Frame(rowTD); td_nav.pack(side="right")
        _td_idx_label = tk.Label(td_nav, text=""); _td_idx_label.pack(side="left", padx=(0,8))
        tk.Button(td_nav, text="◀ Prev", width=7, command=lambda: _bump_td(-1)).pack(side="left", padx=(6,4))
        tk.Button(td_nav, text="Next ▶", width=7, command=lambda: _bump_td(+1)).pack(side="left")

        # Fixed-height wrapper for Time-domain tab
        tab3_fig_wrap = tk.Frame(tab3, height=int(0.9 * avail_h_px))  # adjust 0.9 as needed
        tab3_fig_wrap.pack(fill="x", expand=False)
        tab3_fig_wrap.pack_propagate(False)

        fig3 = plt.Figure(figsize=(fig_w_in, fig_h3_in), dpi=dpi)
        ax_td = fig3.add_subplot(111)
        can3 = FigureCanvasTkAgg(fig3, master=tab3_fig_wrap)
        w3 = can3.get_tk_widget()
        w3.pack(fill="both", expand=True)
        tb3  = NavigationToolbar2Tk(can3, tab3); tb3.update(); tb3.pack(side="top", fill="x")

        ax_td.format_coord = lambda x,y: f"Time: {x:.1f} ms   EMG: {y:.3f} {(self.emg_unit or '')}".rstrip()

        # ── Fix FigureCanvasTkAgg ghost rendering ──────────────────────────────
        # FigureCanvasTkAgg.resize() places a NEW canvas image item (via
        # create_image) every time the widget is resized, without removing the
        # old one.  Old items stay at their original centre positions, showing
        # the right-edge content of the figure as a ghost strip.
        #
        # The cleanup function below is the authoritative fix: it runs AFTER
        # every draw_idle() completes, removes all image items except the most
        # recently created one (highest integer ID), and repositions that item
        # at (0,0) with anchor='nw' so the image always starts at the top-left
        # corner and the viewport clips overflow on the right.
        # This works regardless of which matplotlib version or internal path
        # (create_image / itemconfigure / coords) is in use.
        def _cleanup_canvas(can):
            _tk = can.get_tk_widget()
            try:
                image_items = [i for i in _tk.find_all()
                               if _tk.type(i) == "image"]
            except Exception:
                return
            if not image_items:
                return
            keep = max(image_items)          # most recently created
            for item in image_items:
                if item != keep:
                    try: _tk.delete(item)
                    except Exception: pass
            try:
                _tk.coords(keep, 0, 0)        # top-left origin
                _tk.itemconfig(keep, anchor='nw')  # no centring offset
            except Exception:
                pass

        # Also intercept create_image to place new items at (0,0) nw directly,
        # reducing the window of time when a misplaced item is briefly visible.
        def _patch_create_image(can):
            _tk  = can.get_tk_widget()
            _orig = _tk.create_image
            def _nw(*args, **kwargs):
                img = kwargs.get('image')
                if img is None and len(args) >= 3:
                    img = args[2]
                return _orig(0, 0, image=img, anchor='nw')
            _tk.create_image = _nw

        def _log_cfg(e, _name):
            _tw = {"can1":can1,"can2":can2,"can3":can3}[_name].get_tk_widget()
            _f  = {"can1":fig1,"can2":fig2,"can3":fig3}[_name]
            _items = [(_i, _tw.coords(_i), _tw.itemcget(_i,"anchor"))
                      for _i in _tw.find_all() if _tw.type(_i)=="image"]
            print(f"[CFG] {_name}: event={e.width}x{e.height} "
                  f"widget={_tw.winfo_width()}x{_tw.winfo_height()} "
                  f"fig_px={_f.get_figwidth()*_f.dpi:.0f}x{_f.get_figheight()*_f.dpi:.0f} "
                  f"items={_items}")
        # ── FIX: prevent _update_device_pixel_ratio from firing on <Map> ──
        # On Windows, matplotlib computes device_pixel_ratio = tk_scaling/(96/72).
        # When Tk reports 72 DPI, ratio = 0.75 → fig.dpi drops to 75 → figure
        # renders at 544px into a 725px canvas, leaving a stale-content ghost
        # strip on the right side of every plot.
        # Replacing the <Map> binding with a no-op prevents any DPI change.
        for _can in (can1, can2, can3):
            _patch_create_image(_can)
            _cleanup_canvas(_can)
            _can.get_tk_widget().unbind("<Map>")
            _can.get_tk_widget().bind("<Map>", lambda e: None)
        for _cname, _c in [("can1",can1),("can2",can2),("can3",can3)]:
            _c.get_tk_widget().bind("<Configure>",
                lambda e, n=_cname: _log_cfg(e, n), add="+")

        # ─────────────────── State + wavelet builders ───────────────────
        current_P = current_freqs = current_times = None
        power_unit = "dB"
        w_resp = h_tot = w_gd = gd_ms = None
        emg_f_full = emg_raw_full = None
        events_tfr = {}; events_td = {}

        def _morlet_kernel(f0_hz, fs, s_samples, L=None):
            s_samples = float(max(s_samples, 1e-9))
            if L is None: L = int(np.ceil(12.0 * s_samples)) | 1
            half = L//2; n = np.arange(-half, half+1); tt = n/fs
            gauss = np.exp(-0.5*(n/s_samples)**2); carrier = np.exp(1j*2*np.pi*f0_hz*tt)
            w = gauss*carrier
            w /= np.sqrt(np.sum(np.abs(w)**2) + 1e-20)
            return w

        def _cycles_morlet_tfr(x, fs, freqs, w_cycles, norm_scale=False):
            x = np.asarray(x, float); n = x.size; P = np.zeros((len(freqs), n))
            for i, f0 in enumerate(freqs):
                s_samples = (w_cycles/(2.0*np.pi*float(f0)))*fs
                wv = _morlet_kernel(f0, fs, s_samples); pad = len(wv)//2
                coef = fftconvolve(np.pad(x,(pad,pad),'reflect'), np.conj(wv[::-1]), mode="same")[pad:-pad]
                if norm_scale: coef = coef/np.sqrt(max(s_samples,1e-12))
                P[i,:] = (np.abs(coef)**2)
            return P

        def _fixed_window_morlet_tfr(x, fs, freqs, fixed_ms):
            x = np.asarray(x, float); n = x.size
            s_samples_const = (float(fixed_ms)/1000.0/6.0)*fs  # 6 cycles in fixed window (EEGLAB-like)
            L = (int(np.ceil(12.0*s_samples_const)) | 1)
            P = np.zeros((len(freqs), n))
            for i, f0 in enumerate(freqs):
                wv = _morlet_kernel(f0, fs, s_samples_const, L=L); pad = len(wv)//2
                coef = fftconvolve(np.pad(x,(pad,pad),'reflect'), np.conj(wv[::-1]), mode="same")[pad:-pad]
                P[i,:] = (np.abs(coef)**2)
            return P

        # ───────────────────────────── Build segments ──────────────────────────
        def _rebuild_segments_only():
            nonlocal events_tfr, events_td, emg_raw_full, emg_f_full
            pre_ms  = float(self.pre_time.get()); post_ms = float(self.post_time.get())
            sb = int(round(pre_ms*fs/1000.0)); sa = int(round(post_ms*fs/1000.0))

            emg_raw_full = self.raw_emg.copy()
            emg_f_full   = _apply_pipeline(self.raw_emg)

            src_tfr = emg_raw_full if tfr_use_raw.get() else emg_f_full
            src_td  = emg_raw_full if td_use_raw.get()  else emg_f_full

            events_tfr, events_td = {}, {}
            if self.raw_emg is not None and hasattr(self, "last_times"):
                for stype, times in self.last_stim.items():
                    segs_tfr, segs_td = [], []
                    for t0 in times:
                        idx0  = np.argmin(np.abs(self.last_times - t0))
                        # TFR
                        start = max(0, idx0 - sb); end = min(len(src_tfr), idx0 + sa)
                        seg   = src_tfr[start:end]; t_ax = np.linspace(-pre_ms, post_ms, len(seg))
                        segs_tfr.append((seg, t_ax))
                        # TD
                        start = max(0, idx0 - sb); end = min(len(src_td), idx0 + sa)
                        seg2  = src_td[start:end]; t_ax2 = np.linspace(-pre_ms, post_ms, len(seg2))
                        segs_td.append((seg2, t_ax2))
                    if segs_tfr: events_tfr[stype] = segs_tfr
                    if segs_td:  events_td [stype] = segs_td

            # Keep comboboxes populated and preserve current selection if possible
            keys_tfr = sorted(events_tfr.keys()); dd_tfr["values"] = keys_tfr
            if keys_tfr and dd_tfr.get() not in keys_tfr: dd_tfr.set(keys_tfr[0])
            keys_td  = sorted(events_td.keys());  dd_td["values"]  = keys_td
            if keys_td  and dd_td.get()  not in keys_td:  dd_td.set(keys_td[0])

            # Clamp persisted trial selections within new lengths (both tabs)
            for et, idx in list(self._preview_sel_trial.items()):
                n1 = len(events_tfr.get(et, []))
                n2 = len(events_td .get(et, []))
                n  = max(n1, n2)
                if n == 0: self._preview_sel_trial[et] = 0
                else:      self._preview_sel_trial[et] = int(min(max(0, idx), n-1))
            if dd_td.get():
                sel_trial_idx.set(int(self._preview_sel_trial.get(dd_td.get(), 0)))

        def _recompute_everything():
            nonlocal w_resp, h_tot, w_gd, gd_ms
            _, H, w_gd, gd_ms = _freq_response_curves()
            _rebuild_segments_only()

        # ───────────────────────────── Redraw logic ────────────────────────────
        def _pretty_log_ticks(fmin, fmax):
            mult=[1,2,3,5,7]; ticks=[]
            d0=int(np.floor(np.log10(max(fmin,1e-6)))); d1=int(np.ceil(np.log10(max(fmax,1e-6))))
            for d in range(d0,d1+1):
                for m in mult:
                    v=m*(10**d)
                    if fmin*0.999<=v<=fmax*1.001: ticks.append(v)
            return sorted(set(ticks))

        def _update_tfr_idx_label():
            et = dd_tfr.get()
            n  = len(events_tfr.get(et, []))
            if tfr_avg_trials.get():
                _tfr_idx_label.config(text=f"(Averaging {n} trial{'s' if n!=1 else ''})")
            else:
                cur = int(self._preview_sel_trial.get(et, 0))
                _tfr_idx_label.config(text=f"(Trial {cur+1}/{max(n,0)})")

        def _on_tfr_event_change():
            et = dd_tfr.get()
            if et and et not in self._preview_sel_trial:
                self._preview_sel_trial[et] = 0
            _update_tfr_idx_label()
            _redraw_plots()

        def _bump_tfr(delta: int):
            et = dd_tfr.get()
            n = len(events_tfr.get(et, []))
            if n == 0: return
            if tfr_avg_trials.get():
                tfr_avg_trials.set(False)
            cur = int(self._preview_sel_trial.get(et, 0))
            cur = max(0, min(n-1, cur + delta))
            self._preview_sel_trial[et] = cur
            # Keep time-domain selection in sync when same event
            if dd_td.get() == et and et in events_td:
                sel_td_n = len(events_td[et])
                if sel_td_n:
                    sel_idx_sync = min(cur, sel_td_n - 1)
                    sel_trial_idx.set(sel_idx_sync)
            _update_tfr_idx_label()
            _redraw_plots()

        def _bump_td(delta: int):
            et = dd_td.get()
            n = len(events_td.get(et, []))
            if n == 0: return
            cur = int(self._preview_sel_trial.get(et, 0))
            cur = max(0, min(n-1, cur + delta))
            self._preview_sel_trial[et] = cur
            sel_trial_idx.set(cur)
            if dd_tfr.get() == et:
                _update_tfr_idx_label()
            _redraw_plots()

        def _redraw_plots(update_xlims=False):
            nonlocal current_P, current_freqs, current_times, power_unit, im, cbar

            pre_ms  = float(self.pre_time.get()); post_ms = float(self.post_time.get())
            sb = int(round(pre_ms*fs/1000.0)); sa = int(round(post_ms*fs/1000.0))

            # Tab1: Freq/FFT
            ax1.clear(); ax2.clear(); ax3.clear()
            _prune_axes(fig1, {ax1, ax2, ax3})
            xmin = float(xmin_e.get()); xmax = float(xmax_e.get())
            w, H, w_gd, gd_ms = _freq_response_curves()
            ax1.plot(np.linspace(0, nyq, len(H)), 20*np.log10(np.maximum(np.abs(H), 1e-12)), lw=2)
            ax1.set_ylabel("Gain (dB)")
            ax2.plot(w_gd, gd_ms, lw=1); ax2.set_ylabel("Delay (ms)")
            if self.raw_emg is not None:
                N=len(self.raw_emg); f=np.fft.rfftfreq(N, 1/fs)
                raw=np.abs(np.fft.rfft(self.raw_emg))
                flt=np.abs(np.fft.rfft(_apply_pipeline(self.raw_emg)))
                ax3.plot(f, raw, lw=0.8, label="Raw"); ax3.plot(f, flt, lw=0.8, label="Filt"); ax3.legend(fontsize=12)
                mm=(f>=xmin)&(f<=xmax)
                if mm.any(): ax3.set_ylim(0, max(raw[mm].max(), flt[mm].max())*1.1)
            for ax in (ax1,ax2,ax3): ax.set_xlim(xmin,xmax); ax.grid(ls=":", lw=0.5)
            ax3.set_xlabel("Frequency (Hz)")
            can1.draw_idle()
            can1.get_tk_widget().after_idle(
                lambda _c=can1: _cleanup_canvas(_c))
            if update_xlims: return

            # Tab2: TFR
            # Fully clear both axes so no stale artists accumulate across redraws.
            ax_w.cla()
            ax_m.cla(); ax_m.grid(ls=":", lw=0.5)
            # Clear custom attribute lists (the artists they referenced are now gone)
            for attr in ("_marg_art", "_freq_markers", "_quadmesh_art"):
                setattr(ax_w, attr, [])
            _prune_fig2()

            sel = dd_tfr.get()
            if sel in events_tfr and events_tfr[sel]:
                desired_len = sb + sa
                raw_segments = events_tfr[sel]

                # Ensure uniform length by pad/trim to desired_len
                segs = []
                for seg,_t in raw_segments:
                    seg = np.asarray(seg, float)
                    if len(seg) == desired_len:
                        segs.append(seg)
                    elif len(seg) > desired_len:
                        segs.append(seg[:desired_len])
                    else:
                        segs.append(np.pad(seg, (0, desired_len-len(seg)), mode="reflect"))

                # Clamp requested freq range to (0, nyq]
                fmin_req = float(tfr_fmin_var.get())
                fmax_req = float(tfr_fmax_var.get())
                fmin = max(1.0, min(fmin_req, nyq-1.0))
                fmax = max(fmin+1.0, min(fmax_req, nyq))
                n_freqs = 100

                freqs = (np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
                        if tfr_log_freq.get() else np.linspace(fmin, fmax, n_freqs))
                w_cycles = float(tfr_cycles_var.get()); fixed_ms = float(tfr_fixed_ms.get())

                # Average vs single-trial
                if tfr_avg_trials.get():
                    P_list=[]
                    for s in segs:
                        if tfr_method.get()=="eeglab":
                            P_list.append(_fixed_window_morlet_tfr(s, fs, freqs, fixed_ms))
                        else:
                            P_list.append(_cycles_morlet_tfr(s, fs, freqs, w_cycles=w_cycles,
                                                            norm_scale=bool(tfr_norm_scale.get())))
                    P = np.mean(P_list, axis=0); trial_note = None
                else:
                    cur = int(self._preview_sel_trial.get(sel, 0))
                    cur = max(0, min(len(segs)-1, cur)); self._preview_sel_trial[sel] = cur
                    s = segs[cur]
                    if tfr_method.get()=="eeglab":
                        P = _fixed_window_morlet_tfr(s, fs, freqs, fixed_ms)
                    else:
                        P = _cycles_morlet_tfr(s, fs, freqs, w_cycles=w_cycles,
                                            norm_scale=bool(tfr_norm_scale.get()))
                    trial_note = f"Trial {cur+1}/{len(segs)}"

                # clamp to avoid log10(0)
                P = np.maximum(P, eps)
                t_axis = np.linspace(-pre_ms, post_ms, P.shape[1], endpoint=True)

                # Baseline normalization (dB) with safe clamp
                if tfr_baseline_norm.get():
                    bmask = (t_axis >= -pre_ms) & (t_axis <= -pre_ms/2.0)
                    if not bmask.any(): bmask = (t_axis >= -pre_ms) & (t_axis <= -2*pre_ms/3.0)
                    B = np.maximum(P[:, bmask].mean(axis=1, keepdims=True), eps)
                    P = 10.0*np.log10(P/B); power_unit="dB"
                else:
                    power_unit="a.u."

                current_P = P; current_freqs=freqs; current_times=t_axis

                # Color scale FIRST
                if tfr_colors_mode.get()=="robust":
                    vmin=float(np.nanpercentile(P, 2.0)); vmax=float(np.nanpercentile(P, 98.0)); note=" (robust 2–98%)"
                else:
                    vmin=float(np.nanmin(P)); vmax=float(np.nanmax(P)); note=" (full range)"
                if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmin == vmax):
                    vmin, vmax = (0.0, 1.0) if not np.isfinite(vmin+vmax) else (vmin-1.0, vmax+1.0)

                # Build time and frequency bin edges
                dt_edges = np.empty(P.shape[1] + 1, dtype=float)
                dt_edges[1:-1] = 0.5 * (t_axis[:-1] + t_axis[1:])
                dt_edges[0]    = t_axis[0]  - (t_axis[1] - t_axis[0]) / 2.0
                dt_edges[-1]   = t_axis[-1] + (t_axis[-1] - t_axis[-2]) / 2.0

                if tfr_log_freq.get():
                    f_edges = np.empty(len(freqs) + 1, dtype=float)
                    ratios  = freqs[1:] / freqs[:-1]
                    f_edges[1:-1] = np.sqrt(freqs[:-1] * freqs[1:])
                    f_edges[0]  = freqs[0]  / np.sqrt(ratios[0])
                    f_edges[-1] = freqs[-1] * np.sqrt(ratios[-1])
                else:
                    f_edges = np.empty(len(freqs) + 1, dtype=float)
                    f_edges[1:-1] = 0.5 * (freqs[:-1] + freqs[1:])
                    f_edges[0]    = freqs[0]  - (freqs[1] - freqs[0]) / 2.0
                    f_edges[-1]   = freqs[-1] + (freqs[-1] - freqs[-2]) / 2.0

                # Draw the heatmap. ax_w was already cleared above so there
                # are no stale artists to remove first.
                quad = ax_w.pcolormesh(dt_edges, f_edges, P, shading="auto",
                                    norm=Normalize(vmin=vmin, vmax=vmax))
                im = quad  # keep a valid artist handle for colorbar/hover

                # Re-apply axis labels lost by cla()
                ax_w.set_ylabel("Frequency (Hz)")
                ax_w.set_xlabel("Time (ms)")
                ax_w.yaxis.set_major_locator(MaxNLocator(nbins=7))
                ax_w.xaxis.set_major_locator(MaxNLocator(nbins=7))

                # Enforce y-axis limit up to Nyquist
                ax_w.set_yscale('log' if tfr_log_freq.get() else 'linear')
                ax_w.set_ylim(freqs[0], min(freqs[-1], nyq))
                if tfr_log_freq.get():
                    yticks=_pretty_log_ticks(freqs[0], min(freqs[-1], nyq))
                    ax_w.yaxis.set_major_locator(FixedLocator(yticks))
                    ax_w.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{int(round(y))}"))
                ax_w.grid(ls=":", lw=0.4)

                # Clear ax_c content without detaching it from fig2 (cbar.remove()
                # would call ax_c.remove(), making ax_c.get_figure() return None).
                ax_c.cla()
                cbar = fig2.colorbar(im, cax=ax_c)
                cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                cbar.set_label(f"Power ({power_unit}){note}" + (f" — {trial_note}" if trial_note else ""))

                # Marginal window & statistic
                mstart=float(marg_start_ms_var.get()); mend=float(marg_end_ms_var.get())
                ax_w._marg_art = [ax_w.axvspan(mstart, mend, color='k', alpha=0.08, lw=0)]
                win=(t_axis >= mstart) & (t_axis <= mend)
                if not np.any(win): win=(t_axis >= 0.0) & (t_axis <= post_ms)
                W=P[:,win]
                if marg_stat.get()=="max": marg=np.max(W,axis=1)
                elif marg_stat.get()=="p95": marg=np.percentile(W,95,axis=1)
                else: marg=np.mean(W,axis=1)

                peak_idx=int(np.argmax(marg)); peak_freq=float(freqs[peak_idx])
                marg_lin = np.power(10.0, marg/10.0) if tfr_baseline_norm.get() else (marg+0.0)
                com_freq=float(np.sum(freqs*marg_lin)/np.sum(marg_lin))

                ax_m.set_xlim(freqs[0], min(freqs[-1], nyq))
                if tfr_log_freq.get():
                    ax_m.semilogx(freqs, marg, lw=1.7)
                    xticks=_pretty_log_ticks(freqs[0], min(freqs[-1], nyq))
                    ax_m.xaxis.set_major_locator(FixedLocator(xticks))
                    ax_m.xaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{int(round(y))}"))
                else:
                    ax_m.plot(freqs, marg, lw=1.7)

                ax_m.axvline(peak_freq, color="C3", ls="--", lw=1.2, label="Peak")
                ax_m.axvline(com_freq,  color="tab:purple", ls="-.",  lw=1.2, label="COM")
                ax_m.set_xlabel("Frequency (Hz)", labelpad=4)
                ax_m.set_ylabel(f"Mean power ({'dB' if tfr_baseline_norm.get() else 'a.u.'})")
                ax_m.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax_m.xaxis.set_major_locator(MaxNLocator(nbins=7))
                ax_m.legend(loc="upper left", fontsize=12, frameon=False)
                ax_m.text(0.98, 0.98,
                        f"Window: {mstart:.0f}–{mend:.0f} ms\nPeak: {peak_freq:.0f} Hz\nCOM: {com_freq:.0f} Hz\nStat: {marg_stat.get()}",
                        ha="right", va="top", transform=ax_m.transAxes, fontsize=12)

                ax_w._freq_markers = [
                    ax_w.axhline(peak_freq, color='C3', ls='--', lw=0.9, alpha=0.7),
                    ax_w.axhline(com_freq,  color='tab:purple', ls='-.',  lw=0.9, alpha=0.7),
                ]
            else:
                # No events: keep a valid artist for the colorbar/hover
                current_P=current_freqs=current_times=None
                ax_w.cla()
                im = ax_w.imshow(np.zeros((2,2)), aspect="auto", origin="lower",
                                extent=[-pre_ms, post_ms, 10, min(1000, nyq)])
                ax_w.set_ylabel("Frequency (Hz)"); ax_w.set_xlabel("Time (ms)")
                ax_c.cla()
                cbar = fig2.colorbar(im, cax=ax_c)

            _update_tfr_idx_label()
            can2.draw_idle()
            can2.get_tk_widget().after_idle(
                lambda _c=can2: _cleanup_canvas(_c))

            # Tab3: Time domain (overlay all, highlight selected)
            ax_td.clear(); _prune_axes(fig3, {ax_td})
            sel_td = dd_td.get()
            if sel_td in events_td and events_td[sel_td]:
                n = len(events_td[sel_td])
                cur = int(self._preview_sel_trial.get(sel_td, 0))
                cur = max(0, min(n-1, cur))
                self._preview_sel_trial[sel_td] = cur
                sel_trial_idx.set(cur)

                # plot all in thin grey
                for i, (seg, t_ax) in enumerate(events_td[sel_td]):
                    if i == cur: continue
                    ax_td.plot(t_ax, seg, alpha=0.35, lw=0.7, color="#bbbbbb")
                # highlight selected
                seg_sel, t_sel = events_td[sel_td][cur]
                ax_td.plot(t_sel, seg_sel, lw=1.6, color="C0", label=f"Trial {cur+1}/{n}")
                _td_idx_label.config(text=f"(Trial {cur+1}/{n})")
                ax_td.legend(loc="upper right", fontsize=12, frameon=False)

            ax_td.axvline(0, color='k', ls='--', lw=0.8)
            ax_td.set(title=f"EMG around {sel_td}", xlabel="Time (ms)", ylabel=(self.emg_unit or "EMG"))
            can3.draw_idle()
            can3.get_tk_widget().after_idle(
                lambda _c=can3: _cleanup_canvas(_c))

        # Hover text for TFR (into toolbar)
        def _format_tfr(x, y):
            if x is None or y is None or current_P is None: return ""
            try:
                if not (current_times[0] <= x <= current_times[-1] and current_freqs[0] <= y <= current_freqs[-1]):
                    return f"Time: {x:.1f} ms   Freq: {y:.1f} Hz"
                it = int(np.argmin(np.abs(current_times - x)))
                jf = int(np.argmin(np.abs(current_freqs - y)))
                p = float(current_P[jf, it])
                try:
                    vmin, vmax = im.norm.vmin, im.norm.vmax
                except Exception:
                    vmin, vmax = np.nanmin(current_P), np.nanmax(current_P)
                p_disp = min(max(p, vmin), vmax)
                clipped = " (clipped)" if p_disp != p else ""
                return f"Time: {x:.1f} ms   Freq: {y:.1f} Hz   Power: {p_disp:.4g} {power_unit}{clipped}"
            except Exception:
                return f"Time: {x:.1f} ms   Freq: {y:.1f} Hz"

        ax_w.format_coord = _format_tfr
        for cid in list(fig2.canvas.callbacks.callbacks.get('motion_notify_event', {})):
            fig2.canvas.mpl_disconnect(cid)
        fig2.canvas.mpl_connect(
            "motion_notify_event",
            lambda e: tb2.set_message(ax_w.format_coord(e.xdata, e.ydata))
            if (e.inaxes is ax_w and e.xdata is not None and e.ydata is not None) else None
        )

        # ───────────────────────── Init & bindings ─────────────────────────
        def _after_init_autoselect_and_draw():
            # Process pack layout so widget sizes are available.
            popup.update_idletasks()

            # Use the popup's own configured width as the authoritative
            # figure width.  tabs.winfo_width() can be misleadingly large
            # if any control row forces content wider than the viewport.
            # Subtracting the scrollbar (~17px) and notebook padding (20px)
            # gives a safe figure width guaranteed to fit in the viewport.
            safe_w = base_w - 40
            tab_w  = min(max(tabs.winfo_width(), 400), safe_w)

            # 1. Pin wrapper frames so pack cannot make them wider.
            for _wrap in (tab1_fig_wrap, tab2_fig_wrap, tab3_fig_wrap):
                _wrap.config(width=tab_w)

            # 2. Explicitly constrain each Tk canvas widget to tab_w.
            #    This forces FigureCanvasTkAgg.resize() to fire with exactly
            #    tab_w, placing the canvas image item at (tab_w//2, ...) so
            #    the image spans 0..tab_w and never overflows the viewport.
            for _can in (can1, can2, can3):
                _can.get_tk_widget().config(width=tab_w)

            # 3. Set matplotlib figure widths to match.
            for _fig in (fig1, fig2, fig3):
                _fig.set_size_inches(tab_w / dpi,
                                     _fig.get_figheight(),
                                     forward=False)

            # 4. Process the Configure events that steps 1-2 generate so
            #    FigureCanvasTkAgg.resize() runs before _redraw_plots().
            popup.update_idletasks()
            if self.last_stim:
                first = sorted(self.last_stim)[0]
                if not dd_tfr.get(): dd_tfr.set(first)
                if not dd_td.get():  dd_td.set(first)
            # ── DIAGNOSTIC: print sizes and canvas items ──
            _tk_dpi = popup.winfo_fpixels("1i")  # Tk reported screen DPI
            print(f"[DIAG] base_w={base_w} safe_w={safe_w} tab_w={tab_w}")
            print(f"[DIAG] Tk screen DPI={_tk_dpi:.1f}, hardcoded dpi={dpi}")
            print(f"[DIAG] fig2 size px = {fig2.get_figwidth()*fig2.dpi:.0f} x {fig2.get_figheight()*fig2.dpi:.0f}")
            print(f"[DIAG] fig3 size px = {fig3.get_figwidth()*fig3.dpi:.0f} x {fig3.get_figheight()*fig3.dpi:.0f}")
            print(f"[DIAG] popup={popup.winfo_width()}x{popup.winfo_height()}")
            print(f"[DIAG] tabs.winfo_width()={tabs.winfo_width()}")
            for _name, _can in [("can1",can1),("can2",can2),("can3",can3)]:
                _tw = _can.get_tk_widget()
                _items = [(_i, _tw.type(_i), _tw.coords(_i),
                           _tw.itemcget(_i,"anchor"))
                          for _i in _tw.find_all() if _tw.type(_i)=="image"]
                print(f"[DIAG] {_name}: winfo_width={_tw.winfo_width()} "
                      f"cget_width={_tw.cget('width')} items={_items}")
            # ── END DIAGNOSTIC ──
            _recompute_everything()
            _redraw_plots()
            # ── POST-DRAW DIAGNOSTIC ──
            def _post_draw_diag():
                for _name, _can in [("can1",can1),("can2",can2),("can3",can3)]:
                    _tw = _can.get_tk_widget()
                    _items = [(_i, _tw.type(_i), _tw.coords(_i),
                               _tw.itemcget(_i,"anchor"))
                              for _i in _tw.find_all() if _tw.type(_i)=="image"]
                    print(f"[POST] {_name}: items={_items}")
            popup.after(500, _post_draw_diag)

        dd_tfr.bind("<<ComboboxSelected>>", lambda e: _on_tfr_event_change())
        dd_td .bind("<<ComboboxSelected>>", lambda e: _redraw_plots())

        # When the user switches tabs the newly-visible canvas widget has just
        # received its real size from pack.  Wait one Tk event cycle (after 50 ms)
        # so FigureCanvasTkAgg can process the <Configure> resize event, then
        # redraw so the figure matches the now-correct canvas dimensions.
        def _tab_changed_diag(event):
            def _dump():
                for _ax, _nm in [(ax_w,"ax_w"),(ax_c,"ax_c"),(ax_m,"ax_m")]:
                    _p = _ax.get_position()
                    _fw = fig2.get_figwidth()*fig2.dpi
                    print(f"[POS] {_nm}: x0={_p.x0*_fw:.1f} x1={_p.x1*_fw:.1f} "
                          f"y0={_p.y0*fig2.get_figheight()*fig2.dpi:.1f} "
                          f"in fig2 {_fw:.0f}px wide")
                for _n, _c in [("can1",can1),("can2",can2),("can3",can3)]:
                    _tw = _c.get_tk_widget()
                    _fw = _c.figure.get_figwidth()*_c.figure.dpi
                    _fh = _c.figure.get_figheight()*_c.figure.dpi
                    _items = [(_i, _tw.type(_i), _tw.coords(_i),
                               _tw.itemcget(_i,"anchor"))
                              for _i in _tw.find_all() if _tw.type(_i)=="image"]
                    print(f"[TAB] {_n}: widget={_tw.winfo_width()}x{_tw.winfo_height()} "
                          f"fig_px={_fw:.0f}x{_fh:.0f} items={_items}")
            popup.after(300, _dump)
        tabs.bind("<<NotebookTabChanged>>", _tab_changed_diag, add="+")
        tabs.bind("<<NotebookTabChanged>>",
                  lambda e: popup.after(50, _redraw_plots), add="+")

        # Delay the initial draw so Tkinter has time to fully render the popup
        # and fire all <Configure> resize events before matplotlib draws.
        popup.after(150, _after_init_autoselect_and_draw)

