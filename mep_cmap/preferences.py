"""
mep_cmap.preferences
~~~~~~~~~~~~~~~~~~~~
DPI-aware UI scaling and user preferences.
"""
from __future__ import annotations
import json, platform, sys
from pathlib import Path

PREFS_DIR  = Path.home() / ".mep_cmap"
PREFS_FILE = PREFS_DIR / "preferences.json"
DEFAULTS   = {
    "font_scale":          1.0,
    "latency_profiles":    None,   # None → use LATENCY_PROFILE_DEFAULTS
    "default_latency_key": None,   # None → use DEFAULT_LATENCY_KEY
    "onset_method":        "bigoni",  # "peak_fraction" | "bootstrap" | "bigoni"
    # Peak-fraction method parameters
    "onset_peak_frac":          0.15,
    "onset_min_peak_amplitude": 0.05,
    "onset_slope_threshold":    0.08,
    # Bootstrap method parameters
    "onset_bootstrap_crit":     1.96,
    "onset_bootstrap_n":        500,
    # Bigoni method parameters
    "onset_bigoni_smooth_ms":   2.0,
    "onset_bigoni_min_run_ms":  1.0,
}

# ── Canonical latency profiles ────────────────────────────────────────────────
# Each entry defines the physiological MEP onset search window for a
# stim-type / muscle-group combination, derived from published normative data.
# References: Groppa et al. 2012 (IFCN), Colebatch et al. 1990,
#             Cantone et al. 2023, Miyano et al. 2026.
LATENCY_PROFILE_DEFAULTS = [
    {"stim_type": "TMS",              "muscle": "Deltoid / Trapezius",           "min_ms":  8, "max_ms": 16},
    {"stim_type": "TMS",              "muscle": "Biceps / Triceps brachii",      "min_ms": 12, "max_ms": 20},
    {"stim_type": "TMS",              "muscle": "Trunk / External oblique",      "min_ms": 12, "max_ms": 22},
    {"stim_type": "TMS",              "muscle": "Hand / FDI / APB / ADM",        "min_ms": 18, "max_ms": 28},
    {"stim_type": "TMS",              "muscle": "Forearm (FCR / ECR)",           "min_ms": 16, "max_ms": 26},
    {"stim_type": "TMS",              "muscle": "Vastus lateralis / Quad",       "min_ms": 18, "max_ms": 30},
    {"stim_type": "TMS",              "muscle": "Hamstrings",                    "min_ms": 18, "max_ms": 32},
    {"stim_type": "TMS",              "muscle": "Tibialis anterior / Leg",       "min_ms": 28, "max_ms": 45},
    {"stim_type": "Peripheral nerve", "muscle": "Upper limb (M-wave)",           "min_ms":  2, "max_ms": 12},
    {"stim_type": "Peripheral nerve", "muscle": "Lower limb (M-wave)",           "min_ms":  4, "max_ms": 18},
    {"stim_type": "Custom",           "muscle": "Custom",                        "min_ms": 10, "max_ms": 50},
]

# The profile pre-selected by default in Stage 1a for new stimulus types
DEFAULT_LATENCY_KEY = ("TMS", "Hand / FDI / APB / ADM")

BASE_FONT_SIZES = {
    "TkDefaultFont":      11,
    "TkTextFont":         11,
    "TkFixedFont":        11,
    "TkMenuFont":         11,
    "TkHeadingFont":      12,
    "TkCaptionFont":      12,
    "TkSmallCaptionFont": 10,
    "TkIconFont":         10,
    "TkTooltipFont":      10,
}

class Preferences:
    def __init__(self):
        self._data: dict = dict(DEFAULTS)
        self._dpi_scale: float = 1.0
        self.load()

    def load(self):
        try:
            if PREFS_FILE.exists():
                stored = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
                for k, v in stored.items():
                    if k in DEFAULTS:
                        self._data[k] = v
        except Exception:
            pass

    def save(self):
        try:
            PREFS_DIR.mkdir(parents=True, exist_ok=True)
            PREFS_FILE.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def reset(self):
        self._data = dict(DEFAULTS)
        self.save()

    @property
    def font_scale(self) -> float:
        return float(self._data.get("font_scale", 1.0))

    def set_font_scale(self, value: float):
        self._data["font_scale"] = round(max(0.7, min(1.5, float(value))), 2)
        self.save()

    # ── Latency profiles ──────────────────────────────────────────────────────

    @property
    def latency_profiles(self) -> list:
        """Return the current list of latency profile dicts.

        Falls back to LATENCY_PROFILE_DEFAULTS if the user has not customised
        them, ensuring the list always reflects the latest literature values
        on a fresh install.
        """
        stored = self._data.get("latency_profiles")
        if stored and isinstance(stored, list) and len(stored) > 0:
            return stored
        return [dict(p) for p in LATENCY_PROFILE_DEFAULTS]

    def latency_profiles_as_dict(self) -> dict:
        """Return {(stim_type, muscle): (min_ms, max_ms)} for fast lookup."""
        return {(p["stim_type"], p["muscle"]): (p["min_ms"], p["max_ms"])
                for p in self.latency_profiles}

    def muscle_options(self) -> dict:
        """Return {stim_type: [muscle, ...]} derived from the current profiles."""
        opts: dict = {}
        for p in self.latency_profiles:
            opts.setdefault(p["stim_type"], []).append(p["muscle"])
        return opts

    @property
    def default_latency_key(self) -> tuple:
        """(stim_type, muscle) to pre-select in Stage 1a for new stim types."""
        stored = self._data.get("default_latency_key")
        if stored and isinstance(stored, (list, tuple)) and len(stored) == 2:
            return tuple(stored)
        return DEFAULT_LATENCY_KEY

    def set_latency_prefs(self, profiles: list, default_key: tuple):
        """Persist user-edited latency profiles and the chosen default."""
        self._data["latency_profiles"]    = profiles
        self._data["default_latency_key"] = list(default_key)
        self.save()

    # ── Onset detection method ────────────────────────────────────────────────

    @property
    def onset_method(self) -> str:
        """Active onset detection method key. One of: 'peak_fraction', 'bootstrap'."""
        return str(self._data.get("onset_method", "peak_fraction"))

    # ── Peak-fraction parameters ──────────────────────────────────────────────

    @property
    def onset_peak_frac(self) -> float:
        return float(self._data.get("onset_peak_frac", 0.15))

    @property
    def onset_min_peak_amplitude(self) -> float:
        return float(self._data.get("onset_min_peak_amplitude", 0.05))

    @property
    def onset_slope_threshold(self) -> float:
        return float(self._data.get("onset_slope_threshold", 0.08))

    # ── Bootstrap parameters ──────────────────────────────────────────────────

    @property
    def onset_bootstrap_crit(self) -> float:
        return float(self._data.get("onset_bootstrap_crit", 1.96))

    @property
    def onset_bootstrap_n(self) -> int:
        return int(self._data.get("onset_bootstrap_n", 500))

    @property
    def onset_bigoni_smooth_ms(self) -> float:
        return float(self._data.get("onset_bigoni_smooth_ms", 2.0))

    @property
    def onset_bigoni_min_run_ms(self) -> float:
        return float(self._data.get("onset_bigoni_min_run_ms", 1.0))

    def set_onset_prefs(self, method: str,
                        peak_frac: float, min_peak_amplitude: float,
                        slope_threshold: float,
                        bootstrap_crit: float, bootstrap_n: int,
                        bigoni_smooth_ms: float = 2.0,
                        bigoni_min_run_ms: float = 1.0):
        """Persist all onset detection preferences."""
        self._data["onset_method"]              = method
        self._data["onset_peak_frac"]           = round(float(peak_frac), 4)
        self._data["onset_min_peak_amplitude"]  = round(float(min_peak_amplitude), 4)
        self._data["onset_slope_threshold"]     = round(float(slope_threshold), 4)
        self._data["onset_bootstrap_crit"]      = round(float(bootstrap_crit), 4)
        self._data["onset_bootstrap_n"]         = int(bootstrap_n)
        self._data["onset_bigoni_smooth_ms"]    = round(float(bigoni_smooth_ms), 2)
        self._data["onset_bigoni_min_run_ms"]   = round(float(bigoni_min_run_ms), 2)
        self.save()

    # ── DPI / scaling ─────────────────────────────────────────────────────────

    def detect_dpi(self, root) -> float:
        """
        Use tk's own scaling to get physical DPI.
        Called AFTER the window is visible so Tk reports the correct monitor.
        tk scaling = pt/px.  At 96 DPI: 0.75 pt/px.
        dpi = tk_scaling * 96 / 0.75
        """
        try:
            tk_scale = float(root.tk.call('tk', 'scaling'))
            dpi = tk_scale * 96.0 / 0.75
            dpi = max(40.0, min(300.0, dpi))
        except Exception:
            dpi = 96.0
        self._dpi_scale = dpi / 96.0
        return self._dpi_scale

    @property
    def total_scale(self) -> float:
        return self._dpi_scale * self.font_scale

    def font(self, base_pt: int) -> int:
        return max(8, int(round(base_pt * self.total_scale)))

    def px(self, base_px: int) -> int:
        return max(1, int(round(base_px * self.total_scale)))

    def fig_scale(self, gentle: bool = False) -> float:
        s = self.total_scale
        return 0.8 + 0.2 * s if gentle else s


prefs = Preferences()


def apply_scaling(root):
    """
    Scale all named Tk fonts AND ttk styles to match DPI + user preference.
    Call AFTER the window is visible/maximised for accurate DPI detection.
    """
    from tkinter import font as tkfont
    from tkinter import ttk as _ttk

    prefs.detect_dpi(root)
    sz = prefs.font(11)
    font_spec = ("TkDefaultFont", sz)

    # Named Tk fonts
    for fname in tkfont.names():
        f = tkfont.nametofont(fname)
        base = next((size for key, size in BASE_FONT_SIZES.items()
                     if key in fname), None)
        if base is None:
            try:
                current = abs(f.cget("size"))
                base = max(8, int(round(current / prefs.total_scale))) \
                       if prefs.total_scale > 0 else current
            except Exception:
                base = 11
        try:
            f.configure(size=prefs.font(base))
        except Exception:
            pass

    # ttk styles — Combobox, Notebook tabs, Spinbox, etc.
    style = _ttk.Style(root)
    for s in ("TCombobox","TButton","TEntry","TLabel","TCheckbutton",
              "TRadiobutton","TMenubutton","TSpinbox","TNotebook.Tab","Centered.TNotebook.Tab",
              "TLabelframe.Label","Treeview","Treeview.Heading"):
        try:
            style.configure(s, font=font_spec)
        except Exception:
            pass

    # Combobox dropdown popup listbox
    try:
        root.option_add("*TCombobox*Listbox.font", font_spec)
    except Exception:
        pass

    # Tk Menu widgets
    def _fix_menus(widget):
        try:
            if widget.winfo_class() == "Menu":
                widget.configure(font=font_spec)
        except Exception:
            pass
        try:
            for child in widget.winfo_children():
                _fix_menus(child)
        except Exception:
            pass
    _fix_menus(root)

    try:
        root.option_add("*Menu.font", font_spec)
    except Exception:
        pass


def open_preferences_dialog(root, on_apply=None):
    import tkinter as tk
    from tkinter import ttk

    win = tk.Toplevel(root)
    win.title("Preferences")
    win.transient(root)
    win.resizable(True, True)

    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

    # ── Tab 1: Font & UI ──────────────────────────────────────────────────────
    font_tab = tk.Frame(notebook)
    notebook.add(font_tab, text="Font & UI")

    tk.Label(font_tab, text="UI & Font Scale",
             font=("TkDefaultFont", 11, "bold")).pack(pady=(14, 4))

    scale_var = tk.DoubleVar(value=prefs.font_scale * 100)
    frm = tk.Frame(font_tab); frm.pack(padx=20, pady=4)
    tk.Label(frm, text="Smaller").pack(side="left")
    tk.Scale(frm, from_=70, to=150, resolution=5, orient="horizontal",
             variable=scale_var, length=220, showvalue=False).pack(side="left", padx=6)
    tk.Label(frm, text="Larger").pack(side="left")

    pct_lbl = tk.Label(font_tab); pct_lbl.pack()
    def _update_label(*_): pct_lbl.config(text=f"{int(scale_var.get())}%")
    scale_var.trace_add("write", _update_label); _update_label()

    tk.Label(font_tab, text="Affects fonts, buttons, padding and window sizes.",
             fg="grey").pack(pady=(2, 16))

    # ── Tab 2: Latency Profiles ───────────────────────────────────────────────
    lat_tab = tk.Frame(notebook)
    notebook.add(lat_tab, text="Latency Profiles")

    tk.Label(lat_tab,
             text="Default onset detection windows by muscle group.\n"
                  "The ● Default column sets which profile is pre-selected when a new\n"
                  "stimulus type is configured in Stage 1a. Per-file overrides are\n"
                  "saved independently and are not affected by changes here.",
             justify="left", fg="grey").pack(anchor="w", padx=12, pady=(10, 6))

    canvas_frame = tk.Frame(lat_tab)
    canvas_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    canvas    = tk.Canvas(canvas_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    inner     = tk.Frame(canvas)

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    win.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

    _bold = ("TkDefaultFont", 9, "bold")
    for col, (text, w) in enumerate([
        ("Default", 7), ("Stim type", 18), ("Muscle group", 26),
        ("Min (ms)", 7),  ("Max (ms)", 7),  ("", 3),
    ]):
        tk.Label(inner, text=text, font=_bold, width=w, anchor="w")\
            .grid(row=0, column=col, padx=(4, 2), pady=(4, 2), sticky="w")
    ttk.Separator(inner, orient="horizontal")\
        .grid(row=1, column=0, columnspan=6, sticky="ew", padx=4, pady=2)

    current_key = prefs.default_latency_key
    radio_var   = tk.StringVar(value=f"{current_key[0]}::{current_key[1]}")

    working_profiles = [dict(p) for p in prefs.latency_profiles]
    canonical = {(p["stim_type"], p["muscle"]): (p["min_ms"], p["max_ms"])
                 for p in LATENCY_PROFILE_DEFAULTS}

    row_vars: list[tuple] = []

    for i, profile in enumerate(working_profiles):
        st  = profile["stim_type"]
        mg  = profile["muscle"]
        row = i + 2

        radio_val = f"{st}::{mg}"
        tk.Radiobutton(inner, variable=radio_var, value=radio_val, width=2)\
            .grid(row=row, column=0, padx=(8, 2), sticky="w")

        tk.Label(inner, text=st,  anchor="w", width=18)\
            .grid(row=row, column=1, padx=(2, 4), sticky="w")
        tk.Label(inner, text=mg,  anchor="w", width=26)\
            .grid(row=row, column=2, padx=(2, 4), sticky="w")

        v_min = tk.StringVar(value=str(profile["min_ms"]))
        v_max = tk.StringVar(value=str(profile["max_ms"]))
        tk.Entry(inner, textvariable=v_min, width=6, justify="center")\
            .grid(row=row, column=3, padx=4, sticky="w")
        tk.Entry(inner, textvariable=v_max, width=6, justify="center")\
            .grid(row=row, column=4, padx=4, sticky="w")

        def _make_reset(vm_in, vm_ax, s=st, m=mg):
            def _reset():
                defaults = canonical.get((s, m))
                if defaults:
                    vm_in.set(str(defaults[0]))
                    vm_ax.set(str(defaults[1]))
            return _reset

        reset_btn = tk.Button(inner, text="↺", width=2,
                              command=_make_reset(v_min, v_max),
                              relief="flat", cursor="hand2")
        reset_btn.grid(row=row, column=5, padx=(2, 6))
        row_vars.append((st, mg, v_min, v_max))

    def _reset_all():
        for st, mg, v_min, v_max in row_vars:
            d = canonical.get((st, mg))
            if d:
                v_min.set(str(d[0]))
                v_max.set(str(d[1]))

    tk.Button(lat_tab, text="Reset all to defaults", command=_reset_all)\
        .pack(anchor="e", padx=12, pady=(2, 6))

    # ── Tab 3: Detection Method ───────────────────────────────────────────────
    det_tab = tk.Frame(notebook)
    notebook.add(det_tab, text="Detection")

    tk.Label(det_tab, text="Onset Latency Detection Method",
             font=("TkDefaultFont", 11, "bold")).pack(pady=(14, 4), anchor="w", padx=16)

    tk.Label(det_tab,
             text="Sets the global default method. Individual files can override\n"
                  "this in Stage 1a without affecting the preference here.",
             justify="left", fg="grey").pack(anchor="w", padx=16, pady=(0, 10))

    method_var = tk.StringVar(value=prefs.onset_method)

    # ── Method descriptions ───────────────────────────────────────────────────
    METHOD_DESCRIPTIONS = {
        "peak_fraction": (
            "Peak Fraction\n\n"
            "Finds the largest peak in the MEP window, sets a threshold at a\n"
            "fraction of that peak, then backtracks to find the onset. Fast and\n"
            "works well on clean, high-amplitude MEPs."
        ),
        "bootstrap": (
            "Bootstrap Threshold\n\n"
            "Estimates a noise threshold from the pre-stimulus baseline using a\n"
            "bootstrap distribution, then finds the onset via a peak-anchored\n"
            "backward scan within the physiological latency window. More robust\n"
            "on noisy or low-amplitude signals."
        ),
        "bigoni": (
            "Derivative-based (Bigoni et al. 2022)\n\n"
            "Identifies the onset as the start of the longest sustained positive\n"
            "derivative run in the rising edge of the MEP. Does not rely on\n"
            "pre-stimulus baseline statistics — robust on active-contraction data\n"
            "and biphasic waveforms. Reference: J Neural Eng 19 (2022) 024002."
        ),
    }

    # Radio buttons
    radio_frame = tk.Frame(det_tab)
    radio_frame.pack(anchor="w", padx=16, pady=(0, 6))

    desc_lbl = tk.Label(det_tab, text="", justify="left", fg="#444",
                        wraplength=460, anchor="w")
    desc_lbl.pack(anchor="w", padx=16, pady=(0, 12))

    def _update_desc(*_):
        desc_lbl.config(text=METHOD_DESCRIPTIONS.get(method_var.get(), ""))
        _toggle_param_frames()

    for key, label in [("peak_fraction", "Peak Fraction"),
                        ("bootstrap",    "Bootstrap Threshold"),
                        ("bigoni",       "Derivative-based (Bigoni et al. 2022)")]:
        tk.Radiobutton(radio_frame, text=label, variable=method_var,
                       value=key, command=_update_desc)\
            .pack(anchor="w", pady=2)

    # ── Parameter frames (show/hide based on selection) ───────────────────────
    # Peak-fraction parameters
    pf_frame = tk.LabelFrame(det_tab, text="Peak Fraction parameters", padx=10, pady=8)

    def _pf_row(parent, label, var, row):
        tk.Label(parent, text=label, anchor="w", width=28)\
            .grid(row=row, column=0, sticky="w", pady=3)
        tk.Entry(parent, textvariable=var, width=8, justify="center")\
            .grid(row=row, column=1, padx=8, sticky="w")

    pf_peak_frac_var = tk.StringVar(value=str(prefs.onset_peak_frac))
    pf_min_amp_var   = tk.StringVar(value=str(prefs.onset_min_peak_amplitude))
    pf_slope_var     = tk.StringVar(value=str(prefs.onset_slope_threshold))

    _pf_row(pf_frame, "Peak fraction (0–1)",        pf_peak_frac_var, 0)
    _pf_row(pf_frame, "Min peak amplitude (mV)",    pf_min_amp_var,   1)
    _pf_row(pf_frame, "Slope threshold (mV/ms)",    pf_slope_var,     2)

    # Bootstrap parameters
    bs_frame = tk.LabelFrame(det_tab, text="Bootstrap parameters", padx=10, pady=8)

    bs_crit_var = tk.StringVar(value=str(prefs.onset_bootstrap_crit))
    bs_n_var    = tk.StringVar(value=str(prefs.onset_bootstrap_n))

    _pf_row(bs_frame, "Criterion (SD multiplier)",  bs_crit_var, 0)
    _pf_row(bs_frame, "Bootstrap iterations",       bs_n_var,    1)

    # Bigoni parameters
    bg_frame = tk.LabelFrame(det_tab, text="Derivative-based parameters", padx=10, pady=8)

    bg_smooth_var  = tk.StringVar(value=str(prefs.onset_bigoni_smooth_ms))
    bg_run_var     = tk.StringVar(value=str(prefs.onset_bigoni_min_run_ms))

    _pf_row(bg_frame, "Smoothing window (ms)",      bg_smooth_var, 0)
    _pf_row(bg_frame, "Min positive run (ms)",      bg_run_var,    1)

    tk.Label(bg_frame,
             text="Set smoothing to 0 to disable. Min run filters\n"
                  "single-sample noise spikes from onset selection.",
             fg="grey", justify="left").grid(row=2, column=0, columnspan=2,
                                             sticky="w", pady=(4, 0))

    def _toggle_param_frames():
        m = method_var.get()
        bs_frame.pack_forget()
        pf_frame.pack_forget()
        bg_frame.pack_forget()
        if m == "peak_fraction":
            pf_frame.pack(anchor="w", padx=16, pady=(0, 8), fill="x")
        elif m == "bootstrap":
            bs_frame.pack(anchor="w", padx=16, pady=(0, 8), fill="x")
        elif m == "bigoni":
            bg_frame.pack(anchor="w", padx=16, pady=(0, 8), fill="x")

    # Initialise
    _update_desc()

    # ── Shared Apply / Reset / Cancel row ────────────────────────────────────
    def _apply():
        # Font scale
        prefs.set_font_scale(scale_var.get() / 100.0)
        apply_scaling(root)
        if on_apply:
            on_apply(root)

        # Latency profiles
        updated = []
        for st, mg, v_min, v_max in row_vars:
            try:
                mn = float(v_min.get())
                mx = float(v_max.get())
            except ValueError:
                continue
            updated.append({"stim_type": st, "muscle": mg,
                             "min_ms": mn, "max_ms": mx})

        raw_key = radio_var.get().split("::", 1)
        def_key = tuple(raw_key) if len(raw_key) == 2 else DEFAULT_LATENCY_KEY
        prefs.set_latency_prefs(updated, def_key)

        # Onset detection
        try:
            pf   = float(pf_peak_frac_var.get())
            mpa  = float(pf_min_amp_var.get())
            slp  = float(pf_slope_var.get())
            crit = float(bs_crit_var.get())
            n    = int(bs_n_var.get())
            bsm  = float(bg_smooth_var.get())
            brn  = float(bg_run_var.get())
            prefs.set_onset_prefs(method_var.get(), pf, mpa, slp, crit, n, bsm, brn)
        except ValueError:
            pass

    def _reset_font():
        scale_var.set(100); _apply()

    btn_row = tk.Frame(win)
    btn_row.pack(pady=(6, 12))
    tk.Button(btn_row, text="Apply",           width=10, command=_apply).pack(side="left", padx=4)
    tk.Button(btn_row, text="Reset font 100%", width=14, command=_reset_font).pack(side="left", padx=4)
    tk.Button(btn_row, text="Cancel",          width=10, command=win.destroy).pack(side="left", padx=4)

    win.update_idletasks()
    win.minsize(700, 520)
    try:
        win.state("zoomed")          # Windows / Linux maximise
    except Exception:
        # macOS doesn't support "zoomed" — use screen dimensions instead
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{sw}x{sh}+0+0")
    win.grab_set()
