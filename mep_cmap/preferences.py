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

    # ── 3. Tk Menu widgets ────────────────────────────────────────────────────
    # On Windows, Menu widgets ignore TkMenuFont and use the system menu font
    # unless font= is set explicitly on each menu instance.
    # We walk the widget tree and reconfigure every Menu widget found.
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

    # Also set via option_add so future menus created after this call are correct
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

    # Scrollable table area
    canvas_frame = tk.Frame(lat_tab)
    canvas_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    canvas  = tk.Canvas(canvas_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    inner   = tk.Frame(canvas)

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Enable mouse-wheel scrolling inside the canvas
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    win.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

    # Column headers
    _bold = ("TkDefaultFont", 9, "bold")
    for col, (text, w) in enumerate([
        ("Default", 7), ("Stim type", 18), ("Muscle group", 26),
        ("Min (ms)", 7),  ("Max (ms)", 7),  ("", 3),
    ]):
        tk.Label(inner, text=text, font=_bold, width=w, anchor="w")\
            .grid(row=0, column=col, padx=(4, 2), pady=(4, 2), sticky="w")
    ttk.Separator(inner, orient="horizontal")\
        .grid(row=1, column=0, columnspan=6, sticky="ew", padx=4, pady=2)

    # One radio-group variable — value = "stim_type::muscle"
    current_key = prefs.default_latency_key
    radio_var   = tk.StringVar(value=f"{current_key[0]}::{current_key[1]}")

    # Working copy of profiles (edited in-place by the entries)
    working_profiles = [dict(p) for p in prefs.latency_profiles]

    # Build canonical lookup for reset-to-default per row
    canonical = {(p["stim_type"], p["muscle"]): (p["min_ms"], p["max_ms"])
                 for p in LATENCY_PROFILE_DEFAULTS}

    row_vars: list[tuple] = []   # (stim_type, muscle, v_min, v_max)

    for i, profile in enumerate(working_profiles):
        st  = profile["stim_type"]
        mg  = profile["muscle"]
        row = i + 2   # rows 0=header, 1=separator, then data

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

        # Reset button restores this row to its factory value (if known)
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

    # Reset-all button
    def _reset_all():
        for st, mg, v_min, v_max in row_vars:
            d = canonical.get((st, mg))
            if d:
                v_min.set(str(d[0]))
                v_max.set(str(d[1]))

    tk.Button(lat_tab, text="Reset all to defaults", command=_reset_all)\
        .pack(anchor="e", padx=12, pady=(2, 6))

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

        raw_key   = radio_var.get().split("::", 1)
        def_key   = tuple(raw_key) if len(raw_key) == 2 else DEFAULT_LATENCY_KEY
        prefs.set_latency_prefs(updated, def_key)

    def _reset_font():
        scale_var.set(100); _apply()

    btn_row = tk.Frame(win)
    btn_row.pack(pady=(6, 12))
    tk.Button(btn_row, text="Apply",          width=10, command=_apply).pack(side="left", padx=4)
    tk.Button(btn_row, text="Reset font 100%",width=14, command=_reset_font).pack(side="left", padx=4)
    tk.Button(btn_row, text="Cancel",         width=10, command=win.destroy).pack(side="left", padx=4)

    win.update_idletasks()
    win.minsize(560, 440)
    x = root.winfo_rootx() + (root.winfo_width()  - win.winfo_width())  // 2
    y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{x}+{y}")
    win.grab_set()
