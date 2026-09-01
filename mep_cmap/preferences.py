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
# Detection defaults are NOT restated here. They live in
# mep_cmap.detection.defaults, which is the single source of truth, and are
# merged in below. Restating them was how PipelineConfig and preferences came
# to disagree about onset_method in the first place.
from .detection.defaults import as_pref_defaults as _detection_pref_defaults
from .detection.defaults import (
    DETECTION_DEFAULTS_VERSION as _DETECTION_DEFAULTS_VERSION,
)

DEFAULTS   = {
    "font_scale":          1.0,
    "addons_path":         None,   # user add-ons folder (None -> built-in only)
    "latency_profiles":    None,   # None -> use LATENCY_PROFILE_DEFAULTS
    "default_latency_key": None,   # None -> use DEFAULT_LATENCY_KEY
    # Trials per stimulus type pre-selected in Preview detection's trial
    # chooser. A default, not a cap -- any subset can be chosen there.
    "preview_trials_per_type": 8,
    # Seed for a new stimulus type's epoch window on tab 1a, in ms. The value
    # in force is the row itself; this only decides what a fresh row starts at.
    "default_epoch_ms": [20, 400],
    # Column selection for the narrowed COPY of trials.csv. Off by default:
    # a study that never touches this writes exactly the files it always did.
    #
    # BOTH keys MUST be here. load() keeps only `if k in DEFAULTS`, so a key
    # written to preferences.json without an entry above appears to save,
    # reads back for the rest of the session from the in-memory dict, and is
    # silently discarded on the next start -- a setting that forgets itself
    # overnight and only ever for the analyst, never on the machine it was
    # tested on.
    "trials_selected_enabled": False,
    "trials_selected_groups": ["amplitude", "onset", "prestim"],
}
DEFAULTS.update(_detection_pref_defaults())
# Stamped so a superseded default can be recognised rather than shadowed
# forever by a value the analyst never deliberately chose.
DEFAULTS["detection_defaults_version"] = _DETECTION_DEFAULTS_VERSION

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

def accent_button_kw(kind="green"):
    """Cross-platform styling for coloured action buttons.

    macOS (Aqua) ignores a tk.Button's ``bg``, so ``fg='white'`` would render as
    unreadable white-on-light. There we colour the *text* instead (still readable
    and visually distinct). Windows/Linux honour ``bg``, so we fill the button.

    Usage:  tk.Button(parent, text="Run", command=..., **accent_button_kw("green"))
    """
    colour = {"green": "#5cb85c", "red": "#d9534f",
              "blue": "#2196F3"}.get(kind, "#5cb85c")
    if sys.platform == "darwin":
        return {"fg": colour}
    return {"bg": colour, "fg": "white"}


class Preferences:
    def __init__(self):
        self._data: dict = dict(DEFAULTS)
        self._dpi_scale: float = 1.0
        self.load()

    def load(self):
        self.migration_notes = []
        stored = {}
        try:
            if PREFS_FILE.exists():
                stored = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
                for k, v in stored.items():
                    if k in DEFAULTS:
                        self._data[k] = v
        except Exception:
            stored = {}
        # Bring forward defaults that were raised after this file was written,
        # but only where the stored value is still the one it superseded.
        #
        # The version MUST come from `stored`, the file itself. `self._data`
        # was seeded from DEFAULTS, which carries the current version, so
        # reading it from there makes every install look up to date and the
        # migration a no-op -- which is how a stale 60 ms offset cap survived
        # an upgrade and silently reduced offset detection to 1 trial in 81.
        # A file with no version key predates the stamp and is version 1.
        try:
            from .detection.defaults import migrate_detection_defaults
            self.migration_notes = migrate_detection_defaults(
                self._data, stored_version=stored.get(
                    "detection_defaults_version", 1))
            if self.migration_notes:
                self.save()
        except Exception:
            self.migration_notes = []

    def reset_detection(self):
        """Drop stored detection settings so the canonical defaults apply."""
        from .detection.defaults import reset_detection_defaults
        removed = reset_detection_defaults(self._data)
        for key in removed:
            if key in DEFAULTS:
                self._data[key] = DEFAULTS[key]
        self.save()
        return removed

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
    def default_epoch_ms(self):
        """(pre_ms, post_ms) a new stimulus type's window starts at."""
        v = self._data.get("default_epoch_ms") or [20, 400]
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return 20.0, 400.0

    def set_default_epoch_ms(self, pre, post):
        self._data["default_epoch_ms"] = [float(pre), float(post)]
        self.save()

    @property
    def preview_trials_per_type(self) -> int:
        """How many trials per stimulus type Preview detection pre-selects.

        A starting point, not a limit: the trial chooser lets any subset be
        picked, up to every trial in the file.
        """
        return int(self._data.get("preview_trials_per_type", 8))

    def set_preview_trials_per_type(self, value: int):
        self._data["preview_trials_per_type"] = max(1, min(100, int(value)))
        self.save()

    # ── Selected-column trial file ────────────────────────────────────────────

    @property
    def trials_selected_enabled(self) -> bool:
        """Whether a run also writes the column-narrowed copy of trials.csv."""
        return bool(self._data.get("trials_selected_enabled", False))

    @property
    def trials_selected_groups(self) -> list:
        """Column groups the narrowed copy keeps, as keys from column_groups.

        Unknown keys are dropped on the way out rather than on the way in: a
        group renamed or retired in a later version would otherwise sit in the
        stored preference forever, and asking for a group that no longer exists
        should mean "not selected", not an error mid-run.
        """
        from .column_groups import GROUP_KEYS
        stored = self._data.get("trials_selected_groups")
        if not isinstance(stored, list):
            stored = DEFAULTS["trials_selected_groups"]
        return [k for k in stored if k in GROUP_KEYS]

    def set_trials_selected(self, enabled: bool, groups=None):
        """Persist both keys together.

        One call because they are one decision: a selection with nothing
        enabled and an enable with no selection are both half a setting.
        """
        from .column_groups import GROUP_KEYS
        self._data["trials_selected_enabled"] = bool(enabled)
        if groups is not None:
            self._data["trials_selected_groups"] = [
                k for k in groups if k in GROUP_KEYS]
        self.save()

    @property
    def font_scale(self) -> float:
        return float(self._data.get("font_scale", 1.0))

    def set_font_scale(self, value: float):
        self._data["font_scale"] = round(max(0.7, min(1.5, float(value))), 2)
        self.save()

    # ── Add-ons folder ────────────────────────────────────────────────────────
    @property
    def addons_path(self):
        v = self._data.get("addons_path")
        return v or None

    def set_addons_path(self, value):
        self._data["addons_path"] = (str(value).strip() or None) if value else None
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

    def _det(self, key):
        """Stored detection value, falling back to the canonical default."""
        return self._data.get(key, DEFAULTS.get(key))

    @property
    def onset_method(self) -> str:
        """Active onset detection method key; see detection.ONSET_METHOD_LABELS."""
        return str(self._det("onset_method"))

    # ── Peak-fraction parameters ──────────────────────────────────────────────

    @property
    def onset_peak_frac(self) -> float:
        return float(self._det("onset_peak_frac"))

    @property
    def onset_min_peak_amplitude(self) -> float:
        return float(self._det("onset_min_peak_amplitude"))

    @property
    def onset_slope_threshold(self) -> float:
        return float(self._det("onset_slope_threshold"))

    # ── Bootstrap parameters ──────────────────────────────────────────────────

    @property
    def onset_bootstrap_crit(self) -> float:
        return float(self._det("onset_bootstrap_crit"))

    @property
    def onset_bootstrap_n(self) -> int:
        return int(self._det("onset_bootstrap_n"))

    @property
    def onset_bigoni_smooth_ms(self) -> float:
        return float(self._det("onset_bigoni_smooth_ms"))

    @property
    def onset_bigoni_min_run_ms(self) -> float:
        return float(self._det("onset_bigoni_min_run_ms"))

    @property
    def onset_bigoni_walkback_sd(self) -> float:
        return float(self._det("onset_bigoni_walkback_sd"))

    # ── RMS envelope parameters ───────────────────────────────────────────────

    @property
    def onset_env_window_ms(self) -> float:
        return float(self._det("onset_env_window_ms"))

    @property
    def onset_env_criterion(self) -> float:
        return float(self._det("onset_env_criterion"))

    @property
    def onset_env_significance(self) -> float:
        return float(self._det("onset_env_significance"))

    @property
    def onset_env_n_boot(self) -> int:
        return int(self._det("onset_env_n_boot"))

    @property
    def onset_env_min_run_ms(self) -> float:
        return float(self._det("onset_env_min_run_ms"))

    @property
    def onset_env_min_response_ms(self) -> float:
        return float(self._det("onset_env_min_response_ms"))

    @property
    def onset_env_tkeo(self) -> bool:
        return bool(self._det("onset_env_tkeo"))

    @property
    def onset_env_causal(self) -> bool:
        return bool(self._det("onset_env_causal"))

    @property
    def onset_env_refine(self) -> bool:
        return bool(self._det("onset_env_refine"))

    @property
    def onset_env_refine_window_ms(self) -> float:
        return float(self._det("onset_env_refine_window_ms"))

    @property
    def onset_env_refine_sd(self) -> float:
        return float(self._det("onset_env_refine_sd"))

    @property
    def onset_env_refine_sustain_ms(self) -> float:
        return float(self._det("onset_env_refine_sustain_ms"))

    # ── CUSUM parameters ──────────────────────────────────────────────────────

    @property
    def onset_cusum_k(self) -> float:
        return float(self._det("onset_cusum_k"))

    @property
    def onset_cusum_h(self) -> float:
        return float(self._det("onset_cusum_h"))

    @property
    def onset_cusum_max_accum_ms(self) -> float:
        return float(self._det("onset_cusum_max_accum_ms"))

    @property
    def onset_cusum_min_response_ms(self) -> float:
        return float(self._det("onset_cusum_min_response_ms"))

    @property
    def onset_cusum_tkeo(self) -> bool:
        return bool(self._det("onset_cusum_tkeo"))

    # ── Derivative ratio (Boyles et al. 2026) ─────────────────────────────────

    @property
    def onset_boyles_block_ms(self) -> float:
        return float(self._det("onset_boyles_block_ms"))

    @property
    def onset_boyles_baseline_start_ms(self) -> float:
        return float(self._det("onset_boyles_baseline_start_ms"))

    @property
    def onset_boyles_baseline_end_ms(self) -> float:
        return float(self._det("onset_boyles_baseline_end_ms"))

    @property
    def onset_boyles_amplitude_gate(self) -> float:
        return float(self._det("onset_boyles_amplitude_gate"))

    @property
    def onset_boyles_peak_jitter_ms(self) -> float:
        return float(self._det("onset_boyles_peak_jitter_ms"))

    @property
    def onset_boyles_peak_window_length(self) -> float:
        return float(self._det("onset_boyles_peak_window_length"))

    @property
    def onset_boyles_ratio_cutoff(self) -> float:
        return float(self._det("onset_boyles_ratio_cutoff"))

    @property
    def onset_boyles_max_latency_ms(self) -> float:
        return float(self._det("onset_boyles_max_latency_ms"))

    @property
    def onset_boyles_deriv_check_ms(self) -> float:
        return float(self._det("onset_boyles_deriv_check_ms"))

    @property
    def onset_boyles_deriv_check_duty(self) -> float:
        return float(self._det("onset_boyles_deriv_check_duty"))

    @property
    def onset_boyles_base_deriv_sds(self) -> float:
        return float(self._det("onset_boyles_base_deriv_sds"))

    @property
    def onset_boyles_deriv_window_length(self) -> float:
        return float(self._det("onset_boyles_deriv_window_length"))

    @property
    def onset_boyles_literal(self) -> bool:
        return bool(self._det("onset_boyles_literal"))
    # ── Consensus / agreement ─────────────────────────────────────────────────

    @property
    def onset_methods_median_members(self) -> list:
        val = self._det("onset_methods_median_members")
        return list(val) if val else list(DEFAULTS["onset_methods_median_members"])

    @property
    def onset_agreement(self) -> bool:
        return bool(self._det("onset_agreement"))

    # ── MEP offset ────────────────────────────────────────────────────────────

    @property
    def mep_offset_enabled(self) -> bool:
        return bool(self._det("mep_offset_enabled"))

    @property
    def mep_offset_min_duration_ms(self) -> float:
        return float(self._det("mep_offset_min_duration_ms"))

    @property
    def mep_offset_max_duration_ms(self) -> float:
        return float(self._det("mep_offset_max_duration_ms"))

    @property
    def mep_offset_min_return_ms(self) -> float:
        return float(self._det("mep_offset_min_return_ms"))

    @property
    def mep_offset_env_window_ms(self) -> float:
        return float(self._det("mep_offset_env_window_ms"))

    @property
    def mep_offset_criterion(self) -> float:
        return float(self._det("mep_offset_criterion"))

    @property
    def mep_offset_peak_frac(self) -> float:
        return float(self._det("mep_offset_peak_frac"))

    # ── PTP window anchoring ──────────────────────────────────────────────────

    @property
    def ptp_anchor(self) -> bool:
        return bool(self._det("ptp_anchor"))

    @property
    def ptp_anchor_pre_ms(self) -> float:
        return float(self._det("ptp_anchor_pre_ms"))

    @property
    def ptp_anchor_duration_ms(self) -> float:
        return float(self._det("ptp_anchor_duration_ms"))

    @property
    def ptp_anchor_min_trials(self) -> int:
        return int(self._det("ptp_anchor_min_trials"))

    def set_detection_prefs(self, **kwargs):
        """
        Persist any subset of detection preferences by config field name.

        Complements set_onset_prefs, whose fixed positional signature would
        need extending for every new parameter. Unknown keys are rejected
        rather than silently written, so a typo cannot create a dead
        preference that never reads back.
        """
        from .detection.defaults import DETECTION_DEFAULTS, pref_key_for
        unknown = set(kwargs) - set(DETECTION_DEFAULTS)
        if unknown:
            raise KeyError("unknown detection preference(s): %s"
                           % ", ".join(sorted(unknown)))
        for key, value in kwargs.items():
            self._data[pref_key_for(key)] = value
        self.save()

    def set_onset_prefs(self, method: str,
                        peak_frac: float, min_peak_amplitude: float,
                        slope_threshold: float,
                        bootstrap_crit: float, bootstrap_n: int,
                        bigoni_smooth_ms: float = None,
                        bigoni_min_run_ms: float = None,
                        bigoni_walkback_sd: float = None):
        """Persist the onset detection preferences exposed by the GUI.

        Bigoni parameters default to None rather than to literals, so the
        canonical values in detection.defaults are the only place they are
        written down.
        """
        if bigoni_smooth_ms is None:
            bigoni_smooth_ms = DEFAULTS["onset_bigoni_smooth_ms"]
        if bigoni_min_run_ms is None:
            bigoni_min_run_ms = DEFAULTS["onset_bigoni_min_run_ms"]
        if bigoni_walkback_sd is None:
            bigoni_walkback_sd = DEFAULTS["onset_bigoni_walkback_sd"]
        self._data["onset_method"]              = method
        self._data["onset_peak_frac"]           = round(float(peak_frac), 4)
        self._data["onset_min_peak_amplitude"]  = round(float(min_peak_amplitude), 4)
        self._data["onset_slope_threshold"]     = round(float(slope_threshold), 4)
        self._data["onset_bootstrap_crit"]      = round(float(bootstrap_crit), 4)
        self._data["onset_bootstrap_n"]         = int(bootstrap_n)
        self._data["onset_bigoni_smooth_ms"]    = round(float(bigoni_smooth_ms), 2)
        self._data["onset_bigoni_min_run_ms"]   = round(float(bigoni_min_run_ms), 2)
        self._data["onset_bigoni_walkback_sd"]  = round(float(bigoni_walkback_sd), 2)
        self.save()

    # ── DPI / scaling ─────────────────────────────────────────────────────────

    def detect_dpi(self, root) -> float:
        """
        Use tk's own scaling to get physical DPI.
        Called AFTER the window is visible so Tk reports the correct monitor.
        tk scaling = pt/px.  At 96 DPI: 0.75 pt/px.
        dpi = tk_scaling * 96 / 0.75

        NOT on Aqua. macOS does display scaling itself: Tk reports a scaling of
        1.0 whether or not the screen is Retina, and the system font at its
        native size is already right for the display. Putting 1.0 through the
        formula above yields 128 DPI and a factor of 1.33, so every font in the
        application comes out a third larger than intended and the layout wants
        more width than the screen has. The window then either opens wider than
        the display or, once clamped, clips its own panels.
        """
        try:
            if root.tk.call("tk", "windowingsystem") == "aqua":
                self._dpi_scale = 1.0
                return self._dpi_scale
        except Exception:                   # noqa: BLE001 — not Aqua, carry on
            pass
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
              "Sub.TNotebook.Tab",
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
    # win.transient(root)
    win.resizable(True, True)

    win.rowconfigure(0, weight=1)
    win.rowconfigure(1, weight=0)
    win.columnconfigure(0, weight=1)

    notebook = ttk.Notebook(win)
    notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

    btn_row = tk.Frame(win)
    btn_row.grid(row=1, column=0, pady=(6, 12))

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

    # ── Scrollable body ──────────────────────────────────────────────────────
    # Seven methods plus their parameters, the offset settings and the
    # agreement toggle no longer fit a fixed-height tab on a laptop screen.
    _det_canvas = tk.Canvas(det_tab, highlightthickness=0, borderwidth=0)
    _det_scroll = ttk.Scrollbar(det_tab, orient="vertical",
                                command=_det_canvas.yview)
    _det_body = tk.Frame(_det_canvas)
    _det_body.bind(
        "<Configure>",
        lambda e: _det_canvas.configure(scrollregion=_det_canvas.bbox("all")))
    _det_win = _det_canvas.create_window((0, 0), window=_det_body, anchor="nw")
    _det_canvas.bind(
        "<Configure>",
        lambda e: _det_canvas.itemconfigure(_det_win, width=e.width))
    _det_canvas.configure(yscrollcommand=_det_scroll.set)
    _det_canvas.pack(side="left", fill="both", expand=True)
    _det_scroll.pack(side="right", fill="y")

    def _det_wheel(event):
        _det_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    _det_canvas.bind_all("<MouseWheel>", _det_wheel)

    tk.Label(_det_body, text="Onset Latency Detection Method",
             font=("TkDefaultFont", 11, "bold")).pack(pady=(14, 4), anchor="w", padx=16)

    tk.Label(_det_body,
             text="Sets the global default method. Individual files can override\n"
                  "this in Stage 1a without affecting the preference here.",
             justify="left", fg="grey").pack(anchor="w", padx=16, pady=(0, 10))

    method_var = tk.StringVar(value=prefs.onset_method)

    # ── Method radios, generated from the detection registry ─────────────────
    # Built from ONSET_METHOD_LABELS rather than a literal list, so registering
    # a detector makes it selectable without a matching edit here. The previous
    # hard-coded list is why three methods were reachable from the pipeline but
    # invisible in the GUI.
    from .detection import ONSET_METHOD_HINTS, ONSET_METHOD_LABELS

    radio_frame = tk.Frame(_det_body)
    radio_frame.pack(anchor="w", padx=16, pady=(0, 6))

    desc_lbl = tk.Label(_det_body, text="", justify="left", fg="#444",
                        wraplength=460, anchor="w")
    desc_lbl.pack(anchor="w", padx=16, pady=(0, 12))

    def _update_desc(*_):
        m = method_var.get()
        desc_lbl.config(text="%s\n\n%s" % (
            ONSET_METHOD_LABELS.get(m, m), ONSET_METHOD_HINTS.get(m, "")))
        _toggle_param_frames()

    _ORDER = ["bigoni", "bigoni_walkback", "rms_envelope", "cusum",
              "methods_median", "peak_fraction", "bootstrap"]
    _ordered = [k for k in _ORDER if k in ONSET_METHOD_LABELS]
    _ordered += [k for k in ONSET_METHOD_LABELS if k not in _ordered]
    for key in _ordered:
        tk.Radiobutton(radio_frame, text=ONSET_METHOD_LABELS[key],
                       variable=method_var, value=key, command=_update_desc)\
            .pack(anchor="w", pady=1)

    # ── Row helper ───────────────────────────────────────────────────────────
    def _pf_row(parent, label, var, row, hint=None):
        tk.Label(parent, text=label, anchor="w", width=30)\
            .grid(row=row, column=0, sticky="w", pady=3)
        tk.Entry(parent, textvariable=var, width=8, justify="center")\
            .grid(row=row, column=1, padx=8, sticky="w")
        if hint:
            tk.Label(parent, text=hint, fg="grey", anchor="w")\
                .grid(row=row, column=2, sticky="w", padx=(4, 0))

    def _check(parent, label, var, row, hint=None):
        tk.Checkbutton(parent, text=label, variable=var, anchor="w")\
            .grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        if hint:
            tk.Label(parent, text=hint, fg="grey", anchor="w")\
                .grid(row=row, column=2, sticky="w", padx=(4, 0))

    # ── Applies to every method ──────────────────────────────────────────────
    # min_peak_amplitude gates all seven detectors, but used to sit inside the
    # "Peak Fraction parameters" frame and was therefore hidden whenever any
    # other method was selected -- while still applying.
    common_frame = tk.LabelFrame(_det_body, text="Applies to all methods",
                                 padx=10, pady=8)
    pf_min_amp_var = tk.StringVar(value=str(prefs.onset_min_peak_amplitude))
    _pf_row(common_frame, "Min peak amplitude (mV)", pf_min_amp_var, 0,
            "response smaller than this is not detected")
    common_frame.pack(anchor="w", padx=16, pady=(0, 8), fill="x")

    # ── Peak-fraction ────────────────────────────────────────────────────────
    pf_frame = tk.LabelFrame(_det_body, text="Peak Fraction parameters",
                             padx=10, pady=8)
    pf_peak_frac_var = tk.StringVar(value=str(prefs.onset_peak_frac))
    pf_slope_var     = tk.StringVar(value=str(prefs.onset_slope_threshold))
    _pf_row(pf_frame, "Peak fraction (0-1)",     pf_peak_frac_var, 0)
    _pf_row(pf_frame, "Slope threshold (mV/ms)", pf_slope_var,     1)

    # ── Bootstrap (legacy) ───────────────────────────────────────────────────
    bs_frame = tk.LabelFrame(_det_body, text="Bootstrap parameters",
                             padx=10, pady=8)
    bs_crit_var = tk.StringVar(value=str(prefs.onset_bootstrap_crit))
    bs_n_var    = tk.StringVar(value=str(prefs.onset_bootstrap_n))
    _pf_row(bs_frame, "Criterion (SD multiplier)", bs_crit_var, 0)
    _pf_row(bs_frame, "Bootstrap iterations",      bs_n_var,    1)
    tk.Label(bs_frame,
             text="Retained so analyses run on v1.3.x reproduce exactly.\n"
                  "Its threshold is clipped to a multiple of the baseline\n"
                  "mean, which places onsets early; prefer RMS Envelope\n"
                  "for new work.",
             fg="#a00", justify="left").grid(row=2, column=0, columnspan=3,
                                             sticky="w", pady=(6, 0))

    # ── Bigoni ───────────────────────────────────────────────────────────────
    bg_frame = tk.LabelFrame(_det_body, text="Derivative-based parameters",
                             padx=10, pady=12)
    bg_smooth_var = tk.StringVar(value=str(prefs.onset_bigoni_smooth_ms))
    bg_run_var    = tk.StringVar(value=str(prefs.onset_bigoni_min_run_ms))
    bg_wb_sd_var  = tk.StringVar(value=str(prefs.onset_bigoni_walkback_sd))
    _pf_row(bg_frame, "Smoothing window (ms)", bg_smooth_var, 0)
    _pf_row(bg_frame, "Min positive run (ms)", bg_run_var,    1)
    tk.Label(bg_frame,
             text="Set smoothing to 0 to disable. Min run filters\n"
                  "single-sample noise spikes from onset selection.",
             fg="grey", justify="left").grid(row=2, column=0, columnspan=3,
                                             sticky="w", pady=(4, 0))

    wb_frame = tk.LabelFrame(_det_body, text="Walkback parameters",
                             padx=10, pady=8)
    _pf_row(wb_frame, "Walkback SD multiplier", bg_wb_sd_var, 0)
    tk.Label(wb_frame, text="Lower = earlier onset. Default 1.0.",
             fg="grey", justify="left").grid(row=1, column=0, columnspan=3,
                                             sticky="w", pady=(4, 0))

    # ── RMS envelope ─────────────────────────────────────────────────────────
    env_frame = tk.LabelFrame(_det_body, text="RMS Envelope parameters",
                              padx=10, pady=8)
    env_win_var     = tk.StringVar(value=str(prefs.onset_env_window_ms))
    env_crit_var    = tk.StringVar(value=str(prefs.onset_env_criterion))
    env_sig_var     = tk.StringVar(value=str(prefs.onset_env_significance))
    env_nboot_var   = tk.StringVar(value=str(prefs.onset_env_n_boot))
    env_minrun_var  = tk.StringVar(value=str(prefs.onset_env_min_run_ms))
    env_minresp_var = tk.StringVar(value=str(prefs.onset_env_min_response_ms))
    env_tkeo_var    = tk.BooleanVar(value=prefs.onset_env_tkeo)
    env_causal_var  = tk.BooleanVar(value=prefs.onset_env_causal)
    env_refine_var  = tk.BooleanVar(value=prefs.onset_env_refine)
    env_rwin_var    = tk.StringVar(value=str(prefs.onset_env_refine_window_ms))
    env_rsd_var     = tk.StringVar(value=str(prefs.onset_env_refine_sd))
    env_rsus_var    = tk.StringVar(value=str(prefs.onset_env_refine_sustain_ms))
    _pf_row(env_frame, "Envelope window (ms)",    env_win_var,     0)
    _pf_row(env_frame, "Criterion (SD multiplier)", env_crit_var,  1)
    _pf_row(env_frame, "Significance",            env_sig_var,     2,
            "for the run-length criterion")
    _pf_row(env_frame, "Bootstrap iterations",    env_nboot_var,   3)
    _pf_row(env_frame, "Min above-threshold (ms)", env_minrun_var, 4)
    _pf_row(env_frame, "Min response width (ms)", env_minresp_var, 5,
            "rejects single-sample artefacts; 0 disables")
    _check(env_frame, "Teager-Kaiser preconditioning", env_tkeo_var, 6,
           "sharpens low-SNR onsets")
    _check(env_frame, "Causal window", env_causal_var, 7)
    _check(env_frame, "Refine onset on a short window", env_refine_var, 8,
           "strongly recommended")
    _pf_row(env_frame, "  Refine window (ms)",     env_rwin_var,  9)
    _pf_row(env_frame, "  Refine SD multiplier",   env_rsd_var,  10)
    _pf_row(env_frame, "  Refine sustain (ms)",    env_rsus_var, 11)
    tk.Label(env_frame,
             text="With refinement off, a wide envelope window places the\n"
                  "onset several ms early. Leave it on unless comparing\n"
                  "against a published unrefined implementation.",
             fg="grey", justify="left").grid(row=12, column=0, columnspan=3,
                                             sticky="w", pady=(6, 0))

    # ── CUSUM ────────────────────────────────────────────────────────────────
    cs_frame = tk.LabelFrame(_det_body, text="CUSUM parameters",
                             padx=10, pady=8)
    cs_k_var       = tk.StringVar(value=str(prefs.onset_cusum_k))
    cs_h_var       = tk.StringVar(value=str(prefs.onset_cusum_h))
    cs_accum_var   = tk.StringVar(value=str(prefs.onset_cusum_max_accum_ms))
    cs_minresp_var = tk.StringVar(value=str(prefs.onset_cusum_min_response_ms))
    cs_tkeo_var    = tk.BooleanVar(value=prefs.onset_cusum_tkeo)
    _pf_row(cs_frame, "Allowance k (SD)",        cs_k_var,       0,
            "shifts smaller than this are ignored")
    _pf_row(cs_frame, "Decision interval h (SD)", cs_h_var,      1,
            "higher = fewer false alarms")
    _pf_row(cs_frame, "Max accumulation (ms)",   cs_accum_var,   2,
            "0 = classical unbounded CUSUM")
    _pf_row(cs_frame, "Min response width (ms)", cs_minresp_var, 3)
    _check(cs_frame, "Teager-Kaiser preconditioning", cs_tkeo_var, 4)

    # ── Derivative ratio (Boyles et al. 2026) ────────────────────────────────
    by_frame = tk.LabelFrame(_det_body, text="Derivative Ratio parameters",
                             padx=10, pady=8)
    by_block_var   = tk.StringVar(value=str(prefs.onset_boyles_block_ms))
    by_bstart_var  = tk.StringVar(value=str(prefs.onset_boyles_baseline_start_ms))
    by_bend_var    = tk.StringVar(value=str(prefs.onset_boyles_baseline_end_ms))
    by_gate_var    = tk.StringVar(value=str(prefs.onset_boyles_amplitude_gate))
    by_jitter_var  = tk.StringVar(value=str(prefs.onset_boyles_peak_jitter_ms))
    by_pwin_var    = tk.StringVar(value=str(prefs.onset_boyles_peak_window_length))
    by_cut_var     = tk.StringVar(value=str(prefs.onset_boyles_ratio_cutoff))
    by_maxlat_var  = tk.StringVar(value=str(prefs.onset_boyles_max_latency_ms))
    by_dchk_var    = tk.StringVar(value=str(prefs.onset_boyles_deriv_check_ms))
    by_duty_var    = tk.StringVar(value=str(prefs.onset_boyles_deriv_check_duty))
    by_sds_var     = tk.StringVar(value=str(prefs.onset_boyles_base_deriv_sds))
    by_dwin_var    = tk.StringVar(value=str(prefs.onset_boyles_deriv_window_length))
    by_literal_var = tk.BooleanVar(value=prefs.onset_boyles_literal)
    _pf_row(by_frame, "Slope window (ms)",           by_block_var,  0,
            "either side of each candidate")
    _pf_row(by_frame, "Baseline start (ms pre-stim)", by_bstart_var, 1,
            "clamped to the pre-stim data present")
    _pf_row(by_frame, "Baseline end (ms pre-stim)",  by_bend_var,   2)
    _pf_row(by_frame, "Amplitude gate (ratio)",      by_gate_var,   3,
            "response vs baseline peak-to-peak")
    _pf_row(by_frame, "Peak jitter (ms)",            by_jitter_var, 4,
            "vs the condition average's peak")
    _pf_row(by_frame, "Search back (x peak-trough)", by_pwin_var,   5)
    _pf_row(by_frame, "Ratio cutoff (0-1)",          by_cut_var,    6,
            "fraction of the peak derivative ratio")
    _pf_row(by_frame, "Max latency (ms)",            by_maxlat_var, 7,
            "the tighter of this and the 1a profile applies")
    _pf_row(by_frame, "Initial slope window (ms)",   by_dchk_var,   8)
    _pf_row(by_frame, "  fraction that must exceed", by_duty_var,   9)
    _pf_row(by_frame, "Overall slope (SD above)",    by_sds_var,   10)
    _pf_row(by_frame, "Overall window (x peak-trough)", by_dwin_var, 11)
    _check(by_frame, "Reproduce the published implementation literally",
           by_literal_var, 12)
    tk.Label(by_frame,
             text="Needs a condition average, which the analysis supplies from\n"
                  "the outlier-screened trials of each event type.\n\n"
                  "Three details of the reference MATLAB code do not match its\n"
                  "own comments: the slope window is fixed in samples rather\n"
                  "than milliseconds, so it shrinks as sampling rate rises; the\n"
                  "amplitude gate compares a peak-to-peak against a single peak;\n"
                  "and the peak-jitter gate compares the trial's largest peak\n"
                  "against the average's first peak. All three are corrected\n"
                  "here. Tick the box above only to reproduce the published\n"
                  "method exactly.",
             fg="grey", justify="left").grid(row=13, column=0, columnspan=3,
                                             sticky="w", pady=(6, 0))

    # ── Consensus ────────────────────────────────────────────────────────────
    cons_frame = tk.LabelFrame(_det_body, text="Median across methods \u2014 members",
                               padx=10, pady=8)
    _cons_selected = set(prefs.onset_methods_median_members)
    cons_vars = {}
    _r = 0
    for key in _ordered:
        if key == "methods_median":
            continue
        v = tk.BooleanVar(value=key in _cons_selected)
        cons_vars[key] = v
        tk.Checkbutton(cons_frame, text=ONSET_METHOD_LABELS[key],
                       variable=v, anchor="w")\
            .grid(row=_r, column=0, columnspan=3, sticky="w", pady=1)
        _r += 1
    tk.Label(cons_frame,
             text="The reported onset is the median of the members that\n"
                  "detect one. The median is not a verdict on which method is\n"
                  "right \u2014 it is the middle value, chosen because it resists\n"
                  "one stray member. An odd number of members keeps the median\n"
                  "a value some method actually reported. Bootstrap Threshold\n"
                  "is slow and best left unticked.",
             fg="grey", justify="left").grid(row=_r, column=0, columnspan=3,
                                             sticky="w", pady=(6, 0))

    _METHOD_FRAMES = {
        "peak_fraction":   [pf_frame],
        "bootstrap":       [bs_frame],
        "bigoni":          [bg_frame],
        "bigoni_walkback": [bg_frame, wb_frame],
        "rms_envelope":    [env_frame],
        "cusum":           [cs_frame],
        "boyles":          [by_frame],
        "methods_median":       [cons_frame],
    }

    # ── MEP offset (independent of the onset method) ─────────────────────────
    off_frame = tk.LabelFrame(_det_body, text="MEP Offset & Duration",
                              padx=10, pady=8)
    off_en_var     = tk.BooleanVar(value=prefs.mep_offset_enabled)
    off_mindur_var = tk.StringVar(value=str(prefs.mep_offset_min_duration_ms))
    off_maxdur_var = tk.StringVar(value=str(prefs.mep_offset_max_duration_ms))
    off_ret_var    = tk.StringVar(value=str(prefs.mep_offset_min_return_ms))
    off_win_var    = tk.StringVar(value=str(prefs.mep_offset_env_window_ms))
    off_crit_var   = tk.StringVar(value=str(prefs.mep_offset_criterion))
    off_frac_var   = tk.StringVar(value=str(prefs.mep_offset_peak_frac))
    _check(off_frame, "Detect MEP offset and duration", off_en_var, 0)
    _pf_row(off_frame, "Min duration (ms)",      off_mindur_var, 1)
    _pf_row(off_frame, "Max duration (ms)",      off_maxdur_var, 2,
            "raise for slow or polyphasic responses")
    _pf_row(off_frame, "Min return to baseline (ms)", off_ret_var, 3)
    _pf_row(off_frame, "Envelope window (ms)",   off_win_var,    4)
    _pf_row(off_frame, "Criterion (SD multiplier)", off_crit_var, 5)
    _pf_row(off_frame, "Peak fraction floor (0-1)", off_frac_var, 6,
            "scales the return threshold to the response")
    tk.Label(off_frame,
             text="Where a cortical silent period is detected, its start IS\n"
                  "the end of the MEP and is reported as the offset. Where\n"
                  "there is none, the return to baseline is detected instead,\n"
                  "which also gives AUC an endpoint at rest.",
             fg="grey", justify="left").grid(row=7, column=0, columnspan=3,
                                             sticky="w", pady=(6, 0))
    off_frame.pack(anchor="w", padx=16, pady=(8, 8), fill="x")

    # ── PTP measurement window anchoring ─────────────────────────────────────
    ptpa_frame = tk.LabelFrame(_det_body,
                           text="Amplitude (PTP) Window Anchoring",
                               padx=10, pady=8)
    ptpa_en_var   = tk.BooleanVar(value=prefs.ptp_anchor)
    ptpa_pre_var  = tk.StringVar(value=str(prefs.ptp_anchor_pre_ms))
    ptpa_dur_var  = tk.StringVar(value=str(prefs.ptp_anchor_duration_ms))
    ptpa_min_var  = tk.StringVar(value=str(prefs.ptp_anchor_min_trials))
    _check(ptpa_frame, "Anchor PTP window to each event type's median onset",
           ptpa_en_var, 0)
    _pf_row(ptpa_frame, "Window starts before onset (ms)", ptpa_pre_var, 1)
    _pf_row(ptpa_frame, "Window length from onset (ms)",   ptpa_dur_var, 2)
    _pf_row(ptpa_frame, "Min onsets before anchoring",     ptpa_min_var, 3,
            "below this the file-wide window is kept")
    tk.Label(ptpa_frame,
             text="The PTP window in 1c is one setting for the whole file, but\n"
                  "each event type has its own latency profile. A recording\n"
                  "containing both M-waves and MEPs cannot be measured by one\n"
                  "window: with a 10 ms start, an M-wave beginning at 4 ms has\n"
                  "most of its response excluded from the amplitude. Anchoring\n"
                  "gives each event type a window placed on its own median\n"
                  "onset. The 1c window end is still applied as a ceiling.",
             fg="grey", justify="left").grid(row=4, column=0, columnspan=3,
                                             sticky="w", pady=(6, 0))
    ptpa_frame.pack(anchor="w", padx=16, pady=(0, 8), fill="x")

    # ── Agreement ────────────────────────────────────────────────────────────
    ag_frame = tk.LabelFrame(_det_body, text="Onset Method Agreement",
                             padx=10, pady=8)
    ag_var = tk.BooleanVar(value=prefs.onset_agreement)
    _check(ag_frame, "Compare methods on every trial", ag_var, 0)
    tk.Label(ag_frame,
             text="Runs the member methods alongside the selected method\n"
                  "and reports how far apart they land, as\n"
                  "Onset_Disagreement(ms). Trials where methods diverge are\n"
                  "the ones worth reviewing by hand. Slows detection roughly\n"
                  "in proportion to the number of members.",
             fg="grey", justify="left").grid(row=1, column=0, columnspan=3,
                                             sticky="w", pady=(4, 0))
    ag_frame.pack(anchor="w", padx=16, pady=(0, 12), fill="x")

    def _toggle_param_frames():
        for _f in (pf_frame, bs_frame, bg_frame, wb_frame,
                   env_frame, cs_frame, by_frame, cons_frame):
            _f.pack_forget()
        # Offset and agreement are method-independent and stay put; repacking
        # them keeps them below the method-specific frames.
        off_frame.pack_forget()
        ptpa_frame.pack_forget()
        ag_frame.pack_forget()
        for _f in _METHOD_FRAMES.get(method_var.get(), []):
            _f.pack(anchor="w", padx=16, pady=(0, 8), fill="x")
        off_frame.pack(anchor="w", padx=16, pady=(8, 8), fill="x")
        ptpa_frame.pack(anchor="w", padx=16, pady=(0, 8), fill="x")
        ag_frame.pack(anchor="w", padx=16, pady=(0, 12), fill="x")

    # ── Restore detection defaults ───────────────────────────────────────────
    def _restore_detection_defaults():
        from tkinter import messagebox
        if not messagebox.askyesno(
                "Restore detection defaults",
                "Reset every setting on this tab to its shipped default?\n\n"
                "Latency profiles, fonts and add-on paths are not affected.",
                parent=win):
            return
        prefs.reset_detection()
        messagebox.showinfo(
            "Detection defaults restored",
            "Close and reopen Preferences to see the restored values.",
            parent=win)

    _rst = tk.Frame(_det_body)
    _rst.pack(anchor="w", padx=16, pady=(0, 14))
    tk.Button(_rst, text="Restore detection defaults",
              command=_restore_detection_defaults).pack(side="left")
    tk.Label(_rst,
             text="  Use after updating if a value looks stale: settings saved\n"
                  "  by an earlier version take precedence over new defaults.",
             fg="grey", justify="left").pack(side="left")

    # Initialise
    _update_desc()

    # ── Shared Apply / Reset / Cancel row ────────────────────────────────────
    # ── Tab 4: Add-ons ────────────────────────────────────────────────────────
    addon_tab = tk.Frame(notebook)
    notebook.add(addon_tab, text="Add-ons")
    tk.Label(addon_tab, text="Add-ons folder",
             font=("TkDefaultFont", 11, "bold")).pack(pady=(14, 4))
    tk.Label(addon_tab,
             text="Folder containing your custom analysis add-on modules (.py).\n"
                  "Built-in add-ons always load; this path is for your own, and is\n"
                  "saved so you can keep add-ons anywhere.",
             justify="left", fg="grey").pack(anchor="w", padx=16, pady=(0, 8))
    addons_path_var = tk.StringVar(value=prefs.addons_path or "")
    _ar = tk.Frame(addon_tab); _ar.pack(fill="x", padx=16, pady=(0, 6))
    tk.Entry(_ar, textvariable=addons_path_var).pack(side="left", fill="x", expand=True)
    def _browse_addons():
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select add-ons folder")
        if d:
            addons_path_var.set(d)
    tk.Button(_ar, text="Browse\u2026", command=_browse_addons).pack(side="left", padx=(6, 0))
    tk.Button(addon_tab, text="Clear",
              command=lambda: addons_path_var.set("")).pack(anchor="w", padx=16, pady=(0, 4))

    # ── Tab 5: Trial columns ─────────────────────────────────────────────────
    from .column_groups import GROUPS as _CG_GROUPS
    from .column_groups import (DEPENDENCIES as _CG_DEPS,
                                GROUP_LABELS as _CG_LABELS,
                                PROTECTED as _CG_PROTECTED)

    col_tab = tk.Frame(notebook)
    notebook.add(col_tab, text="Trial columns")

    _cc = tk.Canvas(col_tab, highlightthickness=0, borderwidth=0)
    _cs = ttk.Scrollbar(col_tab, orient="vertical", command=_cc.yview)
    _cb = tk.Frame(_cc)
    _cb.bind("<Configure>",
             lambda e: _cc.configure(scrollregion=_cc.bbox("all")))
    _cw = _cc.create_window((0, 0), window=_cb, anchor="nw")
    _cc.bind("<Configure>", lambda e: _cc.itemconfigure(_cw, width=e.width))
    _cc.configure(yscrollcommand=_cs.set)
    _cc.pack(side="left", fill="both", expand=True)
    _cs.pack(side="right", fill="y")

    tk.Label(_cb, text="Trimmed trials file",
             font=("TkDefaultFont", 11, "bold")).pack(pady=(14, 4),
                                                      anchor="w", padx=16)
    tk.Label(_cb,
             text="_trials.csv always carries every column; this writes an\n"
                  "extra, narrower copy beside it as _trials_selected.csv, which\n"
                  "Second Level can be pointed at instead. Nothing is lost: the\n"
                  "columns below identify every trial, so the trimmed file can\n"
                  "always be merged back against the full one.\n\n"
                  "This is the default for every recording. A single recording\n"
                  "can override it on tab 1c.",
             justify="left", fg="grey").pack(anchor="w", padx=16, pady=(0, 10))

    sel_en_var = tk.BooleanVar(value=prefs.trials_selected_enabled)
    tk.Checkbutton(_cb, text="Also write a trimmed copy of _trials.csv",
                   variable=sel_en_var, anchor="w").pack(anchor="w", padx=16)

    _prot = tk.LabelFrame(_cb, text="Always written", padx=10, pady=8)
    _prot.pack(anchor="w", padx=16, pady=(10, 6), fill="x")
    for _name, _why in _CG_PROTECTED.items():
        _v = tk.BooleanVar(value=True)
        # Ticked and disabled, with the reason beside it. A control that
        # cannot be changed and does not say why reads as a broken checkbox.
        tk.Checkbutton(_prot, text=_name, variable=_v, state="disabled",
                       anchor="w").grid(sticky="w")
        tk.Label(_prot, text=_why, fg="grey", anchor="w").grid(
            row=_prot.grid_size()[1] - 1, column=1, sticky="w", padx=(8, 0))

    _grp = tk.LabelFrame(_cb, text="Choose what else to keep", padx=10, pady=8)
    _grp.pack(anchor="w", padx=16, pady=(0, 10), fill="x")
    sel_group_vars = {}
    for _key, _label, _cols in _CG_GROUPS:
        _v = tk.BooleanVar(value=_key in prefs.trials_selected_groups)
        sel_group_vars[_key] = _v
        _req = _CG_DEPS.get(_key)
        _txt = f"{_label}  ({len(_cols)} column{'s' if len(_cols) != 1 else ''})"
        if _req:
            _txt += f"  \u2014 also selects '{_CG_LABELS.get(_req, _req)}'"
        tk.Checkbutton(_grp, text=_txt, variable=_v,
                       anchor="w").pack(anchor="w")

    tk.Label(_cb,
             text="A group whose members depend on another pulls it in\n"
                  "automatically, and the run log says so.",
             justify="left", fg="grey").pack(anchor="w", padx=16, pady=(0, 12))

    def _apply():
        # Font scale
        prefs.set_font_scale(scale_var.get() / 100.0)
        apply_scaling(root)

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

        # ── Detection ────────────────────────────────────────────────────────
        # Fields are parsed individually so that one malformed entry does not
        # discard every other edit on the tab. The previous single try/except
        # around the whole block silently dropped all onset settings if any one
        # box held a typo -- the dialog closed as though it had saved.
        _bad = []

        def _num(var, key, cast=float):
            try:
                return cast(var.get())
            except (TypeError, ValueError):
                _bad.append(key)
                return None

        _det = {
            "onset_method":                method_var.get(),
            "peak_fraction":               _num(pf_peak_frac_var, "Peak fraction"),
            "min_peak_amplitude":          _num(pf_min_amp_var, "Min peak amplitude"),
            "slope_threshold":             _num(pf_slope_var, "Slope threshold"),
            "onset_bootstrap_crit":        _num(bs_crit_var, "Bootstrap criterion"),
            "onset_bootstrap_n":           _num(bs_n_var, "Bootstrap iterations", int),
            "onset_bigoni_smooth_ms":      _num(bg_smooth_var, "Smoothing window"),
            "onset_bigoni_min_run_ms":     _num(bg_run_var, "Min positive run"),
            "onset_bigoni_walkback_sd":    _num(bg_wb_sd_var, "Walkback SD"),
            "onset_env_window_ms":         _num(env_win_var, "Envelope window"),
            "onset_env_criterion":         _num(env_crit_var, "Envelope criterion"),
            "onset_env_significance":      _num(env_sig_var, "Envelope significance"),
            "onset_env_n_boot":            _num(env_nboot_var, "Envelope iterations", int),
            "onset_env_min_run_ms":        _num(env_minrun_var, "Min above-threshold"),
            "onset_env_min_response_ms":   _num(env_minresp_var, "Envelope min response width"),
            "onset_env_tkeo":              bool(env_tkeo_var.get()),
            "onset_env_causal":            bool(env_causal_var.get()),
            "onset_env_refine":            bool(env_refine_var.get()),
            "onset_env_refine_window_ms":  _num(env_rwin_var, "Refine window"),
            "onset_env_refine_sd":         _num(env_rsd_var, "Refine SD multiplier"),
            "onset_env_refine_sustain_ms": _num(env_rsus_var, "Refine sustain"),
            "onset_cusum_k":               _num(cs_k_var, "CUSUM allowance k"),
            "onset_cusum_h":               _num(cs_h_var, "CUSUM decision interval h"),
            "onset_cusum_max_accum_ms":    _num(cs_accum_var, "CUSUM max accumulation"),
            "onset_cusum_min_response_ms": _num(cs_minresp_var, "CUSUM min response width"),
            "onset_cusum_tkeo":            bool(cs_tkeo_var.get()),
            "onset_boyles_block_ms":            _num(by_block_var, "Boyles slope window"),
            "onset_boyles_baseline_start_ms":   _num(by_bstart_var, "Boyles baseline start"),
            "onset_boyles_baseline_end_ms":     _num(by_bend_var, "Boyles baseline end"),
            "onset_boyles_amplitude_gate":      _num(by_gate_var, "Boyles amplitude gate"),
            "onset_boyles_peak_jitter_ms":      _num(by_jitter_var, "Boyles peak jitter"),
            "onset_boyles_peak_window_length":  _num(by_pwin_var, "Boyles search back"),
            "onset_boyles_ratio_cutoff":        _num(by_cut_var, "Boyles ratio cutoff"),
            "onset_boyles_max_latency_ms":      _num(by_maxlat_var, "Boyles max latency"),
            "onset_boyles_deriv_check_ms":      _num(by_dchk_var, "Boyles initial slope window"),
            "onset_boyles_deriv_check_duty":    _num(by_duty_var, "Boyles initial slope fraction"),
            "onset_boyles_base_deriv_sds":      _num(by_sds_var, "Boyles overall slope SD"),
            "onset_boyles_deriv_window_length": _num(by_dwin_var, "Boyles overall window"),
            "onset_boyles_literal":             bool(by_literal_var.get()),
            "onset_agreement":             bool(ag_var.get()),
            "mep_offset_enabled":          bool(off_en_var.get()),
            "mep_offset_min_duration_ms":  _num(off_mindur_var, "Offset min duration"),
            "mep_offset_max_duration_ms":  _num(off_maxdur_var, "Offset max duration"),
            "mep_offset_min_return_ms":    _num(off_ret_var, "Min return to baseline"),
            "mep_offset_env_window_ms":    _num(off_win_var, "Offset envelope window"),
            "mep_offset_criterion":        _num(off_crit_var, "Offset criterion"),
            "mep_offset_peak_frac":        _num(off_frac_var, "Offset peak fraction"),
            "ptp_anchor":                  bool(ptpa_en_var.get()),
            "ptp_anchor_pre_ms":           _num(ptpa_pre_var, "PTP anchor pre-onset"),
            "ptp_anchor_duration_ms":      _num(ptpa_dur_var, "PTP anchor duration"),
            "ptp_anchor_min_trials":       _num(ptpa_min_var, "PTP anchor min onsets", int),
        }

        _members = [k for k, v in cons_vars.items() if v.get()]
        if _members:
            _det["onset_methods_median_members"] = _members
        elif method_var.get() == "methods_median":
            _bad.append("Median across methods \u2014 members (none selected)")

        # Drop only the fields that failed to parse; keep everything valid.
        prefs.set_detection_prefs(
            **{k: v for k, v in _det.items() if v is not None})

        if _bad:
            from tkinter import messagebox
            messagebox.showwarning(
                "Some settings were not saved",
                "These fields could not be read and kept their previous "
                "values:\n\n  - " + "\n  - ".join(_bad) +
                "\n\nEverything else on the Detection tab was saved.",
                parent=win)

        prefs.set_addons_path(addons_path_var.get())

        # Both keys in one call: an enable with no groups and groups with no
        # enable are each half a setting.
        prefs.set_trials_selected(
            sel_en_var.get(),
            [k for k, v in sel_group_vars.items() if v.get()])

        if on_apply:
            on_apply(root)

    def _reset_font():
        scale_var.set(100); _apply()

    tk.Button(btn_row, text="Apply",           width=10, command=_apply).pack(side="left", padx=4)
    tk.Button(btn_row, text="Reset font 100%", width=14, command=_reset_font).pack(side="left", padx=4)
    tk.Button(btn_row, text="Cancel",          width=10, command=win.destroy).pack(side="left", padx=4)

    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w  = min(900, int(sw * 0.75))
    h  = min(750, int(sh * 0.80))
    x  = (sw - w) // 2
    y  = (sh - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")
    win.minsize(700, 520)
    win.grab_set()
