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
DEFAULTS   = {"font_scale": 1.0}

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
    win.resizable(False, False)

    tk.Label(win, text="UI & Font Scale",
             font=("TkDefaultFont", 11, "bold")).pack(pady=(14, 4))

    scale_var = tk.DoubleVar(value=prefs.font_scale * 100)
    frm = tk.Frame(win); frm.pack(padx=20, pady=4)
    tk.Label(frm, text="Smaller").pack(side="left")
    tk.Scale(frm, from_=70, to=150, resolution=5, orient="horizontal",
             variable=scale_var, length=220, showvalue=False).pack(side="left", padx=6)
    tk.Label(frm, text="Larger").pack(side="left")

    pct_lbl = tk.Label(win); pct_lbl.pack()
    def _update_label(*_): pct_lbl.config(text=f"{int(scale_var.get())}%")
    scale_var.trace_add("write", _update_label); _update_label()

    tk.Label(win, text="Affects fonts, buttons, padding and window sizes.",
             fg="grey").pack(pady=(2, 10))

    def _apply():
        prefs.set_font_scale(scale_var.get() / 100.0)
        apply_scaling(root)
        if on_apply: on_apply(root)

    def _reset():
        scale_var.set(100); _apply()

    btn_row = tk.Frame(win); btn_row.pack(pady=(0, 12))
    tk.Button(btn_row, text="Apply",        width=10, command=_apply).pack(side="left", padx=4)
    tk.Button(btn_row, text="Reset to 100%",width=12, command=_reset).pack(side="left", padx=4)
    tk.Button(btn_row, text="Cancel",       width=10, command=win.destroy).pack(side="left", padx=4)

    win.update_idletasks()
    x = root.winfo_rootx() + (root.winfo_width()  - win.winfo_width())  // 2
    y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{x}+{y}")
    win.grab_set()
