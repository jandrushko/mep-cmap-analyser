"""
mep_cmap.preferences
~~~~~~~~~~~~~~~~~~~~
DPI-aware UI scaling and user preferences.

Preferences are stored in ~/.mep_cmap/preferences.json and persist
across sessions independently of project/data location.

Usage
-----
    from .preferences import prefs, apply_scaling

    # At app startup (after root Tk() is created):
    apply_scaling(root)

    # Get a scaled pixel value:
    prefs.px(24)          # 24 base-px scaled to current DPI + user pref

    # Get a scaled font size:
    prefs.font(10)        # 10pt scaled

    # After user changes preference:
    prefs.set_font_scale(1.2)
    apply_scaling(root)
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Optional


# ── Preferences file location ─────────────────────────────────────────────────
PREFS_DIR  = Path.home() / ".mep_cmap"
PREFS_FILE = PREFS_DIR / "preferences.json"

DEFAULTS = {
    "font_scale": 1.0,   # user multiplier  (0.7 – 1.5)
}


class Preferences:
    """Singleton that holds user preferences and computed DPI scale."""

    def __init__(self):
        self._data: dict = dict(DEFAULTS)
        self._dpi_scale: float = 1.0   # set by detect_dpi()
        self.load()

    # ── Persistence ───────────────────────────────────────────────────────────
    def load(self):
        try:
            if PREFS_FILE.exists():
                stored = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
                for k, v in stored.items():
                    if k in DEFAULTS:
                        self._data[k] = v
        except Exception:
            pass   # corrupt file → silently use defaults

    def save(self):
        try:
            PREFS_DIR.mkdir(parents=True, exist_ok=True)
            PREFS_FILE.write_text(
                json.dumps(self._data, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    def reset(self):
        self._data = dict(DEFAULTS)
        self.save()

    # ── Accessors ─────────────────────────────────────────────────────────────
    @property
    def font_scale(self) -> float:
        return float(self._data.get("font_scale", 1.0))

    def set_font_scale(self, value: float):
        self._data["font_scale"] = round(max(0.7, min(1.5, float(value))), 2)
        self.save()

    # ── DPI detection ─────────────────────────────────────────────────────────
    def detect_dpi(self, root) -> float:
        """
        Detect the physical screen DPI and store as _dpi_scale.
        96 DPI = scale 1.0 (Windows standard baseline).
        Returns the computed dpi_scale.
        """
        dpi = 96.0  # safe fallback

        try:
            system = platform.system()

            if system == "Windows":
                # Ask Windows for the real DPI before Tkinter's virtualisation
                try:
                    import ctypes
                    ctypes.windll.shcore.SetProcessDpiAwareness(1)
                    hdc = ctypes.windll.user32.GetDC(0)
                    dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                    ctypes.windll.user32.ReleaseDC(0, hdc)
                except Exception:
                    dpi = root.winfo_fpixels("1i")

            elif system == "Darwin":
                # winfo_fpixels is reliable on macOS / Retina
                dpi = root.winfo_fpixels("1i")

            else:
                # Linux — winfo_fpixels usually works, fallback to 96
                dpi = root.winfo_fpixels("1i")

        except Exception:
            pass

        # Clamp to a sane range (40–300 DPI)
        dpi = max(40.0, min(300.0, float(dpi)))
        self._dpi_scale = dpi / 96.0
        return self._dpi_scale

    # ── Scaling helpers ───────────────────────────────────────────────────────
    @property
    def total_scale(self) -> float:
        """Combined DPI scale × user font preference."""
        return self._dpi_scale * self.font_scale

    def font(self, base_pt: int) -> int:
        """Return a font size in points scaled for current DPI + user pref."""
        scaled = int(round(base_pt * self.total_scale))
        return max(8, scaled)

    def px(self, base_px: int) -> int:
        """Return a pixel/padding value scaled for current DPI + user pref."""
        scaled = int(round(base_px * self.total_scale))
        return max(1, scaled)

    def fig_scale(self, gentle: bool = False) -> float:
        """
        Matplotlib figure scale factor.
        gentle=True  → use 80% of total_scale (for filter preview / wavelet).
        gentle=False → full total_scale.
        """
        s = self.total_scale
        return 0.8 + 0.2 * s if gentle else s   # gentler: 0.8–1.3 vs 0.7–1.5


# Module-level singleton
prefs = Preferences()


# ── Apply scaling to all named Tk fonts ───────────────────────────────────────
def apply_scaling(root, base_sizes: Optional[dict] = None):
    """
    Resize every named Tkinter font to match the current DPI + user preference.

    base_sizes: optional dict mapping font name fragment → base pt size.
                Defaults cover the standard Tk font names.
    """
    from tkinter import font as tkfont

    if base_sizes is None:
        base_sizes = {
            "TkDefaultFont":    10,
            "TkTextFont":       10,
            "TkFixedFont":      10,
            "TkMenuFont":       10,
            "TkHeadingFont":    10,
            "TkCaptionFont":    11,
            "TkSmallCaptionFont": 9,
            "TkIconFont":        9,
            "TkTooltipFont":     9,
        }

    # Detect DPI on first call (root must already exist)
    if prefs._dpi_scale == 1.0:
        prefs.detect_dpi(root)

    for fname in tkfont.names():
        f = tkfont.nametofont(fname)
        # Look up base size; fall back to current size / total_scale
        # so already-scaled fonts don't drift on repeated calls
        base = None
        for key, size in base_sizes.items():
            if key in fname:
                base = size
                break
        if base is None:
            try:
                current = abs(f.cget("size"))
                base = max(8, int(round(current / prefs.total_scale))) \
                       if prefs.total_scale != 0 else current
            except Exception:
                base = 10

        new_size = prefs.font(base)
        try:
            f.configure(size=new_size)
        except Exception:
            pass
