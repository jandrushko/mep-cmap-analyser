"""
mep_cmap.compat
~~~~~~~~~~~~~~~
Compatibility shims that must run before anything else:

  • Disable Python's cyclic GC at import time so it never fires on a
    BLAS/scipy worker thread and triggers Tcl_AsyncDelete crashes.
  • Provide _np_trapz that resolves to np.trapezoid (NumPy >=2.0) or
    np.trapz (NumPy <2.0).
  • Patch tk.Variable.__del__ and tk.Image.__del__ to swallow the
    RuntimeError that Python 3.12 raises when those objects are GC'd
    from a background thread.
"""

import gc as _gc
_gc.disable()   # manual GC only — see app.py _poll_queue for the schedule
del _gc

# ── DPI awareness (Windows) ───────────────────────────────────────────────────
# Must be called before the first Tk window is created so that Tkinter reports
# the real physical DPI rather than the virtualised 96-DPI value.
# PROCESS_PER_MONITOR_DPI_AWARE (value=2) is the highest level and correctly
# handles mixed-DPI multi-monitor setups.
import sys as _sys
if _sys.platform == "win32":
    try:
        import ctypes as _ctypes
        _ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            _ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
del _sys

import numpy as np
import tkinter as tk

# NumPy 1.x → trapz,  NumPy 2.x → trapezoid
_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# np.ptp() was removed in NumPy 2.0; replace with max - min
def _np_ptp(a, axis=None):
    """Peak-to-peak (max - min) — drop-in replacement for removed np.ptp()."""
    return np.max(a, axis=axis) - np.min(a, axis=axis)

# Restore np.ptp and np.ndarray.ptp for third-party libraries that still
# use these functions removed in NumPy 2.0 (e.g. older Neo versions).
# The module-level np.ptp covers calls like np.ptp(arr).
# The class-level np.ndarray.ptp covers unbound-method calls like
# np.ndarray.ptp(arr, axis=0) which is what older Neo versions use.
if not hasattr(np, "ptp"):
    np.ptp = _np_ptp
if not hasattr(np.ndarray, "ptp"):
    try:
        np.ndarray.ptp = lambda self, axis=None: (
            self.max(axis=axis) - self.min(axis=axis)
        )
    except (AttributeError, TypeError):
        pass


def _apply_tk_patches():
    """Patch tk.Variable and tk.Image __del__ to be thread-safe."""

    def _safe_var_del(self):
        try:
            if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
                self._tk.call("unset", self._name)
        except (RuntimeError, Exception):
            pass

    def _safe_image_del(self):
        try:
            self.tk.call("image", "delete", self.name)
        except (RuntimeError, Exception):
            pass

    tk.Variable.__del__ = _safe_var_del
    tk.Image.__del__    = _safe_image_del


_apply_tk_patches()
