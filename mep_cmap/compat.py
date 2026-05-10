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

import numpy as np
import tkinter as tk

# NumPy 1.x → trapz,  NumPy 2.x → trapezoid
_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


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
