"""
mep_cmap.assets
~~~~~~~~~~~~~~~
Images shipped with the package.

Loading is deliberately total: an image that fails to load returns None and the
caller omits it. A decorative logo is never worth preventing the application
from starting, and the ways it can fail -- a frozen build that did not bundle
the folder, a Tk without PNG support, a corrupted file -- are all invisible
until run time and none of them affect anything the tool does.

Tk holds only a weak reference to a PhotoImage, so an image whose only
reference is a local variable is collected and the widget draws nothing. The
loaded images are cached here, which keeps them alive for the life of the
process and avoids re-decoding the same file for every window that wants it.
"""

from __future__ import annotations

import os
import sys

#: Kept alive: Tk does not hold a strong reference to a PhotoImage, and an
#: image that is garbage collected leaves the widget blank with no error.
_CACHE = {}

_HERE = os.path.dirname(os.path.abspath(__file__))


def asset_path(name: str) -> str:
    """Absolute path to a bundled asset, whether installed or frozen.

    __file__ is not enough on its own. In a PyInstaller build this module is
    imported from the archive, so its __file__ points inside that archive while
    the PNGs are unpacked under sys._MEIPASS -- the path exists as a string,
    os.path.isfile says no, load_photo returns None, and the logo is silently
    absent from the release build while working perfectly from source.

    _MEIPASS is checked FIRST and only when set, matching bids_schema and
    addons, which resolve the schema and the built-in add-ons the same way.
    This module was the one that never got the same treatment.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        frozen = os.path.join(meipass, "mep_cmap", "assets", name)
        if os.path.isfile(frozen):
            return frozen
    return os.path.join(_HERE, name)


def load_photo(name: str):
    """A ``tk.PhotoImage`` for a bundled PNG, or None if it cannot be loaded.

    None rather than an exception: every caller is decoration, and a missing
    image should leave a gap rather than a traceback.

    A cached image is checked before it is handed back. The cache outlives the
    Tk interpreter that built the image: the frozen build runs a splash screen
    on its own ``tk.Tk()``, destroys it, and starts the application on a second
    one, and an image built under the first is a live Python object wrapping a
    Tcl image that no longer exists. Returned unchecked it is not None, so a
    caller's fallback never runs, and the TclError only surfaces when a widget
    tries to draw it -- which is why the About mark was absent from the release
    build and present when run from source, where there is no splash and only
    ever one interpreter. ``width()`` is the cheapest call that reaches Tcl and
    so the cheapest way to find out whether the image is still real.
    """
    if name in _CACHE:
        cached = _CACHE[name]
        if cached is None:
            return None
        try:
            cached.width()
            return cached
        except Exception:
            del _CACHE[name]   # stale interpreter: fall through and rebuild
    try:
        import tkinter as tk

        path = asset_path(name)
        if not os.path.isfile(path):
            _CACHE[name] = None
            return None
        img = tk.PhotoImage(file=path)
    except Exception:
        _CACHE[name] = None
        return None
    _CACHE[name] = img
    return img


def tmsmultilab_logo(size: int = 22):
    """The TMSMultiLab mark, at one of the sizes shipped.

    Sized copies are shipped rather than scaled at run time because Tk's own
    subsampling is integer-factor only and produces visibly ragged edges on a
    figure of concentric circles.
    """
    for candidate in (f"tmsmultilab_{size}.png", "tmsmultilab.png"):
        img = load_photo(candidate)
        if img is not None:
            return img
    return None
