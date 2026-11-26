#!/usr/bin/env python3
"""
MEP-CMAP Analyser — Launcher with Splash Screen
Shows splash screen while loading the main application.

Works with both:
  • Direct run:  python launcher.py
  • PyInstaller: compiled exe entry point
"""

# freeze_support() prevents PyInstaller from re-launching the full GUI
# whenever numpy/scipy spawn a worker subprocess on Windows.
import multiprocessing
multiprocessing.freeze_support()

import sys
import os


def _point_tcl_at_bundled_files():
    """Tell Tcl/Tk where the built app keeps its library files.

    PyInstaller ships them as _internal/_tcl_data and _internal/_tk_data and
    normally sets TCL_LIBRARY / TK_LIBRARY from a runtime hook. The 1.4.10
    Windows build -- identical files, identical hook -- started without them
    set, and the first tkinter window failed with "Can't find a usable
    init.tcl"; setting them by hand made the same build start. Doing it here
    as well, before anything imports tkinter, does not depend on that hook.

    Only in the built app, only if the folders exist, and never overriding a
    value already set to a folder that exists.
    """
    if not getattr(sys, "frozen", False):
        return
    base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    for var, sub in (("TCL_LIBRARY", "_tcl_data"), ("TK_LIBRARY", "_tk_data")):
        folder = os.path.join(base, sub)
        if os.path.isdir(folder) and not os.path.isdir(os.environ.get(var, "")):
            os.environ[var] = folder


_point_tcl_at_bundled_files()

_RELAUNCH_MARK = "MEPCMAP_TCL_RELAUNCHED"


def _relaunch_with_tcl_env(exc):
    """Restart the built app once with the Tcl/Tk folders set from the start.

    The second line of defence. Setting TCL_LIBRARY in-process (above, and in
    PyInstaller's own hook) is normally enough, but with the 1.4.10 Windows
    build only a value present when the process STARTED was seen by Tcl:
    launched with the variable already set, the same build ran. So if the
    first window still fails to find init.tcl, start a fresh copy with the
    variables in its environment, at most once, and leave this one.

    Returns True if a relaunch was started; False if it does not apply.
    """
    if not getattr(sys, "frozen", False) or os.environ.get(_RELAUNCH_MARK):
        return False
    if "init.tcl" not in str(exc):
        return False
    import subprocess
    env = dict(os.environ)
    env[_RELAUNCH_MARK] = "1"
    # A built app starting a fresh copy of itself must say so, or the copy
    # treats PyInstaller's runtime variables as its own (PyInstaller docs,
    # "independent instance of the same application").
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    subprocess.Popen([sys.executable] + sys.argv[1:], env=env,
                     cwd=os.path.dirname(sys.executable))
    return True


if __name__ == '__main__':

    from splash_screen import show_splash
    try:
        splash = show_splash()
    except Exception as _tcl_exc:
        if _relaunch_with_tcl_env(_tcl_exc):
            sys.exit(0)
        raise

    try:
        splash.update_message("Loading numpy...")
        import numpy as np

        splash.update_message("Loading pandas...")
        import pandas as pd

        splash.update_message("Loading matplotlib...")
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        splash.update_message("Loading scipy...")
        from scipy.signal import butter, filtfilt, iirnotch

        splash.update_message("Loading GUI components...")
        import tkinter as tk

        splash.update_message("Initialising application...")

        # Add the directory containing mep_cmap/ to the path when frozen
        if getattr(sys, 'frozen', False):
            sys.path.insert(0, os.path.dirname(sys.executable))
        else:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from mep_cmap.app import TMSAnalysisApp

        splash.close()

        root = tk.Tk()
        root.tk.call('tk', 'scaling', 1.0)

        app = TMSAnalysisApp(root)

        def _on_close():
            try:
                root.quit()
                root.destroy()
            except Exception:
                pass
            sys.exit(0)

        root.protocol("WM_DELETE_WINDOW", _on_close)
        root.mainloop()

    except Exception as e:
        try:
            splash.close()
        except Exception:
            pass

        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Error Starting Application",
            f"Failed to start MEP-CMAP Analyser:\n\n{str(e)}\n\nPlease contact support."
        )

    sys.exit(0)
