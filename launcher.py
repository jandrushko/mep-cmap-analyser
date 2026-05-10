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

if __name__ == '__main__':

    from splash_screen import show_splash
    splash = show_splash()

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
