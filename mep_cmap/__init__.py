"""
mep_cmap
~~~~~~~~
MEP-CMAP Analysis Tool — BIDS-compliant TMS/EMG neurophysiology pipeline.

Package layout
--------------
compat      — gc patch, numpy shim, tk thread-safety
bids        — StudyMetadata, BIDS label sanitisation
utils       — shared helpers (_add_time_and_digmark)
io          — Spike-2 file reading
filters     — EMG filter functions
detection   — MEP onset + CSP bootstrap detection
pipeline       — PipelineConfig + all pipeline_* subfunctions + run_pipeline
inspector      — DataInspectorWindow (per-trial interactive review)
stage2         — Stage2Mixin (group analysis tab)
filter_preview — FilterPreviewMixin (filter preview popup)
app            — TMSAnalysisApp (main GUI, inherits stage2 + filter_preview)

Quickstart
----------
    python -m mep_cmap          # launch the GUI
    from mep_cmap import run_app; run_app()
"""

# compat must be imported first — it disables the cyclic GC and patches tkinter
from . import compat  # noqa: F401

from .bids      import StudyMetadata, TOOL_VERSION
from .pipeline  import PipelineConfig, run_pipeline
from .detection import detect_mep_onset_peak_fraction, detect_csp_bootstrap

__version__ = TOOL_VERSION
__all__ = [
    "run_app",
    "StudyMetadata",
    "PipelineConfig",
    "run_pipeline",
    "detect_mep_onset_peak_fraction",
    "detect_csp_bootstrap",
]


def run_app():
    """Launch the MEP-CMAP Analysis GUI."""
    import sys
    import tkinter as tk

    # Windows: create Tk root before numpy to avoid Tcl_AsyncDelete crash
    pre_root = None
    if sys.platform == "win32":
        pre_root = tk.Tk()
        pre_root.withdraw()

    # Now safe to import matplotlib TkAgg backend
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)

    from .app import TMSAnalysisApp

    root = pre_root if pre_root is not None else tk.Tk()
    root.tk.call("tk", "scaling", 1.0)
    root.deiconify()

    app = TMSAnalysisApp(root)
    root.mainloop()
