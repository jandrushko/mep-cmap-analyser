"""
mep_cmap.detection
~~~~~~~~~~~~~~~~~~
Backward-compatible facade.

All detection logic now lives in the mep_cmap/detection/ subpackage.
This module re-exports everything so that existing imports in pipeline.py,
inspector.py, and any other caller continue to work without modification.

    from .detection import detect_mep_onset_peak_fraction   ← still works
    from .detection import detect_mep_onset_bootstrap       ← still works
    from .detection import detect_mep_onset_bigoni          ← new
    from .detection import detect_csp_bootstrap             ← still works
    from .detection import compute_bootstrap_threshold      ← still works
    from .detection import detect_mep_onset                 ← dispatcher
"""

from .detection import (                          # noqa: F401
    # Onset detectors
    detect_mep_onset_peak_fraction,
    detect_mep_onset_bootstrap,
    detect_mep_onset_bigoni,
    detect_mep_onset_bigoni_walkback,
    # Silent period
    detect_csp_bootstrap,
    # Shared baseline
    compute_bootstrap_threshold,
    # Quantification
    compute_ptp,
    compute_auc,
    compute_prestim_rms,
    compute_prestim_ptp,
    # Dispatcher
    detect_mep_onset,
    # Registry / labels (useful for UI)
    ONSET_METHOD_LABELS,
    DETECTION_VERSION,
)
