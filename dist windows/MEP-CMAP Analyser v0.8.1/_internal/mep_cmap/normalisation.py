"""
mep_cmap.normalisation
~~~~~~~~~~~~~~~~~~~~~~
M-wave / Mmax normalisation and paired-pulse ratio computation.

  • compute_mmax        — robust plateau detection on a recruitment curve
  • apply_normalisation — fill Reference_* and Normalised_PTP columns in
                          the trial-level row lists produced by pipeline.py
"""

from __future__ import annotations
import numpy as np


# ─── Mmax plateau detection ───────────────────────────────────────────────────

def compute_mmax(
        ptp_values: np.ndarray | list,
        plateau_tolerance: float = 0.10,
        min_plateau_trials: int = 1,
) -> dict:
    """
    Robustly estimate Mmax from an array of M-wave PTP amplitudes.

    Handles three real-world scenarios:
      1. Full recruitment curve  → find and average the plateau
      2. A few supramaximal pulses → average those
      3. Single M-wave           → use that value directly

    Algorithm
    ---------
    1.  Find peak PTP (single largest value).
    2.  "Plateau trials" = trials within ``plateau_tolerance`` × peak of peak.
    3.  If ≥ 3 plateau trials  → Mmax = mean of plateau trials.
    4.  Elif 2 plateau trials  → Mmax = mean of those 2.
    5.  Else (only 1 near peak) → Mmax = peak value.

    Parameters
    ----------
    ptp_values        : array of M-wave PTP amplitudes (mV)
    plateau_tolerance : fraction of peak within which trials count as plateau
                        (default 0.10 = 10% of peak)
    min_plateau_trials: minimum trials needed to use mean rather than peak
                        (default 1 — always try to average if possible)

    Returns
    -------
    dict with keys:
        mmax          : float — estimated Mmax in mV
        method        : str   — "plateau_mean" / "peak"
        n_plateau     : int   — number of trials contributing to estimate
        peak_ptp      : float — single largest PTP observed
        plateau_tol   : float — tolerance used
    """
    vals = np.asarray(ptp_values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if len(vals) == 0:
        return dict(mmax=np.nan, method="no_data",
                    n_plateau=0, peak_ptp=np.nan, plateau_tol=plateau_tolerance)

    peak_ptp = float(np.max(vals))
    threshold = peak_ptp * (1.0 - plateau_tolerance)
    plateau   = vals[vals >= threshold]
    n_plateau = int(len(plateau))

    if n_plateau >= min_plateau_trials and n_plateau > 1:
        mmax   = float(np.mean(plateau))
        method = "plateau_mean"
    else:
        mmax   = peak_ptp
        method = "peak"
        n_plateau = 1

    return dict(
        mmax        = mmax,
        method      = method,
        n_plateau   = n_plateau,
        peak_ptp    = peak_ptp,
        plateau_tol = plateau_tolerance,
    )


# ─── Apply normalisation to trial row lists ───────────────────────────────────

def apply_normalisation(
        latency_rows: list[list],
        col: dict,
        stim_ptps:    dict[str, list[float]],
        reference_map: dict[str, str],
        plateau_tolerance: float = 0.10,
        log_callback=print,
) -> None:
    """
    Fill normalisation columns in-place in the trial row list.

    For each stim type with a reference designation, computes:
        Normalised_PTP = trial PTP / reference_mean

    The reference_mean uses plateau detection (compute_mmax) if
    plateau_tolerance > 0, otherwise uses the simple mean of all
    reference trials.

    Parameters
    ----------
    latency_rows    : list of rows indexed by col dict
    col             : {column_name: row_index}
    stim_ptps       : {stim_type: [ptp_val, ...]}
    reference_map   : {stim_type: ref_stim_type}  —  "" / None = no normalisation
    plateau_tolerance: fraction of peak for plateau detection (0 = simple mean)
    """
    if not latency_rows or not reference_map:
        return

    # ── Compute reference mean for each referenced stim ───────────────────────
    ref_means: dict[str, tuple[float, int, str]] = {}  # {ref_stim: (mean, n, method)}
    for ref_st in set(v for v in reference_map.values() if v):
        if ref_st not in stim_ptps:
            continue
        vals = [v for v in stim_ptps[ref_st]
                if v is not None and np.isfinite(v) and v > 0]
        if not vals:
            continue
        if plateau_tolerance > 0:
            r = compute_mmax(vals, plateau_tolerance=plateau_tolerance)
            ref_means[ref_st] = (r["mmax"], r["n_plateau"], r["method"])
            log_callback(
                f"📐 Reference '{ref_st}': {r['mmax']:.3f} mV  "
                f"({r['method']}, {r['n_plateau']} trial(s), "
                f"plateau ≥ {100 - int(plateau_tolerance*100)}% of peak {r['peak_ptp']:.3f} mV)"
            )
        else:
            mean_val = float(np.mean(vals))
            ref_means[ref_st] = (mean_val, len(vals), "mean")
            log_callback(
                f"📐 Reference '{ref_st}': {mean_val:.3f} mV  "
                f"(mean of {len(vals)} trials)"
            )

    # ── Fill rows ─────────────────────────────────────────────────────────────
    ri_type   = col["Reference_Type"]
    ri_mean   = col["Reference_Mean(mV)"]
    ri_n      = col["Reference_N"]
    ri_norm   = col["Normalised_PTP"]
    ri_ptp    = col["PTP(mV)"]
    ri_sttype = col["StimType"]

    for row in latency_rows:
        st       = row[ri_sttype]
        ref_st   = reference_map.get(st, "")
        if not ref_st or ref_st not in ref_means:
            continue
        ptp = row[ri_ptp]
        try:
            ptp_f = float(ptp)
        except (TypeError, ValueError):
            continue

        mean_ref, n_ref, method = ref_means[ref_st]
        if mean_ref <= 0:
            continue

        row[ri_type] = f"{ref_st}_{method}"
        row[ri_mean] = round(mean_ref, 4)
        row[ri_n]    = n_ref
        row[ri_norm] = round(ptp_f / mean_ref, 4)

