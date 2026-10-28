"""
Trial-to-trial variability — built-in first-level (single-file) MEP-CMAP add-on
==============================================================================

Quantifies how much a response measure varies from trial to trial within each
stimulus type of ONE recording, and whether that variation is random or
structured (drifting across the session, or serially dependent between
successive trials).

What this add-on can and cannot tell you
----------------------------------------
Within a single recording there is no between-participant variance, so nothing
here is a reliability coefficient and none of it should be reported as one.
Reliability, ICC, and minimal detectable change require several participants and
come from the group-level `variability_group` add-on on the merged group table.

What one recording DOES support:
  * spread — coefficient of variation, four ways, with confidence intervals
  * outlying trials — robust (median/MAD) and log-scale z scores
  * precision — how tight the condition mean is, and how many trials it would
    take to tighten it to a given percentage
  * serial structure — autocorrelation, an AR model, and a drift test
  * agreement — how far one trial, or one k-trial average, can sit from another
  * the block drift index — near zero means a stationary series, high means
    amplitude shifted systematically across the session

Outputs (new files only; core outputs are never modified)
---------------------------------------------------------
  <prefix>_variability.csv
      One row per trial, carrying File / StimType / Segment so that Second
      Level ▸ Group Analysis joins it onto the core trial table automatically.
      Only columns the core pipeline does NOT already produce are emitted: the
      core already writes Z_PTP_Within and Z_PTP_Pooled, so the classical z
      scores are deliberately not duplicated here.

  <prefix>_variability_by_condition.csv
      One row per stimulus type. Deliberately NOT named "*_summary.csv", which
      Second Level treats as a core output, and it carries no Segment column so
      it is never mistaken for a per-trial sidecar.

  figures/<prefix>_variability_figures/*.png   (optional, off by default)

Method references: Vangel (1996) modified McKay CV interval; Bland & Altman
(1999) limits of agreement; Ljung & Box (1978); Shrout & Fleiss (1979).
"""

import os
import numpy as np
import pandas as pd

try:
    from mep_cmap import variability as V
    _IMPORT_ERROR = None
except Exception as _ex:                  # pragma: no cover - packaging failure
    V = None
    _IMPORT_ERROR = _ex

try:
    # Stimulus types such as ">" or "?" cannot go into a filename as they are.
    from mep_cmap.utils import filename_token as _fname_token
except Exception:                         # pragma: no cover - packaging failure
    def _fname_token(label):
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(label))

ADDON_NAME        = "variability"
ADDON_DESCRIPTION = ("Trial-to-trial variability per stimulus type: coefficient of "
                     "variation, robust z scores, precision of the mean, serial "
                     "dependence and drift, and single-trial limits of agreement")
ADDON_VERSION     = "1.0.0"
ADDON_AUTHOR      = "MEP-CMAP Analyser (built-in)"
ADDON_SCOPE       = "single_file"

ADDON_SETTINGS = [
    {
        "key": "variability_metric",
        "label": "Measure to analyse",
        "help": ("Column of the per-trial table to quantify. Defaults to peak-to-peak "
                 "amplitude. Other useful choices are 'AUC(mV*s)', "
                 "'Adjusted_PTP_QR(mV)', 'Normalised_PTP', or 'cSP_Duration(ms)'. "
                 "One run analyses one measure and overwrites the previous output. "
                 "Columns that were never populated (normalisation not run, cSP "
                 "detection off) are reported as having no usable trials."),
        "type": "str",
        "default": "PTP(mV)",
        "choices": ["PTP(mV)", "AUC(mV*s)", "Adjusted_PTP_QR(mV)",
                    "PTP_Detrended_WithinCond(mV)", "PTP_Detrended_Session(mV)",
                    "PTP_per_PreStimRMS", "Normalised_PTP",
                    "Normalised_Adjusted_PTP_QR", "Latency(ms)",
                    "cSP_Duration(ms)", "cSP_MEP_Ratio(ms/mV)", "PreStimRMS"],
        # Overridden at render time by the columns actually present in
        # this dataset; the list above is the fallback when no dataset
        # is open yet. Editable, so an unlisted column can be typed.
        "choices_from": "trial_columns",
    },
    {
        "key": "variability_block_size",
        "label": "Block size (trials)",
        "help": ("Trials per block for the drift index. Blocks are compared against "
                 "each other, so this many trials should be a meaningful chunk of the "
                 "run; 5 works well for 20-trial conditions."),
        "type": "int",
        "default": 5,
        "min": 2,
        "max": 100,
    },
    {
        "key": "variability_pairing",
        "label": "Pairing for limits of agreement",
        "help": ("How to build pairs from one run of trials. 'single' pairs odd "
                 "against even trials and asks how far one trial can sit from "
                 "another; 'block' pairs consecutive block means; 'half' pairs the "
                 "first half against the second and exposes drift. None of these is "
                 "test-retest agreement, which needs a second session."),
        "type": "str",
        "default": "single",
        "choices": ["single", "block", "half"],
    },
    {
        "key": "variability_robust_z",
        "label": "Robust z threshold",
        "help": ("Trials beyond this median/MAD z are counted as extreme. Unlike a "
                 "classical z score, an outlier cannot inflate its own denominator "
                 "here, so 3.5 is a stricter test than it looks."),
        "type": "float",
        "default": 3.5,
        "min": 1.0,
        "max": 10.0,
    },
    {
        "key": "variability_exclude_flagged",
        "label": "Exclude trials flagged as outliers",
        "help": ("Leave trials marked in Outlier_Decision out of the statistics. "
                 "They still appear in the per-trial output with "
                 "Var_Used_In_Summary set to False, so nothing is hidden."),
        "type": "bool",
        "default": True,
    },
    {
        "key": "variability_cross_metrics",
        "label": "Measures to cross-correlate",
        "help": ("Comma-separated trial-level columns to correlate against each "
                 "other within each condition. Pre-stimulus EMG against amplitude "
                 "is the usual one: it turns part of what looks like trial-to-trial "
                 "noise into something explainable. Leave blank to skip."),
        "type": "str",
        "default": "PTP(mV),Latency(ms),PreStimRMS",
    },
    {
        "key": "variability_figures",
        "label": "Write diagnostic figures",
        "help": ("Off by default because figure rendering dominates the run time "
                 "over a whole dataset. Turn it on for a recording you want to look "
                 "at closely: one figure per stimulus type, into the figures folder."),
        "type": "bool",
        "default": False,
    },
]

# Per-trial output. Names are metric-neutral because the measure is configurable,
# and Var_Metric records which one was analysed. Deliberately avoids every column
# name the core pipeline writes, so the group join never has to namespace them.
TRIAL_COLUMNS = [
    "File", "StimType", "Segment",
    "Var_Metric", "Var_Z_Robust", "Var_Z_Log",
    "Var_Cumulative_Mean", "Var_Used_In_Summary",
]

# Fixed column order for the per-condition file, so downstream code sees the same
# columns whether or not every condition had enough trials to be summarised.
CONDITION_COLUMNS = [
    "File", "StimType", "Metric", "N_Trials_Total", "N_Trials_Used",
    "Mean", "SD", "Median", "Geometric_Mean",
    "CV(%)", "CV_Corrected(%)", "CV_Log(%)", "CV_Robust(%)",
    "MAD_Raw", "MAD_Scaled", "IQR", "IQR_SD_Equivalent",
    "IQR(%_of_median)", "IQR_CV(%)",
    "Outliers_Removed_N", "CV_Outliers_Removed(%)", "CV_Change_When_Trimmed(%)",
    "Jackknife_Max_Influence_On_CV(%)", "Jackknife_Most_Influential_Trial",
    "Jackknife_Single_Trial_Dominates",
    "CV_CI_Lo(%)", "CV_CI_Hi(%)",
    "Mean_CI_Lo", "Mean_CI_Hi", "Mean_CI_Halfwidth(%)", "N_Trials_For_10pct_CI",
    "N_Extreme_Robust_Z", "Max_Abs_Robust_Z",
    "RMSE_Trial_About_Mean", "RMSE(%_of_mean)",
    "Typical_Error", "Typical_Error(%_of_mean)", "Typical_Error_CV(%)",
    "AR_Resid_RMSE", "AR0_RMSE", "AR_RMSE_Reduction(%)",
    "Trials_For_LoA_Within_50pct",
    "ACF_Lag1", "AR_Order", "AR_Phi1", "Ljung_Box_p",
    "Drift_Slope_Per_Trial", "Drift_p", "Drift(%_per_10_trials)",
    "LoA_Pairing", "LoA_N_Pairs", "LoA_Lower", "LoA_Upper", "LoA_Width(%_of_mean)",
    "Ratio_LoA_Lower", "Ratio_LoA_Upper",
    "Block_Drift_ICC1", "Block_Drift_ICC1_Lo", "Block_Drift_ICC1_Hi", "N_Blocks",
]

_FLAG_WORDS = ("flagged", "excluded", "exclude", "reject", "rejected", "outlier")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cfg(config, key, default, cast=None):
    """Read one add-on setting, falling back to the default on anything odd."""
    val = (config or {}).get(key, default)
    if cast is None:
        return val
    try:
        if cast is bool:
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("1", "true", "yes", "on")
        return cast(val)
    except Exception:
        return default


def _is_flagged(series):
    """True where Outlier_Decision marks a trial as excluded.

    'Not flagged' contains the word 'flagged', so a substring test would invert
    the meaning. Matching is done on the whole normalised string instead.
    """
    s = series.astype(str).str.strip().str.lower()
    return s.isin(_FLAG_WORDS)


def _resolve_metric(trials, requested, log):
    """Pick the metric column, falling back to peak-to-peak if it is missing."""
    if requested in trials.columns:
        return requested
    fallback = "PTP(mV)"
    if fallback in trials.columns:
        log(f"{ADDON_NAME}: column '{requested}' not in the trial table — "
            f"falling back to '{fallback}'.")
        return fallback
    numeric = [c for c in trials.columns
               if pd.to_numeric(trials[c], errors="coerce").notna().mean() > 0.5]
    wanted = (f"'{requested}' is not in the trial table"
              if requested == fallback else
              f"neither '{requested}' nor '{fallback}' is in the trial table")
    raise ValueError(
        f"{ADDON_NAME}: {wanted}. Numeric-looking columns are: "
        + ", ".join(numeric[:15]))


def _segments_for(sub):
    """1-based trial index within the condition, matching the core table."""
    if "Segment" in sub.columns:
        seg = pd.to_numeric(sub["Segment"], errors="coerce")
        if seg.notna().all():
            return seg.astype(int).to_numpy()
    return np.arange(1, len(sub) + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def _plot_condition(x, res, tta, metric, stim_type, out_png):
    """Six-panel diagnostic: the series, its shape, its precision, its memory,
    how many trials an average needs, and where the noise sits."""
    import matplotlib.pyplot as plt

    n = x.size
    t = np.arange(1, n + 1)
    m = res.get("mean", float(np.mean(x)))
    sd = res.get("sd", float(np.std(x, ddof=1)))

    fig, axes = plt.subplots(3, 2, figsize=(10, 10.5))
    fig.suptitle(f"Variability: {stim_type} — {metric}", fontweight="bold")

    ax = axes[0, 0]
    ax.axhspan(m - 2 * sd, m + 2 * sd, color="#5DADE2", alpha=0.15, label="mean ± 2 SD")
    ax.axhspan(m - sd, m + sd, color="#5DADE2", alpha=0.25, label="mean ± 1 SD")
    ax.axhline(m, color="#2C3E50", lw=1.2, label=f"mean {m:.3f}")
    ax.plot(t, x, "o-", color="#2C3E50", ms=4, lw=1)
    zr = res.get("z_robust")
    if zr is not None:
        ext = np.isfinite(zr) & (np.abs(zr) > res.get("robust_z_threshold", 3.5))
        if np.any(ext):
            ax.plot(t[ext], x[ext], "o", mfc="none", mec="#C0392B", ms=11, mew=2)
    if res.get("trend_p", 1.0) < 0.05:
        sl = res["trend_slope_per_trial"]
        ax.plot(t, sl * t + (m - sl * t.mean()), "--", color="#C0392B", lw=1.2,
                label=f"drift p={res['trend_p']:.3f}")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_xlabel("Trial number within this condition (collection order)")
    ax.set_ylabel(metric)
    ax.set_title("Trial-ordered series", fontsize=10)

    ax = axes[0, 1]
    ax.hist(x, bins=max(6, int(np.sqrt(n)) + 3), color="#5DADE2", edgecolor="white")
    ax.axvline(m, color="#2C3E50", lw=1.2, label=f"mean {m:.3f}")
    ax.axvline(res.get("median", np.median(x)), color="#95A5A6", lw=1.2, ls=":",
               label=f"median {res.get('median', np.median(x)):.3f}")
    ax.legend(fontsize=8)
    ax.set_xlabel(metric)
    ax.set_ylabel("Number of trials")
    ax.set_title(f"Distribution (CV = {res.get('cv_percent', float('nan')):.1f} %)",
                 fontsize=10)

    ax = axes[1, 0]
    cm = res.get("cumulative_mean")
    if cm is not None:
        ax.plot(t, cm, "-", color="#2C3E50", lw=1.5)
        ax.axhline(m, color="#C0392B", ls="--", lw=1)
        ax.fill_between([1, n], m * 0.95, m * 1.05, color="#C0392B", alpha=0.10)
        if n > 4:
            lo, hi = np.nanmin(cm[3:]), np.nanmax(cm[3:])
            pad = 0.35 * (hi - lo) + 1e-12
            ax.set_ylim(min(lo - pad, m * 0.93), max(hi + pad, m * 1.07))
    ax.set_xlabel("Number of trials averaged, from the first trial onward")
    ax.set_ylabel(f"Running mean of {metric}")
    ax.set_title("Running mean (band = ±5 % of final)", fontsize=10)

    ax = axes[1, 1]
    ar = res.get("_ar")
    if ar is not None:
        r = ar["acf"]
        lags = np.arange(len(r))
        ax.vlines(lags[1:], 0, r[1:], color="#2C3E50", lw=2)
        ax.plot(lags[1:], r[1:], "o", color="#2C3E50", ms=4)
        ax.axhspan(-ar["acf_bound"], ar["acf_bound"], color="#95A5A6", alpha=0.25)
        ax.axhline(0, color="k", lw=0.8)
        lim = max(float(np.max(np.abs(r[1:]))), ar["acf_bound"]) * 1.35
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Correlation between trials this far apart")
        ax.set_title(f"Autocorrelation (AR order {ar['best_order']})", fontsize=10)
    else:
        ax.text(0.5, 0.5, "too few trials", ha="center", va="center",
                transform=ax.transAxes, color="#95A5A6")
        ax.set_axis_off()
    ax.set_xlabel("Lag (how many trials apart)")

    ax = axes[2, 0]
    if tta is not None and len(tta):
        ax.plot(tta["k_trials_averaged"], tta["loa_width_pct_of_mean"], "o-",
                color="#2C3E50", ms=4, label="95% LoA width")
        ax.plot(tta["k_trials_averaged"], tta["SEM_Pct_Of_Mean"], "s--",
                color="#5DADE2", ms=4, label="SEM of the mean")
        ax.axhline(50, ls=":", color="#C0392B", lw=1)
        ax.set_xlabel("Trials averaged per measurement (k)")
        ax.set_ylabel("Spread, as % of the condition mean")
        ax.set_title("How many trials an average needs", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()

    ax = axes[2, 1]
    labels, vals = [], []
    for key, lab in (("sd_sample", "SD of\ntrials"),
                     ("typical_error", "Typical\nerror"),
                     ("ar_resid_rmse", "AR\nresidual")):
        v = res.get(key)
        if v is not None and np.isfinite(v):
            labels.append(lab); vals.append(v)
    if vals:
        bars = ax.bar(labels, vals, color=["#2C3E50", "#5DADE2", "#C0392B"][:len(vals)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_ylabel(f"Noise magnitude ({metric})")
        ax.set_xlabel("Estimator")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title("Noise, three ways", fontsize=10)
    else:
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)




def _n(v, nd=3):
    """Format a number for the caption, or say plainly that it is unavailable."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "not available"
    if not np.isfinite(f):
        return "not available"
    return ("%." + str(nd) + "f") % f


def _caption_condition(x, res, tta, metric, stim_type, file_name):
    """A caption filled with this recording's own numbers.

    Written beside the figure so the panels can be read without going back to
    the documentation, and so the figure can be handed to a colleague or dropped
    into a supplement as it stands. Interpretation warnings appear only when the
    data actually triggers them.
    """
    L = []
    add = L.append
    n = int(x.size)

    add(f"FIGURE: trial-to-trial variability, condition '{stim_type}', {metric}")
    add(f"Recording: {file_name}")
    add(f"Trials analysed: {n}")
    add("=" * 72)
    add("")
    add("WHAT EACH PANEL SHOWS")
    add("")
    add("Top left - Trial-ordered series.")
    add("  Every point is one trial, in the order it was collected. The shaded")
    add("  bands are one and two standard deviations either side of the mean.")
    add(f"  Mean {_n(res.get('mean'))}, SD {_n(res.get('sd'))}.")
    if res.get("n_extreme_robust_z"):
        add(f"  {res['n_extreme_robust_z']} trial(s) ringed in red exceed a robust")
        add("  (median/MAD) z of 3.5. Robust z is used because an outlier cannot")
        add("  inflate its own denominator, unlike a classical z score.")
    if res.get("trend_p") is not None and res["trend_p"] < 0.05:
        add(f"  A red dashed line marks a significant drift across the run:")
        add(f"  {_n(res.get('trend_pct_per_10_trials'), 1)} % of the mean per 10 "
            f"trials (p = {_n(res.get('trend_p'), 4)}).")
    else:
        add("  No significant drift across the run, so no trend line is drawn.")
    add("")
    add("Top right - Distribution.")
    add("  How the trial amplitudes are spread. The solid line is the mean and")
    add("  the dotted line the median; a gap between them indicates skew, which")
    add("  MEP amplitudes usually show.")
    add(f"  Coefficient of variation {_n(res.get('cv_percent'), 1)} %, "
        f"median {_n(res.get('median'))},")
    add(f"  skewness {_n(res.get('skew'), 2)}.")
    add("")
    add("Middle left - Running mean.")
    add("  The average of the first k trials as k increases, showing how quickly")
    add("  the estimate settles. The pink band is 5 % either side of the final")
    add("  mean. A stable recording enters the band early and stays there; a")
    add("  line that climbs or falls most of the way indicates drift rather than")
    add("  random noise.")
    add("")
    add("Middle right - Autocorrelation.")
    add("  Whether a trial predicts the ones that follow it. Each bar is the")
    add("  correlation between trials that many apart. Bars inside the grey band")
    add("  are indistinguishable from chance.")
    add(f"  Lag-1 autocorrelation {_n(res.get('acf_lag1'), 3)}; the selected")
    add(f"  autoregressive order was {res.get('ar_order')}.")
    if res.get("acf_lag1") is not None and np.isfinite(res.get("acf_lag1", np.nan)):
        if res["acf_lag1"] < -0.1:
            add("  Negative means alternation: a large trial tends to be followed")
            add("  by a smaller one.")
        elif res["acf_lag1"] > 0.1:
            add("  Positive means persistence: a large trial tends to be followed")
            add("  by another large one.")
    add("")
    add("Bottom left - How many trials an average needs.")
    add("  Take two independent sets of k trials and average each. The dark line")
    add("  is how far apart those two averages could fall (95 % limits of")
    add("  agreement), as a percentage of the mean. This is the number that")
    add("  should drive how many stimuli a protocol delivers. The light line is")
    add("  the textbook standard error, which assumes independent trials and is")
    add("  correspondingly more optimistic.")
    if tta is not None and len(tta):
        k1 = tta.iloc[0]
        add(f"  At k = 1 the limits span {_n(k1['loa_width_pct_of_mean'], 0)} % of "
            f"the mean.")
        k50 = res.get("trials_for_loa_50")
        if k50 is not None and np.isfinite(k50):
            add(f"  {int(k50)} trials brings that below 50 %.")
        else:
            add("  Within the range tested, no number of trials brings it below")
            add("  50 %, which is itself the finding.")
    add("")
    add("Bottom right - Noise, three ways.")
    add("  Three estimates of noise that are routinely confused.")
    add(f"  SD of trials {_n(res.get('sd_sample'))}: spread of single trials about")
    add("    the condition mean.")
    add(f"  Typical error {_n(res.get('typical_error'))}: measurement noise on a")
    add("    single trial, from paired odd and even trials.")
    add(f"  AR residual {_n(res.get('ar_resid_rmse'))}: what remains after the")
    add("    previous trials are used to predict the next one. The gap between")
    add("    this and the SD is how much the ordering explains")
    add(f"    ({_n(res.get('ar_rmse_reduction_pct'), 1)} % reduction).")
    add("")

    warnings = []
    te, sd = res.get("typical_error"), res.get("sd")
    if te is not None and sd is not None and np.isfinite(te) and np.isfinite(sd) \
            and te > sd:
        warnings.append(
            "Typical error came out LARGER than the SD. Odd-even pairing compares "
            "adjacent trials, so negative autocorrelation inflates the paired "
            "differences. This is a real property of the series rather than an "
            "error, but the typical error is pessimistic here; prefer the limits "
            "of agreement between independent k-trial averages (bottom left).")
    if res.get("trend_p") is not None and res["trend_p"] < 0.05:
        warnings.append(
            "A significant drift inflates the coefficient of variation and the "
            "typical error, because a series that climbs or falls steadily has a "
            "wide spread even when each individual trial is precise. Re-running "
            "on the PTP_Detrended_WithinCond(mV) column and comparing the two CVs "
            "shows how much of the variability is drift rather than noise.")
    if n < 12:
        warnings.append(
            f"With only {n} trials, every quantity here carries wide uncertainty; "
            "the autocorrelation and the limits of agreement especially.")
    if warnings:
        add("READ WITH CARE")
        add("")
        for w in warnings:
            for i, line in enumerate(_wrap(w, 70)):
                add(("  - " if i == 0 else "    ") + line)
            add("")

    add("Produced by the MEP-CMAP Analyser 'variability' add-on "
        f"v{ADDON_VERSION}.")
    add("Nothing here is a reliability coefficient: a single recording has no")
    add("between-participant variance. Reliability, ICC and minimal detectable")
    add("change come from the group-level 'variability_group' add-on.")
    return "\n".join(L)


def _wrap(text, width):
    """Wrap without pulling in textwrap, keeping the module import list short."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(context):
    if V is None:
        raise RuntimeError(
            "the variability add-on needs mep_cmap.variability, which failed to "
            f"import ({_IMPORT_ERROR}). Check that variability.py sits beside "
            "addons.py in the mep_cmap package.")

    log = context.log
    cfg = context.config or {}
    trials = getattr(context, "trials", None)
    if trials is None or len(trials) == 0:
        log(f"{ADDON_NAME}: no per-trial table beside the waveform bundle — "
            f"run the analysis first so <prefix>_trials.csv exists.")
        return []

    metric = _resolve_metric(trials, _cfg(cfg, "variability_metric", "PTP(mV)", str), log)
    block_size = _cfg(cfg, "variability_block_size", 5, int)
    pairing = _cfg(cfg, "variability_pairing", "single", str).strip().lower()
    if pairing not in ("single", "block", "half"):
        log(f"{ADDON_NAME}: pairing '{pairing}' not recognised — using 'single'.")
        pairing = "single"
    robust_z = _cfg(cfg, "variability_robust_z", 3.5, float)
    drop_flagged = _cfg(cfg, "variability_exclude_flagged", True, bool)
    want_figs = _cfg(cfg, "variability_figures", False, bool)
    cross_metrics = [m.strip() for m in
                     _cfg(cfg, "variability_cross_metrics",
                          "PTP(mV),Latency(ms),PreStimRMS", str).split(",")
                     if m.strip()]

    # File is the join key; context.trials is already filtered to this recording.
    file_name = context.bids_prefix
    if "File" in trials.columns and len(trials["File"]):
        file_name = trials["File"].iloc[0]

    stim_col = "StimType" if "StimType" in trials.columns else None
    if stim_col is None:
        log(f"{ADDON_NAME}: the trial table has no StimType column — nothing to group by.")
        return []

    trial_rows = []
    cond_rows = []
    tta_rows = []
    sens_rows = []
    jk_rows = []
    fig_jobs = []
    by_condition_values = {}
    corr_rows = []

    for stim_type, sub in trials.groupby(stim_col, sort=True):
        sub = sub.copy()
        if "Segment" in sub.columns:
            sub = sub.sort_values("Segment", kind="mergesort")
        seg = _segments_for(sub)

        raw = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        used = np.isfinite(raw)
        if drop_flagged and "Outlier_Decision" in sub.columns:
            used = used & ~_is_flagged(sub["Outlier_Decision"]).to_numpy()

        x = raw[used]
        n_used = int(x.size)

        z_rob = np.full(raw.size, np.nan)
        z_log = np.full(raw.size, np.nan)
        cum = np.full(raw.size, np.nan)

        res = {}
        if n_used >= 3:
            res = V.summarise_condition(
                x, block_size=block_size, pairing=pairing,
                robust_z_thresh=robust_z, n_boot=2000, n_mc=1000)
            res["robust_z_threshold"] = robust_z
            if res.get("z_robust") is not None:
                z_rob[used] = res["z_robust"]
            if res.get("z_log") is not None:
                z_log[used] = res["z_log"]
            if res.get("cumulative_mean") is not None:
                cum[used] = res["cumulative_mean"]
            # RMSE family: typical error, and how much the AR model helps
            res.update(V.rmse_family(x, res.get("_ar")))

            # The full dispersion family, so the conventional CV can be read
            # against the robust alternatives rather than on its own.
            res.update(V.dispersion_metrics(x))

            # Does the answer depend on the outliers?
            sens, sinfo = V.dispersion_sensitivity(x, robust_z_thresh=robust_z)
            if len(sens):
                st = sens.copy()
                st.insert(0, "File", file_name)
                st.insert(1, "StimType", stim_type)
                st.insert(2, "Metric", metric)
                st["N_Trials"] = sinfo.get("n_trials")
                st["N_Removed"] = sinfo.get("n_removed")
                sens_rows.append(st)
                res["sens_n_removed"] = sinfo.get("n_removed")
                cvrow = sens[sens["Dispersion_Metric"] == "CV(%)"]
                if len(cvrow):
                    res["sens_cv_trimmed"] = float(cvrow["Outliers_Removed"].iloc[0])
                    res["sens_cv_change"] = float(cvrow["Change(%)"].iloc[0])

            # Is the dispersion a property of the series, or of one trial?
            jk, jinfo = V.jackknife_dispersion(x, robust_z_thresh=robust_z)
            if len(jk):
                jt = jk.copy()
                jt.insert(0, "File", file_name)
                jt.insert(1, "StimType", stim_type)
                jt.insert(2, "Metric", metric)
                jk_rows.append(jt)
                res["jk_max_influence"] = jinfo.get("max_abs_influence")
                res["jk_trial"] = jinfo.get("most_influential_trial")
                res["jk_dominates"] = jinfo.get("single_trial_dominates")
                if jinfo.get("single_trial_dominates"):
                    log(f"{ADDON_NAME}: {stim_type} — trial "
                        f"{jinfo['most_influential_trial']} alone moves the CV from "
                        f"{jinfo['cv_percent']:.1f}% to "
                        f"{jinfo['cv_without_most_influential']:.1f}%; the summary is "
                        f"describing that trial as much as the series.")

            # Trials-to-average: computed on every run, and now actually written
            tta = V.trials_to_average_table(x, n_rep=1000)
            if len(tta):
                t2 = tta.copy()
                t2.insert(0, "File", file_name)
                t2.insert(1, "StimType", stim_type)
                t2.insert(2, "Metric", metric)
                tta_rows.append(t2)
                within50 = tta[tta["loa_width_pct_of_mean"] <= 50.0]
                res["trials_for_loa_50"] = (int(within50["k_trials_averaged"].min())
                                            if len(within50) else np.nan)
            by_condition_values[stim_type] = x

            if want_figs:
                fig_jobs.append((x, res, tta, stim_type))
        else:
            log(f"{ADDON_NAME}: {stim_type} has only {n_used} usable trial(s) — "
                f"summary skipped (3 are needed).")

        for i in range(raw.size):
            trial_rows.append([
                file_name, stim_type, int(seg[i]), metric,
                z_rob[i], z_log[i], cum[i], bool(used[i]),
            ])

        row = {
            "File": file_name,
            "StimType": stim_type,
            "Metric": metric,
            "N_Trials_Total": int(raw.size),
            "N_Trials_Used": n_used,
        }
        if res and "note" not in res:
            row.update({
                "Mean": res.get("mean"),
                "SD": res.get("sd"),
                "Median": res.get("median"),
                "Geometric_Mean": res.get("geometric_mean"),
                "CV(%)": res.get("cv_percent"),
                "CV_Corrected(%)": res.get("cv_corrected_percent"),
                "CV_Log(%)": res.get("cv_log_percent"),
                "CV_Robust(%)": res.get("cv_robust_percent"),
                "MAD_Raw": res.get("mad_raw"),
                "MAD_Scaled": res.get("mad_scaled"),
                "IQR": res.get("iqr"),
                "IQR_SD_Equivalent": res.get("iqr_sd_equivalent"),
                "IQR(%_of_median)": res.get("iqr_percent_of_median"),
                "IQR_CV(%)": res.get("iqr_cv_percent"),
                "Outliers_Removed_N": res.get("sens_n_removed"),
                "CV_Outliers_Removed(%)": res.get("sens_cv_trimmed"),
                "CV_Change_When_Trimmed(%)": res.get("sens_cv_change"),
                "Jackknife_Max_Influence_On_CV(%)": res.get("jk_max_influence"),
                "Jackknife_Most_Influential_Trial": res.get("jk_trial"),
                "Jackknife_Single_Trial_Dominates": res.get("jk_dominates"),
                "CV_CI_Lo(%)": res.get("cv_boot_lo_percent"),
                "CV_CI_Hi(%)": res.get("cv_boot_hi_percent"),
                "Mean_CI_Lo": res.get("ci_lo"),
                "Mean_CI_Hi": res.get("ci_hi"),
                "Mean_CI_Halfwidth(%)": res.get("ci_halfwidth_pct_of_mean"),
                "N_Trials_For_10pct_CI": res.get("n_trials_for_10pct_ci"),
                "N_Extreme_Robust_Z": res.get("n_extreme_robust_z"),
                "Max_Abs_Robust_Z": res.get("max_abs_robust_z"),
                "RMSE_Trial_About_Mean": res.get("rmse_trial_about_mean"),
                "RMSE(%_of_mean)": res.get("rmse_pct_of_mean"),
                "Typical_Error": res.get("typical_error"),
                "Typical_Error(%_of_mean)": res.get("typical_error_pct_of_mean"),
                "Typical_Error_CV(%)": res.get("typical_error_cv_percent"),
                "AR_Resid_RMSE": res.get("ar_resid_rmse"),
                "AR0_RMSE": res.get("ar0_rmse"),
                "AR_RMSE_Reduction(%)": res.get("ar_rmse_reduction_pct"),
                "Trials_For_LoA_Within_50pct": res.get("trials_for_loa_50"),
                "ACF_Lag1": res.get("acf_lag1"),
                "AR_Order": res.get("ar_order"),
                "AR_Phi1": res.get("ar_phi1"),
                "Ljung_Box_p": res.get("ljung_box_p"),
                "Drift_Slope_Per_Trial": res.get("trend_slope_per_trial"),
                "Drift_p": res.get("trend_p"),
                "Drift(%_per_10_trials)": res.get("trend_pct_per_10_trials"),
                "LoA_Pairing": res.get("loa_pairing"),
                "LoA_N_Pairs": res.get("loa_n_pairs"),
                "LoA_Lower": res.get("loa_lo"),
                "LoA_Upper": res.get("loa_hi"),
                "LoA_Width(%_of_mean)": res.get("loa_width_pct_of_mean"),
                "Ratio_LoA_Lower": res.get("ratio_loa_lo"),
                "Ratio_LoA_Upper": res.get("ratio_loa_hi"),
                # Not a reliability coefficient: blocks are the targets, so this
                # says whether amplitude drifted across the run, nothing more.
                "Block_Drift_ICC1": res.get("block_drift_icc1"),
                "Block_Drift_ICC1_Lo": res.get("block_drift_icc1_lo"),
                "Block_Drift_ICC1_Hi": res.get("block_drift_icc1_hi"),
                "N_Blocks": res.get("n_blocks"),
            })
        cond_rows.append(row)

        if cross_metrics and n_used >= 4:
            cm = V.metric_correlations(sub[used] if len(sub) == used.size else sub,
                                       cross_metrics)
            if len(cm):
                cm = cm.copy()
                cm.insert(0, "File", file_name)
                cm.insert(1, "StimType", stim_type)
                corr_rows.append(cm)

    if not trial_rows:
        log(f"{ADDON_NAME}: nothing to write.")
        return []

    written = []

    trial_df = pd.DataFrame(trial_rows, columns=TRIAL_COLUMNS)
    # The group join validates one row per (File, StimType, Segment); duplicate
    # Segment values would otherwise abort the merge for the whole session.
    dupes = trial_df.duplicated(subset=["File", "StimType", "Segment"]).sum()
    if dupes:
        log(f"{ADDON_NAME}: {dupes} duplicate (StimType, Segment) row(s) dropped so "
            f"the group-level join stays one-to-one.")
        trial_df = trial_df.drop_duplicates(subset=["File", "StimType", "Segment"],
                                            keep="first")
    trial_path = os.path.join(context.addons_dir,
                              f"{context.bids_prefix}_{ADDON_NAME}.csv")
    trial_df.to_csv(trial_path, index=False)
    written.append(trial_path)

    cond_df = pd.DataFrame(cond_rows).reindex(columns=CONDITION_COLUMNS)
    cond_path = os.path.join(context.addons_dir,
                             f"{context.bids_prefix}_{ADDON_NAME}_by_condition.csv")
    cond_df.to_csv(cond_path, index=False)
    written.append(cond_path)

    if tta_rows:
        tta_path = os.path.join(context.addons_dir,
                                f"{context.bids_prefix}_{ADDON_NAME}_trials_to_average.csv")
        pd.concat(tta_rows, ignore_index=True).to_csv(tta_path, index=False)
        written.append(tta_path)

    # Contrasts between conditions: a between-condition property, so it gets its
    # own file rather than being forced into the per-condition table.
    if len(by_condition_values) >= 2:
        pairs, omni = V.compare_conditions(by_condition_values)
        if len(pairs):
            pairs = pairs.copy()
            pairs.insert(0, "File", file_name)
            pairs.insert(1, "Metric", metric)
            for k, v in (omni or {}).items():
                pairs["Omnibus_" + k] = v
            con_path = os.path.join(context.addons_dir,
                                    f"{context.bids_prefix}_{ADDON_NAME}_contrasts.csv")
            pairs.to_csv(con_path, index=False)
            written.append(con_path)
            if omni.get("fligner_p") is not None:
                verdict = ("differ" if omni["fligner_p"] < 0.05 else "do not differ")
                log(f"{ADDON_NAME}: conditions {verdict} in spread "
                    f"(Fligner-Killeen p = {omni['fligner_p']:.4f}).")

    if sens_rows:
        sp = os.path.join(context.addons_dir,
                          f"{context.bids_prefix}_{ADDON_NAME}_sensitivity.csv")
        pd.concat(sens_rows, ignore_index=True).to_csv(sp, index=False)
        written.append(sp)

    if jk_rows:
        jp = os.path.join(context.addons_dir,
                          f"{context.bids_prefix}_{ADDON_NAME}_jackknife.csv")
        pd.concat(jk_rows, ignore_index=True).to_csv(jp, index=False)
        written.append(jp)

    if corr_rows:
        cor_path = os.path.join(context.addons_dir,
                                f"{context.bids_prefix}_{ADDON_NAME}_correlations.csv")
        pd.concat(corr_rows, ignore_index=True).to_csv(cor_path, index=False)
        written.append(cor_path)

    done = cond_df[cond_df.get("CV(%)").notna()] if "CV(%)" in cond_df.columns \
        else cond_df.iloc[0:0]
    if len(done):
        bits = ", ".join(f"{r['StimType']} CV={r['CV(%)']:.1f}%"
                         for _, r in done.iterrows())
        log(f"{ADDON_NAME}: {metric} — {bits}")
    log(f"{ADDON_NAME}: {len(trial_df)} trial row(s), {len(cond_df)} condition(s) -> "
        f"{os.path.basename(trial_path)}, {os.path.basename(cond_path)}")

    if want_figs and fig_jobs:
        try:
            import matplotlib
            matplotlib.use("Agg")
            fig_root = context.figures_dir or os.path.join(context.results_dir, "figures")
            fig_dir = os.path.join(fig_root, f"{context.bids_prefix}_{ADDON_NAME}_figures")
            os.makedirs(fig_dir, exist_ok=True)
            for x, res, tta, stim_type in fig_jobs:
                png = os.path.join(fig_dir,
                                   f"{context.bids_prefix}_stim-{_fname_token(stim_type)}_variability.png")
                _plot_condition(x, res, tta, metric, stim_type, png)
                written.append(png)
                cap = png[:-4] + "_caption.txt"
                with open(cap, "w", encoding="utf-8") as fh:
                    fh.write(_caption_condition(x, res, tta, metric, stim_type,
                                                file_name))
                written.append(cap)
            log(f"{ADDON_NAME}: {len(fig_jobs)} figure(s) with captions -> "
                f"{os.path.basename(fig_dir)}/")
        except Exception as ex:
            log(f"{ADDON_NAME}: figure generation skipped ({ex})")

    return written
