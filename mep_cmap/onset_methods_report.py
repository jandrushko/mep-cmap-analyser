"""
mep_cmap.onset_methods_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-file comparison of the onset detectors, as tables and figures.

When onset-method agreement is enabled, every member method runs on every
trial. The pipeline keeps only the median, the spread and the count; the
individual latencies are discarded. This module retains them and turns them
into two tables and three figures, at no extra detection cost -- the detector
runs have already happened.

What this is for
----------------
The recurring practical question is not "which detector is best" -- no detector
wins across resting and active paradigms, SNRs and muscles -- but "which trials
do I need to look at, and does my choice of method matter for this dataset".
A number in a spreadsheet does not answer the second question; seeing where
each method lands on the actual waveform does.

What this is NOT
----------------
Agreement is not accuracy. Detectors that share an assumption can be wrong
together, and two of the members are not independent at all: the walkback
variant starts from the plain derivative method's answer, so their agreement is
partly structural. Tight agreement means the methods concur, not that they are
right. Establishing accuracy needs ground truth -- expert manual marking, or
simulation with a known onset. Every figure produced here carries that caveat
in its subtitle, because a reader who does not already know it is exactly the
reader who will over-read the plot.

Triggering
----------
These outputs follow ``cfg.onset_agreement``, NOT the selected onset method.
The two are independent: agreement runs the member detectors whatever method is
selected, and comparing methods while running the one you trust is the more
useful case -- that is how a method choice gets justified. Selecting
Selecting the median-across-methods detector without enabling agreement
produces no comparison outputs,
and
the pipeline says so rather than writing nothing silently.

  * collect_agreement_rows      -- long-format per trial x method table
  * build_method_summary        -- per stim type x method statistics
  * write_onset_method_tables   -- both CSVs
  * plot_onset_methods_on_trace -- where each method lands on the waveform
  * plot_method_agreement       -- latency by method, per condition
  * plot_bland_altman           -- each method vs the others' median
  * plot_disagreement_distribution
  * write_onset_method_figures  -- all of them, into a subfolder
  * figures_subdir              -- the subfolder this module writes into

Figure layout
-------------
Figures go into ``figures/<bids_prefix>_onset_methods_figures/`` rather than
directly into ``figures/``. This matches the convention the bundled add-ons
already use (``<bids_prefix>_<name>_figures``), and it matters here because a
file with eight stimulus types produces eleven images -- enough to bury the
trace figures the main pipeline writes alongside them.
"""

FIGURE_SUBDIR_SUFFIX = "onset_methods_figures"

import os

import numpy as np
import pandas as pd

METHODS_MEDIAN_KEY = "methods_median"

CAVEAT = ("Agreement between methods is not accuracy: detectors sharing an "
          "assumption can be wrong together.")

# Long-format table columns.
METHOD_COLS = [
    "File", "StimType", "Stim_Label", "Segment", "Method",
    "Latency(ms)", "Detected", "Delta_From_MethodsMedian(ms)",
    # Consensus recomputed WITHOUT this method. See _loo_methods_median.
    "LOO_MethodsMedian(ms)", "Delta_From_LOO_MethodsMedian(ms)",
]

SUMMARY_COLS = [
    "File", "StimType", "Stim_Label", "Method",
    "N_Trials", "N_Detected", "Detection_Rate",
    "Mean_Latency(ms)", "SD_Latency(ms)", "Median_Latency(ms)",
    "Mean_Delta_From_MethodsMedian(ms)", "SD_Delta_From_MethodsMedian(ms)",
    "LoA_Lower(ms)", "LoA_Upper(ms)",
    # Bland-Altman statistics against the leave-one-out methods_median, which is
    # the comparison that is not part-whole. Prefer these for reporting.
    "Bias_vs_LOO(ms)", "SD_Diff_vs_LOO(ms)",
    "LoA_LOO_Lower(ms)", "LoA_LOO_Upper(ms)",
]


def _loo_methods_median(per_method, exclude):
    """Median of the other members, excluding ``exclude``.

    A Bland-Altman plot of a method against the median of all methods is a
    part-whole comparison: the method being assessed is one of the values that
    median is computed from, so it is correlated with its own reference. That pulls
    the bias toward zero and narrows the limits of agreement, making every
    method look better than it is -- and the fewer members there are, the
    worse the distortion. With five members a method contributes a fifth of
    its own comparator.

    Recomputing the median without the method under test removes that.
    Bland & Altman make the same point about comparing a method with a mean
    that includes it; leave-one-out is the standard remedy.

    Returns None when fewer than two other members detected an onset, since a
    single remaining value is not a median across methods.
    """
    others = [v for m, v in per_method.items()
              if m != exclude and v is not None]
    if len(others) < 2:
        return None
    return float(np.median(others))


def collect_agreement_rows(agreement_by_trial, name, custom_labels=None):
    """Flatten per-trial agreement results into long-format rows.

    Parameters
    ----------
    agreement_by_trial : dict
        ``{(stim_type, trial_index): OnsetAgreement}``, as accumulated by
        ``pipeline_quantify_segments``.
    name : str            source file name, for the File column
    custom_labels : dict  stim_type -> analyst's label

    Returns
    -------
    list of dict, one per trial x method.

    Long format rather than one column per method: the member set is a user
    setting, so a wide table would change shape whenever a member is ticked or
    unticked, breaking any script that reads it. Long format also drops
    straight into ggplot or seaborn without reshaping.
    """
    custom_labels = custom_labels or {}
    rows = []
    for (stim_type, idx), ag in sorted(
            agreement_by_trial.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if ag is None:
            continue
        methods_median = ag.consensus_ms
        # The methods median is a DERIVED value, not a member: compute_onset_agreement
        # returns per_method for the member detectors only. Emitting it as a row
        # of its own is what puts it on the figures -- without this the plots
        # showed every method except the one whose value gets reported, which is
        # the single thing an analyst most wants to locate.
        per_method = dict(ag.per_method)
        per_method[METHODS_MEDIAN_KEY] = methods_median
        for method, latency in sorted(per_method.items()):
            delta = None
            if latency is not None and methods_median is not None:
                delta = round(float(latency) - float(methods_median), 3)
            # For the methods median row itself the leave-one-out reference is the
            # median of all members, i.e. the methods median -- so the difference is
            # zero by construction and carries no information.
            loo = (None if method == METHODS_MEDIAN_KEY
                   else _loo_methods_median(ag.per_method, method))
            loo_delta = None
            if latency is not None and loo is not None:
                loo_delta = round(float(latency) - loo, 3)
            rows.append({
                "File": name,
                "StimType": stim_type,
                "Stim_Label": custom_labels.get(stim_type, ""),
                "Segment": idx + 1,
                "Method": method,
                "Latency(ms)": latency,
                "Detected": bool(latency is not None),
                "Delta_From_MethodsMedian(ms)": delta,
                "LOO_MethodsMedian(ms)": round(loo, 3) if loo is not None else None,
                "Delta_From_LOO_MethodsMedian(ms)": loo_delta,
            })
    return rows


def build_method_summary(rows):
    """Per stim type x method statistics, including limits of agreement.

    Limits of agreement are the Bland-Altman 95% interval of each method's
    difference from the median across methods (mean +/- 1.96 SD). They answer the question
    an analyst actually has -- "if I switched to this method, how far could an
    individual trial move" -- which a mean difference alone does not.
    """
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLS)

    df = pd.DataFrame(rows)
    out = []
    for (fname, stim, label, method), g in df.groupby(
            ["File", "StimType", "Stim_Label", "Method"], sort=False):
        lat = pd.to_numeric(g["Latency(ms)"], errors="coerce").dropna()
        dlt = pd.to_numeric(g["Delta_From_MethodsMedian(ms)"],
                            errors="coerce").dropna()
        loo = pd.to_numeric(g["Delta_From_LOO_MethodsMedian(ms)"],
                            errors="coerce").dropna()
        n_tot = len(g)
        d_mean = float(dlt.mean()) if len(dlt) else None
        d_sd = float(dlt.std(ddof=1)) if len(dlt) > 1 else None
        loa_lo = loa_hi = None
        if d_mean is not None and d_sd is not None:
            loa_lo = round(d_mean - 1.96 * d_sd, 3)
            loa_hi = round(d_mean + 1.96 * d_sd, 3)
        out.append({
            "File": fname, "StimType": stim, "Stim_Label": label,
            "Method": method,
            "N_Trials": n_tot,
            "N_Detected": int(len(lat)),
            "Detection_Rate": round(len(lat) / n_tot, 3) if n_tot else None,
            "Mean_Latency(ms)": round(float(lat.mean()), 3) if len(lat) else None,
            "SD_Latency(ms)": round(float(lat.std(ddof=1)), 3) if len(lat) > 1 else None,
            "Median_Latency(ms)": round(float(lat.median()), 3) if len(lat) else None,
            "Mean_Delta_From_MethodsMedian(ms)": round(d_mean, 3) if d_mean is not None else None,
            "SD_Delta_From_MethodsMedian(ms)": round(d_sd, 3) if d_sd is not None else None,
            "LoA_Lower(ms)": loa_lo,
            "LoA_Upper(ms)": loa_hi,
            "Bias_vs_LOO(ms)": round(float(loo.mean()), 3) if len(loo) else None,
            "SD_Diff_vs_LOO(ms)": (round(float(loo.std(ddof=1)), 3)
                                   if len(loo) > 1 else None),
            "LoA_LOO_Lower(ms)": (round(float(loo.mean() - 1.96 * loo.std(ddof=1)), 3)
                                  if len(loo) > 1 else None),
            "LoA_LOO_Upper(ms)": (round(float(loo.mean() + 1.96 * loo.std(ddof=1)), 3)
                                  if len(loo) > 1 else None),
        })
    return pd.DataFrame(out, columns=SUMMARY_COLS)


def write_onset_method_tables(rows, results_out, bids_prefix):
    """Write the long-format and summary CSVs. Returns the paths written."""
    if not rows:
        return []
    os.makedirs(results_out, exist_ok=True)
    written = []

    from .results_layout import result_path
    long_path = result_path(results_out,
                            f"{bids_prefix}_onset_methods.csv")
    pd.DataFrame(rows, columns=METHOD_COLS).to_csv(long_path, index=False)
    written.append(long_path)

    summ = build_method_summary(rows)
    if len(summ):
        summ_path = result_path(
            results_out, f"{bids_prefix}_onset_method_summary.csv")
        summ.to_csv(summ_path, index=False)
        written.append(summ_path)
    return written


# ── Figures ───────────────────────────────────────────────────────────────────
# Figures are built with matplotlib.figure.Figure + FigureCanvasAgg rather than
# pyplot. run_pipeline executes on a worker thread, and touching the global
# pyplot state from there triggers Tcl async-handler cleanup from the wrong
# thread -- the hard "Tcl_AsyncDelete" crash on Windows.

def _new_figure(figsize):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _method_colours(methods):
    # matplotlib.cm.get_cmap is deprecated from 3.7 and removed in 3.11, but
    # matplotlib.colormaps only exists from 3.6. pyproject requires >=3.7, so
    # prefer the new API and fall back for the older supported versions.
    try:
        import matplotlib
        cmap = matplotlib.colormaps["tab10"]
    except (AttributeError, KeyError, ImportError):
        from matplotlib import cm
        cmap = cm.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(methods)}


def figures_subdir(figures_out, bids_prefix, create=False):
    """Path of the subfolder this module writes its figures into.

    Mirrors the add-on convention, ``figures/<bids_prefix>_<name>_figures``,
    so the comparison images sit together instead of interleaving with the
    pipeline's own trace figures.
    """
    path = os.path.join(figures_out, f"{bids_prefix}_{FIGURE_SUBDIR_SUFFIX}")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _display(method):
    """Readable name for a method key, for figure text only.

    Tables keep the key: it is what a script matches on, and renaming it would
    break anyone's analysis. Figures are read by people, so they get the label.
    Unknown keys fall through unchanged rather than raising, so a method added
    without a short label still plots.
    """
    try:
        from .detection import ONSET_METHOD_SHORT_LABELS
        return ONSET_METHOD_SHORT_LABELS.get(method, method)
    except Exception:
        return method


def _safe(stim_type):
    # The shared rule. The old replace-with-underscore mapped "=", ">" and "?"
    # all to "_", so one stimulus type's figure overwrote another's.
    from .utils import filename_token
    return filename_token(stim_type)


def plot_onset_methods_on_trace(stim_type, segs, fs, pre_ms, rows,
                                figures_out, bids_prefix, unit="mV",
                                label="", selected_method=None):
    """Where each method's onset lands on this condition's waveform.

    The median waveform is drawn rather than the mean: a single trial with a
    late or absent response drags a mean and would move the very landmarks the
    figure exists to locate.

    Per-trial onsets appear as a strip beneath the trace, because two methods
    can share a median while disagreeing wildly trial to trial -- which is the
    case worth seeing and the one a summary statistic hides.
    """
    segs = np.asarray(segs, dtype=float)
    if segs.ndim != 2 or segs.shape[0] == 0 or not rows:
        return None

    sub = [r for r in rows if r["StimType"] == stim_type]
    if not sub:
        return None
    methods = sorted({r["Method"] for r in sub})
    colours = _method_colours(methods)

    n = segs.shape[1]
    t = np.linspace(-float(pre_ms), (n / float(fs)) * 1000.0 - float(pre_ms),
                    n, endpoint=False)
    median_wave = np.median(segs, axis=0)

    # The strip's y-tick labels are method names plus a detection count, and
    # the longest ("Median across methods  (20/20)") overflowed a hard-coded
    # left margin and was clipped. Size the margin from the labels that will
    # actually be drawn: a fixed value is a guess that goes stale the moment a
    # method with a longer name is added.
    fig_w = 11.0
    label_chars = max(len(_display(m)) for m in methods) + len("  (00/00)")
    # ~4.7 pt per character at 8 pt in the default sans face, plus tick padding.
    left_margin = min(0.34, 0.045 + (label_chars * 4.7) / (fig_w * 72.0))

    fig = _new_figure((fig_w, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.5], hspace=0.30,
                          left=left_margin, right=0.97, top=0.88, bottom=0.09)
    ax = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(t, median_wave, color="0.25", lw=1.4, zorder=3,
            label="median waveform")
    lo = np.percentile(segs, 25, axis=0)
    hi = np.percentile(segs, 75, axis=0)
    ax.fill_between(t, lo, hi, color="0.75", alpha=0.45, zorder=1,
                    label="IQR across trials")

    # Vertical lines carry their value in the legend rather than as rotated
    # text on the axes. With five or more methods landing within a few ms of
    # each other -- which is the normal case -- rotated annotations overlap
    # into an unreadable stack exactly where the figure is most informative.
    handles = []
    for m in methods:
        vals = [r["Latency(ms)"] for r in sub
                if r["Method"] == m and r["Latency(ms)"] is not None]
        if not vals:
            continue
        med = float(np.median(vals))
        is_consensus = (m == METHODS_MEDIAN_KEY)
        is_selected = (selected_method is not None and m == selected_method)
        # The value that ends up in Latency(ms) is drawn solid and heavy; the
        # rest are dashed references. Without this the figure shows five equally
        # weighted candidates and no indication of which one was reported.
        tag = ""
        if is_selected:
            tag = "  ← reported"
        elif is_consensus:
            tag = "  (median)"
        ln = ax.axvline(
            med,
            ls="-" if (is_consensus or is_selected) else "--",
            color="black" if (is_consensus or is_selected) else colours[m],
            lw=2.4 if (is_consensus or is_selected) else 1.4,
            alpha=1.0 if (is_consensus or is_selected) else 0.85,
            zorder=5 if (is_consensus or is_selected) else 4,
            label=f"{_display(m)} \u2014 {med:.1f} ms{tag}")
        handles.append(ln)

    # Detection counts belong on the tick labels, not as text anchored to the
    # axis limits: limits are not final until the artists are drawn, so text
    # placed at get_xlim()[0] lands outside the axes.
    ytick_labels = []
    for i, m in enumerate(methods):
        vals = [r["Latency(ms)"] for r in sub
                if r["Method"] == m and r["Latency(ms)"] is not None]
        n_tot = len({r["Segment"] for r in sub if r["Method"] == m})
        ax_strip.scatter(vals, np.full(len(vals), i),
                         s=18, color=colours[m], alpha=0.75,
                         edgecolors="none")
        ytick_labels.append(f"{_display(m)}  ({len(vals)}/{n_tot})")

    ax_strip.set_yticks(range(len(methods)))
    ax_strip.set_yticklabels(ytick_labels, fontsize=8)
    ax_strip.set_ylim(-0.6, len(methods) - 0.4)
    ax_strip.set_xlabel("Time re: stimulus (ms)")
    ax_strip.set_title("Per-trial onsets (detected / total)", fontsize=9,
                       loc="left", color="0.3")
    ax_strip.grid(axis="x", alpha=0.25)
    ax.set_ylabel(f"Amplitude ({unit})")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
              title="median onset", title_fontsize=8, ncol=2)

    title = f"Onset methods — {stim_type}"
    if label:
        title += f" ({label})"
    fig.suptitle(title, y=0.975, fontsize=12)
    fig.text(0.5, 0.935, CAVEAT, ha="center", fontsize=8, color="0.35")

    # Focus on the response, not the whole post-stimulus window.
    all_on = [r["Latency(ms)"] for r in sub if r["Latency(ms)"] is not None]
    if all_on:
        ax.set_xlim(min(all_on) - 15, max(all_on) + 45)

    path = os.path.join(figures_subdir(figures_out, bids_prefix, create=True),
                        f"{bids_prefix}_stim-{_safe(stim_type)}_onset_methods.png")
    fig.savefig(path, dpi=150)
    return path


def plot_method_agreement(rows, figures_out, bids_prefix):
    """Latency by method, one panel per stimulus type."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    stims = list(dict.fromkeys(df["StimType"]))
    methods = sorted(df["Method"].unique())
    colours = _method_colours(methods)

    ncol = min(3, len(stims))
    nrow = int(np.ceil(len(stims) / ncol))
    fig = _new_figure((4.6 * ncol, 3.6 * nrow + 0.6))
    rng = np.random.default_rng(0)   # jitter must be reproducible

    for k, stim in enumerate(stims):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        g = df[df["StimType"] == stim]
        for i, m in enumerate(methods):
            vals = pd.to_numeric(g[g["Method"] == m]["Latency(ms)"],
                                 errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            ax.scatter(np.full(len(vals), i) + rng.uniform(-.13, .13, len(vals)),
                       vals, s=14, alpha=0.65, color=colours[m],
                       edgecolors="none")
            ax.hlines(np.median(vals), i - .28, i + .28,
                      color="black", lw=1.6, zorder=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_display(m) for m in methods],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(str(stim), fontsize=10)
        if k % ncol == 0:
            ax.set_ylabel("Onset latency (ms)")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Onset latency by detection method", fontsize=12)
    fig.text(0.5, 0.005, CAVEAT + "  Black bars are medians.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = os.path.join(figures_subdir(figures_out, bids_prefix, create=True),
                        f"{bids_prefix}_onset_method_agreement.png")
    fig.savefig(path, dpi=150)
    return path


def plot_disagreement_distribution(agreement_by_trial, figures_out,
                                   bids_prefix):
    """Spread between methods, for choosing a manual-review cutoff.

    Plotted per stimulus type as well as pooled: spread depends on response
    amplitude and baseline quality, both of which vary by condition, so a
    single pooled cutoff can be wrong for every condition in the file.
    """
    recs = [(st, ag.spread_ms) for (st, _i), ag in agreement_by_trial.items()
            if ag is not None and ag.spread_ms is not None]
    if not recs:
        return None
    df = pd.DataFrame(recs, columns=["StimType", "Spread"])

    fig = _new_figure((10, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(df["Spread"], bins=24, color="#4C72B0", alpha=0.85)
    med = float(df["Spread"].median())
    ax1.axvline(med, color="black", ls="--", lw=1.4)
    ax1.annotate(f"median {med:.1f} ms", xy=(med, ax1.get_ylim()[1] * 0.92),
                 xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax1.set_xlabel("Onset_Disagreement (ms)")
    ax1.set_ylabel("Trials")
    ax1.set_title("All trials", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    stims = list(dict.fromkeys(df["StimType"]))
    rng = np.random.default_rng(0)
    for i, st in enumerate(stims):
        v = df[df["StimType"] == st]["Spread"].values
        ax2.scatter(np.full(len(v), i) + rng.uniform(-.13, .13, len(v)),
                    v, s=14, alpha=0.65, color="#4C72B0", edgecolors="none")
        if len(v):
            ax2.hlines(np.median(v), i - .28, i + .28, color="black", lw=1.6)
    ax2.set_xticks(range(len(stims)))
    ax2.set_xticklabels([str(s) for s in stims], fontsize=8)
    ax2.set_ylabel("Onset_Disagreement (ms)")
    ax2.set_title("By stimulus type", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Disagreement between onset methods", fontsize=12)
    fig.text(0.5, 0.005,
             "High-disagreement trials are the ones worth reviewing by hand. "
             "Spread depends on amplitude and baseline quality, so a cutoff "
             "chosen on one condition may not suit another.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = os.path.join(figures_subdir(figures_out, bids_prefix, create=True),
                        f"{bids_prefix}_onset_disagreement.png")
    fig.savefig(path, dpi=150)
    return path


def plot_bland_altman(rows, figures_out, bids_prefix, use_loo=True):
    """Bland-Altman of each method against the median of the others.

    One panel per method. Horizontal axis is the mean of the two estimates,
    vertical axis their difference, with the bias and the 95% limits of
    agreement drawn across. Points are coloured by stimulus type, because a
    method can be unbiased overall while being systematically early on one
    condition and late on another -- a pooled bias of zero would hide exactly
    the pattern worth acting on.

    The reference is the LEAVE-ONE-OUT median by default: the method under
    test is excluded from the median it is compared against. Comparing a method
    with a composite that contains it is part-whole, which drags the bias
    toward zero and narrows the limits, flattering every method. Pass
    ``use_loo=False`` to reproduce the naive version for comparison.

    What the plot cannot show
    ------------------------
    Neither reference is truth. A method that agrees closely with the others is
    typical, not correct; if the members share a systematic error the bias will
    read as zero. Interpret the SPREAD (limits of agreement) as the practical
    consequence of a method choice, and treat the bias as a description of
    where a method sits relative to its peers, not as an error.
    """
    if not rows:
        return None
    df = pd.DataFrame(rows)
    diff_col = ("Delta_From_LOO_MethodsMedian(ms)" if use_loo
                else "Delta_From_MethodsMedian(ms)")
    ref_col = "LOO_MethodsMedian(ms)" if use_loo else None
    if diff_col not in df.columns:
        return None

    df = df[pd.to_numeric(df[diff_col], errors="coerce").notna()].copy()
    df[diff_col] = pd.to_numeric(df[diff_col], errors="coerce")
    df["Latency(ms)"] = pd.to_numeric(df["Latency(ms)"], errors="coerce")
    if ref_col:
        df[ref_col] = pd.to_numeric(df[ref_col], errors="coerce")
        df["_mean"] = (df["Latency(ms)"] + df[ref_col]) / 2.0
    else:
        df["_mean"] = df["Latency(ms)"] - df[diff_col] / 2.0
    df = df[df["_mean"].notna()]
    if df.empty:
        return None

    methods = [m for m in sorted(df["Method"].unique()) if m != "methods_median"]
    if not methods:
        return None
    stims = list(dict.fromkeys(df["StimType"]))
    stim_colours = _method_colours(stims)

    ncol = min(3, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    fig = _new_figure((4.9 * ncol, 3.9 * nrow + 0.9))

    for k, m in enumerate(methods):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        g = df[df["Method"] == m]
        d = g[diff_col].values
        bias = float(np.mean(d))
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

        for st in stims:
            gs_ = g[g["StimType"] == st]
            if len(gs_):
                ax.scatter(gs_["_mean"], gs_[diff_col], s=18, alpha=0.7,
                           color=stim_colours[st], edgecolors="none",
                           label=str(st) if k == 0 else None)

        ax.axhline(0, color="0.6", lw=1.0, ls=":")
        ax.axhline(bias, color="black", lw=1.6)
        ax.axhline(lo, color="black", lw=1.1, ls="--")
        ax.axhline(hi, color="black", lw=1.1, ls="--")
        xr = ax.get_xlim()[1]
        for y, lab in ((bias, f"bias {bias:+.2f}"),
                       (hi, f"+1.96 SD {hi:+.2f}"),
                       (lo, f"-1.96 SD {lo:+.2f}")):
            ax.annotate(lab, xy=(xr, y), xytext=(-3, 2),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=7, color="0.25")
        ax.set_title(f"{_display(m)}   (n={len(d)})", fontsize=10)
        ax.set_xlabel("Mean of method and reference (ms)", fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel("Method − reference (ms)", fontsize=9)
        ax.grid(alpha=0.22)

    # Prose, not an identifier. An earlier repair pass converted this back into
    # the registry key and it appeared as "methods_median" in the figure title.
    ref_name = ("leave-one-out median of the other methods" if use_loo
                else "median across all methods")
    fig.suptitle(f"Bland-Altman: each method vs the {ref_name}", fontsize=12)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.045),
                   title="stimulus type", title_fontsize=8)
    fig.text(0.5, 0.008,
             "Reference excludes the method under test, so the comparison is "
             "not part-whole. " + CAVEAT,
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    suffix = "" if use_loo else "_vs_consensus"
    path = os.path.join(figures_subdir(figures_out, bids_prefix, create=True),
                        f"{bids_prefix}_onset_bland_altman{suffix}.png")
    fig.savefig(path, dpi=150)
    return path


def write_onset_method_figures(rows, agreement_by_trial, segs_by_type, fs,
                               pre_ms, figures_out, bids_prefix, unit="mV",
                               custom_labels=None, selected_method=None,
                               log_callback=print):
    """Produce all comparison figures. Returns the paths written."""
    if not rows:
        return []
    custom_labels = custom_labels or {}
    figures_subdir(figures_out, bids_prefix, create=True)
    written = []

    for stim_type, segs in (segs_by_type or {}).items():
        try:
            p = plot_onset_methods_on_trace(
                stim_type, segs, fs, pre_ms, rows, figures_out, bids_prefix,
                unit=unit, label=custom_labels.get(stim_type, ""),
                selected_method=selected_method)
            if p:
                written.append(p)
        except Exception as exc:
            log_callback(f"   ⚠️ Onset-method figure for '{stim_type}' failed "
                         f"({type(exc).__name__}: {exc})")

    for fn, args in ((plot_method_agreement, (rows, figures_out, bids_prefix)),
                     (plot_bland_altman, (rows, figures_out, bids_prefix)),
                     (plot_disagreement_distribution,
                      (agreement_by_trial, figures_out, bids_prefix))):
        try:
            p = fn(*args)
            if p:
                written.append(p)
        except Exception as exc:
            log_callback(f"   ⚠️ {fn.__name__} failed "
                         f"({type(exc).__name__}: {exc})")
    return written
