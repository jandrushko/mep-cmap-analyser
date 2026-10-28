"""
Group variability and reliability — built-in second-level MEP-CMAP add-on
=========================================================================

Runs on the merged group table (`group_level_LME_ready.csv`) and answers the
questions a single recording structurally cannot: how much of the variance in a
measure is real between-participant signal, how much is session-to-session
instability, how much is trial-to-trial noise, and therefore how many stimuli
and how many visits a protocol actually needs.

Sections produced, per stimulus type
------------------------------------
1. Amplitude and within-recording CV across the dataset
2. Variance components — between target, between session, trial to trial
3. Reliability, SEM and MDC95 of a measurement averaged over k trials and m
   sessions (a generalisability D study)
4. Classical ICC(1,1), ICC(2,1), ICC(3,1) and their k-forms on recording means
5. Session-to-session agreement, explicitly labelled as reliability or as change

How design factors are used
---------------------------
Factors are read from `study_design.json` (the Second Level design) and each is
classified by whether it varies within a participant:

  * BETWEEN-target factors (group, arm, sex) are safe to stratify on. Every
    participant keeps all of their sessions inside a stratum, so between-session
    reliability survives and you get it separately per group.

  * WITHIN-target factors (timepoint, pre/post) are the repeated-measures axis
    itself. Stratifying on one would leave a single session per participant and
    destroy the very comparison it describes, so it is used instead to LABEL the
    session axis and to decide whether a session pair measures reliability or
    measures change.

That distinction matters for interpretation. Two sessions either side of an
intervention differ by measurement error PLUS whatever the intervention did, so
their agreement is not test-retest reliability and will look poor even when the
measurement is excellent. This add-on says which of the two it is rather than
letting the reader assume.

Outputs (new files only; the group table is never modified)
-----------------------------------------------------------
  <prefix>_variability_group.csv             one row per stimulus type x stratum
  <prefix>_variability_group_reliability.csv the D-study table, long format
  <prefix>_variability_group_recordings.csv  per-recording means and CVs
  <prefix>_variability_group_report.txt      the readable narrative
  figures/<prefix>_variability_group_figures/*.png

Method references: Searle et al. (1992) unbalanced variance components; Shrout &
Fleiss (1979) and McGraw & Wong (1996) ICC; Bland & Altman (1999); Brennan
(2001) generalisability theory.
"""

import os
import json
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

ADDON_NAME        = "variability_group"
ADDON_DESCRIPTION = ("Dataset-level reliability: variance components, how many trials "
                     "and sessions are needed, ICC, and session-to-session agreement, "
                     "optionally split by your between-participant design factors")
ADDON_VERSION     = "1.0.0"
ADDON_AUTHOR      = "MEP-CMAP Analyser (built-in)"
ADDON_SCOPE       = "group_level"

ADDON_SETTINGS = [
    {
        "key": "variability_metric",
        "label": "Measure to analyse",
        "help": ("Column of the group table to quantify. Defaults to peak-to-peak "
                 "amplitude; 'AUC(mV*s)', 'Normalised_PTP' and 'cSP_Duration(ms)' "
                 "are other common choices. Columns that were never populated in "
                 "this dataset are reported as having no usable trials."),
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
        "choices_from": "group_columns",
    },
    {
        "key": "variability_icc_target",
        "label": "Reliability target",
        "help": ("What counts as one entity being measured. 'subject_limb' treats "
                 "left and right as separate targets, which is usually right since "
                 "two limbs of one person are two different corticospinal pathways. "
                 "'subject' pools them."),
        "type": "str",
        "default": "subject_limb",
        "choices": ["subject_limb", "subject"],
    },
    {
        "key": "variability_split_by",
        "label": "Split results by",
        "help": ("'auto' reports the whole dataset plus one breakdown per "
                 "between-participant design factor, each computed separately "
                 "rather than crossed. 'none' reports the whole dataset only. Or "
                 "name factors directly, comma separated, e.g. 'Group'. "
                 "Within-participant factors such as Timepoint are never split on; "
                 "they label the session axis instead."),
        "type": "str",
        "default": "auto",
    },
    {
        "key": "variability_exclude_flagged",
        "label": "Exclude trials flagged as outliers",
        "help": ("Leave trials marked in Outlier_Decision out of the estimates. The "
                 "group table keeps outliers by design so the analyst can decide, "
                 "and variance components are sensitive to them."),
        "type": "bool",
        "default": True,
    },
    {
        "key": "variability_figures",
        "label": "Write figures",
        "help": "One summary figure per stimulus type, into the figures folder.",
        "type": "bool",
        "default": True,
    },
]

# A stratum with fewer targets than this is reported but not decomposed: the
# between-target component would be noise dressed up as an estimate.
MIN_TARGETS = 3

TRIALS_GRID = (5, 10, 15, 20, 30)
SESSIONS_GRID = (1, 2)

_FLAG_WORDS = ("flagged", "excluded", "exclude", "reject", "rejected", "outlier")
_POOLED = "(all)"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cfg(config, key, default, cast=None):
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
    """'Not flagged' contains the word 'flagged', so match the whole string."""
    return series.astype(str).str.strip().str.lower().isin(_FLAG_WORDS)


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(f):
        return "n/a"
    return ("%." + str(nd) + "f") % f


def _read_design_factors(results_dir, log):
    """Design factor names from the Second Level study_design.json, if present."""
    path = os.path.join(results_dir, "study_design.json")
    if not os.path.exists(path):
        log(f"{ADDON_NAME}: no study_design.json beside the group table — "
            f"design factors will be inferred from the table's own columns.")
        return []
    try:
        with open(path, encoding="utf-8") as f:
            design = json.load(f)
        return [str(c) for c in (design.get("group_columns") or [])]
    except Exception as ex:
        log(f"{ADDON_NAME}: could not read study_design.json ({ex}).")
        return []


# Identifier and bookkeeping columns, and the prefixes of measurement blocks the
# pipeline and add-ons write. None of these is a study-design factor.
_NOT_A_FACTOR = {
    "File", "participant_id", "session", "task", "StimType", "Stim_Label",
    "Stim_Role", "Segment", "Segment_Overall", "Trial", "Outlier_Decision",
    "Limb", "Measure", "Manual_Note", "_target", "Clipped", "Units_Assumed",
}
_METRIC_PREFIXES = ("EMGComp_", "Var_", "Reference_", "Normalised_", "Z_",
                    "PTP_", "cSP_", "Adjusted_", "MEP_", "PreStim_")


def _infer_design_factors(df, candidates):
    """Guess which columns are study-design factors, when no design file exists.

    The group loader labels every non-numeric column a design column, which
    sweeps up measurements that merely contain a sentinel string: `Latency(ms)`
    holds 'Not Detected', `cSP_Duration(ms)` holds 'Not Marked'. Treating those
    as factors produced nonsense such as latency being reported as a
    repeated-measures axis.

    A real factor is categorical, has a handful of levels, and is not a
    measurement. Columns that are mostly numeric once coerced, carry a unit in
    their name, or belong to a known measurement block are rejected.

    This is a fallback. When study_design.json exists its factor list is used
    verbatim and none of this guessing happens.
    """
    out = []
    for c in candidates:
        if c in _NOT_A_FACTOR or c.startswith(_METRIC_PREFIXES):
            continue
        if "(" in c and ")" in c:          # a unit in the name means a measurement
            continue
        if c not in df.columns:
            continue
        col = df[c]
        # Mostly coercible to numbers means it is a measurement carrying
        # sentinel strings, not a categorical factor.
        if pd.to_numeric(col, errors="coerce").notna().mean() > 0.5:
            continue
        n_levels = col.dropna().nunique()
        if not (2 <= n_levels <= 20):      # constants split nothing; IDs are not factors
            continue
        out.append(c)
    return out


def _resolve_metric(df, requested, log):
    if requested in df.columns:
        return requested
    fallback = "PTP(mV)"
    if fallback in df.columns:
        log(f"{ADDON_NAME}: column '{requested}' not in the group table — "
            f"falling back to '{fallback}'.")
        return fallback
    numeric = [c for c in df.columns
               if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    wanted = (f"'{requested}' is not in the group table" if requested == fallback
              else f"neither '{requested}' nor '{fallback}' is in the group table")
    raise ValueError(f"{ADDON_NAME}: {wanted}. Numeric-looking columns are: "
                     + ", ".join(numeric[:15]))


def _build_records(df, metric, target_col, session_col, file_col):
    """One record per recording: one participant, one session, one limb.

    Participant and limb are carried separately from `target`, because the
    target may be either of them depending on the reliability-target setting,
    and a reader needs to see which recording a row refers to regardless.
    """
    records = []
    keys = [c for c in (target_col, session_col, file_col) if c in df.columns]
    for key, sub in df.groupby(keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rec = {"target": str(key[0]),
               "session": str(key[1]) if len(key) > 1 else "ses-1",
               "x": vals}
        rec["file"] = str(key[2]) if len(key) > 2 else rec["target"]

        def _first(col):
            if col in sub.columns:
                vv = sub[col].dropna()
                if len(vv):
                    return str(vv.iloc[0])
            return ""
        rec["subject"] = _first("participant_id")
        rec["limb"] = _first("Limb")
        records.append(rec)
    return records


def _recording_label(rec):
    """Compact identifier for one recording: subject, session, limb.

    Kept short because it becomes a y-axis tick. Empty parts are dropped rather
    than printed blank, so a dataset without a Limb column reads cleanly.
    """
    parts = [rec.get("subject") or rec.get("target", ""),
             rec.get("session", "")]
    limb = str(rec.get("limb") or "")
    if limb:
        parts.append(limb.replace("limb-", ""))
    return "  ".join(p for p in parts if p)


def _recording_table(records):
    """Per-recording mean, SD and CV, for the spread-across-recordings section."""
    rows = []
    for r in records:
        x = r["x"]
        m = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
        rows.append({
            "Label": _recording_label(r),
            "Subject": r.get("subject", ""), "Session": r["session"],
            "Limb": r.get("limb", ""), "Target": r["target"],
            "File": r.get("file"),
            "N_Trials": int(x.size), "Mean": m, "SD": sd,
            "CV(%)": 100.0 * sd / m if m else np.nan,
        })
    return pd.DataFrame(rows)


def _target_by_session_matrix(records):
    """Targets x sessions matrix of recording means, complete cases only."""
    rows = {}
    for r in records:
        rows.setdefault(r["target"], {}).setdefault(r["session"], []).append(
            float(np.mean(r["x"])))
    flat = {t: {s: float(np.mean(v)) for s, v in d.items()} for t, d in rows.items()}
    mat = pd.DataFrame(flat).T
    if mat.empty:
        return mat
    return mat.reindex(sorted(mat.columns), axis=1).dropna(axis=0, how="any")


# ─────────────────────────────────────────────────────────────────────────────
# One stratum
# ─────────────────────────────────────────────────────────────────────────────
def _analyse_stratum(records, df_stratum, within_factor, session_col):
    """Variance components, reliability, ICC and session agreement for one slice."""
    out = {"n_recordings": len(records),
           "n_targets": len(set(r["target"] for r in records)),
           "n_sessions": len(set(r["session"] for r in records))}

    rec_tbl = _recording_table(records)
    out["recordings"] = rec_tbl
    if len(rec_tbl):
        out["mean_of_means"] = float(rec_tbl["Mean"].mean())
        out["sd_of_means"] = float(rec_tbl["Mean"].std(ddof=1)) if len(rec_tbl) > 1 else np.nan
        out["cv_median"] = float(rec_tbl["CV(%)"].median())
        out["cv_iqr_lo"] = float(rec_tbl["CV(%)"].quantile(0.25))
        out["cv_iqr_hi"] = float(rec_tbl["CV(%)"].quantile(0.75))

    # Do recordings differ in how noisy they are, and is serial structure a
    # property of the paradigm rather than of one session?
    arrays = [r["x"] for r in records]
    out["spread"] = V.spread_across_recordings(arrays)

    r1s, drifts, tes, ttas, r1_ns = [], [], [], [], []
    for r in records:
        x = r["x"]
        if x.size >= 6:
            ar = V.ar_analysis(x)
            if "error" not in ar:
                r1s.append(ar["r1"])
                r1_ns.append(x.size)
                drifts.append(ar["trend_pct_per_10_trials"])
        te = V.typical_error(x)
        if np.isfinite(te) and np.mean(x):
            tes.append(100.0 * te / float(np.mean(x)))
        if x.size >= 4:
            ttas.append(V.trials_to_average_table(x, n_rep=400))
    out["serial"] = V.serial_structure_across_recordings(r1s, drifts,
                                                         n_trials=r1_ns)
    if tes:
        out["typical_error_pct_median"] = float(np.median(tes))
        out["typical_error_pct_q25"] = float(np.percentile(tes, 25))
        out["typical_error_pct_q75"] = float(np.percentile(tes, 75))
    out["pooled_trials"] = V.pooled_trials_to_average(ttas)

    # Does noise scale with amplitude, and therefore should the analysis be
    # done on the log scale? Measured rather than assumed.
    out["scaling"] = V.mean_variance_scaling(records)

    # Does the choice of dispersion metric change which recordings look most
    # variable, or only rescale them?
    labels = [_recording_label(r) for r in records]
    out["metric_table"], out["metric_pairs"] = V.compare_dispersion_metrics(
        records, labels=labels)

    if out["n_targets"] < MIN_TARGETS:
        out["skipped"] = (f"only {out['n_targets']} target(s); at least "
                          f"{MIN_TARGETS} are needed to separate between-target "
                          f"variance from noise")
        return out

    vc = V.variance_components(records)
    out["vc"] = vc
    if "error" not in vc:
        out["reliability"] = V.reliability_table(vc, trials=TRIALS_GRID,
                                                 sessions=SESSIONS_GRID)
        for tgt in (0.75, 0.90):
            out[f"trials_for_{int(tgt * 100)}"] = V.trials_for_reliability(vc, tgt, 1)
        # Ceiling with one session: no number of trials can beat this.
        sa, sb = vc.get("var_target", np.nan), vc.get("var_session", 0.0) or 0.0
        out["ceiling_1_session"] = sa / (sa + sb) if (sa + sb) > 0 else np.nan

    mat = _target_by_session_matrix(records)
    out["matrix"] = mat
    if mat.shape[0] >= 2 and mat.shape[1] >= 2:
        out["icc"] = V.icc_from_matrix(mat.values)
        s_a, s_b = mat.columns[0], mat.columns[1]
        a, b = mat[s_a].to_numpy(), mat[s_b].to_numpy()
        out["session_pair"] = (str(s_a), str(s_b))
        out["ba"] = V.bland_altman(a, b)
        out["ba_ratio"] = V.bland_altman_ratio(a, b)
        out["pair_role"] = V.session_pair_role(
            df_stratum, s_a, s_b, within_factor=within_factor,
            session_col=session_col)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def _plot_stratum(res, metric, stim_type, out_png):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Group variability: {stim_type} — {metric}", fontweight="bold")

    ax = axes[0, 0]
    rt = res.get("recordings")
    if rt is not None and len(rt):
        s = rt.sort_values("Mean").reset_index(drop=True)
        y = np.arange(len(s))
        err = 1.96 * s["SD"] / np.sqrt(s["N_Trials"].clip(lower=1))
        ax.errorbar(s["Mean"], y, xerr=err, fmt="o", color="#2C3E50",
                    ecolor="#5DADE2", ms=4, elinewidth=1.5)
        ax.axvline(s["Mean"].mean(), color="#C0392B", ls="--", lw=1)
        # Name the rows while they are still legible. Past roughly 30 recordings
        # the ticks collide and an unlabelled axis is the honest choice; the
        # recordings CSV carries the same labels for looking one up.
        labels = list(s["Label"]) if "Label" in s.columns else []
        if labels and len(labels) <= 30:
            ax.set_yticks(y)
            ax.set_yticklabels(labels,
                               fontsize=(8 if len(labels) <= 14 else 6))
            ax.set_ylabel("")
        else:
            ax.set_yticks([])
            ax.set_ylabel(f"Each row is one recording, sorted by mean "
                          f"({len(s)} recordings)")
        ax.set_ylim(-0.7, len(s) - 0.3)
        ax.set_xlabel(f"Mean {metric} for one recording, with 95% CI")
    ax.set_title("Recording means", fontsize=10)

    ax = axes[0, 1]
    vc = res.get("vc")
    if vc and "error" not in vc:
        labels, vals, cols = [], [], []
        for key, lab, col in (("pct_target", "Between\ntargets", "#2C3E50"),
                              ("pct_session", "Between\nsessions", "#5DADE2"),
                              ("pct_trial", "Trial to\ntrial", "#C0392B")):
            v = vc.get(key)
            if v is not None and np.isfinite(v):
                labels.append(lab); vals.append(v); cols.append(col)
        bars = ax.bar(labels, vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                    ha="center", fontsize=9)
        ax.set_ylabel("Share of total variance (%)")
        ax.set_xlabel("Level of the design")
        ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
    ax.set_title("Where the variance sits", fontsize=10)

    ax = axes[1, 0]
    rel = res.get("reliability")
    if rel is not None and len(rel):
        for m, style, col in ((1, "-", "#2C3E50"), (2, "--", "#5DADE2")):
            sub = rel[rel["sessions"] == m]
            if len(sub):
                ax.plot(sub["trials_per_session"], sub["reliability"], style,
                        marker="o", ms=4, color=col, label=f"{m} session(s)")
        ax.axhline(0.90, ls=":", color="#C0392B", lw=1)
        ax.axhline(0.75, ls=":", color="#95A5A6", lw=1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    ax.set_xlabel("Trials averaged per session (k)")
    ax.set_ylabel("Reliability of the averaged measurement")
    ax.set_title("Reliability of an averaged measurement", fontsize=10)

    ax = axes[1, 1]
    ba = res.get("ba")
    if ba and "error" not in ba:
        ax.scatter(ba["avg"], ba["diff"], color="#5DADE2", edgecolor="#2C3E50", s=40)
        ax.axhline(ba["bias"], color="#2C3E50", lw=1.4)
        ax.axhline(ba["loa_lo"], color="#C0392B", ls="--", lw=1.2)
        ax.axhline(ba["loa_hi"], color="#C0392B", ls="--", lw=1.2)
        ax.axhline(0, color="k", lw=0.7)
        sp = res.get("session_pair", ("", ""))
        role = (res.get("pair_role") or {}).get("role", "unknown")
        ax.set_xlabel(f"Mean of the two sessions ({metric})")
        ax.set_ylabel("Difference between sessions")
        ax.set_title(f"{sp[0]} vs {sp[1]} — measures {role}", fontsize=10)
    else:
        ax.text(0.5, 0.5, "needs two sessions", ha="center", va="center",
                transform=ax.transAxes, color="#95A5A6")
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)




def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def _caption_stratum(res, metric, stim_type, prefix):
    """Caption filled with this dataset's own numbers, written beside the figure."""
    L = []
    add = L.append
    vc = res.get("vc") or {}
    icc = res.get("icc") or {}
    ba = res.get("ba") or {}
    role = res.get("pair_role") or {}

    add(f"FIGURE: group variability and reliability, condition '{stim_type}', {metric}")
    add(f"Dataset: {prefix}")
    add(f"{res.get('n_recordings')} recordings from {res.get('n_targets')} targets "
        f"across {res.get('n_sessions')} session(s)")
    add("=" * 72)
    add("")
    add("WHAT EACH PANEL SHOWS")
    add("")
    add("Top left - Recording means.")
    add("  One row per recording, sorted by mean amplitude, with a 95 % interval")
    add("  showing how precisely that recording's own mean is pinned down. The")
    add("  dashed line is the dataset mean. Wide scatter between rows is what")
    add("  makes a measure able to tell participants apart; wide intervals")
    add("  within a row mean individual recordings are noisy.")
    add("")
    add("Top right - Where the variance sits.")
    add("  Total variance split into three levels. This is the panel that")
    add("  answers design questions.")
    add(f"  Between targets {_fmt(vc.get('pct_target'), 1)} %: real, stable")
    add("    differences between participants, the signal a study needs.")
    add(f"  Between sessions {_fmt(vc.get('pct_session'), 1)} %: the same target")
    add("    measured on another day. More trials in one visit cannot reduce this.")
    add(f"  Trial to trial {_fmt(vc.get('pct_trial'), 1)} %: the noise that")
    add("    averaging more stimuli does remove.")
    add("")
    add("Bottom left - Reliability of an averaged measurement.")
    add("  How reliable a measurement becomes as more trials are averaged, shown")
    add("  separately for one and two sessions. Dotted lines mark the")
    add("  conventional 0.75 and 0.90 benchmarks.")
    t75, t90 = res.get("trials_for_75"), res.get("trials_for_90")
    for tgt, k in ((0.75, t75), (0.90, t90)):
        if k is not None and np.isfinite(k):
            add(f"  Reliability {tgt:.2f} is reached at {int(k)} trials in one session.")
        else:
            add(f"  Reliability {tgt:.2f} cannot be reached in a single session no")
            add(f"    matter how many trials are collected; the ceiling is "
                f"{_fmt(res.get('ceiling_1_session'), 3)}.")
    add("")
    add("Bottom right - Session against session.")
    if ba and "error" not in ba:
        add("  Each point is one target measured twice. The horizontal axis is the")
        add("  average of its two sessions, the vertical axis the difference")
        add("  between them. The solid line is the mean difference and the dashed")
        add("  lines the 95 % limits of agreement.")
        sp = res.get("session_pair", ("", ""))
        add(f"  {sp[0]} against {sp[1]}: bias {_fmt(ba.get('bias'))}, limits "
            f"{_fmt(ba.get('loa_lo'))} to {_fmt(ba.get('loa_hi'))}")
        add(f"  ({_fmt(ba.get('loa_width_pct_of_mean'), 0)} % of the mean).")
    else:
        add("  Empty for this dataset: comparing a target against itself on a")
        add("  second occasion needs at least two sessions per target.")
    add("")
    add("KEY NUMBERS")
    add("")
    if icc:
        ci = icc.get("ICC2_ci", (np.nan, np.nan))
        add(f"  ICC(2,1) {_fmt(icc.get('ICC2'), 3)} "
            f"[{_fmt(ci[0], 3)}, {_fmt(ci[1], 3)}] for a single recording")
        if "SEM" in icc:
            add(f"  SEM {_fmt(icc.get('SEM'))}, MDC95 {_fmt(icc.get('MDC95'))}: a")
            add("    change smaller than the MDC95 cannot be distinguished from")
            add("    measurement error in an individual.")
    if res.get("typical_error_pct_median") is not None:
        add(f"  Typical error, median across recordings "
            f"{_fmt(res.get('typical_error_pct_median'), 1)} % of the mean.")
    add("")

    warn = []
    if role.get("role") == "change":
        warn.append(
            "The two sessions differ on your within-participant design factor, so "
            "the bottom right panel measures CHANGE, not test-retest reliability. "
            "It reflects measurement error plus whatever really changed between "
            "visits, and will look wide even if the measurement is excellent. The "
            "ICC above inherits the same caveat.")
    elif role.get("role") == "reliability_assumed":
        warn.append(
            "No design factor distinguishes the two sessions, so they are treated "
            "as repeat measurements. If something was in fact meant to change "
            "between visits, record it as a Second Level design factor so it can "
            "be accounted for.")
    if vc.get("clamped"):
        warn.append(
            f"The {vc['clamped']} variance component estimated below zero and was "
            "set to zero. That means the design cannot separate that level from "
            "noise, not that the variance is genuinely absent.")
    if not vc.get("has_session_level", True):
        warn.append(
            "Only one session per target, so session-level variance cannot be "
            "estimated and everything here assumes a single visit.")
    if res.get("n_targets", 99) < 5:
        warn.append(
            f"Only {res.get('n_targets')} targets. Between-target variance, and "
            "therefore every reliability figure derived from it, is poorly "
            "determined.")
    sp_ = res.get("spread") or {}
    if sp_.get("n_atypical_cv"):
        warn.append(
            f"{sp_['n_atypical_cv']} recording(s) have an atypical coefficient of "
            "variation relative to the rest. Worth inspecting before they are "
            "allowed to influence the variance components.")
    se_ = res.get("serial") or {}
    if se_.get("r1_p") is not None and se_["r1_p"] < 0.05:
        warn.append(
            f"Lag-1 autocorrelation averages {_fmt(se_.get('r1_mean_fisher'), 3)} "
            f"across recordings (p = {_fmt(se_.get('r1_p'), 4)}), corrected for "
            "small-sample bias, so trial ordering has a systematic effect and "
            "trials are not fully exchangeable.")
    if warn:
        add("READ WITH CARE")
        add("")
        for w in warn:
            for i, line in enumerate(_wrap(w, 70)):
                add(("  - " if i == 0 else "    ") + line)
            add("")

    add(f"Produced by the MEP-CMAP Analyser '{ADDON_NAME}' add-on v{ADDON_VERSION}.")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def _report_stratum(L, res, metric, label):
    add = L.append
    add("-" * 78)
    add(f"  {label}")
    add("-" * 78)
    add(f"  recordings {res['n_recordings']}   targets {res['n_targets']}   "
        f"sessions {res['n_sessions']}")
    if "mean_of_means" in res:
        add(f"  mean of recording means  {_fmt(res['mean_of_means'])} "
            f"(SD {_fmt(res.get('sd_of_means'))})")
        add(f"  within-recording CV      median {_fmt(res.get('cv_median'), 2)} % "
            f"[{_fmt(res.get('cv_iqr_lo'), 2)}, {_fmt(res.get('cv_iqr_hi'), 2)}]")

    sp = res.get("spread") or {}
    if sp.get("levene_p") is not None:
        add(f"  equality of spread across recordings: Levene p = "
            f"{_fmt(sp['levene_p'], 4)}, Fligner-Killeen p = "
            f"{_fmt(sp.get('fligner_p'), 4)}")
        if sp.get("n_atypical_cv"):
            add(f"    {sp['n_atypical_cv']} recording(s) have an atypical CV "
                f"(robust z above 3.5); worth a look before they influence the")
            add("    variance components")
    if "typical_error_pct_median" in res:
        add(f"  typical error (odd vs even trials, within recording)")
        add(f"    median {_fmt(res['typical_error_pct_median'], 2)} % of the mean "
            f"[{_fmt(res.get('typical_error_pct_q25'), 2)}, "
            f"{_fmt(res.get('typical_error_pct_q75'), 2)}]")

    se = res.get("serial") or {}
    if se.get("r1_p") is not None:
        add(f"  serial structure across recordings")
        add(f"    lag-1 autocorrelation  median {_fmt(se.get('r1_median'), 3)} "
            f"(range {_fmt(se.get('r1_min'), 3)} to {_fmt(se.get('r1_max'), 3)})")
        add(f"    mean against zero      r = {_fmt(se.get('r1_mean_fisher'), 3)}, "
            f"t({se.get('r1_df')}) = {_fmt(se.get('r1_t'), 2)}, "
            f"p = {_fmt(se.get('r1_p'), 4)} (Fisher z transformed)")
        if se.get("bias_corrected"):
            add(f"      corrected for the -1/(n-1) small-sample bias of the sample")
            add(f"      autocorrelation (mean bias "
                f"{_fmt(se.get('mean_null_bias'), 3)}); without it, short runs look")
            add("      systematically negative when nothing is going on")
        if se.get("drift_p") is not None:
            add(f"    drift                  median "
                f"{_fmt(se.get('drift_median_pct_per_10'), 1)} % of the mean per 10 "
                f"trials, p = {_fmt(se.get('drift_p'), 4)}")

    if "skipped" in res:
        add(f"  [not decomposed] {res['skipped']}")
        add("")
        return

    vc = res.get("vc") or {}
    if "error" in vc:
        add(f"  variance components: {vc['error']}")
    else:
        add("")
        add("  Variance components")
        add(f"    between targets   {_fmt(vc.get('var_target'), 6)}  "
            f"({_fmt(vc.get('pct_target'), 1)} % of total)")
        if vc.get("has_session_level"):
            add(f"    between sessions  {_fmt(vc.get('var_session'), 6)}  "
                f"({_fmt(vc.get('pct_session'), 1)} %)")
        else:
            add("    between sessions  not identifiable (one session per target)")
        add(f"    trial to trial    {_fmt(vc.get('var_trial'), 6)}  "
            f"({_fmt(vc.get('pct_trial'), 1)} %)")
        if vc.get("clamped"):
            add(f"    [caution] the {vc['clamped']} component estimated below zero and "
                f"was set to zero,")
            add("              which means the design cannot separate it from noise")
        if res["n_targets"] < 5:
            add(f"    [caution] only {res['n_targets']} targets, so these shares are "
                f"poorly determined")
        add("    The trial share is the noise that averaging more stimuli removes;")
        add("    the session share is not, and needs repeat visits instead.")

    rel = res.get("reliability")
    if rel is not None and len(rel):
        add("")
        add("  Reliability of an averaged measurement")
        add("    sessions  trials  reliability     SEM   SEM %   MDC95  MDC95 %")
        for _, r in rel.iterrows():
            add("    %8d  %6d  %11s  %6s  %5s  %6s  %7s" % (
                int(r["sessions"]), int(r["trials_per_session"]),
                _fmt(r["reliability"], 3), _fmt(r["SEM"], 4),
                _fmt(r["SEM_pct_of_mean"], 1), _fmt(r["MDC95"], 4),
                _fmt(r["MDC95_pct_of_mean"], 1)))
        for tgt in (75, 90):
            k = res.get(f"trials_for_{tgt}")
            if k is not None and np.isfinite(k):
                add(f"    reliability {tgt / 100:.2f} reached at {int(k)} trials "
                    f"in one session")
            else:
                add(f"    reliability {tgt / 100:.2f} is unreachable in one session "
                    f"(ceiling {_fmt(res.get('ceiling_1_session'), 3)}); more")
                add("      sessions are needed, not more stimuli")

    sc = res.get("scaling") or {}
    if "error" in sc:
        add("")
        add(f"  Noise scaling with amplitude: {sc['error']}")
    elif sc.get("slope") is not None:
        add("")
        add("  Does noise scale with amplitude (raw or log scale?)")
        add(f"    log(SD) on log(mean) slope  {_fmt(sc.get('slope'), 3)} "
            f"[{_fmt(sc.get('slope_ci_lo'), 3)}, {_fmt(sc.get('slope_ci_hi'), 3)}]"
            f", R2 = {_fmt(sc.get('r_squared'), 3)}")
        add("    0 = additive (SD constant), 1 = multiplicative (CV constant),")
        add("    0.5 = variance proportional to the mean")
        for line in _wrap(str(sc.get("interpretation", "")), 68):
            add("    " + line)
        if sc.get("cv_vs_mean_p") is not None:
            add(f"    CV against amplitude: slope "
                f"{_fmt(sc.get('cv_vs_mean_slope'), 2)} per log unit, "
                f"p = {_fmt(sc.get('cv_vs_mean_p'), 4)}")

    mp = res.get("metric_pairs")
    if mp is not None and len(mp):
        add("")
        add("  Do the dispersion metrics agree with each other?")
        add("    metric A       vs metric B        rho   medians        ranking")
        for _, r in mp.iterrows():
            add("    %-14s vs %-14s %5s  %5s / %-5s  %s" % (
                r["Metric_A"], r["Metric_B"], _fmt(r["Spearman_rho"], 2),
                _fmt(r["Median_A"], 1), _fmt(r["Median_B"], 1),
                "preserved" if r["Ranking_Preserved"] else "CHANGED"))
        if not bool(mp["Ranking_Preserved"].all()):
            add("    At least one pair ranks recordings differently, so the choice")
            add("    of metric is a finding to justify rather than a convention.")

    pt = res.get("pooled_trials")
    if pt is not None and len(pt) and "loa_width_pct_of_mean_median" in pt.columns:
        add("")
        add("  How many trials to average (median across recordings)")
        add("    k   95% LoA width between two independent k-trial averages")
        for _, r in pt.iterrows():
            add("    %-3d %6s %% [%s, %s]" % (
                int(r["k_trials_averaged"]),
                _fmt(r["loa_width_pct_of_mean_median"], 1),
                _fmt(r.get("loa_width_pct_of_mean_q25"), 1),
                _fmt(r.get("loa_width_pct_of_mean_q75"), 1)))

    icc = res.get("icc")
    if icc and "error" not in icc:
        add("")
        add(f"  ICC on recording means ({icc['n_targets']} targets x "
            f"{icc['k_measures']} sessions)")
        for key, cikey, lab in (("ICC1", "ICC1_ci", "ICC(1,1) one-way random"),
                                ("ICC2", "ICC2_ci", "ICC(2,1) absolute agreement"),
                                ("ICC3", "ICC3_ci", "ICC(3,1) consistency")):
            ci = icc.get(cikey, (np.nan, np.nan))
            add(f"    {lab:<30} {_fmt(icc.get(key), 3)} "
                f"[{_fmt(ci[0], 3)}, {_fmt(ci[1], 3)}]")
        add(f"    {'ICC(2,k)':<30} {_fmt(icc.get('ICC2k'), 3)}")
        if "SEM" in icc:
            add(f"    SEM {_fmt(icc['SEM'])}   MDC95 {_fmt(icc['MDC95'])}")

    ba = res.get("ba")
    if ba and "error" not in ba:
        sp = res.get("session_pair", ("", ""))
        role = res.get("pair_role") or {}
        add("")
        add(f"  Session-to-session agreement: {sp[0]} vs {sp[1]}")
        rname = {"reliability": "TEST-RETEST RELIABILITY",
                 "reliability_assumed": "TEST-RETEST RELIABILITY (assumed)",
                 "change": "CHANGE, NOT RELIABILITY"}.get(
                     role.get("role"), str(role.get("role", "unknown")).upper())
        add(f"    this pair measures {rname}")
        add(f"      {role.get('reason', '')}")
        if role.get("role") == "change":
            add("      so these limits are NOT test-retest reliability; they will look")
            add("      wide even if the measurement itself is excellent")
        add(f"    bias                 {_fmt(ba['bias'])} "
            f"[{_fmt(ba['bias_ci_lo'])}, {_fmt(ba['bias_ci_hi'])}], "
            f"p = {_fmt(ba['bias_p'], 4)}")
        add(f"    95% limits           {_fmt(ba['loa_lo'])} to {_fmt(ba['loa_hi'])} "
            f"({_fmt(ba['loa_width_pct_of_mean'], 1)} % of the mean)")
        add(f"    proportional bias    slope {_fmt(ba['prop_bias_slope'], 3)}, "
            f"p = {_fmt(ba['prop_bias_p'], 4)}")
        bar = res.get("ba_ratio")
        if bar:
            add(f"    ratio limits         x{_fmt(bar['ratio_loa_lo'], 3)} to "
                f"x{_fmt(bar['ratio_loa_hi'], 3)}")
    add("")


def _flatten_row(stim_type, metric, factor, level, res):
    """One tidy row per stimulus type and stratum for the summary CSV."""
    vc = res.get("vc") or {}
    icc = res.get("icc") or {}
    ba = res.get("ba") or {}
    role = res.get("pair_role") or {}
    sp = res.get("session_pair", (None, None))
    row = {
        "StimType": stim_type, "Metric": metric,
        "Stratum_Factor": factor, "Stratum_Level": level,
        "N_Recordings": res.get("n_recordings"),
        "N_Targets": res.get("n_targets"),
        "N_Sessions": res.get("n_sessions"),
        "Mean_Of_Recording_Means": res.get("mean_of_means"),
        "SD_Of_Recording_Means": res.get("sd_of_means"),
        "CV_Median(%)": res.get("cv_median"),
        "CV_IQR_Lo(%)": res.get("cv_iqr_lo"),
        "CV_IQR_Hi(%)": res.get("cv_iqr_hi"),
        "Skipped_Reason": res.get("skipped"),
        "Spread_Levene_p": (res.get("spread") or {}).get("levene_p"),
        "Spread_Fligner_p": (res.get("spread") or {}).get("fligner_p"),
        "N_Atypical_CV_Recordings": (res.get("spread") or {}).get("n_atypical_cv"),
        "Typical_Error_Median(%)": res.get("typical_error_pct_median"),
        "Noise_Scaling_Slope": (res.get("scaling") or {}).get("slope"),
        "Noise_Scaling_Slope_Lo": (res.get("scaling") or {}).get("slope_ci_lo"),
        "Noise_Scaling_Slope_Hi": (res.get("scaling") or {}).get("slope_ci_hi"),
        "Noise_Scaling_R2": (res.get("scaling") or {}).get("r_squared"),
        "Noise_Scaling_Verdict": (res.get("scaling") or {}).get("interpretation"),
        "R1_Median": (res.get("serial") or {}).get("r1_median"),
        "R1_Mean_Fisher": (res.get("serial") or {}).get("r1_mean_fisher"),
        "R1_p": (res.get("serial") or {}).get("r1_p"),
        "Drift_Median(%_per_10_trials)": (res.get("serial") or {}).get("drift_median_pct_per_10"),
        "Drift_p": (res.get("serial") or {}).get("drift_p"),
        "Var_Target": vc.get("var_target"),
        "Var_Session": vc.get("var_session"),
        "Var_Trial": vc.get("var_trial"),
        "Pct_Var_Target": vc.get("pct_target"),
        "Pct_Var_Session": vc.get("pct_session"),
        "Pct_Var_Trial": vc.get("pct_trial"),
        "Var_Components_Clamped": vc.get("clamped"),
        "Session_Level_Identifiable": vc.get("has_session_level"),
        "Reliability_Ceiling_1_Session": res.get("ceiling_1_session"),
        "Trials_For_Reliability_0.75": res.get("trials_for_75"),
        "Trials_For_Reliability_0.90": res.get("trials_for_90"),
        "ICC1_1": icc.get("ICC1"), "ICC2_1": icc.get("ICC2"),
        "ICC3_1": icc.get("ICC3"), "ICC2_k": icc.get("ICC2k"),
        "ICC_SEM": icc.get("SEM"), "ICC_MDC95": icc.get("MDC95"),
        "Session_A": sp[0], "Session_B": sp[1],
        "Session_Pair_Measures": role.get("role"),
        "Session_Bias": ba.get("bias"),
        "Session_LoA_Lower": ba.get("loa_lo"),
        "Session_LoA_Upper": ba.get("loa_hi"),
        "Session_LoA_Width(%_of_mean)": ba.get("loa_width_pct_of_mean"),
    }
    for key, cikey in (("ICC1_1", "ICC1_ci"), ("ICC2_1", "ICC2_ci"),
                       ("ICC3_1", "ICC3_ci")):
        ci = icc.get(cikey, (np.nan, np.nan))
        row[key + "_Lo"], row[key + "_Hi"] = ci[0], ci[1]
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(context):
    if V is None:
        raise RuntimeError(
            "the group variability add-on needs mep_cmap.variability, which failed "
            f"to import ({_IMPORT_ERROR}). Check that variability.py sits beside "
            "addons.py in the mep_cmap package.")

    log = context.log
    cfg = context.config or {}
    df = context.group_table
    if df is None or len(df) == 0:
        log(f"{ADDON_NAME}: the group table is empty.")
        return []

    metric = _resolve_metric(df, _cfg(cfg, "variability_metric", "PTP(mV)", str), log)
    icc_target = _cfg(cfg, "variability_icc_target", "subject_limb", str).strip().lower()
    split_spec = _cfg(cfg, "variability_split_by", "auto", str).strip()
    drop_flagged = _cfg(cfg, "variability_exclude_flagged", True, bool)
    want_figs = _cfg(cfg, "variability_figures", True, bool)

    part_col = "participant_id" if "participant_id" in df.columns else None
    if part_col is None:
        log(f"{ADDON_NAME}: the group table has no participant_id column.")
        return []
    session_col = "session" if "session" in df.columns else None
    stim_col = "StimType" if "StimType" in df.columns else None
    if stim_col is None:
        log(f"{ADDON_NAME}: the group table has no StimType column.")
        return []
    file_col = "File" if "File" in df.columns else None

    work = df.copy()
    if session_col is None:
        work["session"] = "ses-1"
        session_col = "session"
        log(f"{ADDON_NAME}: no session column — treating the dataset as single-session.")

    n_before = len(work)
    work = work[pd.to_numeric(work[metric], errors="coerce").notna()]
    if drop_flagged and "Outlier_Decision" in work.columns:
        work = work[~_is_flagged(work["Outlier_Decision"])]
    dropped = n_before - len(work)
    if dropped:
        log(f"{ADDON_NAME}: {dropped} of {n_before} trial row(s) excluded "
            f"(non-numeric or flagged).")
    if work.empty:
        log(f"{ADDON_NAME}: no usable trials after exclusions.")
        return []

    # Target: two limbs of one participant are two corticospinal pathways.
    #
    # Built defensively because a blank Limb is common — BIDS-ified data often
    # leaves it empty. Concatenating a missing value produces NaN for the whole
    # target, and groupby silently drops NaN keys, so every recording would
    # disappear and the add-on would report nothing at all rather than failing.
    _subject = work[part_col].astype(str).str.strip().fillna("")
    _subject = _subject.replace({"nan": "", "None": "", "<NA>": "", "NaN": ""})
    target_desc = "participant"
    if icc_target == "subject_limb" and "Limb" in work.columns:
        # fillna first: .str.strip() leaves pandas NA as NA rather than a
        # string, so a dict replace never sees it and the blank goes undetected.
        _limb = (work["Limb"].astype(str).str.strip().fillna("")
                 .replace({"nan": "", "None": "", "<NA>": "", "NaN": ""}))
        n_blank = int((_limb == "").sum())
        if n_blank == len(_limb):
            work["_target"] = _subject
            log(f"{ADDON_NAME}: no Limb recorded, so each participant is one "
                f"target (the 'subject_limb' setting has nothing to separate).")
        else:
            work["_target"] = np.where(_limb.eq(""), _subject,
                                       _subject + "_" + _limb)
            target_desc = "participant and limb"
            if n_blank:
                log(f"{ADDON_NAME}: {n_blank} row(s) have no Limb; those "
                    f"participants are treated as a single target.")
    else:
        work["_target"] = _subject
    target_col = "_target"

    if work["_target"].isna().any() or (work["_target"] == "").any():
        log(f"{ADDON_NAME}: some rows have no participant identifier and were "
            f"dropped.")
        work = work[work["_target"].notna() & (work["_target"] != "")]
        if work.empty:
            log(f"{ADDON_NAME}: no rows left after dropping unidentified "
                f"recordings.")
            return []

    # Which design factors exist, and how each may be used
    factors = _read_design_factors(context.results_dir, log)
    if not factors:
        factors = _infer_design_factors(work,
                                        getattr(context, "design_columns", []) or [])
        if factors:
            log(f"{ADDON_NAME}: inferred design factor(s): {', '.join(factors)}.")
        else:
            log(f"{ADDON_NAME}: no usable design factors found, so results are "
                f"not split. Assign factors in Second Level > Group Analysis and "
                f"save the design to enable splitting.")
    roles = V.classify_design_factors(work, factors, target_col=target_col,
                                      session_col=session_col)
    between = [f for f, i in roles.items() if i.get("role") == "between_target"]
    within = [f for f, i in roles.items() if i.get("role") == "within_target"]
    within_factor = within[0] if within else None

    if roles:
        for f, i in roles.items():
            log(f"{ADDON_NAME}: design factor '{f}' is {i.get('role')} "
                f"({i.get('reason')}).")
    if within_factor:
        log(f"{ADDON_NAME}: '{within_factor}' labels the session axis rather than "
            f"splitting it, so between-session reliability is preserved.")

    if split_spec.lower() == "none":
        split_factors = []
    elif split_spec.lower() == "auto":
        split_factors = list(between)
    else:
        asked = [s.strip() for s in split_spec.split(",") if s.strip()]
        split_factors = []
        for f in asked:
            role = roles.get(f, {}).get("role")
            if role == "between_target":
                split_factors.append(f)
            elif role == "within_target":
                log(f"{ADDON_NAME}: '{f}' varies within a participant, so splitting "
                    f"on it would leave one session per participant and destroy "
                    f"between-session reliability. Using it as the session label "
                    f"instead.")
            else:
                log(f"{ADDON_NAME}: '{f}' is not a usable between-participant factor "
                    f"({role or 'not found'}). Available: "
                    f"{', '.join(between) if between else 'none'}.")

    # ── analyse ─────────────────────────────────────────────────────────────
    summary_rows, rel_rows, rec_rows, fig_jobs, pooled_rows = [], [], [], [], []
    metric_rows, metric_pair_rows = [], []
    L = []
    L.append("=" * 78)
    L.append(f"GROUP VARIABILITY AND RELIABILITY   metric: {metric}")
    L.append(f"targets defined by {target_desc}")
    L.append("=" * 78)
    L.append(f"  trials {len(work)}   recordings "
             f"{work.groupby([target_col, session_col]).ngroups}   "
             f"targets {work[target_col].nunique()}   "
             f"sessions {work[session_col].nunique()}")
    if within_factor:
        L.append(f"  '{within_factor}' identifies the repeated-measures axis")
    L.append(f"  splitting by: "
             f"{', '.join(split_factors) if split_factors else 'nothing (pooled only)'}")
    L.append("")

    for stim_type, sub_all in work.groupby(stim_col, sort=True):
        L.append("=" * 78)
        L.append(f"STIMULUS TYPE {stim_type}")
        L.append("=" * 78)

        slices = [(_POOLED, _POOLED, sub_all)]
        for f in split_factors:
            for lvl, sub_lvl in sub_all.groupby(f, sort=True):
                slices.append((f, str(lvl), sub_lvl))

        for factor, level, sub in slices:
            records = _build_records(sub, metric, target_col, session_col, file_col)
            if not records:
                continue
            res = _analyse_stratum(records, sub, within_factor, session_col)
            label = ("whole dataset" if factor == _POOLED
                     else f"{factor} = {level}")
            _report_stratum(L, res, metric, label)
            summary_rows.append(_flatten_row(stim_type, metric, factor, level, res))

            rel = res.get("reliability")
            if rel is not None and len(rel):
                r = rel.copy()
                r.insert(0, "StimType", stim_type)
                r.insert(1, "Stratum_Factor", factor)
                r.insert(2, "Stratum_Level", level)
                rel_rows.append(r)

            mt = res.get("metric_table")
            if mt is not None and len(mt):
                mt = mt.copy()
                mt.insert(0, "StimType", stim_type)
                mt.insert(1, "Stratum_Factor", factor)
                mt.insert(2, "Stratum_Level", level)
                metric_rows.append(mt)
            mpr = res.get("metric_pairs")
            if mpr is not None and len(mpr):
                mpr = mpr.copy()
                mpr.insert(0, "StimType", stim_type)
                mpr.insert(1, "Stratum_Factor", factor)
                mpr.insert(2, "Stratum_Level", level)
                metric_pair_rows.append(mpr)

            pt = res.get("pooled_trials")
            if pt is not None and len(pt):
                pt = pt.copy()
                pt.insert(0, "StimType", stim_type)
                pt.insert(1, "Stratum_Factor", factor)
                pt.insert(2, "Stratum_Level", level)
                pooled_rows.append(pt)

            rt = res.get("recordings")
            if rt is not None and len(rt) and factor == _POOLED:
                rt = rt.copy()
                rt.insert(0, "StimType", stim_type)
                rec_rows.append(rt)

            if want_figs and factor == _POOLED and "skipped" not in res:
                fig_jobs.append((res, stim_type))

    if not summary_rows:
        log(f"{ADDON_NAME}: nothing to summarise.")
        return []

    # ── write ───────────────────────────────────────────────────────────────
    written = []
    base = os.path.join(context.results_dir, f"{context.bids_prefix}_{ADDON_NAME}")

    pd.DataFrame(summary_rows).to_csv(base + ".csv", index=False)
    written.append(base + ".csv")

    if rel_rows:
        pd.concat(rel_rows, ignore_index=True).to_csv(
            base + "_reliability.csv", index=False)
        written.append(base + "_reliability.csv")

    if metric_rows:
        pd.concat(metric_rows, ignore_index=True).to_csv(
            base + "_dispersion_by_recording.csv", index=False)
        written.append(base + "_dispersion_by_recording.csv")

    if metric_pair_rows:
        pd.concat(metric_pair_rows, ignore_index=True).to_csv(
            base + "_dispersion_metric_agreement.csv", index=False)
        written.append(base + "_dispersion_metric_agreement.csv")

    if pooled_rows:
        pd.concat(pooled_rows, ignore_index=True).to_csv(
            base + "_trials_to_average.csv", index=False)
        written.append(base + "_trials_to_average.csv")

    if rec_rows:
        pd.concat(rec_rows, ignore_index=True).to_csv(
            base + "_recordings.csv", index=False)
        written.append(base + "_recordings.csv")

    L.append("=" * 78)
    with open(base + "_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    written.append(base + "_report.txt")

    for row in summary_rows:
        if row["Stratum_Factor"] == _POOLED and row.get("ICC2_1") is not None:
            log(f"{ADDON_NAME}: {row['StimType']} — trial variance "
                f"{_fmt(row.get('Pct_Var_Trial'), 1)} %, ICC(2,1) "
                f"{_fmt(row.get('ICC2_1'), 3)}, session pair measures "
                f"{row.get('Session_Pair_Measures')}")
    log(f"{ADDON_NAME}: {len(summary_rows)} stratum row(s) -> "
        f"{os.path.basename(base)}.csv and report")

    if want_figs and not fig_jobs:
        log(f"{ADDON_NAME}: no figure written — every stratum was too small to "
            f"decompose (at least {MIN_TARGETS} targets are needed, and this "
            f"dataset has "
            f"{work[target_col].nunique()}). Process more recordings, or set "
            f"'Reliability target' to 'subject' if limbs were split.")

    if want_figs and fig_jobs:
        # Each figure and each caption is guarded separately. Wrapping the whole
        # loop in one try meant a single failure silently cost every later
        # figure as well, and left a stray PNG with no caption beside it.
        fig_dir = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            fig_root = context.figures_dir or os.path.join(context.results_dir,
                                                           "figures")
            fig_dir = os.path.join(fig_root,
                                   f"{context.bids_prefix}_{ADDON_NAME}_figures")
            os.makedirs(fig_dir, exist_ok=True)
        except Exception as ex:
            log(f"{ADDON_NAME}: could not prepare the figures folder ({ex}); "
                f"figures and captions skipped.")
            fig_dir = None

        n_png = n_cap = 0
        if fig_dir:
            for res, stim_type in fig_jobs:
                png = os.path.join(fig_dir, f"{context.bids_prefix}_stim-{_fname_token(stim_type)}"
                                            f"_{ADDON_NAME}.png")
                try:
                    _plot_stratum(res, metric, stim_type, png)
                    written.append(png)
                    n_png += 1
                except Exception as ex:
                    log(f"{ADDON_NAME}: figure for {stim_type} skipped "
                        f"({type(ex).__name__}: {ex})")

                cap_path = png[:-4] + "_caption.txt"
                try:
                    text = _caption_stratum(res, metric, stim_type,
                                            context.bids_prefix)
                    with open(cap_path, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    written.append(cap_path)
                    n_cap += 1
                except Exception as ex:
                    log(f"{ADDON_NAME}: caption for {stim_type} skipped "
                        f"({type(ex).__name__}: {ex})")

            log(f"{ADDON_NAME}: {n_png} figure(s) and {n_cap} caption(s) -> "
                f"{os.path.basename(fig_dir)}/")

    return written
