"""
mep_cmap.normalisation
~~~~~~~~~~~~~~~~~~~~~~
M-wave / Mmax normalisation and paired-pulse ratio computation.

  • compute_mmax          — robust plateau detection on a recruitment curve
  • apply_normalisation   — fill Reference_* and Normalised_PTP columns in
                            the trial-level row lists produced by pipeline.py
  • fit_qr_compensation   — per-sample quantile-regression compensation for
                            spinal motoneurone excitability (Carson, 2026)
  • apply_emg_compensation — fill Adjusted_PTP_QR(mV) + EMGComp_* diagnostic
                            columns in the trial-level row lists

EMG excitability compensation
-----------------------------
Implements the quantile-regression (QR) method of Carson (2026, J Physiol
604.14, 5731–5757) for compensating MEP/evoked-response amplitude for the
positive covariation with pre-stimulus ('background') r.m.s. EMG that reflects
the momentary excitability of the spinal motoneurone pool. Per participant
sample (one StimType at a single TMS intensity) the amplitude is regressed on
pre-stimulus RMS via median (tau=0.5) quantile regression; each trial's
residual is expressed relative to an uncertainty-weighted reference value
(a blend of the regression intercept and the central tendency of the fitted
values). The adjusted amplitude is the estimate of what the response would
have been had background excitability been at that common reference level.
Units are unchanged (mV).

Correspondence with the author's reference code
-----------------------------------------------
Verified against ``annotated_QR_example_code.R`` and ``example_data.csv``
(Carson 2026, https://doi.org/10.5281/zenodo.20037178). Because QR estimates a
conditional MEDIAN, the reference value must be anchored on ``median(fitted)``
and the second standard error evaluated at the RMS value that produces it, not
on the arithmetic mean of the raw amplitudes. Anchoring on the mean inflates
the reference for right-skewed MEP samples (mean > median) and pushes every
adjusted value upward. On the author's example data the median anchor
reproduces his reference value of 27.15 uV exactly; a mean anchor gives 35.84.
The corresponding LSR script (``annotated_LSR_example_code.R``) uses
``mean(fitted)``, which for least squares is identically ``mean(y)`` — the two
scripts are consistent, they simply match the central tendency to the model.
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
        key_of=None,
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
    stim_ptps       : {group_key: [ptp_val, ...]}
    reference_map   : {group_key: ref_group_key}  —  "" / None = no normalisation
    plateau_tolerance: fraction of peak for plateau detection (0 = simple mean)
    key_of          : row -> the group key reference_map is keyed by. Defaults
                      to the StimType column, which is the group key only for
                      a recording without conditions. With conditions a row
                      says StimType=B, Condition=haptic while tab 1a says
                      B·haptic, so the caller passes the rule that rebuilds the
                      key (pipeline.group_key_for); without it every
                      condition-split row was skipped and left blank.

    Anything designated on tab 1a that cannot be applied is logged with a ⚠️:
    a reference with no usable trials, and a group that ends up with no
    normalised trial. A blank Normalised_PTP column is never silent.
    """
    if not latency_rows or not reference_map:
        return

    # ── Compute reference mean for each referenced stim ───────────────────────
    ref_means: dict[str, tuple[float, int, str]] = {}  # {ref_stim: (mean, n, method)}
    for ref_st in sorted(set(v for v in reference_map.values() if v)):
        _users = sorted(k for k, v in reference_map.items() if v == ref_st)
        if ref_st not in stim_ptps:
            log_callback(
                f"⚠️  Normalisation: reference '{ref_st}' (chosen on tab 1a "
                f"for {', '.join(_users)}) has no trials in this file, so "
                f"{'that group is' if len(_users) == 1 else 'those groups are'} "
                f"not normalised.")
            continue
        vals = [v for v in stim_ptps[ref_st]
                if v is not None and np.isfinite(v) and v > 0]
        if not vals:
            log_callback(
                f"⚠️  Normalisation: reference '{ref_st}' has no usable PTP "
                f"values (none above zero), so {', '.join(_users)} "
                f"{'is' if len(_users) == 1 else 'are'} not normalised.")
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
    ri_type      = col["Reference_Type"]
    ri_mean      = col["Reference_Mean(mV)"]
    ri_n         = col["Reference_N"]
    ri_norm      = col["Normalised_PTP"]
    ri_ptp       = col["PTP(mV)"]
    ri_sttype    = col["StimType"]
    ri_rms       = col.get("PreStimRMS")
    ri_ptp_rms   = col.get("PTP_per_PreStimRMS")
    ri_norm_rms  = col.get("Normalised_PTP_per_PreStimRMS")
    ri_adj       = col.get("Adjusted_PTP_QR(mV)")
    ri_norm_adj  = col.get("Normalised_Adjusted_PTP_QR")

    if key_of is None:
        def key_of(row):
            return row[ri_sttype]
    n_done: dict[str, int] = {}

    for row in latency_rows:
        st       = key_of(row)
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
        n_done[st] = n_done.get(st, 0) + 1
        # Normalised_PTP / PreStimRMS
        if ri_norm_rms is not None and ri_rms is not None:
            try:
                _rms = float(row[ri_rms])
                _norm_ptp = ptp_f / mean_ref
                row[ri_norm_rms] = round(_norm_ptp / _rms, 4) if _rms > 0 else None
            except (TypeError, ValueError):
                pass
        # Normalised_Adjusted_PTP_QR = Adjusted_PTP_QR / raw reference mean.
        # Denominator is deliberately the RAW reference mean (M-wave/Mmax is not
        # excitability-compensated), so this is the excitability-compensated PTP
        # expressed as a fraction of the muscle's maximal response.
        if ri_norm_adj is not None and ri_adj is not None:
            try:
                _adj = float(row[ri_adj])
                row[ri_norm_adj] = round(_adj / mean_ref, 4)
            except (TypeError, ValueError):
                pass

    # A group set to be normalised to a reference that WAS found, yet with no
    # trial normalised, is the silent failure this check exists for: the rows
    # did not match tab 1a's group key. Say so rather than leave a blank column.
    for target, ref_st in sorted(reference_map.items()):
        if ref_st and ref_st in ref_means and not n_done.get(target):
            log_callback(
                f"⚠️  Normalisation: '{target}' was set to be normalised to "
                f"'{ref_st}', but no '{target}' trial was normalised. Its "
                f"Normalised_PTP column is blank.")


# ─── EMG excitability compensation (Carson 2026, quantile regression) ──────────

def fit_qr_compensation(
        rms:  np.ndarray | list,
        ptp:  np.ndarray | list,
        tau:  float = 0.5,
        min_trials: int = 8,
        low_n_trials: int = 15,
) -> dict:
    """
    Per-sample quantile-regression compensation of evoked-response amplitude
    for pre-stimulus ('background') r.m.s. EMG (Carson, 2026).

    Regresses ``ptp`` on ``rms`` using median (tau=0.5) quantile regression,
    then expresses each trial's residual relative to an uncertainty-weighted
    reference value. The reference is a blend of the regression intercept and
    the central tendency of the FITTED values, weighted by the relative
    standard error of the fitted ordinate at the intercept (RMS = 0) versus at
    the RMS value that yields that central tendency:

        ct        = median(fitted)            (QR estimates a conditional median)
        x_ct      = RMS at which fitted == ct (identically median(RMS), see note)
        SEi       = SE of the fitted ordinate at RMS = 0     (the intercept)
        SEc       = SE of the fitted ordinate at RMS = x_ct
        SET       = SEi + SEc
        Wi        = 1 - SEi/SET + SEc/SET     (weight on the intercept)
        Wc        = SEi/SET - SEc/SET         (weight on the central tendency)
        reference = Wi * intercept + Wc * ct
        adjusted  = residual + reference

    Because the fitted values are a monotone linear function of RMS, the median
    of the fitted values is attained at the median of RMS, so ``x_ct`` is
    computed as ``median(rms)`` directly. This is algebraically identical to the
    author's ``(median(fitted) - intercept) / slope`` and avoids dividing by a
    near-zero slope.

    Matching the anchor to the model matters. QR centres its residuals on the
    conditional median, so anchoring the reference on the arithmetic mean of the
    raw amplitudes mixes scales: MEP samples are strongly right-skewed, mean >
    median, and every adjusted value in the sample is inflated by the
    difference. The equivalent LSR implementation anchors on ``mean(fitted)``,
    which for least squares is identically ``mean(y)``.

    As the standard error of the intercept grows, Wi -> 0 and the reference
    converges on the central tendency of the fitted values. Independently, as
    the slope tends to zero the intercept itself converges on that central
    tendency, so the reference converges regardless of Wi. The scheme is
    self-limiting in both cases: where no association is present, the
    adjustment vanishes.

    Standard errors of the fitted ordinate are computed analytically from the
    parameter covariance: SE_fit(x) = sqrt([1, x] · cov(β) · [1, x]ᵀ). The
    author converts rq() prediction CIs to SEs instead; only the RATIOS
    SEi/SET and SEc/SET enter the weights, so any common scaling cancels.

    Parameters
    ----------
    rms          : pre-stimulus r.m.s. EMG per trial (same order as ``ptp``)
    ptp          : evoked-response peak-to-peak amplitude per trial (mV)
    tau          : quantile for QR (0.5 = median; matches Carson 2026)
    min_trials   : minimum trials required to attempt a fit (paper floor = 8)
    low_n_trials : below this n the fit is still performed but ``low_n`` is set,
                   because with two parameters the standard errors, and hence
                   Wi, are unstable (the paper's median sample was 20 trials)

    Returns
    -------
    dict with keys:
        status      : "ok" | "insufficient_n" | "degenerate_rms" | "error"
                      | "unavailable"
        method      : "qr" (with a suffix for non-ok status)
        n           : int   — trials used in the fit
        low_n       : bool  — n < low_n_trials (weights poorly determined)
        slope       : float — QR slope, amplitude units per RMS unit | None
                      (PTP and PreStimRMS share a unit, so a slope of ~26
                      corresponds to the paper's 26 µV per µV)
        intercept   : float — QR intercept (mV)         | None
        w_intercept : float — Wi in [0, 1]              | None
        central     : float — median of the fitted values (mV) | None
        adjustment  : float — reference - median(fitted), mV | None
        pseudo_r2   : float — Koenker–Machado pseudo-R²  | None
        rho_pre     : float — Spearman rho(ptp, rms) before adjustment
        rho_post    : float — Spearman rho(adjusted, rms) (≈ 0 when adequate)
        reference   : float — reference value (mV)      | None
        adjusted    : list[float] — per-trial adjusted amplitude (mV);
                      falls back to the raw ptp values for any non-"ok" status
    """
    rms = np.asarray(rms, dtype=float)
    ptp = np.asarray(ptp, dtype=float)

    def _fallback(status: str, suffix: str) -> dict:
        # No adjustment possible → adjusted == raw (clearly flagged via status).
        return dict(
            status=status, method=f"qr_{suffix}", n=int(len(ptp)),
            low_n=bool(len(ptp) < low_n_trials),
            slope=None, intercept=None, w_intercept=None, central=None,
            adjustment=0.0, pseudo_r2=None, rho_pre=None, rho_post=None,
            reference=(float(np.median(ptp)) if len(ptp) else None),
            adjusted=ptp.tolist(),
        )

    if len(ptp) < min_trials:
        return _fallback("insufficient_n", "insufficient_n")
    if not np.all(np.isfinite(rms)) or not np.all(np.isfinite(ptp)):
        # Callers pass pre-cleaned arrays; guard anyway.
        return _fallback("error", "error")
    if np.std(rms) < 1e-12:
        # Predictor has (near) zero variance — regression is undefined.
        return _fallback("degenerate_rms", "degenerate_rms")

    try:
        import statsmodels.api as sm
    except Exception:
        return _fallback("unavailable", "unavailable")

    try:
        X = sm.add_constant(rms)                       # columns [const, rms]
        res = sm.QuantReg(ptp, X).fit(q=tau, max_iter=5000)
        b0 = float(res.params[0])
        b1 = float(res.params[1])
        cov = np.asarray(res.cov_params(), dtype=float)

        def _se_fit(x0: float) -> float:
            v = np.array([1.0, x0])
            return float(np.sqrt(max(v @ cov @ v, 0.0)))

        fitted = np.asarray(res.predict(X), dtype=float)

        # QR estimates a conditional median, so the reference is anchored on the
        # median of the fitted values and the second SE evaluated at the RMS
        # that produces it. fitted is monotone in rms, so that RMS is median(rms)
        # exactly — algebraically identical to the author's
        # (median(fitted) - intercept) / slope, without the near-zero division.
        x_ct = float(np.median(rms))
        ct   = b0 + b1 * x_ct                          # == median(fitted)

        se_i = _se_fit(0.0)
        se_c = _se_fit(x_ct)
        se_t = se_i + se_c
        if not np.isfinite(se_t) or se_t <= 0:
            return _fallback("error", "error")

        w_i = 1.0 - (se_i / se_t) + (se_c / se_t)      # weight on intercept
        # SEi >= SEc in every well-behaved fit, so Wi is confined to [0, 1];
        # clamp as a cheap guard against a pathological covariance estimate and
        # keep the two weights complementary.
        w_i = float(np.clip(w_i, 0.0, 1.0))
        w_c = 1.0 - w_i                                # weight on median(fitted)
        reference = w_i * b0 + w_c * ct

        residuals = ptp - fitted
        adjusted = residuals + reference

        try:
            pseudo_r2 = float(res.prsquared)
        except Exception:
            pseudo_r2 = None

        # rho before and after adjustment: the association being removed, and
        # the residual association left behind (should be ~0 when adequate).
        rho_pre = rho_post = None
        try:
            from scipy.stats import spearmanr
            if np.std(ptp) > 1e-12:
                _rho, _ = spearmanr(ptp, rms)
                rho_pre = float(_rho) if np.isfinite(_rho) else None
            if np.std(adjusted) > 1e-12:
                _rho, _ = spearmanr(adjusted, rms)
                rho_post = float(_rho) if np.isfinite(_rho) else None
        except Exception:
            pass

        return dict(
            status="ok", method="qr", n=int(len(ptp)),
            low_n=bool(len(ptp) < low_n_trials),
            slope=round(b1, 6), intercept=round(b0, 6),
            w_intercept=round(w_i, 4), central=round(ct, 6),
            adjustment=round(reference - ct, 4),
            pseudo_r2=(round(pseudo_r2, 4) if pseudo_r2 is not None else None),
            rho_pre=(round(rho_pre, 4) if rho_pre is not None else None),
            rho_post=(round(rho_post, 4) if rho_post is not None else None),
            reference=round(reference, 6),
            adjusted=[round(float(a), 6) for a in adjusted],
        )
    except Exception:
        return _fallback("error", "error")


# Outlier_Decision values that take a trial out of the regression sample.
# "Kept" means flagged by the z-screen but retained by the reviewer, and
# "Not flagged" was never flagged; both belong in the fit. Retaining as many
# datapoints as possible is one of the stated benefits of the method, so only
# trials actively removed or excluded are dropped.
EXCLUDED_DECISIONS: tuple[str, ...] = ("Removed", "Excluded")


def apply_emg_compensation(
        latency_rows: list[list],
        col: dict,
        exclude_stims=None,
        method: str = "qr",
        tau: float = 0.5,
        min_trials: int = 8,
        extra_group_cols=None,
        excluded_decisions: tuple[str, ...] = EXCLUDED_DECISIONS,
        log_callback=print,
) -> None:
    """
    Fill EMG-excitability-compensation columns in-place in a trial row list.

    Groups rows by StimType (each StimType is assumed to be a single TMS
    intensity — the paper's requirement, since MEP amplitude variance changes
    with stimulator output) plus any columns named in ``extra_group_cols``, and
    fits :func:`fit_qr_compensation` on each group. A group must be a single
    *sample* in the paper's sense: one participant, one intensity, one block or
    condition. If a recording contains, say, pre- and post-intervention trials
    under one StimType label, pass the column that distinguishes them via
    ``extra_group_cols`` so they are fitted separately, as the paper does.

    Trials whose Outlier_Decision is in ``excluded_decisions`` are left out of
    the fit and receive no compensation values. All other trials, including
    those flagged by the z-screen but kept, contribute to the fit.

    Columns written (by name; silently skipped if absent from ``col``):
        Adjusted_PTP_QR(mV), EMGComp_Method, EMGComp_N, EMGComp_Slope,
        EMGComp_Intercept, EMGComp_InterceptWeight, EMGComp_Adjustment(mV),
        EMGComp_PseudoR2, EMGComp_Rho_Pre, EMGComp_Rho_Post

    Parameters
    ----------
    latency_rows       : list of rows indexed by ``col``
    col                : {column_name: row_index}
    exclude_stims      : iterable of StimType values to skip (M-wave / Mmax
                         references — not spinally mediated and often
                         multi-intensity)
    method             : compensation backend; only "qr" is supported in v1
    tau                : quantile for QR (0.5 = median)
    min_trials         : minimum trials per sample to attempt a fit
    extra_group_cols   : additional column names forming the sample key
    excluded_decisions : Outlier_Decision values that drop a trial from the fit
    """
    if not latency_rows:
        return
    if method != "qr":
        log_callback(f"⚠️  EMG compensation: unknown method '{method}' — skipped")
        return

    exclude = set(exclude_stims or ())

    ri_st   = col.get("StimType")
    ri_ptp  = col.get("PTP(mV)")
    ri_rms  = col.get("PreStimRMS")
    ri_out  = col.get("Outlier_Decision")
    ri_adj  = col.get("Adjusted_PTP_QR(mV)")
    if ri_st is None or ri_ptp is None or ri_rms is None or ri_adj is None:
        return  # required columns not present in this schema

    ri_meth = col.get("EMGComp_Method")
    ri_n    = col.get("EMGComp_N")
    ri_slp  = col.get("EMGComp_Slope")
    ri_int  = col.get("EMGComp_Intercept")
    ri_wi   = col.get("EMGComp_InterceptWeight")
    ri_adjm = col.get("EMGComp_Adjustment(mV)")
    ri_pr2  = col.get("EMGComp_PseudoR2")
    ri_rho0 = col.get("EMGComp_Rho_Pre")
    ri_rho  = col.get("EMGComp_Rho_Post")

    ri_extra = [col[c] for c in (extra_group_cols or ()) if c in col]
    dropped  = set(excluded_decisions or ())

    def _in_sample(row) -> bool:
        return ri_out is None or row[ri_out] not in dropped

    # ── Group trial indices by sample key ─────────────────────────────────────
    groups: dict = {}
    for i, row in enumerate(latency_rows):
        st = row[ri_st]
        if st in exclude:
            continue
        if not _in_sample(row):
            continue
        try:
            _p = float(row[ri_ptp])
            _r = float(row[ri_rms])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(_p) and np.isfinite(_r)):
            continue
        key = (st,) + tuple(row[j] for j in ri_extra)
        groups.setdefault(key, []).append(i)

    # ── Fit + write per group ─────────────────────────────────────────────────
    for key, idxs in groups.items():
        st    = key[0]
        label = " / ".join(str(k) for k in key
                           if k is not None and str(k) not in ("", "n/a"))
        rms_vals = [float(latency_rows[i][ri_rms]) for i in idxs]
        ptp_vals = [float(latency_rows[i][ri_ptp]) for i in idxs]

        r = fit_qr_compensation(rms_vals, ptp_vals, tau=tau, min_trials=min_trials)

        adj = r["adjusted"]
        for k, i in enumerate(idxs):
            row = latency_rows[i]
            row[ri_adj] = adj[k]
            if ri_meth is not None: row[ri_meth] = r["method"]
            if ri_n    is not None: row[ri_n]    = r["n"]
            if ri_slp  is not None: row[ri_slp]  = r["slope"]
            if ri_int  is not None: row[ri_int]  = r["intercept"]
            if ri_wi   is not None: row[ri_wi]   = r["w_intercept"]
            if ri_adjm is not None: row[ri_adjm] = r["adjustment"]
            if ri_pr2  is not None: row[ri_pr2]  = r["pseudo_r2"]
            if ri_rho0 is not None: row[ri_rho0] = r["rho_pre"]
            if ri_rho  is not None: row[ri_rho]  = r["rho_post"]

        if r["status"] == "ok":
            log_callback(
                f"🧮 EMG compensation '{label}': n={r['n']}, "
                f"slope={r['slope']:.3f}, Wi={r['w_intercept']:.2f}, "
                f"adjustment={r['adjustment']:+.4f} mV, "
                f"pseudo-R²={r['pseudo_r2']}, "
                f"rho {r['rho_pre']} → {r['rho_post']}"
            )
            if r["low_n"]:
                log_callback(
                    f"⚠️  EMG compensation '{label}': only {r['n']} trials — the "
                    f"intercept weighting is poorly determined below ~15 "
                    f"(Carson's median sample was 20)"
                )
            if r["rho_post"] is not None and abs(r["rho_post"]) > 0.10:
                log_callback(
                    f"⚠️  EMG compensation '{label}': residual rho="
                    f"{r['rho_post']:+.2f} — the association was not fully "
                    f"removed; inspect the fit before using adjusted values"
                )
            if r["slope"] is not None and r["slope"] < 0:
                log_callback(
                    f"ℹ️  EMG compensation '{label}': slope is negative, so the "
                    f"reference sits above the sample median and adjusted "
                    f"amplitudes are LARGER than unadjusted. This is expected "
                    f"(74 of Carson's 182 participants behaved this way)."
                )
        else:
            log_callback(
                f"🧮 EMG compensation '{label}': {r['status']} "
                f"(n={r['n']}) — adjusted = raw amplitude"
            )
