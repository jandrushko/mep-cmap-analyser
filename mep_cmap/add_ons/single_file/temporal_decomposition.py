"""
MEP-CMAP Analyser — Temporal MEP Decomposition add-on
=====================================================

Splits each MEP into successive time bins from onset and aggregates them into an
EARLY and a LATE phase, following the temporal-decomposition approach used to
dissociate fast-conducting corticospinal from slower, polysynaptic
cortico-reticulospinal transmission.

Rationale
---------
The surface MEP is not a single descending volley. The earliest portion is
attributed to fast-conducting corticospinal output, whereas later portions are
attributed to slower and/or polysynaptic routes — notably cortico-reticulospinal
transmission. The two portions can be modulated independently (Tapia et al.,
J Neurophysiol 2023; Chvatal-Reid et al., Exp Brain Res 2022), which is the
empirical basis for quantifying them separately.

The 8 ms early/late boundary and the 20 ms total window follow the published
convention (2 ms bins from MEP onset to 20 ms; initial 8 ms = early-phase,
subsequent 12 ms = late-phase). Both are SETTINGS, not constants — the bin
profile is the primary, assumption-free output and the early/late split is a
derived convenience. Report the bins.

Important caveats (read before interpreting)
--------------------------------------------
* The 8 ms boundary is a convention validated mainly in lower-limb postural
  muscles. In muscles with dense monosynaptic corticomotoneuronal input the
  early/late dissociation is weaker.
* Late-phase area is not independent of MEP size: bigger MEPs last longer.
  Normalise (Mmax, or the phase ratio) or model MEP amplitude as a covariate.
* Onset error propagates 1:1 into the boundary. TD_Onset_Source records whether
  the onset was auto-detected or Inspector-edited so this can be audited.
* Reticulospinal contributions are state-dependent — far more prominent under
  voluntary contraction / postural load than at rest.

Method
------
For each trial:
  1. Onset is taken from the core per-trial table (`Latency(ms)`), joined by
     (StimType, Segment). This is the tool's single source of truth for onset —
     the add-on never re-detects it.
  2. The analysis window runs from onset to onset + window_ms, CLAMPED to the
     detected MEP offset (`cSP_MEP_Offset(ms)`) when clamping is enabled and an
     offset exists. Without the clamp the late window can run past the end of
     the MEP into the silent period, where area is a floor effect rather than
     signal.
  3. |EMG| is integrated by the trapezoidal rule. Integration uses a cumulative
     integral evaluated at exact bin edges by linear interpolation, so bin
     boundaries are not quantised to the sample grid and results are stable
     across sampling rates.
  4. Background EMG is subtracted as mean|EMG| over the pre-stimulus baseline
     multiplied by each window's duration. Values may go slightly negative;
     they are NOT clipped, because clipping hides floor effects.

     The correction is fully reversible from the output file — every area is
     recoverable as `corrected + TD_Baseline_Amp * duration_ms / 1000` — so a
     reviewer can inspect the uncorrected values without a re-run, and no raw
     duplicate columns are needed.

Outputs
-------
`<prefix>_temporal_decomposition.csv` — one row per trial. Every measurement
column is prefixed `TD_` so it can be joined into the group table without
colliding with core column names. Join keys (File / StimType / Segment) are
NOT prefixed, so they match `_trials.csv` exactly.

Optional per-condition diagnostic figures into the session's `figures/` folder.

This add-on writes NEW FILES ONLY and never touches core outputs.
"""

import os

import numpy as np

try:
    import pandas as pd
except Exception:                                    # pragma: no cover
    pd = None


# ── Required metadata ────────────────────────────────────────────────────────
ADDON_NAME        = "temporal_decomposition"
ADDON_DESCRIPTION = ("Temporal MEP decomposition: 2 ms bins from onset, aggregated "
                     "into early (corticospinal) and late (cortico-reticulospinal) "
                     "phases, clamped to the detected MEP offset")
ADDON_VERSION     = "1.0.0"
ADDON_AUTHOR      = "MEP-CMAP Analyser (built-in)"
ADDON_SCOPE       = "single_file"

ADDON_SETTINGS = [
    {
        "key": "td_bin_ms", "label": "Bin width (ms)", "type": "float",
        "default": 2.0, "min": 0.25, "max": 10.0,
        "help": ("Width of each successive bin measured from MEP onset. The "
                 "published convention is 2 ms."),
    },
    {
        "key": "td_window_ms", "label": "Window from onset (ms)", "type": "float",
        "default": 20.0, "min": 2.0, "max": 100.0,
        "help": ("Total analysis window from MEP onset. Published convention is "
                 "20 ms. Truncated further if the MEP offset clamp applies."),
    },
    {
        "key": "td_boundary_ms", "label": "Early/late boundary (ms)", "type": "float",
        "default": 8.0, "min": 0.5, "max": 50.0,
        "help": ("Boundary between early and late phase, measured from onset. "
                 "Published convention is 8 ms. This is a convention, not a "
                 "physiological landmark — the bin columns are the primary output."),
    },
    {
        "key": "td_clamp_to_offset", "label": "Clamp to MEP offset", "type": "bool",
        "default": True,
        "help": ("yes/no. Truncate the window at cSP_MEP_Offset(ms) when a MEP "
                 "offset was detected, so the late phase cannot extend into the "
                 "silent period. Strongly recommended."),
    },
    {
        "key": "td_baseline_correct", "label": "Subtract baseline EMG", "type": "bool",
        "default": True,
        "help": ("yes/no. Subtract mean rectified pre-stimulus EMG x window "
                 "duration from each area. Essential for MEPs collected during "
                 "voluntary contraction."),
    },
    {
        "key": "td_baseline_end_ms", "label": "Baseline ends at (ms)", "type": "float",
        "default": -2.0, "min": -200.0, "max": -0.5,
        "help": ("End of the pre-stimulus baseline window, relative to the "
                 "stimulus. The window starts at the beginning of the epoch. "
                 "Keep it clear of stimulus artefact."),
    },
    {
        "key": "td_stim_types", "label": "Only these stim types", "type": "str",
        "default": "",
        "help": ("Comma-separated stim types to analyse; blank = all. The "
                 "early/late interpretation is specific to TMS-evoked responses "
                 "— running it on M-wave or H-reflex conditions produces numbers "
                 "with no corticospinal/reticulospinal meaning. Name your TMS "
                 "conditions here to exclude peripheral-stimulation conditions."),
    },
    {
        "key": "td_figures", "label": "Write figures", "type": "bool",
        "default": True,
        "help": "yes/no. Per-condition diagnostic figure: mean rectified trace "
                "with early/late shading, plus the mean bin profile.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _as_bool(v, default):
    """Coerce a config value to bool. Settings arrive as bool from the GUI but
    may be raw strings if the add-on is driven from a script."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y", "t")


def _as_float(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f


def _num_or_none(v):
    """Parse a per-trial table cell to float, or None.

    The core table writes the literal string 'Not Detected' when onset detection
    failed, and leaves cSP columns blank when no cSP was found. Both must map to
    None rather than raising or silently becoming NaN-as-zero.
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in ("not detected", "nan", "none", "na"):
            return None
        try:
            v = float(s)
        except ValueError:
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def _cumulative_integral(rect, t_s):
    """Cumulative trapezoidal integral of `rect` over `t_s`, with a leading 0.

    Returned array is the same length as `rect`, so np.interp against `t_s`
    gives the integral from the epoch start to any arbitrary time. Area between
    two times is then a difference of two interpolations — exact at the bin
    edges rather than quantised to the sample grid.
    """
    if rect.size < 2:
        return np.zeros_like(rect)
    steps = np.diff(t_s) * (rect[:-1] + rect[1:]) * 0.5
    out = np.empty_like(rect, dtype=float)
    out[0] = 0.0
    np.cumsum(steps, out=out[1:])
    return out


def _integral_to(rect, cum, t_ms, x_ms):
    """Integral of |EMG| from the epoch start to an arbitrary time `x_ms`.

    Whole sample intervals come from the precomputed cumulative integral; the
    final partial interval is closed with a proper trapezoid against the signal
    value interpolated at `x_ms`. Interpolating the cumulative directly would
    instead assume |EMG| is constant across that partial interval, which biases
    every bin edge that falls between samples.
    """
    x_ms = float(np.clip(x_ms, t_ms[0], t_ms[-1]))
    i = int(np.searchsorted(t_ms, x_ms, side="right") - 1)
    i = min(max(i, 0), rect.size - 1)
    if i >= rect.size - 1 or x_ms <= t_ms[i]:
        return float(cum[i])
    span = t_ms[i + 1] - t_ms[i]
    frac = (x_ms - t_ms[i]) / span
    v_x  = rect[i] + frac * (rect[i + 1] - rect[i])
    partial = (x_ms - t_ms[i]) / 1000.0 * (rect[i] + v_x) * 0.5
    return float(cum[i] + partial)


def _area(rect, cum, t_ms, a_ms, b_ms):
    """Integral of |EMG| between a_ms and b_ms, exact at both edges."""
    if b_ms <= a_ms:
        return 0.0
    return _integral_to(rect, cum, t_ms, b_ms) - _integral_to(rect, cum, t_ms, a_ms)


def _bin_edges(bin_ms, window_ms):
    """Bin edges in ms from onset. The final partial bin is dropped so every
    reported bin has identical width and the columns stay comparable."""
    n = int(np.floor(window_ms / bin_ms + 1e-9))
    return np.arange(n + 1, dtype=float) * bin_ms


def _bin_label(lo, hi, unit):
    def _f(x):
        return f"{x:g}".replace(".", "p")
    return f"TD_Bin_{_f(lo)}_{_f(hi)}({unit}*s)"


# ─────────────────────────────────────────────────────────────────────────────
# Per-trial lookup of onset / offset from the core table
# ─────────────────────────────────────────────────────────────────────────────
def _build_marker_lookup(trials, log):
    """Map (StimType, Segment) -> {'onset', 'offset', 'note'}.

    `Segment` in the core per-trial table is 1-based and indexes directly into
    the waveform stack (Segment - 1), because the stack is `segs_all` — every
    trial, outliers included, in acquisition order. Joining on Segment rather
    than on row position is what keeps this correct if the table is ever
    filtered or re-sorted.
    """
    lut = {}
    if trials is None or pd is None or getattr(trials, "empty", True):
        return lut
    cols = set(trials.columns)
    need = {"StimType", "Segment"}
    if not need.issubset(cols):
        log(f"{ADDON_NAME}: per-trial table lacks {sorted(need - cols)}; "
            f"cannot align onsets to waveforms.")
        return lut
    onset_col  = "Latency(ms)"          if "Latency(ms)"          in cols else None
    offset_col = "cSP_MEP_Offset(ms)"   if "cSP_MEP_Offset(ms)"   in cols else None
    note_col   = "Manual_Note"          if "Manual_Note"          in cols else None
    if onset_col is None:
        log(f"{ADDON_NAME}: per-trial table has no 'Latency(ms)' column.")
        return lut
    for rec in trials.to_dict("records"):
        seg = _num_or_none(rec.get("Segment"))
        if seg is None:
            continue
        lut[(str(rec.get("StimType")), int(round(seg)))] = {
            "onset":  _num_or_none(rec.get(onset_col)),
            "offset": _num_or_none(rec.get(offset_col)) if offset_col else None,
            "note":   rec.get(note_col) if note_col else None,
            "file":   rec.get("File"),
        }
    return lut


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def _write_figure(stim_type, traces, t_ms, onsets, edges, boundary_ms,
                  bin_means, unit, out_dir, prefix, log):
    """Three-panel diagnostic figure. Returns the written path, or None.

    Panel 1 (stimulus-aligned) is the interpretive anchor: it shows where the
    detected onset falls relative to the stimulus artefact, which is the only
    way to tell an onset that is genuinely early from a response that is simply
    slow. Panels 2 and 3 are onset-aligned, matching how the numbers are
    actually computed.

    Note that averaging onset-aligned traces smears the artefact in panel 2 in
    proportion to onset jitter, and averaging stimulus-aligned traces smears the
    MEP in panel 1 for the same reason. Comparing the two panels is therefore
    informative about jitter in its own right.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                            # pragma: no cover
        log(f"{ADDON_NAME}: figures unavailable ({e}); skipping.")
        return None

    pairs = [(tr, on) for tr, on in zip(traces, onsets) if on is not None]
    if not pairs:
        return None
    rect = [np.abs(tr) for tr, _ in pairs]
    ons  = np.array([on for _, on in pairs], dtype=float)
    med  = float(np.median(ons))
    window_ms = float(edges[-1])

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7.5, 9.2))

    # ── Panel 1: stimulus-aligned, no realignment ───────────────────────────
    stim_mean = np.mean(np.vstack(rect), axis=0)
    ax0.plot(t_ms, stim_mean, lw=1.1, color="#222222")
    ax0.axvline(0.0, color="#777777", lw=1.4, ls="-")
    ax0.axvspan(med, med + boundary_ms, color="#4C72B0", alpha=0.18)
    ax0.axvspan(med + boundary_ms, med + window_ms, color="#C44E52", alpha=0.18)
    ax0.axvline(med, color="k", lw=1.0, ls="--")
    _lo = max(float(t_ms[0]), -10.0)
    _hi = min(float(t_ms[-1]), med + window_ms + 15.0)
    ax0.set_xlim(_lo, _hi)
    _top = ax0.get_ylim()[1]
    ax0.annotate("stimulus", xy=(0.0, _top), xytext=(2, -2),
                 textcoords="offset points", va="top", fontsize=8, color="#555555")
    ax0.annotate(f"median onset {med:.1f} ms", xy=(med, _top), xytext=(3, -2),
                 textcoords="offset points", va="top", fontsize=8)
    ax0.set_xlabel("Time from stimulus (ms)")
    ax0.set_ylabel(f"|EMG| ({unit})")
    ax0.set_title(f"{stim_type} — stimulus-aligned mean (n={len(pairs)})")

    # ── Panel 2: onset-aligned ──────────────────────────────────────────────
    grid = np.arange(-5.0, window_ms + 5.0, 0.1)
    stack = [np.interp(grid, t_ms - on, r) for r, on in zip(rect, ons)]
    mean_trace = np.mean(np.vstack(stack), axis=0)

    ax1.plot(grid, mean_trace, lw=1.2, color="#222222")
    ax1.axvspan(0, boundary_ms, color="#4C72B0", alpha=0.22,
                label=f"Early (0-{boundary_ms:g} ms)")
    ax1.axvspan(boundary_ms, window_ms, color="#C44E52", alpha=0.22,
                label=f"Late ({boundary_ms:g}-{window_ms:g} ms)")
    ax1.axvline(0, color="k", lw=0.8, ls="--")
    ax1.set_xlabel("Time from MEP onset (ms)")
    ax1.set_ylabel(f"|EMG| ({unit})")
    ax1.set_title("Onset-aligned mean rectified response")
    ax1.legend(fontsize=8, frameon=False)

    # ── Panel 3: bin profile ────────────────────────────────────────────────
    centres = (edges[:-1] + edges[1:]) / 2.0
    widths  = np.diff(edges) * 0.85
    colours = ["#4C72B0" if c < boundary_ms else "#C44E52" for c in centres]
    ax2.bar(centres, bin_means, width=widths, color=colours)
    ax2.set_xlabel("Time from MEP onset (ms)")
    ax2.set_ylabel(f"Area ({unit}*s)")
    ax2.set_title("Mean area per bin")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    try:
        # The shared rule: collision-free, and leaves working names alone.
        from mep_cmap.utils import filename_token
        safe = filename_token(stim_type)
    except Exception:                       # add-on used outside the package
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(stim_type))
    path = os.path.join(out_dir, f"{prefix}_{ADDON_NAME}_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(context):
    if pd is None:
        raise RuntimeError(f"{ADDON_NAME} requires pandas.")

    cfg     = context.config or {}
    log     = context.log
    # Resolved per stimulus type inside the loop below: types may be epoched
    # over different windows, and one axis applied to all of them meets traces
    # of another length.
    _t_ms_for = getattr(context, "time_ms_for", None)
    unit    = context.unit or "mV"

    bin_ms      = _as_float(cfg.get("td_bin_ms"),      2.0)
    window_ms   = _as_float(cfg.get("td_window_ms"),   20.0)
    boundary_ms = _as_float(cfg.get("td_boundary_ms"), 8.0)
    clamp       = _as_bool(cfg.get("td_clamp_to_offset"),  True)
    basecorr    = _as_bool(cfg.get("td_baseline_correct"), True)
    base_end_ms = _as_float(cfg.get("td_baseline_end_ms"), -2.0)
    want_figs   = _as_bool(cfg.get("td_figures"), True)
    only_raw    = str(cfg.get("td_stim_types") or "").strip()
    only_types  = {t.strip() for t in only_raw.split(",") if t.strip()}

    if bin_ms <= 0 or window_ms <= 0:
        raise ValueError(f"{ADDON_NAME}: bin width and window must be positive.")
    if not (0 < boundary_ms < window_ms):
        raise ValueError(
            f"{ADDON_NAME}: early/late boundary ({boundary_ms:g} ms) must fall "
            f"inside the analysis window (0-{window_ms:g} ms).")

    edges = _bin_edges(bin_ms, window_ms)
    if edges.size < 2:
        raise ValueError(f"{ADDON_NAME}: window is narrower than one bin.")
    # Snap the boundary onto a bin edge so early + late == the sum of the bins.
    # Round HALF UP in bin units: with 2 ms bins a requested 7 ms goes to 8, not
    # 6. Ties must not silently shorten the early phase. k is then constrained to
    # [1, n_bins-1] so neither phase can end up empty.
    n_bins  = edges.size - 1
    k       = int(np.floor(boundary_ms / bin_ms + 0.5))
    k       = min(max(k, 1), n_bins - 1) if n_bins >= 2 else 1
    snapped = float(edges[k])
    if abs(snapped - boundary_ms) > 1e-9:
        log(f"{ADDON_NAME}: early/late boundary {boundary_ms:g} ms snapped to "
            f"{snapped:g} ms so it coincides with a bin edge.")
    boundary_ms = snapped
    window_ms   = float(edges[-1])

    bin_cols = [_bin_label(edges[i], edges[i + 1], unit)
                for i in range(edges.size - 1)]

    # The baseline window depends on the time axis, which is per stimulus
    # type, so the mask is built inside the loop below. A type epoched with a
    # short lead-in may have too few baseline samples while another in the same
    # recording has plenty; disabling correction for the file on account of one
    # of them would discard it where it was perfectly available.

    lut = _build_marker_lookup(getattr(context, "trials", None), log)
    if not lut:
        raise ValueError(
            f"{ADDON_NAME}: could not read MEP onsets from the per-trial table. "
            f"Run the First Level analysis for this file first — this add-on "
            f"reuses the pipeline's onsets and never re-detects them.")

    file_name = None
    for v in lut.values():
        if v.get("file") is not None:
            file_name = v["file"]
            break
    if file_name is None:
        file_name = context.bids_prefix

    rows, fig_paths = [], []

    skipped_types = []
    for stim_type, stack in sorted(context.segments.items()):
        t_ms = np.asarray(_t_ms_for(stim_type) if _t_ms_for
                          else context.time_ms, dtype=float)
        t_s  = t_ms / 1000.0
        base_mask = t_ms <= base_end_ms
        _basecorr = basecorr
        if _basecorr and base_mask.sum() < 2:
            log(f"{ADDON_NAME}: '{stim_type}' has fewer than 2 samples before "
                f"{base_end_ms:g} ms; baseline correction disabled for this "
                f"stimulus type.")
            _basecorr = False
        if only_types and str(stim_type) not in only_types:
            skipped_types.append(str(stim_type))
            continue
        stack = np.asarray(stack, dtype=float)
        traces_for_fig, onsets_for_fig, bins_for_fig = [], [], []

        for i in range(stack.shape[0]):
            trace   = stack[i]
            segment = i + 1                       # 1-based, matches _trials.csv
            mk      = lut.get((str(stim_type), segment), {})
            onset   = mk.get("onset")
            offset  = mk.get("offset")

            row = {
                "File":            file_name,
                "StimType":        stim_type,
                "Segment":         segment,
                "TD_Trial_Index":  i,
                "TD_Unit":         unit,
                "TD_Onset(ms)":    onset,
                "TD_Bin_Width(ms)":     bin_ms,
                "TD_Boundary(ms)":      boundary_ms,
                "TD_Window_Requested(ms)": window_ms,
            }
            for c in bin_cols:
                row[c] = np.nan

            if onset is None:
                row.update({"TD_Valid": False,
                            "TD_Reject_Reason": "no MEP onset in per-trial table"})
                rows.append(row)
                continue

            # ── Window: onset -> onset + window, clamped ────────────────────
            t_start = onset
            t_end   = onset + window_ms
            clamped_by = ""

            if clamp and offset is not None and offset > t_start:
                if offset < t_end:
                    t_end = offset
                    clamped_by = "MEP offset"
            epoch_end = float(t_ms[-1])
            if t_end > epoch_end:
                t_end = epoch_end
                clamped_by = "epoch end" if not clamped_by else clamped_by + " + epoch end"

            if t_end <= t_start:
                row.update({"TD_Valid": False,
                            "TD_Reject_Reason": "analysis window collapsed "
                                                "(MEP offset at or before onset)"})
                rows.append(row)
                continue

            rect = np.abs(trace)
            cum  = _cumulative_integral(rect, t_s)

            base_amp = float(np.mean(rect[base_mask])) if _basecorr else 0.0

            def _corrected(a_ms, b_ms):
                """Area between two times, minus background EMG over that span."""
                b_ms = min(b_ms, t_end)
                if b_ms <= a_ms:
                    return None
                raw = _area(rect, cum, t_ms, a_ms, b_ms)
                return raw - base_amp * (b_ms - a_ms) / 1000.0

            # ── Bins (NaN when a bin is not fully inside the window) ─────────
            n_valid_bins = 0
            bin_vals = []
            for j, col in enumerate(bin_cols):
                lo = t_start + edges[j]
                hi = t_start + edges[j + 1]
                if hi <= t_end + 1e-9:
                    val = _corrected(lo, hi)
                    row[col] = val
                    bin_vals.append(val)
                    n_valid_bins += 1
                else:
                    bin_vals.append(np.nan)

            # ── Phase aggregates ────────────────────────────────────────────
            early_end = min(t_start + boundary_ms, t_end)
            early     = _corrected(t_start, early_end)
            late      = _corrected(early_end, t_end)
            early_dur = max(0.0, early_end - t_start)
            late_dur  = max(0.0, t_end - early_end)

            total = None
            if early is not None or late is not None:
                total = (early or 0.0) + (late or 0.0)

            def _per_ms(area, dur):
                return (area / (dur / 1000.0)) if (area is not None and dur > 0) else None

            row.update({
                "TD_Window_End(ms)":        round(t_end - t_start, 4),
                "TD_Clamped_By":            clamped_by,
                "TD_MEP_Offset(ms)":        offset,
                "TD_Early_Duration(ms)":    round(early_dur, 4),
                "TD_Late_Duration(ms)":     round(late_dur, 4),
                f"TD_Early_Area({unit}*s)": early,
                f"TD_Late_Area({unit}*s)":  late,
                f"TD_Total_Area({unit}*s)": total,
                # Duration-normalised: comparable even when the clamp shortens
                # the late window. Prefer these when windows differ in length.
                f"TD_Early_Mean_Amp({unit})": _per_ms(early, early_dur),
                f"TD_Late_Mean_Amp({unit})":  _per_ms(late,  late_dur),
                f"TD_Baseline_Amp({unit})":   base_amp if _basecorr else None,
                "TD_N_Bins_Valid":          n_valid_bins,
                "TD_N_Bins_Total":          len(bin_cols),
                "TD_Valid":                 True,
                "TD_Reject_Reason":         "",
            })

            # Ratios are computed on BASELINE-CORRECTED areas. This is required,
            # not cosmetic: background EMG contributes area in proportion to
            # window duration, so with an 8 ms early and 12 ms late window it
            # loads 1.5x more into the late phase. Uncorrected, Early_Fraction
            # would fall as background EMG rises — a confound that tracks
            # contraction intensity, i.e. exactly the manipulation under study.
            #
            # Correction can drive a phase to or below zero when there is no
            # detectable response above background. That is a real result, not
            # an error, so it is neither clipped nor hidden: Early_Fraction may
            # exceed 1 when the late phase sits at or below baseline, and
            # TD_Late_At_Baseline flags every such trial so they can be excluded
            # or modelled explicitly rather than silently skewing a group mean.
            row["TD_Late_At_Baseline"]  = (late is not None and late <= 0)
            row["TD_Early_At_Baseline"] = (early is not None and early <= 0)
            row["TD_Early_Fraction"]  = (early / total) if (
                early is not None and total not in (None, 0) and total > 0) else None
            row["TD_Early_Late_Ratio"] = (early / late) if (
                early is not None and late is not None and late > 0) else None

            if late_dur < (window_ms - boundary_ms) - 1e-6:
                row["TD_Reject_Reason"] = ("late window shortened by clamp — "
                                           "use duration-normalised columns")

            rows.append(row)
            traces_for_fig.append(trace)
            onsets_for_fig.append(onset)
            bins_for_fig.append(bin_vals)

        if want_figs and bins_for_fig:
            with np.errstate(invalid="ignore"):
                bin_means = np.nanmean(
                    np.array(bins_for_fig, dtype=float), axis=0)
            bin_means = np.nan_to_num(bin_means, nan=0.0)
            _fig_root = (getattr(context, "figures_dir", None)
                         or os.path.join(context.results_dir, "figures"))
            _fig_dir = os.path.join(
                _fig_root, f"{context.bids_prefix}_{ADDON_NAME}_figures")
            p = _write_figure(stim_type, traces_for_fig, t_ms, onsets_for_fig,
                              edges, boundary_ms, bin_means, unit,
                              _fig_dir, context.bids_prefix, log)
            if p:
                fig_paths.append(p)

    if not rows:
        log(f"{ADDON_NAME}: no trials found in the results bundle — nothing written.")
        return []

    key_cols  = ["File", "StimType", "Segment"]
    meta_cols = ["TD_Trial_Index", "TD_Unit", "TD_Onset(ms)", "TD_MEP_Offset(ms)",
                 "TD_Window_End(ms)", "TD_Clamped_By",
                 "TD_Early_Duration(ms)", "TD_Late_Duration(ms)"]
    val_cols  = [f"TD_Early_Area({unit}*s)", f"TD_Late_Area({unit}*s)",
                 f"TD_Total_Area({unit}*s)", f"TD_Early_Mean_Amp({unit})",
                 f"TD_Late_Mean_Amp({unit})", "TD_Early_Fraction",
                 "TD_Early_Late_Ratio", f"TD_Baseline_Amp({unit})",
                 "TD_Early_At_Baseline", "TD_Late_At_Baseline"]
    tail_cols = ["TD_Bin_Width(ms)", "TD_Boundary(ms)", "TD_Window_Requested(ms)",
                 "TD_N_Bins_Valid", "TD_N_Bins_Total", "TD_Valid", "TD_Reject_Reason"]

    df = pd.DataFrame(rows)
    ordered = key_cols + meta_cols + val_cols + bin_cols + tail_cols
    df = df.reindex(columns=ordered + [c for c in df.columns if c not in ordered])
    df = df.sort_values(["StimType", "Segment"]).reset_index(drop=True)

    # addons_dir creates itself on access, so no makedirs is needed here.
    out_path = os.path.join(context.addons_dir,
                            f"{context.bids_prefix}_{ADDON_NAME}.csv")
    df.to_csv(out_path, index=False)

    n_ok = int(df["TD_Valid"].sum()) if "TD_Valid" in df else 0
    n_clamped = int((df.get("TD_Clamped_By", pd.Series(dtype=str))
                     .fillna("") != "").sum())
    log(f"{ADDON_NAME}: {n_ok}/{len(df)} trials quantified "
        f"({n_clamped} window(s) clamped); early/late boundary {boundary_ms:g} ms, "
        f"{len(bin_cols)} x {bin_ms:g} ms bins.")
    if n_ok < len(df):
        log(f"{ADDON_NAME}: {len(df) - n_ok} trial(s) skipped — see TD_Reject_Reason.")
    if skipped_types:
        log(f"{ADDON_NAME}: not analysed (excluded by 'Only these stim types'): "
            + ", ".join(sorted(skipped_types)))

    return [out_path] + fig_paths
