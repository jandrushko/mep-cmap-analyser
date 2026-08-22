"""
mep_cmap.detection.csp_detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cortical silent period detection on the RMS envelope.

  • detect_csp_bootstrap

Relationship to envelope_stats
------------------------------
This detector predates ``envelope_stats`` and carried its own copies of the
envelope, the baseline statistics and the chance run-length bootstrap. All
three now come from that module, which corrects three defects the local
copies had:

  * The envelope was built with a ZERO-PADDED convolution (``mode='same'``).
    The pre-stimulus window sits at the start of the segment, so the padding
    contaminated exactly the samples the baseline mean and SD are computed
    from. ``compute_rms_envelope`` pads by reflection instead, which
    reproduces the local variance and keeps the baseline honest.
  * The run-length bootstrap resampled sample by sample. A moving-window RMS
    envelope is autocorrelated by construction, so an i.i.d. resample reports
    chance runs of one or two samples and the stated significance level was
    not the one being applied. ``bootstrap_runlength_criterion`` uses a
    circular block bootstrap, which is the corrected form.
  * Run lengths were extracted with a uint8 wraparound trick inside a Python
    loop over every bootstrap iteration.

Offset definition
-----------------
The silent period ends when voluntary EMG RETURNS AND STAYS BACK, not at the
first sample that crosses back over the threshold. ``min_return_ms`` sets how
long the envelope must remain above threshold before a return counts.

This is the part that matters clinically. Breakthrough EMG, a brief burst part
way through an otherwise complete silent period, is common during sustained
contraction. The previous implementation collected every qualifying suppressed
epoch and returned ``valid_epochs[0]``, so a burst split one silent period in
two and the reported duration was only the part before the burst.
``min_return_ms`` was accepted as an argument, carried on ``PipelineConfig``,
exposed in the interface and passed by every caller -- and never read.
"""

import numpy as np

from dataclasses import dataclass

from .envelope_stats import (bootstrap_runlength_criterion,
                             compute_rms_envelope,
                             find_sustained_run)

#: Floor on the suppression threshold, as a fraction of the pre-stimulus
#: envelope mean. Without it a very steady baseline gives a small SD and hence
#: a threshold so close to the baseline that ordinary variation reads as
#: suppression. It is a keyword argument rather than a module constant because
#: on low-force contractions with a noisy baseline it becomes the binding
#: constraint rather than a safety net, and an analyst who cannot see it has no
#: way to know that the value they set for ``criterion`` stopped mattering.
DEFAULT_MIN_THRESHOLD_FRAC = 0.5

#: Fraction of the return window that must be above threshold for the silent
#: period to be considered over. See ``_first_sustained_return``.
DEFAULT_RETURN_DUTY = 0.75


def _first_sustained_return(above, lo, min_run, duty):
    """
    First index at or after ``lo`` that is above threshold AND begins a window
    of ``min_run`` samples at least ``duty`` of which are above threshold.

    Why a duty fraction rather than an unbroken run
    -----------------------------------------------
    Requiring every sample of the return window to exceed the threshold biases
    the offset LATE, and badly. The threshold sits at mu - criterion*sd of the
    baseline envelope, so once EMG is back roughly 2.5% of samples fall below
    it by chance. Over an 80-sample window (40 ms at 2 kHz) an unbroken run
    survives with probability 0.975**80 ~= 0.13, so the detector discards
    several genuine returns before one happens to be clean.

    Measured on the synthetic trial this was developed against: a silent
    period whose true offset was 180 ms was reported at 211.5 ms by the
    unbroken-run form and at 185.5 ms with a duty fraction, the remaining
    5.5 ms being the envelope window's own smearing.

    A duty fraction still rejects breakthrough EMG, which is what ``min_run``
    exists for. An 8 ms burst occupies a fifth of a 40 ms window and cannot
    reach any sensible duty.
    """
    above = np.asarray(above, dtype=bool)
    n = above.size
    lo = max(0, int(lo))
    if lo >= n:
        return None

    if min_run <= 1:
        hit = np.flatnonzero(above[lo:])
        return int(lo + hit[0]) if hit.size else None

    if min_run > n - lo:
        # Not enough room left for a qualifying return to be observed at all.
        return None

    kernel = np.ones(int(min_run), dtype=float) / float(min_run)
    frac = np.convolve(above.astype(float), kernel, mode="valid")
    cand = np.flatnonzero(frac >= float(duty))
    cand = cand[cand >= lo]
    if cand.size:
        cand = cand[above[cand]]
    return int(cand[0]) if cand.size else None



def detect_csp_bootstrap(
        emg_seg, fs, time_axis, *,
        pre_ms=100, search_start_ms=40, search_end_ms=400,
        min_silence_ms=25, min_return_ms=40,
        max_onset_ms=None,
        criterion=1.96, significance=0.99,
        n_boot=1000, rms_window_ms=10, seed=42,
        min_threshold_frac=DEFAULT_MIN_THRESHOLD_FRAC,
        return_duty=DEFAULT_RETURN_DUTY,
        reason_out=None):
    """
    Per-trial cortical silent period detector.

    The threshold is applied to the RMS envelope rather than to raw samples.
    On single trials voluntary EMG oscillates rapidly, so the raw sample SD is
    comparable to the mean; the envelope averages over oscillation cycles and
    gives a stable suppression threshold.

    Algorithm
    ---------
    1.  RMS envelope over ``rms_window_ms``.
    2.  Normalise by the pre-stimulus envelope mean, so baseline ~= 1.0 and
        the threshold reads as a fraction of background EMG.
    3.  Suppression threshold = max(mu - criterion*sd, min_threshold_frac).
    4.  Chance-calibrated minimum run length from a circular block bootstrap
        of the pre-stimulus envelope, floored at ``min_silence_ms``.
    5.  ONSET  = start of the first run below threshold that is at least that
        long.
    6.  OFFSET = start of the first sustained return of EMG at or after the
        onset: a window of ``min_return_ms`` that is at least ``return_duty``
        above threshold. Shorter excursions are breakthrough EMG and do not
        end the silent period.

    Parameters
    ----------
    emg_seg            : 1-D EMG segment
    fs                 : sampling frequency in Hz
    time_axis          : time axis in ms, same length as emg_seg
    pre_ms             : pre-stimulus baseline duration to use (ms)
    search_start_ms    : start of the cSP search window (ms post-stim)
    search_end_ms      : end of the cSP search window (ms post-stim)
    min_silence_ms     : minimum suppression duration to qualify (ms)
    min_return_ms      : EMG must stay back this long to end the cSP (ms)
    max_onset_ms       : latest time (ms) at which the cSP may begin. A
                         suppression starting after this is not the silent
                         period following THIS response, so the trial is
                         rejected rather than measured. None disables the
                         check. This is a bound on the ONSET, never on the
                         search window -- capping the window instead would
                         truncate the duration, which is the measurement.
    criterion          : SD multiplier for the suppression threshold
    significance       : bootstrap percentile for the run-length criterion
    n_boot             : bootstrap iterations
    rms_window_ms      : RMS envelope window (ms); also the bootstrap block
    seed               : RNG seed, so detection is reproducible
    min_threshold_frac : floor on the threshold, as a fraction of baseline
    return_duty        : fraction of the return window that must be above
                         threshold for the silent period to be over
    reason_out         : optional list; a success/failure message is appended

    Returns
    -------
    (start_idx, end_idx) as indices into ``emg_seg``, or None.
    """

    def _fail(msg):
        if reason_out is not None:
            reason_out.append(msg)
        return None

    emg_seg = np.asarray(emg_seg, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    if emg_seg.size != time_axis.size:
        return _fail("Segment and time axis lengths differ")
    if emg_seg.size == 0:
        return _fail("Empty segment")

    # Reflection-padded envelope. The window width doubles as the block length
    # for the bootstrap below: adjacent envelope samples share all but one
    # input sample, so the block must span the window for the resample to
    # preserve that dependence.
    rms_win = max(1, int(round(rms_window_ms * fs / 1000.0)))
    smooth = compute_rms_envelope(emg_seg, fs, window_ms=rms_window_ms)

    prestim_mask = (time_axis >= -pre_ms) & (time_axis < 0.0)
    if int(prestim_mask.sum()) < 10:
        return _fail("Too few pre-stim samples - increase pre-stim window")

    pre_env = smooth[prestim_mask]
    pre_mean = float(pre_env.mean())
    if not np.isfinite(pre_mean) or pre_mean < 1e-12:
        return _fail("Pre-stim signal is flat - no valid baseline")

    norm_env = smooth / pre_mean
    norm_pre = pre_env / pre_mean
    base_mu = float(norm_pre.mean())
    base_sd = float(norm_pre.std(ddof=1)) if norm_pre.size > 1 else 1e-6
    base_sd = max(base_sd, 1e-9)

    suppress_thresh = max(base_mu - criterion * base_sd,
                          float(min_threshold_frac))

    # Chance-calibrated minimum silence, floored at the analyst's
    # min_silence_ms. The floor normally dominates, which is intended: it is a
    # physiological statement about what counts as a silent period, and the
    # bootstrap is the statistical guard underneath it.
    min_sil_samp = max(2, int(round(min_silence_ms * fs / 1000.0)))
    criterion_samples = bootstrap_runlength_criterion(
        norm_pre, criterion=criterion, significance=significance,
        n_boot=n_boot, seed=seed, tail="lower",
        min_samples=min_sil_samp, block_samples=rms_win)

    si = int(np.searchsorted(time_axis,
                             max(search_start_ms, float(time_axis[0]))))
    ei = int(np.searchsorted(time_axis,
                             min(search_end_ms, float(time_axis[-1]))))
    if si >= ei:
        return _fail("Search window empty - check Search start/end settings")

    search_env = norm_env[si:ei]

    # ── Onset: first sustained suppression ────────────────────────────────
    onset_local = find_sustained_run(search_env, suppress_thresh,
                                     criterion_samples, above=False)
    if onset_local is None:
        return _fail(f"No suppression >= {min_silence_ms:g} ms found in "
                     f"search window")

    # The suppression has to belong to THIS response. One that begins long
    # after the MEP is something else -- a pause in the contraction, a
    # movement artefact settling -- and measuring it would report a silent
    # period the stimulus did not cause.
    onset_ms = float(time_axis[int(si + onset_local)])
    if max_onset_ms is not None and onset_ms > float(max_onset_ms):
        return _fail(f"Suppression begins at {onset_ms:.0f} ms, later than the "
                     f"{float(max_onset_ms):.0f} ms limit set by 'Max offset "
                     f"from MEP 2nd peak' - not treated as the silent period "
                     f"for this response")

    # ── Offset: first SUSTAINED return of EMG ─────────────────────────────
    # Anchored at the onset, so an excursion above threshold before the silent
    # period began cannot be mistaken for its end. A run shorter than
    # min_return_ms is breakthrough EMG and is stepped over.
    #
    # A return criterion shorter than the envelope window is unenforceable. A
    # moving-window RMS cannot rise and fall faster than its own window, so
    # asking it to confirm a return over fewer samples than that window spans
    # is asking a question the signal cannot answer: every value below the
    # window behaves alike, and the setting looks broken.
    #
    # This is easy to hit at high sampling rates, where the two are set in ms
    # but compared in samples. At 5 kHz a 10 ms window is 50 samples and a
    # 2 ms return criterion is 10, so 2 ms, 5 ms and 40 ms all gave the same
    # answer. Raised to the window and SAID so, rather than silently honoured
    # or silently ignored.
    min_ret_samp = max(1, int(round(min_return_ms * fs / 1000.0)))
    clamped_return_ms = None
    if min_ret_samp < rms_win:
        clamped_return_ms = rms_win * 1000.0 / fs
        min_ret_samp = rms_win

    offset_local = _first_sustained_return(
        search_env > suppress_thresh, onset_local, min_ret_samp, return_duty)

    # No qualifying return inside the window. The silent period did not end
    # where the search did -- it was not observed to end at all -- so the
    # duration is reported as a lower bound rather than as a measurement that
    # happens to equal the window width.
    truncated = offset_local is None
    if truncated:
        offset_local = search_env.size

    csp_start_idx = int(si + onset_local)
    csp_end_idx = int(min(si + offset_local, smooth.size - 1))

    dur_ms = (csp_end_idx - csp_start_idx) * 1000.0 / fs
    if dur_ms < min_silence_ms:
        return _fail(f"Detected epoch too short (< {min_silence_ms:g} ms)")

    if reason_out is not None:
        t_on = float(time_axis[csp_start_idx])
        if truncated:
            reason_out.append(
                f"Detected - onset ~{t_on:.0f} ms, duration >= {dur_ms:.0f} ms "
                f"(EMG had not returned for {min_return_ms:g} ms before the "
                f"end of the search window, so the offset is a lower bound - "
                f"consider raising Search end)")
        else:
            reason_out.append(f"Detected - onset ~{t_on:.0f} ms, "
                              f"duration ~{dur_ms:.0f} ms")
        if clamped_return_ms is not None:
            reason_out.append(
                f"Min return was raised from {min_return_ms:g} ms to "
                f"{clamped_return_ms:g} ms: a return cannot be confirmed over "
                f"less than the {rms_window_ms:g} ms RMS window, so any "
                f"shorter value behaves identically. Set Min return to at "
                f"least the RMS window for it to have an effect.")

    return csp_start_idx, csp_end_idx


# ── One entry point for both callers ─────────────────────────────────────────
#
# The pipeline and the Data Inspector each used to build this detector's
# arguments themselves, from twenty-odd lines of near-identical code. They
# drifted, and the drift was invisible because both sides looked reasonable in
# isolation:
#
#   * the inspector capped search_end_ms at second_peak + max_mep_offset_ms.
#     With that field at its default the search window was ~100 ms wide, so no
#     silent period longer than ~100 ms could be found AT ALL and every
#     reviewed trial came back truncated -- while the pipeline, which never
#     applied the cap, reported the real duration for the same trial. A comment
#     above the inspector's copy stated that the cap was not applied there.
#   * rms_window_ms was passed by the inspector and was not a field on
#     PipelineConfig at all, so the analysis silently used the function default
#     whatever the interface said.
#
# A review tool that disagrees with the analysis it is reviewing is worse than
# no review tool. So neither caller computes these arguments any more: both
# call detect_csp_for_trial, and there is one place where the mapping from
# user settings to detector arguments is written down.

@dataclass(frozen=True)
class CspSettings:
    """
    The user-facing cortical silent period settings, as one value.

    Frozen so a caller cannot adjust a field for its own use and hand a
    different configuration to the detector than the one the analysis ran.
    """
    min_silence_ms:     float = 25.0
    min_return_ms:      float = 40.0
    criterion:          float = 1.96
    significance:       float = 0.99
    n_boot:             int   = 1000
    search_end_ms:      float = 400.0
    max_mep_offset_ms:  float = 100.0
    rms_window_ms:      float = 10.0
    seed:               int   = 42
    min_threshold_frac: float = DEFAULT_MIN_THRESHOLD_FRAC
    return_duty:        float = DEFAULT_RETURN_DUTY

    _FIELDS = ("min_silence_ms", "min_return_ms", "criterion", "significance",
               "n_boot", "search_end_ms", "max_mep_offset_ms",
               "rms_window_ms", "seed", "min_threshold_frac", "return_duty")

    @classmethod
    def from_source(cls, src, prefix="csp_"):
        """
        Read settings off a PipelineConfig, a saved-session dict, or anything
        else exposing ``csp_<field>``.

        Missing fields keep the class default rather than raising, so a session
        file written before a field existed still loads. Fields are read by
        NAME from a single list, so adding one to the dataclass is enough to
        have it carried by every caller.
        """
        get = (src.get if hasattr(src, "get")
               else lambda k, d=None: getattr(src, k, d))
        vals = {}
        for name in cls._FIELDS:
            # rms_window_ms and seed have no csp_ prefix on some sources.
            for key in (f"{prefix}{name}", name):
                v = get(key, None)
                if v is not None:
                    vals[name] = v
                    break
        return cls(**vals)


def detect_csp_for_trial(emg_seg, fs, time_axis, settings, *,
                         second_peak_ms, pre_ms, reason_out=None):
    """
    Detect the cortical silent period for one trial.

    THE entry point. Both the pipeline and the Data Inspector call this, so
    what the analyst sees while reviewing a trial is what the analysis did to
    it, by construction rather than by two implementations agreeing.

    Parameters
    ----------
    emg_seg        : 1-D EMG segment
    fs             : sampling frequency in Hz
    time_axis      : time axis in ms, same length as emg_seg
    settings       : CspSettings
    second_peak_ms : time of the later of the two PTP peaks (ms). The search
                     starts here, so the detector can never place the cSP
                     onset inside the MEP.
    pre_ms         : pre-stimulus baseline duration to use (ms)
    reason_out     : optional list; a success/failure message is appended

    Returns
    -------
    (start_idx, end_idx) into ``emg_seg``, or None.
    """
    second_peak_ms = float(second_peak_ms)

    # The search END is the analyst's search_end_ms, bounded only by the
    # segment. max_mep_offset_ms bounds the ONSET (see detect_csp_bootstrap),
    # because a cap on the window would truncate the duration being measured.
    search_end_ms = min(float(settings.search_end_ms),
                        float(np.asarray(time_axis)[-1]))

    max_onset_ms = (None if settings.max_mep_offset_ms is None
                    else second_peak_ms + float(settings.max_mep_offset_ms))

    return detect_csp_bootstrap(
        emg_seg, fs, time_axis,
        pre_ms=pre_ms,
        search_start_ms=second_peak_ms,
        search_end_ms=search_end_ms,
        max_onset_ms=max_onset_ms,
        min_silence_ms=settings.min_silence_ms,
        min_return_ms=settings.min_return_ms,
        criterion=settings.criterion,
        significance=settings.significance,
        n_boot=settings.n_boot,
        rms_window_ms=settings.rms_window_ms,
        seed=settings.seed,
        min_threshold_frac=settings.min_threshold_frac,
        return_duty=settings.return_duty,
        reason_out=reason_out,
    )
