"""
mep_cmap.detection.artefact
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Keeping onset detection out of the stimulus artefact.

The onset search starts at a fixed floor (pipeline.ARTEFACT_FLOOR_MS, 2 ms)
or the stimulus type's minimum latency. A peripheral-nerve artefact can last
longer than that: on an FDI M-wave recruitment curve (5 kHz, 20-450 Hz) it
ran to about 3 ms and then drifted back towards baseline for several more.
Detectors that compare the signal with the pre-stimulus baseline (RMS
envelope, CUSUM) saw that drift as a response from the first sample of the
window and returned the window start on EVERY trial; on trials below motor
threshold, and on small M-waves near it, they outvoted the detectors that
were right, and the methods-median reported the window edge as the onset.

Two rules, applied wherever an onset is detected (dispatch_onset and
compute_onset_agreement, so the run, the Inspector and the preview agree):

``settled_search_start``
    Start the search where the trace has stopped MOVING: the first point,
    from the nominal start, at which the sample-to-sample change has stayed
    within the pre-stimulus noise for ``hold_ms``. Slope rather than level,
    because the artefact's tail flattens at an offset from baseline before it
    returns. Searched only up to the largest deflection in the window, so a
    response that begins before the artefact has settled leaves the start
    where it was rather than pushing it past the response.

``on_window_edge``
    A latency on the first sample of the search (within a quarter of a
    millisecond, or one and a half samples) is not an onset: the signal was
    already above the detector's threshold when the search began, so the
    detector cannot show the response started there. The window edge reported
    as a latency, with implausibly little spread across trials, is the
    signature pipeline.onset_search_window warns about.
"""

import numpy as np

# Pre-stimulus guard: the last stretch before the pulse can hold the leading
# edge of the artefact or a conditioning pulse; it is not baseline noise.
_GUARD_MS = 2.0

# Searches starting at or after this many ms post-stimulus are past any
# stimulus artefact: the artefact rules never apply there (artefact_at_start).
# Peripheral-nerve profiles start at 2-4 ms; TMS profiles at 8 ms or later.
ARTEFACT_WINDOW_MS = 8.0


def settled_search_start(signal, fs, pre_ms, search_start_ms, search_end_ms,
                         *, k=3.0, hold_ms=0.6, artefact_mult=3.0,
                         level_frac=0.2):
    """The onset search start, moved past any stimulus artefact still settling.

    Parameters
    ----------
    signal          : 1-D array, the trace (pre-stimulus + post-stimulus)
    fs              : sampling rate, Hz
    pre_ms          : ms of pre-stimulus data in ``signal`` (where the pulse is)
    search_start_ms : nominal search start, ms after the stimulus
    search_end_ms   : search end, ms after the stimulus
    k               : settled = |slope| within ``k`` robust SDs of the
                      pre-stimulus slope
    hold_ms         : for at least this long

    Returns ``search_start_ms`` or a later time; never earlier, and unchanged
    whenever the trace is too short, the baseline is unusable, or no settled
    stretch precedes the largest deflection.
    """
    try:
        x = np.asarray(signal, dtype=float).ravel()
        fs = float(fs)
        s0 = int(round(float(pre_ms) * fs / 1000.0))
        i0 = s0 + int(round(float(search_start_ms) * fs / 1000.0))
        i1 = min(len(x), s0 + int(round(float(search_end_ms) * fs / 1000.0)))
    except (TypeError, ValueError):
        return search_start_ms
    guard = int(round(_GUARD_MS * fs / 1000.0))
    hold = max(2, int(round(hold_ms * fs / 1000.0)))
    if s0 - guard < 10 or i0 < 1 or i1 - i0 <= 2 * hold:
        return search_start_ms
    base = x[:s0 - guard]
    d_base = np.diff(base)
    mad = np.median(np.abs(d_base - np.median(d_base)))
    sd = 1.4826 * mad if mad > 0 else float(np.std(d_base))
    if not np.isfinite(sd) or sd <= 0:
        return search_start_ms
    thr = k * sd
    level = float(np.median(base))
    # Only up to the largest deflection: settling is looked for BEFORE the
    # response, never after it.
    cap = i0 + int(np.argmax(np.abs(x[i0:i1] - level)))
    if cap - i0 < 2:
        return search_start_ms
    # Slope averaged over ``hold_ms``. A strict run of calm samples was tried
    # first and never fired on real FDI M-waves: the artefact's tail keeps
    # drifting at just above the noise slope and dips below it for only a few
    # samples before the M-wave rises. The average falls through the
    # threshold where the drift ends.
    d = np.abs(np.diff(x[i0 - 1:cap + 1]))          # d[j] = |x[i0+j] - x[i0+j-1]|
    w = max(3, int(round(hold_ms * fs / 1000.0)))
    sm = np.convolve(d, np.ones(w) / w, mode="same")
    # Act only when an artefact is plainly still settling at the nominal
    # start: a slope several times the settled threshold, far beyond
    # background EMG. Without this, a TMS trace only slightly short of still
    # at a 10 ms start had its search moved up to the MEP's first dip, and its
    # onset reported a phase late.
    if sm[0] <= artefact_mult * thr:
        return search_start_ms
    # Settled = quiet AND near the baseline level. Quiet alone also describes
    # every peak and trough of the response itself: on a large-artefact
    # M-wave, slope-only settling latched on to the M-wave's first trough and
    # put the onset after it.
    maxdev = float(np.max(np.abs(x[i0:i1] - level)))
    near = np.abs(x[i0:cap + 1] - level) <= level_frac * maxdev
    below = np.nonzero((sm <= thr) & near[:len(sm)])[0]
    if below.size == 0:
        return search_start_ms
    start = i0 + int(below[0])
    if start <= i0 or start >= cap:
        return search_start_ms
    return round((start - s0) * 1000.0 / fs, 4)


def edge_tolerance_ms(fs):
    """How close to the search's lower bound counts as ON it."""
    try:
        return max(0.25, 1.5 * 1000.0 / float(fs))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.25


def on_window_edge(latency_ms, lower_ms, fs):
    """True if ``latency_ms`` sits on the search's lower bound."""
    if latency_ms is None or lower_ms is None:
        return False
    try:
        return float(latency_ms) <= float(lower_ms) + edge_tolerance_ms(fs)
    except (TypeError, ValueError):
        return False


def lower_bound_ms(search_start_ms, min_latency_ms):
    """The earliest latency a detector could report: start or physiological floor."""
    vals = [v for v in (search_start_ms, min_latency_ms) if v is not None]
    try:
        return max(float(v) for v in vals) if vals else None
    except (TypeError, ValueError):
        return None


def plateau_corrected(signal, fs, pre_ms, search_start_ms, *, k=5.0,
                      win_ms=0.5, calm_mult=3.0, min_frac=0.10):
    """The trace with a post-artefact plateau moved to the baseline level.

    After a large peripheral-nerve artefact the trace can sit on a FLAT
    stretch well away from the pre-stimulus baseline before the M-wave starts
    (lower-limb Mmax: about -1.4 mV from 3 to 6 ms). Detectors that compare
    the signal with the pre-stimulus baseline (RMS envelope, CUSUM, peak
    fraction) read that offset as a response from the first sample of the
    search, so they reported the plateau as the onset, and the M-wave's real
    start a few milliseconds later was never seen.

    When the search starts on such a plateau -- level more than ``k`` robust
    SDs from baseline, slope calm -- a copy is returned with everything from
    the stimulus onward shifted by the plateau's offset, so detectors measure
    departure FROM the plateau. The original is never modified, and amplitude
    is not measured on the copy. Otherwise the signal is returned unchanged.
    """
    try:
        x = np.asarray(signal, dtype=float).ravel()
        fs = float(fs)
        s0 = int(round(float(pre_ms) * fs / 1000.0))
        i0 = s0 + int(round(float(search_start_ms) * fs / 1000.0))
    except (TypeError, ValueError):
        return signal
    guard = int(round(_GUARD_MS * fs / 1000.0))
    w = max(3, int(round(win_ms * fs / 1000.0)))
    if s0 - guard < 10 or i0 < 1 or i0 + w + 1 >= len(x):
        return signal
    base = x[:s0 - guard]
    level = float(np.median(base))
    mad = np.median(np.abs(base - level))
    sd = 1.4826 * mad if mad > 0 else float(np.std(base))
    d_base = np.diff(base)
    dmad = np.median(np.abs(d_base - np.median(d_base)))
    dsd = 1.4826 * dmad if dmad > 0 else float(np.std(d_base))
    if not (np.isfinite(sd) and sd > 0 and np.isfinite(dsd) and dsd > 0):
        return signal
    seg = x[i0:i0 + w]
    offset = float(np.mean(seg)) - level
    calm = float(np.mean(np.abs(np.diff(x[i0 - 1:i0 + w])))) <= calm_mult * 3.0 * dsd
    # Large relative to the trial, not only to the baseline noise. A resting
    # TMS trace sits a few noise-SDs off baseline at a 10 ms start but that is
    # 1-4 % of its MEP; shifting it changed which TMS onsets the members found.
    # A post-artefact plateau is 13-35 % of its M-wave.
    biggest = float(np.max(np.abs(x[i0:] - level))) if len(x) > i0 else 0.0
    if (abs(offset) <= k * sd or not calm
            or abs(offset) < min_frac * biggest):
        return signal
    # A plateau, not a tail still recovering. Shifting a decaying tail by its
    # level at the search start made its continued recovery look like a
    # rising response, and three detectors reported it as the onset. The
    # level over the next ``win_ms`` must stay within 15 % of the offset: a
    # flat lower-limb Mmax plateau does; an exponential recovery (time
    # constant ~2.5 ms: ~18 % per half millisecond) does not. Half a
    # millisecond, not one: an M-wave can begin ~1.4 ms after the search
    # start, and a longer look-ahead would run into it.
    if i0 + 2 * w > len(x):
        return signal
    nxt = float(np.mean(x[i0 + w:i0 + 2 * w])) - level
    if abs(nxt - offset) > 0.15 * abs(offset):
        return signal
    y = x.copy()
    y[s0:] -= offset
    return y


def artefact_at_start(signal, fs, pre_ms, search_start_ms, *, k=3.0,
                      hold_ms=0.6, artefact_mult=3.0, level_k=5.0,
                      min_frac=0.10):
    """True if a stimulus artefact is still present where the search starts.

    Either the trace is still moving steeply there (slope more than
    ``artefact_mult`` times the settled threshold of ``settled_search_start``,
    far beyond background EMG) or it sits well off baseline -- a flat plateau
    or a tail still recovering -- by more than ``level_k`` noise SDs and a
    tenth of the trial's largest deflection.

    Every artefact rule -- settled start, plateau correction, window edge and
    the majority quorum -- applies only when this is True. Applied to every
    trial they blanked clear TMS MEPs that one or two detectors had found
    (a 1.2 mV MEP among them) and moved others the wrong way; a trace that
    starts quiet near baseline, the ordinary TMS case, is analysed exactly as
    before.
    """
    try:
        x = np.asarray(signal, dtype=float).ravel()
        fs = float(fs)
        # A stimulus artefact lasts a few milliseconds. A search that starts
        # at or after ARTEFACT_WINDOW_MS (every TMS profile: 8 ms and later)
        # cannot be inside one; there the plateau test fired on three small,
        # noisy MEPs and moved one of them several milliseconds the wrong way.
        if float(search_start_ms) >= ARTEFACT_WINDOW_MS:
            return False
        s0 = int(round(float(pre_ms) * fs / 1000.0))
        i0 = s0 + int(round(float(search_start_ms) * fs / 1000.0))
    except (TypeError, ValueError):
        return False
    guard = int(round(_GUARD_MS * fs / 1000.0))
    w = max(3, int(round(hold_ms * fs / 1000.0)))
    if s0 - guard < 10 or i0 < 1 or i0 + w + 1 >= len(x):
        return False
    d_base = np.diff(x[:s0 - guard])
    mad = np.median(np.abs(d_base - np.median(d_base)))
    sd = 1.4826 * mad if mad > 0 else float(np.std(d_base))
    if np.isfinite(sd) and sd > 0:
        slope0 = float(np.mean(np.abs(np.diff(x[i0 - 1:i0 + w]))))
        if slope0 > artefact_mult * k * sd:
            return True
    # Or still well off baseline, flat or not. Gating on "flat plateau" alone
    # missed a tail still recovering -- neither steep nor flat -- and sent it
    # down the old path, where it was reported as the onset. Off baseline by
    # more than ``level_k`` noise SDs AND a tenth of the trial's largest
    # deflection (as plateau_corrected requires), so background EMG at a
    # small response is not mistaken for one.
    base = x[:s0 - guard]
    level = float(np.median(base))
    lmad = np.median(np.abs(base - level))
    lsd = 1.4826 * lmad if lmad > 0 else float(np.std(base))
    if not (np.isfinite(lsd) and lsd > 0):
        return False
    offset = float(np.mean(x[i0:i0 + w])) - level
    biggest = float(np.max(np.abs(x[i0:] - level)))
    return abs(offset) > level_k * lsd and abs(offset) >= min_frac * biggest
