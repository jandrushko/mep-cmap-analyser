"""
A silent period whose end was never observed.

When the EMG does not return before the end of the search window, the
detector has found a LOWER BOUND, not a duration. It says so internally and
always has; what it could not do was tell the caller, so the pipeline wrote
the bound into cSP_Duration(ms) as though it had been measured.

On a real study that produced 386 ms values against a 400 ms window, on 210
trials across 28 files, indistinguishable from a genuine 386 ms silent
period. On M-wave recordings -- which have no cortical silent period at all --
it happened on nearly every trial, so the column's mean described the width of
the search window rather than anything about the muscle.
"""

import numpy as np
import pytest

from mep_cmap.detection.csp_detection import (CspSettings,
                                              detect_csp_for_trial)

FS = 2000.0
PRE_MS = 100.0
POST_MS = 400.0


def _trial(return_at_ms=None):
    """Background EMG, suppressed from 40 ms, returning only if asked."""
    n = int((PRE_MS + POST_MS) * FS / 1000.0)
    t = np.linspace(-PRE_MS, POST_MS, n, endpoint=False)
    rng = np.random.default_rng(0)
    sig = rng.normal(0.0, 1.0, n)          # tonic contraction
    silent = t >= 40.0
    if return_at_ms is not None:
        silent &= t < return_at_ms
    sig[silent] *= 0.01                    # suppression
    return sig, t


def _detect(return_at_ms):
    sig, t = _trial(return_at_ms)
    trunc = []
    got = detect_csp_for_trial(
        sig, FS, t, CspSettings(),
        second_peak_ms=25.0, pre_ms=PRE_MS, truncated_out=trunc)
    return got, trunc


def test_a_silent_period_that_ends_is_not_truncated():
    got, trunc = _detect(return_at_ms=180.0)
    assert got is not None
    assert trunc == [], "a measured silent period must not be flagged"


def test_emg_that_never_returns_is_flagged():
    """
    The detector still returns indices -- the onset is real and worth showing
    in the Inspector -- but the caller is told the end was never seen.
    """
    got, trunc = _detect(return_at_ms=None)
    assert got is not None
    assert trunc == [True]


def test_the_flag_is_opt_in():
    """Callers that do not pass a list are unaffected."""
    sig, t = _trial(return_at_ms=None)
    assert detect_csp_for_trial(sig, FS, t, CspSettings(),
                                second_peak_ms=25.0, pre_ms=PRE_MS) is not None


def test_a_truncated_duration_would_have_been_the_window_width():
    """
    Why this matters, made concrete: without the flag the reported duration is
    the distance to the end of the search, which looks like a plausible number
    and is not one.
    """
    got, trunc = _detect(return_at_ms=None)
    start, end = got
    sig, t = _trial(return_at_ms=None)
    duration_ms = (end - start) * 1000.0 / FS
    assert trunc == [True]
    assert duration_ms > 250, ("the bound should be large, which is exactly "
                               "why it must not be reported as a measurement")


# ── the invariant that holds whatever csp_types contains ─────────────────────

def test_the_pipeline_writes_a_duration_only_when_not_truncated():
    """
    THE guard, and the reason the csp_types question is not urgent.

    A recording can end up listed as a cSP type without any session recording
    it: on the study this was built against, every M-wave whose session held
    manually placed cSP markers came back with csp_types=['A'], while the four
    with no markers came back empty. Where that inference happens has not been
    found, and until it is, this is what stops it doing harm.

    An M-wave has no cortical silent period, so the search never finds a
    return, so every unmarked trial is truncated -- and with this guard it
    contributes nothing rather than the width of the search window. The
    inference may still be wrong, but it can no longer fabricate a number.
    """
    import inspect

    from mep_cmap import pipeline

    src = inspect.getsource(pipeline)
    i = src.index("if stim_type in cfg.csp_types")
    body = src[i:i + 1600]

    assert "truncated_out=_trunc" in body, (
        "the pipeline no longer asks the detector whether the silent period "
        "was observed to end")
    assert "if _csp_auto is not None and not _trunc:" in body, (
        "a truncated detection can reach cSP_Duration(ms), where it is "
        "indistinguishable from a measured one")


def test_a_stored_marker_is_unaffected_by_the_guard():
    """
    Deliberately narrow. The guard covers DETECTION only; a marker the analyst
    placed is a decision and is read from segments_metadata further up, so
    those trials still report their duration on an M-wave or anywhere else.

    This is why removing the inference is a separate question rather than a
    fix: it would change which trials are measured, not whether a fabricated
    number can appear.
    """
    import inspect

    from mep_cmap import pipeline

    src = inspect.getsource(pipeline)
    stored = src.index('if mk in segments_metadata and "silent_start_idx"')
    guarded = src.index("if _csp_auto is not None and not _trunc:")
    assert stored < guarded, (
        "the stored-marker branch must come first, so detection fills gaps "
        "rather than overruling a decision")
