"""
AUC must close its window from whatever found the end of the response.

The end of the MEP is the same physical event whether it was found as a
return of EMG to baseline (at rest) or as the start of a silent period
(during contraction); resolve_mep_offset reports one offset with a source
beside it precisely because of that.

The AUC branch used to accept only source == "envelope", on the assumption
that a "csp_start" offset always arrived with stored Inspector metadata and
was handled by an earlier branch. That stopped being true once the pipeline
began detecting silent periods for unreviewed trials: those carry a
csp_start offset and no stored markers, matched no branch, and lost their AUC
silently. Eleven of twenty B trials on one recording came out blank while
carrying a perfectly good latency and offset.
"""

import inspect

from mep_cmap import pipeline


def _auc_chain():
    src = inspect.getsource(pipeline)
    i = src.index("auc_val = None")
    return src[i:i + 3500]


def test_a_csp_start_offset_can_close_the_auc_window():
    chain = _auc_chain()
    i = chain.index("mep_offset_src in")
    guard = chain[i:i + 120]
    assert "csp_start" in guard, (
        "an AUC window cannot be closed by a silent-period start, so trials "
        "whose cSP was detected rather than stored lose their AUC")
    assert "envelope" in guard, "resting-state AUC has been dropped"


def test_the_auc_branch_does_not_require_a_manual_latency():
    """
    man_lat is None on a trial nobody reviewed. Requiring it excluded exactly
    the trials this branch exists to serve.
    """
    chain = _auc_chain()
    i = chain.index("mep_offset_src in")
    cond = chain[max(0, i - 220):i]
    assert "auto_lat is not None" in cond, (
        "the branch still requires a manual latency, so unreviewed trials "
        "with a detected onset are skipped")


def test_stored_windows_still_take_precedence():
    """Order matters: a window the analyst set must not be recomputed."""
    chain = _auc_chain()
    assert chain.index('"auc_start_idx" in segments_metadata') < \
        chain.index("mep_offset_src in"), (
        "the stored AUC window is no longer checked first")
    assert chain.index('"silent_start_idx" in segments_metadata') < \
        chain.index("mep_offset_src in"), (
        "stored onset/cSP markers are no longer checked before detection")


def test_every_auc_reaches_the_summary():
    """
    auc_vals_all backs the per-condition mean. A branch computing an AUC
    without appending would show it per trial and drop it from the summary.
    """
    chain = _auc_chain()
    assert chain.count("auc_val = compute_auc(") + chain.count("_np_trapz") \
        <= chain.count("auc_vals_all.append(auc_val)") + 1, (
        "an AUC branch computes a value it never adds to the summary list")
