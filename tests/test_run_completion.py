"""
Telling a finished analysis from a crashed one, without a person watching.

The batch re-runner drives one file at a time and has to know, before it loads
the next, whether the last one actually wrote its outputs. Getting this wrong
is not a cosmetic failure: the runner would mark a crashed file complete, or
load the next file while the previous one's session was still being written.

app.py cannot be imported here (matplotlib's Tk backend), so these read the
source, as test_unattended_load.py does.
"""

import pathlib

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _between(start, end, after=0):
    a = SRC.index(start, after)
    return SRC[a:SRC.index(end, a)]


def test_the_worker_reports_a_crash_as_an_outcome_not_only_a_log_line():
    """
    Until this existed the except branch put a log line and nothing else, so a
    crash was visible to a reader and invisible to a caller.
    """
    body = _between("except Exception as e:", "finally:")
    assert '"failed"' in body


def test_both_outcomes_reach_the_latch():
    poller = _between("def _poll_queue", "except queue.Empty")
    assert 'elif msg == "done":' in poller
    assert 'elif msg == "failed":' in poller
    assert poller.count("self._run_finished.set()") == 2


def test_the_latch_is_set_after_the_autosave_not_before():
    """
    THE ordering rule. The latch releases a runner to load the next file, and
    loading calls _reset_state_for_new_file, which empties segments_metadata.
    Set before the autosave, the previous file's manual edits are written out
    after they have already been cleared -- so a batch run destroys exactly
    the work it exists to preserve.
    """
    done = _between('elif msg == "done":', 'elif msg == "failed":')
    assert "_autosave_session()" in done
    assert done.index("_autosave_session()") < done.index("_run_finished.set()")
    assert done.index("mark_complete()") < done.index("_run_finished.set()")


def test_a_failed_run_is_not_recorded_as_complete():
    """
    A file that raised part-way has outputs in an unknown state. Marking it
    complete makes a half-written analysis indistinguishable from a finished
    one, six months later.
    """
    failed = _between('elif msg == "failed":', "except queue.Empty")
    # The call forms, not the bare names: the branch's own comment explains
    # why neither happens here, and naming them in prose must not trip this.
    assert "mark_complete()" not in failed
    assert "_autosave_session()" not in failed


def test_the_latch_is_cleared_before_a_run_starts():
    """
    Left set from the previous file, a caller waiting on it returns at once
    and reports the PREVIOUS file's outcome as this one's.
    """
    body = _between("def run_analysis_start", "threading.Thread")
    assert "_run_finished.clear()" in body
    assert "self._run_outcome = None" in body
