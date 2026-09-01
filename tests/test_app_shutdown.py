"""
Tests for cancelling pending Tk callbacks at exit.

Several callbacks in app.py reschedule themselves indefinitely -- the message
queue poller every 75 ms, the file-load progress poller every 80 ms. Nothing
cancelled them and nothing marked the window as closing, so quitting left them
queued against an interpreter that was already being torn down:

    invalid command name "12995804736_poll_queue"
        while executing "12995804736_poll_queue" ("after" script)

Harmless in itself, but it is noise printed at exactly the moment a real error
would also appear, and it makes a clean run look broken.

app.py cannot be imported here (matplotlib's Tk backend), so these read the
source. The scheduler's own logic is exercised directly with a stub.
"""

import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _load_file_entry_body():
    """Source of the file-load body only.

    `threading.Thread(target=_worker, ...)` appears in more than one method, so
    anchoring on it finds the wrong region and the assertions silently examine
    unrelated code.

    Anchored on _load_file_entry_INNER: _load_file_entry itself is now a
    re-entrancy guard around it, and the wait loop these tests are about lives
    in the inner method. Anchoring on the outer one slices the guard alone and
    every assertion below passes vacuously.
    """
    a = SRC.index("def _load_file_entry_inner")
    b = SRC.index("\n    def ", a + 10)
    return SRC[a:b]


def _self_rescheduling_calls():
    """Callbacks that schedule themselves again -- the ones that can outlive exit."""
    out = []
    for name in ("_poll_queue", "_poll"):
        for m in re.finditer(rf"\.after\(\s*\d+\s*,\s*(self\.)?{name}\b", SRC):
            out.append(m.group(0))
    return out


def test_self_rescheduling_pollers_go_through_the_tracked_scheduler():
    """
    A raw root.after here cannot be cancelled, because its id is discarded the
    moment it is created.
    """
    assert not _self_rescheduling_calls(), (
        f"these reschedule via an untracked root.after: {_self_rescheduling_calls()}"
    )
    assert "self._schedule(75, self._poll_queue)" in SRC
    assert "self._schedule(80, _poll)" in SRC


def test_a_close_handler_is_bound_to_the_window_button():
    """
    Without WM_DELETE_WINDOW the close button bypasses any cleanup, so the
    common way of quitting is the one that produces the errors.
    """
    assert 'self.root.protocol("WM_DELETE_WINDOW", self._shutdown)' in SRC


def test_the_exit_menu_uses_the_same_path_as_the_close_button():
    """Two ways out must not mean two behaviours."""
    assert 'command=self._shutdown' in SRC
    assert 'label="Exit",          command=self.root.quit' not in SRC


def test_shutdown_cancels_every_tracked_callback():
    a = SRC.index("def _shutdown")
    b = SRC.index("\n    def ", a + 10)
    body = SRC[a:b]
    assert "after_cancel" in body
    assert "self._closing = True" in body
    assert "_after_ids.clear()" in body


def test_scheduler_refuses_to_queue_anything_once_closing():
    a = SRC.index("def _schedule")
    b = SRC.index("def _shutdown")
    body = SRC[a:b]
    assert 'if getattr(self, "_closing", False):' in body
    # and the callback itself re-checks, since it may have been queued before
    # the flag was set
    assert body.count('_closing') >= 2


def test_scheduler_logic_with_a_stub():
    """Exercise the recorded-id behaviour without a display."""
    class FakeRoot:
        def __init__(self):
            self.pending = {}
            self.cancelled = []
            self._n = 0

        def after(self, ms, fn):
            self._n += 1
            self.pending[self._n] = fn
            return self._n

        def after_cancel(self, i):
            self.cancelled.append(i)
            self.pending.pop(i, None)

        def quit(self):
            pass

    import types as _t

    ns = {}
    a = SRC.index("    def _schedule")
    b = SRC.index("    def _make_window_adaptive")
    src = "class App:\n" + SRC[a:b]
    exec(compile(src, "<scheduler>", "exec"), {"tk": _t.SimpleNamespace(TclError=Exception)}, ns)
    App = ns["App"]

    app = App()
    app.root = FakeRoot()
    app._closing = False
    app._after_ids = set()

    fired = []
    i = app._schedule(10, lambda: fired.append(1))
    assert i in app._after_ids

    # firing removes the id and runs the callback
    app.root.pending[i]()
    assert fired == [1]
    assert i not in app._after_ids

    # after shutdown, nothing new is queued and pending ids are cancelled
    j = app._schedule(10, lambda: fired.append(2))
    app._shutdown()
    assert app._closing is True
    assert j in app.root.cancelled
    assert app._after_ids == set()
    assert app._schedule(10, lambda: fired.append(3)) is None
    assert fired == [1]


def test_the_file_load_wait_guards_an_empty_result():
    """
    Regression: adding a `_closing` clause to this wait loop gave it a way to
    exit without the worker having appended anything, and the line after it
    read `_result[0][0]` unguarded. Double-clicking a file in the queue then
    raised IndexError and took the handler down with it.

    Any loop that can exit for a reason other than "the work finished" needs
    the code after it to cope with the work not having finished.
    """
    body = _load_file_entry_body()
    assert "if not _result:" in body, (
        "the empty-result case must be handled before indexing"
    )
    assert "return" in body


def test_the_file_load_wait_cannot_spin_forever():
    """A reader that never returns must not lock the interface indefinitely."""
    body = _load_file_entry_body()
    assert "_deadline" in body
    assert "time.time() >" in body


def test_an_inspector_failure_does_not_hang_the_analysis():
    """
    The worker thread blocks on _last_outlier_result. If opening the window
    raises, nothing ever sets it and the run stops dead with no message.
    """
    a = SRC.index('elif msg == "show-inspector":')
    b = SRC.index('elif msg == "bidsify-convert-done":', a)
    body = SRC[a:b]
    assert "try:" in body and "except Exception" in body
    assert "self._last_outlier_result = {}" in body, (
        "the worker must be released so the analysis can finish"
    )
    assert "_log_gui(" in body, "the failure must be reported to the analyst"


def test_the_file_load_is_not_re_entrant():
    """
    The wait loop pumps the event loop with root.update(), which processes
    every pending event -- a queued selection, a second click, a menu command.
    Any of those can call _load_file_entry again while the first call is still
    inside the loop. The inner call opens the channel-assignment and metadata
    dialogues and returns; the outer one resumes and opens them a second time.
    Nothing errors, so the only symptom is being asked everything twice.
    """
    a = SRC.index("def _load_file_entry(")
    guard = SRC[a:SRC.index("def _load_file_entry_inner")]
    assert "_loading_file" in guard, "the guard flag is gone"
    assert "return" in guard, "a re-entrant call must return, not proceed"
    assert "finally" in guard, "the flag must be cleared even if the load raises"


def test_the_wait_loop_is_inside_the_guard():
    """The loop is what makes re-entry possible, so it must be guarded."""
    assert "while not _ready[0]" in _load_file_entry_body()
