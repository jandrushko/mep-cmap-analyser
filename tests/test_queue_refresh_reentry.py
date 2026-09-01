"""
The queue redraw must not re-enter itself.

The filter variables carry write traces that call the redraw, and the redraw
sets one of them to keep the Type dropdown's counts current. Without a guard
the inner call clears the tree and inserts every row, then the outer call
resumes its own loop and inserts an iid that now exists. Tk raises
"Item ... already exists", the queue is left half drawn, and every later
action that redraws raises again.

app.py cannot be imported here (matplotlib's Tk backend), so these read the
source, as test_app_shutdown.py does.
"""

import pathlib

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _guard():
    a = SRC.index("def _queue_refresh(self)")
    return SRC[a:SRC.index("def _queue_refresh_inner", a)]


def _body():
    a = SRC.index("def _queue_refresh_inner")
    return SRC[a:SRC.index("\n    def ", a + 10)]


def test_the_redraw_is_guarded_against_re_entry():
    guard = _guard()
    assert "_queue_refreshing" in guard
    assert "return" in guard, "a re-entrant call must return, not proceed"
    assert "finally" in guard, "the flag must clear even if the redraw raises"


def test_the_insert_loop_is_inside_the_guard():
    """The loop is what breaks on re-entry, so it must be the guarded part."""
    body = _body()
    assert "tree.delete(" in body
    assert 'tree.insert("", "end", iid=fe.id' in body


def test_the_type_filter_is_only_set_when_it_changes():
    """
    Belt and braces. The guard makes an unconditional set harmless, but a
    write trace firing on every redraw would still churn every widget bound
    to that variable for no reason.
    """
    body = _body()
    a = body.index("_qf_type_dd[")
    window = body[a:a + 1200]
    assert "if self._qf_type.get() != want:" in window


def test_the_reason_is_written_down_where_the_set_happens():
    """
    The next person to touch this will not guess that assigning to a
    StringVar redraws the queue.
    """
    body = _body()
    a = body.index("_qf_type_dd[")
    window = body[a:a + 1200].lower()
    assert "trace" in window and "re-enter" in window
