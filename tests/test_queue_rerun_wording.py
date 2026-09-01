"""
What the queue's rerun choices tell the analyst.

Three routes reprocess a completed file and they differ in exactly one way
that matters: whether the analyst's own work survives. Load & process and
Mark as rerun keep it; Reset & reprocess deletes it. None of those three
verbs says so, and getting it wrong is unrecoverable in one direction --
manual marker edits and assigned conditions cannot be regenerated.

app.py cannot be imported here (matplotlib's Tk backend), so these read the
source, as test_app_shutdown.py does.
"""

import pathlib

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _menu_block():
    a = SRC.index("# Right-click context menu")
    return SRC[a:SRC.index("def _show_ctx", a)]


def _already_done_block():
    a = SRC.index('"Already done"')
    return SRC[a:SRC.index("parent=self.root)", a)]


# ── the menu ─────────────────────────────────────────────────────────────────

def test_each_rerun_route_says_what_it_does_to_the_analysts_edits():
    """
    The destructive one is not distinguishable from the other two by its verb
    alone: "reset" and "reprocess" both sound like re-running.
    """
    block = _menu_block()
    assert "Load & process  (keeps your edits)" in block
    assert "Mark as rerun  (no changes yet)" in block
    assert "(discards your edits)" in block


def test_the_destructive_route_is_still_marked_as_such():
    """The ellipsis promises a confirmation, and one is in fact shown."""
    block = _menu_block()
    assert "Reset & reprocess from scratch…" in block
    assert "_queue_reset_file" in block


# ── the dialogue ─────────────────────────────────────────────────────────────

def test_the_rerun_prompt_says_edits_are_restored():
    """
    This path restores the saved session before running, so it repeats the
    analysis as it was left. Someone expecting a fresh detection would read
    an unchanged result as evidence that nothing needed changing.
    """
    block = _already_done_block()
    assert "restored" in block
    assert "manual edits" in block


def test_the_rerun_prompt_names_the_from_scratch_route():
    """
    It lives on a context menu, so there is no way to discover it from this
    dialogue unless the dialogue says so.
    """
    assert "Reset & reprocess from scratch" in _already_done_block()


def test_the_rerun_prompt_no_longer_claims_it_is_for_changing_settings():
    """
    The old wording, "useful for changing settings", described the opposite of
    what happens: the saved settings are restored over anything just changed.
    """
    block = _already_done_block()
    assert "Useful for changing settings" not in block
