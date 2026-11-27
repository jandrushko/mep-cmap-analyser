"""
Column explanations on tab 1a.

The settings that need a paragraph had one place to receive it: prose above
the table. That cost the vertical space the table needed and sat nowhere near
the field it described, so reading it meant holding a column name in mind while
scrolling. The explanation now lives beside the column.
"""

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
APP = (PKG / "app.py").read_text(encoding="utf-8")
TIP = (PKG / "tooltips.py").read_text(encoding="utf-8")


def _headers():
    src = APP[APP.index('headers = ["Stim"'):]
    return ast.literal_eval(src[:src.index("]") + 1].split("=", 1)[1].strip())


def _column_help():
    """Read COLUMN_HELP from source.

    app.py cannot be imported by the suite -- it needs a working matplotlib Tk
    backend -- so the dict is parsed rather than imported, as every other test
    of that module does.
    """
    tree = ast.parse(APP)
    for node in tree.body:
        if isinstance(node, ast.Assign) and \
                any(getattr(t, "id", "") == "COLUMN_HELP" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("COLUMN_HELP not found in app.py")


# ── coverage ─────────────────────────────────────────────────────────────────

def test_every_column_has_an_explanation():
    COLUMN_HELP = _column_help()
    missing = [h for h in _headers() if h not in COLUMN_HELP]
    assert not missing, f"no help text for: {missing}"


def test_no_help_text_is_orphaned():
    """Keyed by the exact heading, so a renamed column loses its icon.

    Better that than an icon still showing an explanation of the setting the
    column used to be.
    """
    COLUMN_HELP = _column_help()
    orphans = [k for k in COLUMN_HELP if k not in _headers()]
    assert not orphans, f"help text for columns that no longer exist: {orphans}"


def test_the_explanations_say_something():
    COLUMN_HELP = _column_help()
    for col, text in COLUMN_HELP.items():
        assert len(text) > 60, f"{col}: too short to be worth an icon"
        assert col.split(" (")[0].lower() not in text[:len(col) + 2].lower(), \
            f"{col}: starts by restating its own name"


def test_the_timing_columns_keep_their_detail():
    """The paragraphs removed from the header must not have been lost."""
    COLUMN_HELP = _column_help()
    assert "RMS guard" in COLUMN_HELP["Gap (ms)"]
    assert "SICI" in COLUMN_HELP["Gap (ms)"]
    assert "BEFORE the marker" in COLUMN_HELP["Delay (ms)"]
    assert "Detect delays" in COLUMN_HELP["Delay (ms)"]


def test_the_window_columns_distinguish_themselves_from_the_baseline():
    """Three settings on two tabs look like pre-stimulus settings."""
    COLUMN_HELP = _column_help()
    assert "Pre-stim for analysis" in COLUMN_HELP["Pre (ms)"]
    assert "tab 1c" in COLUMN_HELP["Pre (ms)"]


def test_the_header_prose_was_replaced_not_duplicated():
    a = APP.index("Configure labels, colours, and analysis options")
    block = APP[a:a + 900]
    assert "conditioning pulse" not in block, \
        "the gap paragraph is in the tooltip now; two copies will diverge"
    assert "\\u24d8" in block or "\u24d8" in block, \
        "the header should say the icons are there"


# ── the widget ───────────────────────────────────────────────────────────────

def test_hover_and_click_are_both_offered():
    """Hovering suits a reminder; pinning suits actually reading a paragraph."""
    assert '"<Enter>"' in TIP and '"<Leave>"' in TIP and '"<Button-1>"' in TIP
    assert "_pinned" in TIP


def test_the_tooltip_cannot_raise():
    """It may outlive the widget it describes when a tab is rebuilt."""
    tree = ast.parse(TIP)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Tooltip":
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if fn.name in ("__init__",):
                    continue
                src = ast.unparse(fn)
                if "winfo" in src or "destroy" in src or "after" in src:
                    assert "TclError" in src, \
                        f"Tooltip.{fn.name} touches Tk without guarding it"
            return
    raise AssertionError("Tooltip class not found")


def test_it_unbinds_when_the_widget_goes():
    assert '"<Destroy>"' in TIP


def test_it_stays_on_screen():
    """A tooltip on the last column would otherwise open past the edge."""
    assert "winfo_screenwidth()" in TIP


def test_the_icon_is_its_own_widget():
    """A tooltip nobody knows is there explains nothing."""
    body = TIP[TIP.index("def attach_info_icon"):]
    assert "\u24d8" in body
    assert "cursor=\"hand2\"" in body


def test_the_delay_is_long_enough_not_to_flash():
    from mep_cmap.tooltips import HOVER_DELAY_MS
    assert 250 <= HOVER_DELAY_MS <= 900


@pytest.mark.parametrize("col", ["Gap (ms)", "Delay (ms)", "Pre (ms)",
                                 "Post (ms)", "Detect CSP", "Min lat (ms)"])
def test_the_explanations_say_why_not_only_what(col):
    """A restatement of the label is not an explanation.

    Each of these should tell the analyst what goes wrong if it is set badly,
    which is the part that cannot be guessed from the column name.
    """
    COLUMN_HELP = _column_help()
    text = COLUMN_HELP[col].lower()
    assert any(w in text for w in
               ("wrong", "fails", "not", "too", "rather than", "otherwise",
                "does not", "no ")), f"{col}: says what but not why"


# ── tab 1c ───────────────────────────────────────────────────────────────────

def _field_help():
    tree = ast.parse(APP)
    for node in tree.body:
        if isinstance(node, ast.Assign) and \
                any(getattr(t, "id", "") == "FIELD_HELP" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("FIELD_HELP not found in app.py")


def test_every_1c_help_entry_is_used():
    """An unused entry is text nobody will ever read."""
    unused = [k for k in _field_help() if f'FIELD_HELP["{k}"]' not in APP]
    assert not unused, f"help written but never attached: {unused}"


def test_the_csp_footnote_was_replaced_not_duplicated():
    """It described three of eight fields and re-wrapped on every resize."""
    assert "Z-score: threshold multiplier" not in APP
    fh = _field_help()
    assert "1.96" in fh["csp_criterion"]
    assert "99th percentile" in fh["csp_significance"]
    assert "second peak" in fh["csp_max_offset"]


def test_the_amplitude_window_help_says_onset_is_separate():
    """The commonest misreading of that field, and the one the old prose led
    with: the window bounds where amplitude is measured, not where onset may
    be found."""
    fh = _field_help()
    assert "NOT limited" in fh["ptp_start"] or "not limited" in fh["ptp_start"]
    assert "tab 1a" in fh["ptp_start"]


def test_the_icon_glyph_is_defined_once():
    """It has to be consistent enough to be learned."""
    assert "INFO_ICON" in TIP
    assert TIP.count('"\\u24d8"') <= 1


def test_no_widget_claims_help_it_does_not_have():
    """The icon is a promise. A suffix added without a Tooltip behind it is a
    field that appears to explain itself and does not."""
    import re
    for m in re.finditer(r'text="([^"]*\\u24d8[^"]*)"', APP):
        line_start = APP.rfind("\n", 0, m.start()) + 1
        line = APP[line_start:APP.index("\n", m.end())]
        assert "label_with_help" in line or "check_with_help" in line, (
            f"icon without a tooltip: {line.strip()[:70]}")


def test_the_helpers_work_on_python_39():
    """A backslash inside an f-string expression is 3.12 or later.

    pyproject declares >=3.9, and this compiles cleanly on the interpreter
    running the tests, so the failure would only appear for a user on an
    older one.
    """
    import re
    assert not re.search(r'f"[^"\n]*\{[^}\n]*\\[^}\n]*\}', TIP), \
        "backslash inside an f-string expression"


# ── settings whose meaning depends on state ──────────────────────────────────

def test_the_amplitude_note_states_the_state_only():
    """Five lines of grey prose repeated the standing explanation on every
    visit to the tab, to say the one thing that had changed.

    Checked by measuring what the note is set to: a state line is short, and
    the paragraph explaining what the state means belongs to the icon.
    """
    import re

    body = APP[APP.index("def _refresh_ptp_note"):]
    body = body[:body.index("self._refresh_ptp_note = ")]
    for line in ("Amplitude window anchoring is ON (set in",
                 "Amplitude window anchoring is OFF (set in"):
        assert line in body
    notes = re.findall(r"_ptp_note\.config\(\s*text=(.*?)\)\n", body, re.S)
    assert len(notes) == 2, "one note per state"
    for n in notes:
        assert len(n) < 240, f"the on-screen note is a paragraph again: {n[:60]}"


def test_the_amplitude_tooltip_follows_the_anchoring_state():
    """A tooltip describing anchoring as off while it is on is worse than
    none."""
    body = APP[APP.index("def _refresh_ptp_note"):]
    body = body[:body.index("self._refresh_ptp_note = ")]
    assert "_tip.set_text(_detail)" in body
    assert body.count("_detail = (") == 2, "one explanation per state"


def test_the_tooltip_text_can_be_replaced():
    body = TIP[TIP.index("def set_text"):]
    body = body[:body.index("\n    # ")]
    assert "self.text = text" in body
    assert "_show()" in body, "an open window must be refreshed, not left stale"


def test_helpers_expose_their_tooltip():
    """Otherwise a caller has to keep a second reference beside every label."""
    for fn in ("attach_info_icon", "label_with_help", "check_with_help"):
        body = TIP[TIP.index(f"def {fn}"):]
        end = TIP.find("\ndef ", TIP.index(f"def {fn}") + 10)
        body = body[:end - TIP.index(f"def {fn}")] if end > 0 else body
        assert ".tooltip = " in body, f"{fn} does not expose its Tooltip"


def test_the_anchored_start_is_not_confused_with_the_1a_profile():
    """It is each type's own median DETECTED onset, which the profile bounds
    but does not determine. Labelling it as a 1a value would be wrong in a way
    that is hard to notice."""
    body = APP[APP.index("def _refresh_ptp_note"):]
    body = body[:body.index("self._refresh_ptp_note = ")]
    assert "does not\n                           determine" in body or \
           "bounds but does not" in body


# ── the tooltip must not take the click ──────────────────────────────────────

def test_a_pinning_tooltip_does_not_swallow_the_click():
    """Returning "break" stopped the event reaching the widget.

    On a checkbutton that made the box impossible to tick: the tooltip opened
    and the state never changed. Nothing here is worth protecting from a
    second handler, so the click is passed on in every case.
    """
    body = TIP[TIP.index("def _on_click"):]
    body = body[:body.index("\n    def ")]
    assert 'return "break"' not in body


def test_controls_get_hover_only():
    """The click belongs to the control."""
    body = TIP[TIP.index("def check_with_help"):]
    assert "pin_on_click=False" in body


def test_pinning_is_opt_out_not_removed():
    """A label has no click behaviour of its own, and a paragraph is easier to
    read pinned than held under a motionless pointer."""
    body = TIP[TIP.index("def __init__"):]
    body = body[:body.index("\n    def ")]
    assert "pin_on_click: bool = True" in body
    assert "if self._pin_on_click:" in body


def test_the_click_binding_is_absent_when_pinning_is_off():
    """Not merely inert: an unbound handler cannot intercept anything."""
    body = TIP[TIP.index("def __init__"):]
    body = body[:body.index("\n    def ")]
    i = body.index("if self._pin_on_click:")
    j = body.index('"<Button-1>"')
    assert i < j, "the binding must sit inside the guard"
