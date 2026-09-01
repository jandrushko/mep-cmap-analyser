"""
Attributes reprocess_ui.py borrows from FileEntry and DatasetSession.

test_no_missing_attributes.py checks names read off `self`, which is most of
app.py but not this: reprocess_ui walks the queue and reads attributes off
FileEntry objects, and a wrong name there is invisible until a study is open
and the menu item is clicked.

That is exactly how `fe.file_id` reached a running application. The dataclass
field is `id`, the tests never touched a real FileEntry, and nothing failed
until the run was started against a real dataset on another machine.

Parsed rather than searched, so a name in a comment or a docstring does not
count as a use.
"""

import ast
import pathlib

import pytest

from mep_cmap import dataset_session as DS

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "reprocess_ui.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _attrs_read_off(var_name):
    """Attribute names read off a local called `var_name`."""
    out = set()
    for node in ast.walk(TREE):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == var_name):
            out.add(node.attr)
    return out


def _file_entry_names():
    names = set(DS.FileEntry.__dataclass_fields__)
    names |= {a for a in dir(DS.FileEntry) if not a.startswith("_")}
    return names


def test_every_file_entry_attribute_used_exists():
    used = _attrs_read_off("fe")
    missing = sorted(used - _file_entry_names())
    assert not missing, (
        f"reprocess_ui.py reads {missing} off a FileEntry, which has no such "
        f"attribute. The identifier is `id`, not `file_id`.")


def test_the_identifier_is_id():
    """
    Pinned because it is the one that got through, and because `file_id` is
    the more natural name to reach for -- DatasetSession.get_file takes a
    parameter of that name, which is what made the wrong guess plausible.
    """
    assert "id" in DS.FileEntry.__dataclass_fields__
    assert "file_id" not in DS.FileEntry.__dataclass_fields__


def test_the_dataset_attributes_used_exist():
    """
    `files` is assigned in __init__ rather than declared on the class, so it
    is checked on an instance. Checking the class would pass a name that does
    not exist and fail one that does.
    """
    ds = DS.DatasetSession()
    for name in ("files", "get_file", "save", "json_path"):
        assert hasattr(ds, name), f"DatasetSession has no {name!r}"


def test_the_status_constants_used_exist():
    assert DS.STATUS_COMPLETE and DS.STATUS_NEEDS_REVIEW


def test_the_currency_check_reads_the_live_schema():
    """
    LAT_COLS imported from the pipeline, not copied into reprocess_currency.
    A copy would answer yesterday's question: every column added to the trial
    file would silently stop counting as missing, and the check would report
    a study up to date while the outputs were behind.
    """
    body = SRC[SRC.index("def _reprocess_currency_records"):]
    body = body[:body.index("def _reprocess_trials_header")]
    assert "from .pipeline import LAT_COLS" in body


def test_confirmation_is_ensured_on_both_adoption_paths():
    """
    Settings and confirmation are separate records and a session can carry one
    without the other. One file in the study did -- chan_settings ['2'],
    chan_confirmed [] -- written by a run interrupted after the setup was
    entered and before it was confirmed.

    The early return for "settings already present" skipped the confirming
    step, _validate_analysis_setup refused at its second guard, and that one
    file failed out of a hundred and ten.
    """
    body = SRC[SRC.index("def _reprocess_adopt_flat_setup"):]
    body = body[:body.index("def _reprocess_analyse")]
    assert body.count("_reprocess_confirm_channel(idx)") == 2, (
        "one of the two adoption paths returns without confirming the "
        "channel, so a session carrying settings but no confirmation is "
        "refused")


def test_confirming_sets_both_flags():
    """
    _validate_analysis_setup reads _chan_confirmed at one guard and
    _labels_tab_confirmed at another. Satisfying one is not enough.
    """
    body = SRC[SRC.index("def _reprocess_confirm_channel"):]
    body = body[:body.index("def _reprocess_analyse")]
    assert "_chan_confirmed.add(idx)" in body
    assert "_labels_tab_confirmed = True" in body


def test_the_currency_check_offers_a_scope_rather_than_a_second_command():
    """
    One menu item, with "everything" or "only what is behind" chosen in the
    confirmation. Two commands made the analyst pick between names when the
    difference is a reason: a change to how a value is COMPUTED adds no
    column, so every file still looks up to date while none of them is, and
    the narrower command would have found nothing to do.

    The counts show either way, so they are visible even when the answer is
    "everything".
    """
    app = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "app.py").read_text(encoding="utf-8")
    assert "self.reprocess_study" in app
    assert "reprocess_outdated" not in app

    body = SRC[SRC.index("def _reprocess_confirm"):]
    body = body[:body.index("def _reprocess_name")]
    assert "_reprocess_currency_records()" in body
    assert 'value="outdated"' in body and 'value="all"' in body


def test_the_confirmation_returns_what_to_run_not_just_yes():
    """
    The scope choice is part of the answer, so returning a bool would drop
    it and run everything whatever was ticked.

    Matched on the scope variable rather than on one line of the assignment:
    the assignment moved onto two lines when the archive scope was added, and
    a guard that breaks on reformatting tests the layout rather than the
    contract.
    """
    body = SRC[SRC.index("def _reprocess_confirm"):]
    body = body[:body.index("def _reprocess_name")]
    assert 'result["run"]' in body and "outdated" in body
    assert 'return result["run"], result["archive"]' in body


def test_the_confirmation_offers_archiving_without_a_second_menu_item():
    """
    Archiving belongs to the same job as the re-run and is chosen inside the
    one confirmation, not by adding an entry to the File menu.
    """
    body = SRC[SRC.index("def _reprocess_confirm"):]
    body = body[:body.index("def _reprocess_name")]
    assert 'value="archive"' in body
    assert "archive_var" in body


def test_archiving_is_planned_again_after_the_run():
    """
    The run creates the current copies, so a plan made before it would leave
    every session it refreshed looking like the only copy there is.
    """
    body = SRC[SRC.index("def _reprocess_archive_outputs"):]
    body = body[:body.index("# ── the run ")]
    assert "_reprocess_archive_plan()" in body
