"""
Setup ▸ Conditions — the surface.

The rules are tested in test_conditions.py without a display. What is checked
here is the wiring: that the tab reaches the model rather than reimplementing
it, that Apply cannot write a table the model would refuse, and that the events
file it writes is the one the reader looks for.
"""

import ast
import csv
import json
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"
TAB = (PKG / "conditions_tab.py").read_text(encoding="utf-8")
APP = (PKG / "app.py").read_text(encoding="utf-8")


def _method(name, src=TAB):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant):
                body = body[1:]
            return "\n".join(ast.unparse(n) for n in body)
    raise AssertionError(f"{name} not found")


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_tab_sits_between_dataset_and_bidsify():
    """Conditions describe the recording, so they are decided before it is
    converted or analysed."""
    i = APP.index('text="Dataset"')
    j = APP.index('text="Conditions"')
    k = APP.index('text="BIDS-ify"')
    assert i < j < k


def test_the_mixin_is_registered_where_attributes_are_checked():
    """A mixin absent from that list reads as undefined attributes -- the same
    failure David's fork still has for its own tab."""
    t = (pathlib.Path(__file__).parent / "test_no_missing_attributes.py"
         ).read_text(encoding="utf-8")
    assert "conditions_tab.py" in t


def test_the_tab_does_not_reimplement_the_rules():
    """Parsing, splitting and completeness live in conditions.py, which is
    tested without a display. A second copy here would be the copy that drifts.
    """
    for fn, call in (("_cond_split", "C.split_row("),
                     ("_cond_autofill", "C.autofill("),
                     ("_cond_rename", "C.sanitise_name("),
                     ("_cond_apply", "C.validate(")):
        assert call in _method(fn), f"{fn} should delegate to {call}"


def test_apply_validates_before_writing_anything():
    """A file written from a table the model would refuse is a file that
    disagrees with the analysis."""
    body = _method("_cond_apply")
    assert body.index("C.validate(") < body.index("to_event_rows")


def test_apply_composes_through_the_one_function():
    assert "C.group_events(event_rows)" in _method("_cond_apply")


def test_apply_hands_the_groups_to_the_analysis():
    body = _method("_cond_apply")
    assert "self.condition_event_rows = event_rows" in body
    assert "self.condition_map = decoded" in body


def test_a_failed_write_does_not_discard_the_assignment():
    """The conditions are still applied to the session; only the file is
    missing, and the message says so."""
    body = _method("_cond_apply")
    i = body.index("write_events_tsv_beside")
    j = body.index("self.condition_event_rows")
    assert i < j, "the assignment must survive a write failure"
    assert "could not be written" in body


# ── the events file ──────────────────────────────────────────────────────────

def test_the_written_path_matches_what_the_reader_looks_for():
    """What this writes must be what it reads on the next load."""
    from mep_cmap.conditions_tab import events_tsv_path
    from mep_cmap.formats.edf import _events_tsv_path
    for name in ("sub-01_ses-1_task-x_emg.edf", "sub-01_run-2_emg.edf"):
        assert events_tsv_path("/data/" + name) == _events_tsv_path("/data/" + name)


def test_the_file_round_trips_through_the_reader(tmp_path):
    from mep_cmap import conditions as C
    from mep_cmap.conditions_tab import write_events_tsv_beside
    from mep_cmap.formats.edf import _read_events_tsv

    rec = tmp_path / "sub-01_task-x_emg.edf"
    rec.write_bytes(b"")
    stim = {"A": [1.0, 2.0, 3.0, 4.0]}
    rows = C.validate(C.split_row(C.rows_from_events(stim), 0, [0, 1],
                                  new_condition="pre",
                                  keep_condition="post"), stim)
    ev = C.to_event_rows(rows, stim)
    path = write_events_tsv_beside(str(rec), ev)

    back = _read_events_tsv(path)
    assert back == {"A": [1.0, 2.0, 3.0, 4.0]}, \
        "the reader groups by trial_type; conditions come through group_events"

    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh, delimiter="\t"))
    assert [r["condition"] for r in got] == ["pre", "pre", "post", "post"]
    groups, decoded = C.group_events(got)
    assert sorted(groups) == ["A\u00b7post", "A\u00b7pre"]
    assert decoded["A\u00b7pre"] == ("A", "pre")


def test_an_excluded_trial_is_written_as_na(tmp_path):
    from mep_cmap import conditions as C
    from mep_cmap.conditions_tab import write_events_tsv_beside

    rec = tmp_path / "sub-01_task-x_emg.edf"
    rec.write_bytes(b"")
    stim = {"A": [1.0, 2.0]}
    rows = [C.ConditionRow("A", "keep", (0,)),
            C.ConditionRow("A", "", (1,), excluded=True)]
    path = write_events_tsv_beside(str(rec), C.to_event_rows(rows, stim))
    with open(path, encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh, delimiter="\t"))
    assert [r["trial_type"] for r in got] == ["A", "n/a"]


def test_the_sidecar_documents_the_extra_column(tmp_path):
    """condition is not a BIDS-defined column, and an undocumented extra column
    is one only its author can interpret."""
    from mep_cmap import conditions as C
    from mep_cmap.conditions_tab import write_events_tsv_beside

    rec = tmp_path / "sub-01_task-x_emg.edf"
    rec.write_bytes(b"")
    path = write_events_tsv_beside(
        str(rec), C.to_event_rows([C.ConditionRow("A", "pre", (0,))],
                                  {"A": [1.0]}))
    side = pathlib.Path(path).with_suffix(".json")
    meta = json.loads(side.read_text(encoding="utf-8"))
    assert set(meta) == {"onset", "duration", "trial_type", "condition"}
    assert "Description" in meta["condition"]
    assert "n/a" in meta["trial_type"]["Description"], \
        "the sidecar should say what n/a means, since it marks an exclusion"


# ── the review pane ──────────────────────────────────────────────────────────

def test_no_detection_markers_are_drawn():
    """These are raw epochs before any analysis.

    Onset and amplitude belong to Preview detection, which shows one trial at a
    time because a dozen sets of markers on one axes cannot be read.
    """
    body = _method("_cond_draw")
    for marker in ("onset", "PTP", "offset", "dispatch_onset"):
        assert marker not in body


def test_several_conditions_are_drawn_in_distinct_colours():
    """Comparing two is how a split is judged before it is applied."""
    body = _method("_cond_draw")
    assert "_CYCLE[n % len(_CYCLE)]" in body
    assert "legend" in body


def test_a_large_selection_falls_back_to_the_average():
    """Two hundred translucent lines is neither readable nor quick."""
    from mep_cmap.conditions_tab import OVERLAY_LIMIT
    assert 50 <= OVERLAY_LIMIT <= 300
    body = _method("_cond_draw")
    assert "len(usable) <= OVERLAY_LIMIT" in body
    assert "len(usable) > OVERLAY_LIMIT" in body


def test_the_review_window_matches_the_analysis_window():
    """What is judged here is what will be measured."""
    body = _method("_cond_load_segments")
    assert "window_samples(" in body
    # ast.unparse normalises quotes, so match on the key rather than the call
    assert "window_map" in body


def test_the_trial_list_keeps_its_selection():
    """Without exportselection=False, selecting in the table clears the list:
    Tk hands the X selection to whichever widget was touched last."""
    assert "exportselection=False" in TAB


def test_reloading_is_explicit():
    """Returning to check a waveform should not silently undo an assignment."""
    assert 'text="⟳ Reload from file"' in TAB
    assert "_cond_reload" in _method("_build_conditions_tab")


@pytest.mark.parametrize("fn", ["_cond_split", "_cond_set_from_selection",
                                "_cond_autofill"])
def test_edits_that_need_one_row_say_so(fn):
    body = _method(fn)
    assert "showinfo" in body or "showwarning" in body


# ── populating ───────────────────────────────────────────────────────────────

def test_the_tab_reads_the_active_file_variable():
    """It read txt_file_path, which does not exist.

    The attribute holding the open recording is file_path. Reading a name that
    is never set is not an error -- getattr guards it -- so the tab simply
    reported that no file was loaded, for every file.
    """
    assert "txt_file_path" not in TAB
    assert "self.file_path.get()" in TAB


def test_the_tab_populates_when_it_is_shown():
    """The table is built from the file's events, which do not exist when the
    tab is constructed."""
    s2 = (PKG / "stage2.py").read_text(encoding="utf-8")
    assert "tab_conditions" in s2
    assert "_cond_tab_shown()" in s2


def test_an_assignment_survives_leaving_the_tab_and_returning():
    """Rebuilding on every visit would discard the work silently.

    Which is why Reload is a button: going away to check a waveform is not a
    request to start again.
    """
    body = _method("_cond_tab_shown")
    assert "_cond_source_path" in body
    assert "if self._cond_rows and" in body


def test_the_table_is_rebuilt_when_the_recording_changes():
    body = _method("_cond_tab_shown")
    assert "!= path" in body or "== path" in body
    assert "_cond_reload()" in body


# ── the order of the workflow ────────────────────────────────────────────────

def test_loading_a_file_lands_on_conditions():
    """What a stimulus type is FOR is decided before how it is detected.

    Twenty pulses labelled A may be two timepoints; configuring latency
    windows for "A" before saying so means configuring them again for each
    half afterwards.
    """
    body = _method("_build_labels_tab", src=APP)
    assert "self.nb_setup.select(self.tab_conditions)" in body


def test_confirming_goes_on_to_the_labels_tab():
    body = _method("_cond_apply")
    assert "_build_labels_tab(sorted(self._cond_groups_in_range(groups)))" in body
    # The navigation is done here, not inside the rebuild: that method runs on
    # every rebuild, so navigating from it sent channel switches and confirms
    # somewhere unexpected too.
    assert "nb_stage1.select(self.tab1b_frame)" in body


def test_a_failed_rebuild_does_not_navigate():
    """Landing on a setup table that was not rebuilt would show the previous
    grouping as though it were the new one."""
    body = _method("_cond_apply")
    i = body.index("Could not rebuild")
    j = body.index("nb_stage1.select(self.tab1b_frame)")
    assert "return" in body[i:j]


def test_the_rebuild_does_not_navigate():
    """_build_labels_tab runs on opening a file, switching channel, confirming
    a channel and applying conditions. Navigating from it meant every one of
    those jumped somewhere -- confirming a channel advanced to the next and was
    then thrown to the Conditions tab, which read as the advance having stopped
    working. Where to go depends on why it was rebuilt, so the caller decides.
    """
    body = _method("_build_labels_tab", src=APP)
    # The jump may live here, but only behind the flag: an ungated select
    # would fire on every channel switch and every confirm.
    i = body.index("nb_setup.select(self.tab_conditions)")
    guard = body.rindex("_go_to_conditions_after_load", 0, i)
    assert i - guard < 400, "the jump is not gated on a file having been opened"


def test_opening_a_file_is_what_goes_to_conditions():
    assert "_go_to_conditions_after_load = True" in APP
    body = _method("_build_labels_tab", src=APP)
    assert "_go_to_conditions_after_load" in body


def test_the_jump_happens_once_per_file():
    """Left set, every later rebuild would jump again."""
    body = _method("_build_labels_tab", src=APP)
    assert "_go_to_conditions_after_load = False" in body


def test_the_button_says_what_it_does_next():
    """Both of them, and each names its own destination.

    "Confirm events & continue" did not say where it went. There are now two
    destinations, because BIDS-ifying before analysing is a real order of work
    and the tab previously assumed everyone analyses first.
    """
    assert "Confirm & continue to First Level" in TAB
    assert "Confirm & continue to BIDS-ify" in TAB


def test_both_destinations_commit_through_one_function():
    """Two commit paths would be two callers of one rule -- validation, the
    per-channel epoch check, writing the events -- and the second would
    eventually skip a step the first enforces."""
    assert TAB.count("def _cond_apply(") == 1
    assert 'destination="first_level"' in TAB
    assert 'destination="bidsify"' in TAB


def test_both_buttons_are_enabled_together():
    """An invalid table must not become committable by choosing the other
    destination."""
    body = _method("_cond_refresh_status")
    assert "_cond_apply_btn.config" not in body
    assert body.count("_cond_apply_state") >= 3


def test_the_labels_tab_is_rebuilt_whichever_destination():
    """Conditions change the per-type windows 1a is built from, so skipping the
    rebuild on the way to BIDS-ify would leave it describing the last
    grouping."""
    body = _method("_cond_apply")
    rebuild = body.find("_build_labels_tab")
    branch = body.find('destination == \'bidsify\'')
    if branch == -1:
        branch = body.find('destination == "bidsify"')
    assert rebuild != -1 and branch != -1
    assert rebuild < branch, "the rebuild must happen before the branch"


def test_navigation_falls_back_if_the_tab_is_missing():
    """A layout change should not strand a loaded file on no tab at all."""
    body = _method("_build_labels_tab", src=APP)
    assert "except Exception:" in body
    i = body.index("nb_setup.select(self.tab_conditions)")
    assert "nb_stage1.select" in body[i:]


# ── undo ─────────────────────────────────────────────────────────────────────

def test_every_edit_goes_through_the_history():
    """An edit that bypassed it would be the one that cannot be taken back,
    and the analyst would not know which until they tried."""
    for fn in ("_cond_split", "_cond_set_from_selection", "_cond_rename",
               "_cond_autofill", "_cond_toggle_exclude", "_cond_delete"):
        body = _method(fn)
        assert "_cond_commit(" in body, f"{fn} edits without recording it"
        assert "self._cond_rows =" not in body, \
            f"{fn} assigns the table directly, bypassing the history"


def test_a_refused_edit_leaves_nothing_on_the_stack():
    """An undo that does nothing visible teaches the analyst the button is
    unreliable, and the next press is the one that goes too far."""
    body = _method("_cond_split")
    i = body.index("except C.ConditionError")
    j = body.index("_cond_commit(")
    assert i < j, "the failure path must return before anything is recorded"
    assert "return" in body[i:j]


def test_undo_and_redo_are_symmetrical():
    u, r = _method("_cond_undo"), _method("_cond_redo")
    assert "_cond_redo_stack.append" in u
    assert "_cond_undo_stack.append" in r


def test_a_new_edit_discards_the_redo_stack():
    """Redoing onto a table that has since changed would apply an edit to rows
    it was never made against."""
    assert "_cond_redo_stack.clear()" in _method("_cond_commit")


def test_the_history_is_bounded():
    from mep_cmap.conditions_tab import UNDO_DEPTH
    assert 10 <= UNDO_DEPTH <= 200
    assert "del self._cond_undo_stack[:-UNDO_DEPTH]" in _method("_cond_commit")


def test_reloading_clears_the_history():
    """Undoing past a reload would restore rows belonging to a recording that
    is no longer open."""
    body = _method("_cond_reload")
    assert "_cond_undo_stack.clear()" in body
    assert "_cond_redo_stack.clear()" in body


def test_the_button_names_what_it_will_undo():
    """'Undo split' is a different proposition from 'Undo delete'."""
    body = _method("_cond_refresh_history_buttons")
    assert "_cond_undo_stack[-1][-1]" in body
    # ast.unparse normalises quotes, so match the value not the literal
    assert "disabled" in body


def test_the_shortcuts_are_bound_to_the_widgets_not_the_window():
    """Ctrl+Z in a dialogue's entry field should still be the entry's own."""
    body = _method("_build_conditions_tab")
    assert "self._cond_tree, self._cond_list" in body
    assert "self.root.bind" not in body


def test_the_shortcuts_are_bound_after_both_widgets_exist():
    """Bound beside the table, the listbox did not yet exist."""
    i = TAB.index("for _w in (self._cond_tree, self._cond_list):")
    j = TAB.index("self._cond_list = tk.Listbox")
    assert j < i


def test_the_history_holds_frozen_rows():
    """A snapshot is the list itself rather than a copy that has to be kept
    honest, because ConditionRow is frozen and every helper returns a new list.
    """
    from mep_cmap.conditions import ConditionRow
    assert getattr(ConditionRow, "__dataclass_params__").frozen


# ── the review pane's channel and window ─────────────────────────────────────

def test_the_pane_draws_the_chosen_channel_not_the_first():
    """Whether trials belong together is judged on the muscle they were
    recorded from.

    The pane opened on whichever channel the analysis started with, so on a
    recording whose first channel is not the one under study it showed a trace
    with nothing to do with the question being asked.
    """
    body = _method("_cond_load_segments")
    assert "_cond_selected_channel_idx()" in body
    assert 'params["channel_idx"]' not in body


def test_the_picker_offers_the_analysed_channels():
    body = _method("_cond_channel_names")
    assert "_analysis_channel_indices()" in body


def test_the_picker_opens_on_the_current_channel():
    body = _method("_cond_refresh_channel_picker")
    # read through getattr, so match the attribute name rather than the access
    assert "channel_idx" in body
    assert "_cond_chan.set(" in body


def test_changing_channel_drops_the_cached_epochs():
    """Reusing them would draw the previous channel's waveforms under the new
    channel's name -- the failure this control exists to fix, reintroduced."""
    body = _method("_cond_channel_changed")
    assert "self._cond_segments = {}" in body
    assert "_cond_draw()" in body


def test_the_viewing_window_does_not_change_the_analysis():
    """Widening the view to see whether a late component belongs to a
    condition is a question about looking; silently re-epoching the analysis
    to answer it would be a surprising thing for a viewing control to do."""
    body = _method("_cond_window_changed")
    assert "window_map" not in body
    assert "_snapshot_analysis_params" not in body


def test_each_side_of_the_window_overrides_independently():
    """Widening only the tail should not require restating the lead-in."""
    body = _method("_cond_load_segments")
    assert "if _vpre is not None:" in body
    assert "if _vpost is not None:" in body


def test_a_blank_or_bad_window_falls_back_to_the_analysis_one():
    body = _method("_cond_view_window_ms")
    assert "except (TypeError, ValueError)" in body
    assert "return None" in body


def test_the_axis_names_the_unit_that_was_read():
    body = _method("_cond_draw")
    assert "_cond_unit" in body


def test_the_marker_choice_narrows_the_conditions_table():
    """162 Trigger comments and 6 Start Task: choosing Trigger and then being
    offered both is the failure _configured_events exists to prevent, one
    caller further along."""
    body = _method("_configured_events", src=APP)
    assert "self.marker_choice.get()" in body
    assert "ALL_MARKERS" in body


# ── the pane must show what the analysis will see ────────────────────────────

def test_the_review_pane_filters_as_the_analysis_does():
    """Raw epochs carry whatever DC offset and drift the amplifier had.

    That stacks the trials at different baselines, which makes an overlay
    unreadable -- and shows the analyst something the analysis never sees,
    which is the failure a review pane exists to prevent rather than commit.
    """
    body = _method("_cond_load_segments")
    assert "pipeline_apply_filters(" in body
    assert "FILTER_CFG_FIELDS" in body


def test_the_filter_field_list_is_shared_not_copied():
    """Two copies drift the moment a filter setting is added to one of them."""
    pipe = (PKG / "pipeline.py").read_text(encoding="utf-8")
    prev = (PKG / "preview.py").read_text(encoding="utf-8")
    assert "FILTER_CFG_FIELDS = (" in pipe
    assert "FILTER_CFG_FIELDS = (" not in prev, \
        "preview should import the list, not redefine it"
    assert "from .pipeline import FILTER_CFG_FIELDS" in prev


def test_unfilterable_data_is_drawn_and_said_so():
    """A filter that cannot be applied is not a reason to draw nothing."""
    body = _method("_cond_load_segments")
    assert "Showing unfiltered data" in body


def test_the_viewing_window_is_clamped_on_a_stitched_file():
    """A pre-epoched recording is stitched from its own epochs with
    mirror-padded guard bands between them.

    A window longer than an epoch reads that padding, which draws as plausible
    signal and is not signal at all, or reaches into the neighbouring trial.
    The analysis clamps for exactly this reason, and a typed viewing window
    would otherwise walk straight past that protection.
    """
    body = _method("_cond_load_segments")
    assert "get_epoch_bounds(" in body
    assert "clamp_window_map(" in body
    assert "_vpre = _bp" in body and "_vpost = _bq" in body


def test_a_continuous_file_is_not_clamped():
    """There are no epoch bounds to clamp to, and a recording is not stitched."""
    body = _method("_cond_load_segments")
    assert "if _bounds:" in body
    assert "self._cond_clamped = None" in body


def test_clamping_is_reported_in_the_pane():
    """A view that silently drew less than was asked for would look like the
    window control not working."""
    body = _method("_cond_draw")
    assert "_cond_clamped" in body
    assert "pre-epoched file" in body


# ── setting the epoch from what is on screen ─────────────────────────────────

def test_the_table_shows_each_conditions_epoch():
    body = _method("_cond_refresh_table")
    assert "_win_cells(row)" in body
    assert "Pre (ms)" in TAB and "Post (ms)" in TAB


def test_epochs_are_held_per_channel():
    """A condition is a property of the TRIAL -- trial 5 is 'pre' whichever
    muscle is looked at -- so the table is the same for every channel. An
    epoch is a property of the RESPONSE, and a hand muscle and a leg muscle
    legitimately want different windows.
    """
    body = _method("_build_conditions_tab")
    assert "self._cond_epochs = {}" in body
    assert "_cond_epochs_for_channel" in TAB


def test_the_epoch_columns_follow_the_reviewed_channel():
    """They show the channel they were judged against."""
    body = _method("_cond_refresh_table")
    assert "_cond_epochs_for_channel()" in body
    assert "_cond_refresh_table()" in _method("_cond_channel_changed")


def test_the_epoch_comes_from_the_window_being_viewed():
    """The point of setting it here is that the decision can be seen: whether
    the response is truncated, whether a silent period runs past the end. On
    the labels tab the same numbers are typed blind."""
    body = _method("_cond_set_epoch")
    assert "_cond_view_window_ms()" in body


def test_the_epoch_taken_is_the_one_actually_drawn():
    """On a stitched file the view is clamped to the stored epoch, and handing
    the analysis a window the recording cannot supply would undo that."""
    body = _method("_cond_set_epoch")
    assert "_cond_clamped" in body
    assert "min(pre, clamp[0])" in body and "min(post, clamp[1])" in body


def test_setting_an_epoch_can_cover_every_analysed_channel():
    """Setting one and finding it on one channel of two is the fault this
    replaced."""
    body = _method("_cond_set_epoch")
    assert "_cond_epoch_all_chans" in body
    assert "_cond_channel_names()" in body


def test_the_scope_defaults_to_every_channel():
    """One window for a recording is the ordinary case; differing windows per
    muscle is the exception, and the exception should be the thing opted in
    to."""
    body = _method("_build_conditions_tab")
    assert "_cond_epoch_all_chans = tk.BooleanVar(value=True)" in body


def test_setting_an_epoch_with_no_view_says_what_to_do():
    assert "Type a viewing window" in _method("_cond_set_epoch")


def test_the_epoch_can_be_cleared_back_to_the_default():
    body = _method("_cond_clear_epoch")
    assert "book.pop(" in body
    assert "_cond_push_epochs(" in body


def test_confirming_writes_every_channels_epochs():
    """window_map is PER CHANNEL state held in _chan_settings.

    Writing self.window_map alone reaches only the channel currently selected,
    which is why an epoch set here appeared on one channel and not the other.
    """
    body = _method("_cond_apply")
    assert "self._cond_epochs or {}" in body
    assert "_chan_settings.setdefault(ch, {})" in body


def test_the_hand_off_merges_rather_than_replaces():
    """A window set on the labels tab for a stimulus type this table did not
    give one to is still the analyst's setting.

    Counted occurrences of the merge before, one per branch, when the current
    channel was special-cased. That special case was the bug -- its epochs went
    to the live map and were then overwritten by a snapshot restore -- so there
    is one branch now, writing the snapshot for every channel and the live map
    as well. The rule tested is that neither destination is REPLACED.
    """
    body = _method("_cond_apply")
    assert "merged.update(book)" in body, "snapshots must merge, not overwrite"
    assert "_live.update(book)" in body, "the live map must merge, not overwrite"
    # A bare assignment of either from `book` alone would discard the rest.
    assert "= dict(book)" not in body
    assert "window_map = book" not in body


def test_setting_an_epoch_is_undoable():
    """Epochs share the table's history, so one Undo means the last thing
    done rather than the last thing done to one of two structures."""
    for fn in ("_cond_set_epoch", "_cond_clear_epoch"):
        assert "_cond_push_epochs(" in _method(fn)
    for fn in ("_cond_undo", "_cond_redo"):
        assert "self._cond_epochs" in _method(fn)


def test_the_history_entries_carry_both_structures():
    body = _method("_cond_commit")
    assert "copy.deepcopy(self._cond_epochs)" in body


def test_the_button_label_reads_the_last_element():
    """The history entry gained a field; a fixed index would have started
    naming the epoch book instead of the action."""
    body = _method("_cond_refresh_history_buttons")
    assert "[-1][-1]" in body


# ── why Confirm is unavailable ───────────────────────────────────────────────

def test_the_reason_has_its_own_label():
    """It shared the review pane's label, which the drawing code rewrites on
    every selection -- so the reason appeared and was overwritten by
    "10 trial(s) from 1 condition(s)" before it could be read, leaving a
    disabled button with no explanation anywhere on the tab.
    """
    body = _method("_cond_refresh_status")
    assert "_cond_block.config" in body
    assert "_cond_note.config" not in body


def test_the_reason_sits_beside_the_button_it_explains():
    """Which is where anyone looks when a button will not click."""
    body = _method("_build_conditions_tab")
    i = body.index("_cond_apply_btn")
    j = body.index("_cond_block")
    assert abs(body[i:j].count("\\n")) < 20, \
        "the explanation should be built with the button, not elsewhere"


def test_an_empty_table_says_so_too():
    body = _method("_cond_refresh_status")
    assert "No conditions to apply" in body


def test_loose_trials_can_be_given_rows():
    """Reporting the problem and offering no way to act on it means retyping
    trial ranges the tool already knows."""
    body = _method("_cond_add_unassigned")
    assert "C.unassigned(" in body
    assert "_cond_commit(" in body
    assert "Add unassigned" in TAB


def test_adding_when_nothing_is_loose_says_so():
    assert "already belongs to a condition" in _method("_cond_add_unassigned")


# ── which channel the setup starts on ────────────────────────────────────────

def test_confirming_hands_over_on_the_first_analysed_channel():
    """It handed over on whichever channel happened to be current, so
    confirming from the Conditions tab could land on EMG 2 with EMG 1 never
    configured -- and the per-channel advance then had nothing to advance to,
    because the channel it would have started from was already behind it.
    """
    body = _method("_cond_apply")
    assert "_analysis_channel_indices()" in body
    assert "_chans[0]" in body


def test_the_hand_over_clears_previous_confirmations():
    """A channel marked confirmed from before would be skipped by the advance,
    leaving it configured for the old grouping."""
    body = _method("_cond_apply")
    assert "self._chan_confirmed = set()" in body


def test_per_channel_epochs_walk_every_channel_first():
    """An epoch set on EMG 1 alone leaves EMG 2 on the file-wide window, and
    the analyst would not find out until the results disagreed."""
    body = _method("_cond_apply")
    assert "_cond_epoch_all_chans.get()" in body
    assert "_cond_visited_chans" in body


def test_the_walk_says_how_many_are_left():
    body = _method("_cond_apply")
    assert "still to review" in body


def test_the_walk_stops_before_writing_anything():
    """Half-reviewed channels must not produce an events file."""
    body = _method("_cond_apply")
    i = body.index("still to review")
    j = body.index("to_event_rows")
    assert i < j, "the walk must return before the events file is built"


def test_the_visited_set_resets_for_the_next_run():
    """Left populated, a second pass would skip every channel."""
    body = _method("_cond_apply")
    assert "_cond_visited_chans = set()" in body


def test_applying_to_all_channels_does_not_walk():
    """The common case is one window for the recording; walking every channel
    to confirm that would be ceremony."""
    body = _method("_cond_apply")
    assert "if not self._cond_epoch_all_chans.get():" in body


def test_the_header_names_the_recording_after_a_restore():
    """Restoring the rows without touching the header left "No recording
    loaded" above a table full of conditions, which contradicts itself and
    reads as the restore having half worked."""
    body = _method("_cond_tab_shown")
    assert "_cond_status.config" in body
    assert "restored" in body


def test_the_empty_header_says_what_to_do():
    assert "open one from Setup" in TAB


def test_the_channel_is_chosen_before_the_table_is_built():
    """It was chosen afterwards, so the table was built for whichever channel
    happened to be current and rebuilt again by the switch -- and the switch
    was skipped when the index already matched, leaving the wrong channel
    displayed with no way to notice."""
    body = _method("_cond_apply")
    i = body.index("self.channel_idx = _first")
    j = body.index("_build_labels_tab(sorted(self._cond_groups_in_range(groups)))")
    assert i < j, "the channel must be selected before the rebuild"


def test_the_channel_is_set_unconditionally():
    """Guarding on the index already matching left a table built for another
    channel in place."""
    body = _method("_cond_apply")
    assert "if self.channel_idx != _first" not in body
