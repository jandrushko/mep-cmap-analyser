"""
The unattended flag through the file-load path.

Five points in _browse_file_path ask the analyst something. A batch re-run has
to get past all five without a person there, and the dangerous way to do that
is to suppress modal dialogues wholesale: every one of them leaves a variable
set to whatever it happened to hold, and the run continues against it.

So each point is threaded individually and reads its answer from the restored
session. These tests pin that, and pin that the interactive path is unchanged
when the flag is absent.

app.py cannot be imported here (matplotlib's Tk backend), so these read the
source, as test_app_shutdown.py does.
"""

import pathlib

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "mep_cmap" / "app.py").read_text(encoding="utf-8")


def _between(start, end, after=0):
    a = SRC.index(start, after)
    return SRC[a:SRC.index(end, a)]


def test_every_decision_point_is_threaded():
    """
    Five, not four: the crop range, the SMR channel assignment, the general
    channel assignment, the amplitude unit, and the study metadata. Missing
    one means a batch run stops on a modal dialogue with nobody watching.

    The study metadata was the one missed the first time. It sits at the very
    END of _browse_file_path, after every choice the planner checks, so a file
    that planned perfectly cleanly still stopped there.
    """
    assert SRC.count('getattr(self, "_unattended", None)') >= 5


def test_the_flag_defaults_to_absent_everywhere_it_is_read():
    """
    Read with getattr and a None default, so an app that never sets it -- the
    normal case -- behaves exactly as before rather than raising.
    """
    for chunk in SRC.split('getattr(self, "_unattended"')[1:]:
        assert chunk.lstrip().startswith(", None)")


# ── the amplitude unit ───────────────────────────────────────────────────────

def test_an_unrecorded_unit_is_not_guessed():
    """
    Every amplitude in the file scales by this. The suggestion is good enough
    to lead with when a person can disagree with it, and not good enough to
    apply silently across a study.
    """
    body = _between("def _ask_epoched_unit", "dlg = tk.Toplevel")
    assert "emg_unit" in body
    assert "not guessing" in body
    assert body.count("return None") >= 1


def test_a_recorded_unit_is_reused_without_asking():
    body = _between("def _ask_epoched_unit", "dlg = tk.Toplevel")
    assert "return _saved" in body


# ── the channel assignment ───────────────────────────────────────────────────

def test_the_substituted_choice_goes_through_the_same_apply_path():
    """
    _apply_choice is what the dialogue calls. Routing the unattended answer
    through it too means there is ONE place a channel choice takes effect,
    rather than a second path that can drift from the first.
    """
    body = _between("_reopen_assignment = lambda", "# All channels available")
    assert "_show_assign_dlg()" in body, "the interactive path must remain"
    assert "_apply_choice()" in body
    assert body.index("_apply_choice()") > body.index("_show_assign_dlg()")


def test_the_unattended_channel_is_checked_against_this_file():
    """
    A name from the session that this recording does not have must fall
    through to the dialogue rather than being applied.
    """
    body = _between("_reopen_assignment = lambda", "# All channels available")
    assert '_un.get("channel_choice") in chan_list' in body


def test_smr_keeps_its_own_substitution():
    """SMR has a separate dialogue, so it needs its own, or it still blocks."""
    body = _between("def _show_smr_dialog", "if not _chosen:")
    assert "_show_smr_dialog()" in body
    # Checked against `analogue`, this branch's channel list. chan_list is not
    # bound until later in the method, so naming it here raised NameError the
    # first time a batch reached an SMR file -- and nowhere else. Asserted on
    # the unattended block alone: the dialogue above it mentions chan_list for
    # its own reasons.
    assert '_un.get("channel_choice") in analogue' in body
    block = body[body.index("_un = getattr"):]
    assert "chan_list" not in block


# ── the crop range ───────────────────────────────────────────────────────────

def test_a_saved_range_is_reused_without_asking():
    body = _between("_has_saved_range   = bool", "# ── Filter stim types")
    assert 'getattr(self, "_unattended", None) is not None' in body
    assert 'choice = "reuse"' in body


def test_a_session_with_no_saved_range_analyses_the_whole_recording():
    """
    NOT a prompt, and not an assumption. "Use whole file" is IMPLEMENTED as
    setting crop_ranges, crop_start and crop_end all to None, so a session
    carrying none of them is recording a whole-file analysis. 134 of the 135
    sessions in the study are that case.

    This asked instead, on the reasoning that the planner would refuse such
    files and a stray one should ask rather than assume. The planner does not
    refuse them, so a run of 110 files put up 110 modal dialogues and analysed
    nothing.
    """
    body = _between("_has_saved_range   = bool", "# ── Filter stim types")
    guarded = body[:body.index("elif _has_saved_range")]
    assert "self.crop_ranges = None" in guarded
    assert "whole recording" in guarded


def test_the_interactive_path_still_asks_about_the_range():
    """Only the unattended branch changed."""
    body = _between("_has_saved_range   = bool", "# ── Filter stim types")
    assert "Analyse whole recording?" in body
    assert "Select new range" in body


# ── nothing moves the analyst's view during a batch ──────────────────────────

def test_the_load_does_not_jump_to_the_conditions_tab_unattended():
    """
    Right when a file is opened by hand, wrong a hundred times in a row: the
    tab flicks on every file and the run looks like it is waiting to be
    answered.

    It is also side-effectful. The Conditions tab autosaves the session when
    conditions are confirmed there, so navigating into it during a run puts a
    write to the session on a path that should only be reading it -- and four
    sessions were rewritten that way on the first real run.
    """
    body = _between("_go_to_conditions_after_load", "# No navigation here.")
    assert 'getattr(self, "_unattended", None) is None' in body
    assert body.index('_unattended') < body.index("nb_setup.select")


def test_a_refused_setup_neither_pops_a_dialogue_nor_navigates():
    """
    The three setup guards each showed a modal and then moved the tabs. A
    modal stops a batch dead on a file nobody is watching; the navigation
    yanks the view once per refused file.

    run_analysis_start must still return False, so the runner records the file
    as failed and carries on.
    """
    body = _between("def _setup_refused", "def _unconfigured_analysis_channels")
    assert 'getattr(self, "_unattended", None) is not None' in body
    guarded = body[:body.index("messagebox.showwarning")]
    assert "self.log(" in guarded
    assert "return" in guarded


def test_the_guards_all_route_through_it():
    """
    One place, because three copies of the same two lines is how one of them
    gets updated and the others do not.
    """
    body = _between("def _validate_analysis_setup", "def _setup_refused")
    assert body.count("self._setup_refused(") == 3
    # None of them may still navigate or warn on their own.
    assert "self.notebook.select(self.stage1_outer)" not in body


# ── the study metadata ───────────────────────────────────────────────────────

def test_restored_metadata_is_reused_without_asking():
    """
    _apply_loaded_session restores study_metadata before the load reaches
    here, so the dialogue would be asking for something already known -- and
    it grab_sets and waits, which stops a batch dead on its first file.
    """
    body = _between("def prompt_study_metadata", "parsed = self._parse_bids")
    assert "participant_id" in body
    assert "from the saved session" in body


def test_a_session_without_a_participant_id_is_not_filled_in():
    """
    Output paths are built from it, and _remembered_meta deliberately carries
    the PREVIOUS file's metadata across files. Falling back to that would
    write one recording's results under another participant's identifier --
    which looks like a successful run.
    """
    body = _between("def prompt_study_metadata", "parsed = self._parse_bids")
    assert "not " in body and "guessing" in body
    # Both branches of the guard leave the method: one having reused the
    # restored identifier, one having refused. Neither may fall through to
    # the dialogue below, which is where _remembered_meta would be read.
    assert body.count("return") >= 2


def test_the_interactive_path_still_builds_the_dialogue():
    """The guard returns early only when the flag is set."""
    body = _between("def prompt_study_metadata", "def _show_bidsify_preview")
    assert "tk.Toplevel" in body
    assert "wait_window" in body
