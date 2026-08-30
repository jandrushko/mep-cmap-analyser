"""
BDF is the default container for BIDS-ify.

EMG is recorded at amplifier gains that put a 16-bit container's quantisation
step within range of the signal itself, and the post-conversion read-back check
tolerates one step -- so an EDF conversion can verify while having lost more
than the analyst would accept. BDF's 24 bits remove the question.

The default is stated in five places (the plan, the planner, the low-level
writer, the per-study state and the dialogue). They must agree: a caller that
omits the argument anywhere would otherwise write a different format from the
one the analyst chose.
"""

import pathlib

import pytest

from mep_cmap.bidsify import DatasetLayout, Plan, plan_bidsify, write_recording
from mep_cmap.bidsify_dialog import assemble_result
from mep_cmap.bidsify_state import BidsifyState


def _sig(fn):
    import inspect
    return inspect.signature(fn).parameters["container"].default


def test_a_new_study_defaults_to_bdf(tmp_path):
    assert BidsifyState.load_or_create(str(tmp_path)).container == "BDF"


def test_the_plan_defaults_to_bdf(tmp_path):
    plan = Plan(layout=DatasetLayout(rawdata_root=str(tmp_path / "rawdata")))
    assert plan.container == "BDF"


def test_the_planner_defaults_to_bdf():
    assert _sig(plan_bidsify) == "BDF"


def test_the_writer_defaults_to_bdf():
    """The low-level default matters too: a caller that omits it would
    otherwise write the other format."""
    assert _sig(write_recording) == "BDF"


def test_the_dialogue_falls_back_to_bdf_on_an_unrecognised_value():
    r = assemble_result("TMS", {}, "not-a-container", 50, "A")
    assert r.container == "BDF"


def test_an_explicit_choice_is_still_honoured():
    """Both formats remain available; only the starting point changed."""
    assert assemble_result("TMS", {}, "EDF", 50, "A").container == "EDF"
    assert assemble_result("TMS", {}, "BDF", 50, "A").container == "BDF"


def test_a_study_that_chose_edf_keeps_it(tmp_path):
    """The default applies to NEW studies. One already converted as EDF must
    not silently switch container half way through."""
    st = BidsifyState.load_or_create(str(tmp_path))
    st.container = "EDF"
    st.save()
    assert BidsifyState.load_or_create(str(tmp_path)).container == "EDF"


def test_the_dialogue_offers_bdf_first():
    """The order shown should match the default, or the first radio button
    disagrees with the one that is selected."""
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mep_cmap" / "bidsify_dialog.py").read_text(encoding="utf-8")
    assert 'for c in ("BDF", "EDF")' in src


def test_both_containers_are_still_accepted_by_the_planner(tmp_path):
    lay = DatasetLayout(rawdata_root=str(tmp_path / "rawdata"))
    for c in ("EDF", "BDF"):
        assert plan_bidsify([], lay, container=c).container == c
    with pytest.raises(ValueError):
        plan_bidsify([], lay, container="WAV")
