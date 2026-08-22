"""
Overlaying a condition's trials on one set of axes.

The rule that matters: two conditions may share axes only if they resolve to
the same epoch. Overlaying different epochs would draw two time bases against
one axis, showing a latency difference that does not exist.
"""

import ast
import pathlib
import re

import numpy as np
import pytest

from mep_cmap.overlay import MAX_INDIVIDUAL_TRACES, band_of, trace_alpha
from mep_cmap.pipeline import PipelineConfig, overlay_groups, resolve_window

PKG = pathlib.Path(__file__).resolve().parent.parent / "mep_cmap"


# ── what may share axes ──────────────────────────────────────────────────────

def test_one_condition_always_overlays_with_itself():
    cfg = PipelineConfig(pre_ms=20, post_ms=400)
    out = overlay_groups(cfg, ["A"])
    assert out["A"][2] == ""
    assert out["A"][1] == (20.0, 400.0)


def test_conditions_sharing_an_epoch_may_overlay():
    cfg = PipelineConfig(
        pre_ms=20, post_ms=400,
        condition_map={"A|pre": ("A", "pre"), "A|post": ("A", "post")})
    out = overlay_groups(cfg, ["A|pre", "A|post"])
    assert set(out) == {"A"}
    keys, epoch, reason = out["A"]
    assert reason == ""
    assert epoch == (20.0, 400.0)
    # Sorted, not in the order the keys arrived: the menu must read the same
    # way whichever order the segments dict happened to be built in.
    assert keys == sorted(["A|pre", "A|post"])


def test_conditions_with_different_epochs_are_refused():
    """The whole point. Drawing these together would put a response at 30 ms
    in one condition somewhere else on the axis from a response at 30 ms in
    the other."""
    cfg = PipelineConfig(
        pre_ms=20, post_ms=400,
        window_map={"A|post": (20, 100)},
        condition_map={"A|pre": ("A", "pre"), "A|post": ("A", "post")})
    _keys, epoch, reason = overlay_groups(cfg, ["A|pre", "A|post"])["A"]
    assert epoch is None
    assert reason


def test_the_refusal_names_the_epochs_that_differ():
    """An option that is simply absent reads as a limitation of the tool. One
    that says why reads as a property of the recording."""
    cfg = PipelineConfig(
        pre_ms=20, post_ms=400,
        window_map={"A|post": (20, 100)},
        condition_map={"A|pre": ("A", "pre"), "A|post": ("A", "post")})
    reason = overlay_groups(cfg, ["A|pre", "A|post"])["A"][2]
    assert "A|pre" in reason and "A|post" in reason
    assert "400" in reason and "100" in reason


def test_comparison_is_on_the_resolved_epoch_not_the_configuration():
    """A condition carrying an explicit copy of the file-wide pair resolves
    identically and must overlay. Comparing window_map entries would refuse
    it, which is the same fault as reading a file-wide value where a per-type
    one was needed."""
    cfg = PipelineConfig(
        pre_ms=20, post_ms=400,
        window_map={"A|post": (20, 400)},
        condition_map={"A|pre": ("A", "pre"), "A|post": ("A", "post")})
    assert resolve_window(cfg, "A|pre") == resolve_window(cfg, "A|post")
    assert overlay_groups(cfg, ["A|pre", "A|post"])["A"][2] == ""


def test_a_partial_window_entry_still_resolves():
    """window_map may carry a None for one side, meaning "use the file-wide
    value for this one"."""
    cfg = PipelineConfig(
        pre_ms=20, post_ms=400,
        window_map={"A|post": (None, 400)},
        condition_map={"A|pre": ("A", "pre"), "A|post": ("A", "post")})
    assert overlay_groups(cfg, ["A|pre", "A|post"])["A"][2] == ""


def test_different_stim_types_are_never_grouped_together():
    """They are different responses in different muscles or protocols, not
    conditions of one thing."""
    cfg = PipelineConfig(pre_ms=20, post_ms=400)
    out = overlay_groups(cfg, ["A", "B"])
    assert set(out) == {"A", "B"}
    assert len(out["A"][0]) == 1 and len(out["B"][0]) == 1


def test_an_event_delay_difference_does_not_block_an_overlay():
    """A delay moves each type's t=0 onto the actual stimulus, so both are
    correctly aligned. Only the extents matter."""
    cfg = PipelineConfig(
        pre_ms=20, post_ms=400,
        delay_ms_map={"A|pre": 2.0, "A|post": -1.5},
        condition_map={"A|pre": ("A", "pre"), "A|post": ("A", "post")})
    assert overlay_groups(cfg, ["A|pre", "A|post"])["A"][2] == ""


def test_three_conditions_where_one_differs_are_refused_as_a_set():
    cfg = PipelineConfig(
        pre_ms=20, post_ms=400,
        window_map={"A|c": (20, 80)},
        condition_map={"A|a": ("A", "a"), "A|b": ("A", "b"),
                       "A|c": ("A", "c")})
    keys, epoch, reason = overlay_groups(cfg, ["A|a", "A|b", "A|c"])["A"]
    assert len(keys) == 3
    assert epoch is None and reason


# ── drawing helpers ──────────────────────────────────────────────────────────

def test_alpha_falls_as_traces_are_added():
    """Fixed opacity fails at both ends: faint at five traces, a solid block
    at sixty."""
    assert trace_alpha(1) > trace_alpha(10) > trace_alpha(60)


def test_alpha_never_becomes_invisible():
    """A trace too faint to see is not a trace."""
    assert trace_alpha(10000) >= 0.08


def test_alpha_never_exceeds_one():
    for n in (0, 1, 2, 3):
        assert 0.0 < trace_alpha(n) <= 1.0


def test_the_band_keeps_the_outlier():
    """min/max rather than mean/SD, and rather than a subsample of trials: a
    single trial unlike the rest is the thing an overlay is read for, and both
    alternatives hide it."""
    traces = [np.zeros(10), np.zeros(10), np.zeros(10)]
    traces[1] = np.full(10, 5.0)
    lo, hi, med = band_of(traces)
    assert hi.max() == 5.0
    assert lo.min() == 0.0
    assert med.max() == 0.0


def test_the_band_median_is_not_the_mean():
    traces = [np.zeros(5), np.zeros(5), np.full(5, 9.0)]
    _lo, _hi, med = band_of(traces)
    assert med.tolist() == [0.0] * 5


def test_many_traces_switch_to_a_band_automatically():
    assert MAX_INDIVIDUAL_TRACES > 1
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    assert "n_total > MAX_INDIVIDUAL_TRACES" in src


# ── the overlay must not become a second detector ────────────────────────────

def test_the_overlay_detects_nothing():
    """Both the traces and the onsets come from what the preview already
    computed. A second detection path would be a second answer."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    for forbidden in ("dispatch_onset", "pipeline_detect_onsets",
                      "detect_mep_onset"):
        assert forbidden not in src


def test_onsets_come_from_the_seed():
    """The seed holds what the analysis detector returned. Re-deriving here
    would let the overlay disagree with the trial view beside it."""
    src = (PKG / "preview.py").read_text(encoding="utf-8")
    body = src[src.index("def _preview_overlay_payload"):]
    body = body[:body.index("def _preview_combined")]
    assert 'seed.get((key, disp))' in body
    assert "onset_idx" in body


def test_the_menu_is_built_from_the_shared_rule():
    """What is offered and what can be drawn must come from one answer."""
    src = (PKG / "preview.py").read_text(encoding="utf-8")
    body = src[src.index("def _preview_combined"):]
    assert "overlay_groups(cfg, keys)" in body


def test_a_refused_combination_is_named_not_hidden():
    """An option that is simply absent reads as a limitation of the tool; one
    that says why reads as a property of the recording."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("class CombinedPreviewWindow"):]
    # Excluded from the choices, since it cannot be drawn honestly...
    assert "for lbl, _m, reason in options if not reason" in body
    # ...but its reason is still put on screen.
    assert "if reason]" in body


def test_the_combined_window_cannot_take_the_preview_down():
    """It is a view. A fault in it must not cost the analyst the
    trial-by-trial review they asked for."""
    src = (PKG / "preview.py").read_text(encoding="utf-8")
    i = src.index("self._preview_combined(payload")
    assert "try:" in src[i - 200:i]
    assert "Combined preview unavailable" in src[i:i + 400]
    # The fallback is the single-trial view on its own, not nothing.
    assert "_open_inspector_preview(" in src[i:i + 1200]


def test_nothing_is_saved_by_the_overlay():
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    for forbidden in ("to_csv", "savefig", "json.dump", "open("):
        assert forbidden not in src


# ── the stale docstring ──────────────────────────────────────────────────────

def test_the_module_no_longer_claims_anchoring_is_unfaithful():
    """It said anchored types preview with the file-wide window. That stopped
    being true when the preview began detecting over every trial of a type,
    and the docstring went on saying it."""
    src = (PKG / "preview.py").read_text(encoding="utf-8")
    head = src[:src.index('"""', src.index('"""') + 3)]
    assert "preview with the file-wide window" not in head
    assert "Amplitude window anchoring IS faithful" in head


def test_the_preview_still_detects_over_every_trial():
    """The reason anchoring is faithful. If this regressed, the docstring
    above would be wrong again."""
    src = (PKG / "preview.py").read_text(encoding="utf-8")
    assert '(payload.get("every") or {}).get(_st) or _segs' in src


# ── offsets, cSP and the blanking gap in the preview ─────────────────────────

PREVIEW = (PKG / "preview.py").read_text(encoding="utf-8")


def test_the_preview_config_carries_the_gap_and_the_csp_assignment():
    """Both were missing. The gap moves the window PreStimRMS is measured
    over, and csp_types decides whether the end of the MEP is the start of a
    silent period or a return to baseline -- which changes the offset, its
    recorded provenance, and the duration derived from it."""
    body = PREVIEW[PREVIEW.index("_cfg = PipelineConfig("):]
    body = body[:body.index("_fs = payload")]
    assert "gap_ms_map=" in body
    assert "csp_types=" in body


def test_offsets_and_csp_use_the_analysis_detectors():
    """Not reimplemented. A preview computing offsets its own way would be a
    second answer, and the trial view below the overlay would disagree with
    the strip above it."""
    body = PREVIEW[PREVIEW.index("def _preview_detect_extras"):]
    body = body[:body.index("def _preview_prestim_window_ms")]
    assert ("from .detection.csp_detection import CspSettings, "
            "detect_csp_for_trial") in body
    assert "from .detection.offset_detection import resolve_mep_offset" in body


def test_the_silent_period_is_found_before_the_offset():
    """The offset rule takes a detected cSP start as the end of the MEP: the
    two are one physical event. Reversed, every trial with a silent period
    would report a baseline return instead."""
    body = PREVIEW[PREVIEW.index("def _preview_detect_extras"):]
    body = body[:body.index("def _preview_prestim_window_ms")]
    assert body.index("detect_csp_for_trial(") < body.index("resolve_mep_offset(")
    assert "csp_start_ms=csp_start_ms" in body


def test_the_offset_seeds_the_field_the_analysis_reads():
    """mep_offset_idx, silent_start_idx and silent_end_idx are the names the
    pipeline and the Inspector both look for."""
    body = PREVIEW[PREVIEW.index("def _preview_detect_extras"):]
    for field in ("mep_offset_idx", "silent_start_idx", "silent_end_idx"):
        assert field in body


def test_a_configured_csp_that_finds_nothing_says_why():
    """A silent period assigned but never found is a setting to look at, and
    detect_csp already says which one through reason_out. Discarding that left
    "none found" indistinguishable from a search window that never covered the
    suppression, or from a raised exception."""
    body = PREVIEW[PREVIEW.index("def _preview_detect_extras"):]
    assert "reason_out=_why" in body
    assert "No silent period was found" in body
    assert "csp_reasons" in body


def test_cSP_not_being_assigned_is_distinguished_from_not_being_found():
    """Two different situations with the same symptom: an empty offset row."""
    body = PREVIEW[PREVIEW.index("def _preview_detect_extras"):]
    assert "not assigned to any stimulus type" in body


def test_the_onset_count_counts_onsets_not_seed_entries():
    """The seed also holds offsets and silent periods, and those are created
    for trials whose onset detection found nothing. Counting entries reported
    'B 20/20' on a type with six onsets and twenty silent periods -- the
    opposite of what that line exists to say, and it moved when cSP settings
    changed while the anchor median stayed put."""
    body = PREVIEW[PREVIEW.index("# Say what was seeded, positively."):]
    body = body[:body.index("self.log(\"   Onsets pre-detected")]
    assert '_m.get("onset_idx") is not None' in body
    # Both the numerator and the "you placed" set must agree on what counts.
    assert body.count('_m.get("onset_idx") is not None') == 2


def test_the_window_clears_the_gap_of_the_types_being_drawn():
    """The blanking gap is PER stimulus type. Taken over every key in the
    file, one type's 20 ms gap displaced the baseline band on every other
    type in the recording -- a shaded window simply wrong wherever it was not
    the type it came from."""
    cfg = _Cfg(100.0, {"A": 20.0, "B": 0.0, "C": 0.0})
    # Drawing B alone must not inherit A's gap.
    assert _prestim_window(cfg, ["B"]) == (-100.0, 0.0)
    assert _prestim_window(cfg, ["A"]) == (-120.0, -20.0)


def test_the_baseline_window_is_recomputed_when_the_type_changes():
    """Fixed at construction it would keep the first type's gap for the life
    of the window."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("    def _redraw"):]
    body = body[:body.index("    def _on_pick_trial")]
    assert "groups, window, baseline = self.groups_for(" in body
    assert "window, baseline)" in body
    # set_groups must accept it, or the recomputed value goes nowhere.
    panel = src[src.index("    def set_groups"):]
    assert "prestim_window_ms=None" in panel[:200]


def test_the_conditions_overlaid_share_one_gap():
    """Several conditions of a type can carry different gaps; the larger one
    is used, since the smaller would shade samples it excludes."""
    cfg = _Cfg(100.0, {"A|pre": 10.0, "A|post": 40.0})
    assert _prestim_window(cfg, ["A|pre", "A|post"]) == (-140.0, -40.0)


def test_the_strip_has_a_row_for_the_silent_period_end():
    """Its START is the MEP offset row: one physical event, already drawn.
    Drawing it twice would show one finding as two."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("if self.show_rug.get():"):]
    body = body[:body.index("self._draw_ptp_window()")]
    assert 'g.get("csp_end_ms")' in body
    assert body.count("self.ax_rug.plot(") == 3
    assert '"cSP end", "offset", "onset"' in src


def test_the_csp_end_survives_a_trial_filter():
    """Traces, onsets, offsets and cSP ends are parallel lists. Filtering one
    and not another attaches a trial's landmark to a different trial."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("def _filter_groups"):]
    body = body[:body.index("def _sync_inspector_type")]
    assert 'csp_end_ms=_pick(g.get("csp_end_ms"))' in body


def test_the_preview_supplies_only_the_csp_end():
    body = PREVIEW[PREVIEW.index("def _preview_overlay_payload"):]
    body = body[:body.index("def _preview_combined")]
    assert "silent_end_idx" in body
    assert '"csp_end_ms"' in body
    # The start would duplicate the offset row.
    assert "silent_start_idx" not in body


def test_the_arrow_keys_work_wherever_focus_sits():
    """bind() on the window only fires when the focused widget lets the event
    propagate, and the canvas, the listbox and the combobox all consume
    arrows -- so it worked only after clicking the lower plot."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    assert 'self.win.bind_all("<Left>"' in src
    assert 'self.win.bind_all("<Right>"' in src


def test_the_application_wide_binding_is_released_on_close():
    """bind_all outlives the window otherwise, and would go on stepping trials
    in a window with no inspector to step."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("    def close(self):"):]
    assert "unbind_all" in body
    # Before the window is destroyed, or the widget it is called on is gone.
    assert body.index("unbind_all") < body.index("self.win.destroy()")


def test_only_text_entry_keeps_its_arrows():
    """Arrows move a cursor there and stealing them would make typing
    impossible. The listbox gives its arrows up deliberately: its selection
    now drives the trial below, so the two would fight."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("def _step_trial"):]
    body = body[:body.index("def _members_for")]
    assert "tk.Entry" in body and "tk.Text" in body
    assert "Listbox" not in body


def test_picking_one_trial_moves_the_trial_view():
    """An unambiguous request to look at it. Several is a request to compare
    them, and there is no single trial to show."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("    def _redraw"):]
    body = body[:body.index("    def _on_pick_trial")]
    assert "len(pairs) == 1" in body
    assert "self._on_pick_trial(key, number)" in body


def test_rectification_is_display_only():
    """The amplitude window, the baseline band and the onset strip are all
    drawn from values computed on the RAW signal, because that is what the
    analysis measures. Rectifying is a way of looking at a response, not a way
    of measuring one."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("    def draw(self):"):]
    body = body[:body.index("    def _draw_ptp_window")]
    assert "np.abs(traces)" in body
    # Nothing else in the drawing path is rectified.
    assert body.count("np.abs(") == 1


def test_rectification_precedes_the_median_and_the_band():
    """A rectified average is the average of the rectified trials, not the
    rectified average of the raw ones. The two differ wherever trials disagree
    in sign, which on a biphasic response is most of the epoch."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("    def draw(self):"):]
    body = body[:body.index("    def _draw_ptp_window")]
    i_rect = body.index("np.abs(traces)")
    assert i_rect < body.index("band_of(traces)")
    assert i_rect < body.index("np.median(traces")


def test_a_rectified_plot_says_so():
    """Pasted into a slide it is otherwise indistinguishable from a
    monophasic response, with markers around it computed on the raw signal."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    assert "RECTIFIED (display only)" in src


def test_rectifying_matches_taking_the_absolute_value():
    """The band and the median must follow the rectified traces, not the raw
    ones."""
    import numpy as _np
    traces = [_np.array([-2.0, 1.0]), _np.array([2.0, -1.0])]
    lo, hi, med = band_of([_np.abs(t) for t in traces])
    assert lo.tolist() == [2.0, 1.0]
    assert hi.tolist() == [2.0, 1.0]
    assert med.tolist() == [2.0, 1.0]
    # Rectifying AFTER the median would give zeros here, which is the
    # difference the ordering above exists to avoid.
    _lo, _hi, raw_med = band_of(traces)
    assert _np.abs(raw_med).tolist() == [0.0, 0.0]


def test_the_csp_detector_is_imported_by_its_real_name():
    """It is detect_csp_for_trial, the shared entry point the pipeline and the
    Inspector also use. Importing a name that does not exist raised ImportError
    inside the extras stage, which the guard reported but which meant no silent
    period was ever seeded. Asserted by IMPORTING it rather than by matching
    the source, so a rename fails here rather than at run time."""
    from mep_cmap.detection.csp_detection import (CspSettings,
                                                  detect_csp_for_trial)
    assert callable(detect_csp_for_trial)
    assert "detect_csp_for_trial" in PREVIEW
    assert "CspSettings" in PREVIEW


def test_every_csp_keyword_the_preview_passes_exists():
    """A renamed parameter would raise inside a guard and read as 'no silent
    period found'.

    The preview used to name every detector parameter itself. It now passes a
    CspSettings, so the settings are checked as FIELDS of that object; the
    keywords left are the three per-trial values plus reason_out."""
    import inspect
    from mep_cmap.detection.csp_detection import (CspSettings,
                                                  detect_csp_for_trial)
    params = set(inspect.signature(detect_csp_for_trial).parameters)
    for kw in ("emg_seg", "fs", "time_axis", "settings",
               "second_peak_ms", "pre_ms", "reason_out"):
        assert kw in params, kw

    fields = set(CspSettings.__dataclass_fields__)
    for name in ("min_silence_ms", "min_return_ms", "criterion",
                 "significance", "n_boot", "search_end_ms",
                 "max_mep_offset_ms", "rms_window_ms"):
        assert name in fields, name


def test_every_offset_keyword_the_preview_passes_exists():
    import inspect
    from mep_cmap.detection.offset_detection import resolve_mep_offset
    params = set(inspect.signature(resolve_mep_offset).parameters)
    for kw in ("onset_ms", "csp_start_ms", "csp_enabled", "manual_offset_ms"):
        assert kw in params, kw


def test_an_offset_needs_an_onset():
    """Verified against the detector, not assumed: the offset row can only
    ever be as full as the onset row, so an empty offset strip beside a thinly
    populated onset strip is the expected consequence rather than a fault in
    the offset stage."""
    import numpy as _np
    from mep_cmap.detection.offset_detection import resolve_mep_offset
    fs = 5000.0
    sb = int(120 * fs / 1000)
    # A NOISY baseline, not a flat one. The envelope detector thresholds the
    # return against the baseline's own variability, so a perfectly zero
    # baseline gives it nothing to work with and it declines on every trial --
    # which would make this test pass or fail for a reason unrelated to the
    # onset.
    _rng = _np.random.default_rng(0)
    seg = _rng.normal(0.0, 0.01, sb + int(400 * fs / 1000))
    i0, i1 = sb + int(15 * fs / 1000), sb + int(45 * fs / 1000)
    seg[i0:i1] += 2.0 * _np.sin(_np.linspace(0, _np.pi * 3, i1 - i0))
    common = dict(pre_ms=120.0, search_end_ms=400.0, min_duration_ms=5.0,
                  max_duration_ms=100.0, min_return_ms=5.0,
                  env_window_ms=5.0, criterion=1.96, peak_frac=0.0,
                  manual_offset_ms=None, csp_start_ms=None, csp_enabled=False)
    assert resolve_mep_offset(seg, fs, onset_ms=None, **common).offset_ms is None
    assert resolve_mep_offset(seg, fs, onset_ms=15.0,
                              **common).offset_ms is not None


def test_the_overlay_note_counts_the_landmarks_it_drew():
    """An empty offset row is otherwise ambiguous between the detector
    finding none, them never being seeded, and the strip not drawing them."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("def _update_note"):]
    assert "onset(s)," in body and "offset(s)" in body


def test_one_control_selects_the_event_type_for_both_panels():
    """Two dropdowns for one decision could disagree, and did: the overlay
    showing B while the trial view showed A reads as the trial view
    contradicting the summary above it, not as two controls out of step."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("class CombinedPreviewWindow"):]
    # The Inspector's own dropdown is disabled, not removed: nothing about
    # how it redraws should change.
    assert 'self.inspector.dd_event.configure(state="disabled")' in body
    # ...and it is driven from the single control instead.
    assert "def _sync_inspector_type" in body
    assert "self._sync_inspector_type()" in body


def test_changing_trials_does_not_reset_the_trial_view():
    """Only a change of CONDITION re-points the trial view. Otherwise picking
    trials in the list would keep throwing away the segment being read."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("    def _redraw"):]
    body = body[:body.index("    def _on_pick_trial")]
    i_guard = body.index("if refill:", body.index("set_groups"))
    i_sync = body.index("self._sync_inspector_type()")
    assert i_guard < i_sync


def test_an_overlay_of_several_conditions_shows_the_first_below():
    """The trial view can only show one. A view claiming to show all of them
    would be the misleading option."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("def _sync_inspector_type"):]
    body = body[:body.index("def _redraw")]
    assert "members[0]" in body


def test_an_extras_failure_cannot_discard_the_onsets():
    """They were computed successfully before the extras stage ran. Sharing
    one handler meant a cSP fault emptied the seed, showing an empty strip
    beside a trial view with a perfectly good latency -- which points at the
    wrong stage entirely."""
    i = PREVIEW.index("self._preview_detect_extras(payload")
    guard = PREVIEW[i - 700:i]
    assert "Guarded SEPARATELY" in guard
    # Only the INNER handler, up to where the outer one begins. The outer
    # handler empties the seed by design, so a loose window would read its
    # code and pass regardless of what the inner one does.
    inner = PREVIEW[i:PREVIEW.index("        except Exception as exc:", i)]
    assert "onsets are" in inner and "unaffected" in inner
    assert "_seed = {}" not in inner


class _Cfg:
    def __init__(self, prestim_ms, gap_ms_map=None):
        self.prestim_ms = prestim_ms
        self.gap_ms_map = gap_ms_map or {}


def _prestim_window(cfg, keys):
    """Call the method without constructing the app."""
    import types
    src = PREVIEW[PREVIEW.index("    def _preview_prestim_window_ms"):]
    mod = ast.parse("\n".join(l[4:] for l in src.splitlines()))
    ns = {}
    exec(compile(mod, "<x>", "exec"), {}, ns)
    return ns["_preview_prestim_window_ms"](None, cfg, keys)


def test_the_shaded_window_is_the_prestim_one_not_the_detector_baseline():
    """They are DIFFERENT intervals whenever a gap is set. The analysis cuts
    two things per trial: the epoch, whose pre-stimulus part the detectors
    threshold against, and a separate pre-stimulus segment ending a gap before
    the stimulus, which is what PreStimRMS is computed from. Drawing one while
    labelling it the other is how a 50 ms gap came to look as though it had no
    effect."""
    assert _prestim_window(_Cfg(100.0), ["A"]) == (-100.0, 0.0)
    assert _prestim_window(_Cfg(100.0, {"A": 50.0}), ["A"]) == (-150.0, -50.0)


def test_the_window_clears_the_largest_gap_across_the_drawn_types():
    """One shaded band for a plot that may hold several conditions; the
    smaller gap would draw over samples the larger one excludes."""
    cfg = _Cfg(100.0, {"A": 10.0, "B": 40.0})
    assert _prestim_window(cfg, ["A", "B"]) == (-140.0, -40.0)


def test_a_malformed_gap_does_not_break_the_drawing():
    cfg = _Cfg(100.0, {"A": "not a number"})
    assert _prestim_window(cfg, ["A"]) == (-100.0, 0.0)


def test_the_strip_draws_offsets_in_their_own_row():
    """Onsets and offsets together say what the DURATION distribution looks
    like, which neither says alone."""
    src = (PKG / "overlay.py").read_text(encoding="utf-8")
    body = src[src.index("if self.show_rug.get():"):]
    body = body[:body.index("self._draw_ptp_window()")]
    assert 'g.get("offsets_ms")' in body
    # Onset, offset and cSP end: three rows at three heights, never stacked
    # on one line where they could not be told apart.
    heights = re.findall(r"np\.full\(len\([a-z]+\), ([0-9.]+)\)", body)
    assert len(set(heights)) == 3


def test_the_overlay_payload_carries_offsets():
    body = PREVIEW[PREVIEW.index("def _preview_overlay_payload"):]
    body = body[:body.index("def _preview_combined")]
    assert "mep_offset_idx" in body
    assert '"offsets_ms"' in body
