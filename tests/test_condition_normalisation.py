"""
Normalisation and EMG compensation with conditions (v1.4.10).

A recording with A and B each split into two conditions, and B normalised to
A per condition on tab 1a (B·haptic -> A·haptic). Trial rows carry the
stimulus and its condition apart (StimType=B, Condition=haptic), and both
stages looked rows up by StimType alone:

* normalisation found no instruction for any condition-split row and left
  Normalised_PTP blank without a warning;
* EMG compensation pooled both conditions of a stimulus type into one fit.
"""

import pathlib

import numpy as np

from mep_cmap.pipeline import (LAT_COLS, PipelineConfig, group_key_for,
                               split_group_key)
from mep_cmap.normalisation import apply_normalisation, apply_emg_compensation

COL = {c: i for i, c in enumerate(LAT_COLS)}
PKG = pathlib.Path(__file__).resolve().parents[1] / "mep_cmap"

CMAP = {"A·haptic": ("A", "haptic"), "A·no_haptic": ("A", "no_haptic"),
        "B·haptic": ("B", "haptic"), "B·no_haptic": ("B", "no_haptic")}
REFS = {"B·haptic": "A·haptic", "B·no_haptic": "A·no_haptic"}


def _cfg():
    cfg = PipelineConfig()
    cfg.condition_map = dict(CMAP)
    cfg.reference_map = dict(REFS)
    return cfg


def _row(stim, cond, ptp, rms=0.01):
    r = [None] * len(LAT_COLS)
    r[COL["StimType"]] = stim
    r[COL["Condition"]] = cond
    r[COL["PTP(mV)"]] = ptp
    r[COL["PreStimRMS"]] = rms
    r[COL["Outlier_Decision"]] = "Not flagged"
    return r


def _recording():
    # Reference means: A·haptic 2.0, A·no_haptic 4.0. B is 1.0 throughout, so
    # the correct ratios are 0.5 (haptic) and 0.25 (no_haptic); pooling A
    # would give 1/3 for both.
    return ([_row("A", "haptic", v) for v in (1.0, 2.0, 3.0)]
            + [_row("A", "no_haptic", v) for v in (3.0, 4.0, 5.0)]
            + [_row("B", "haptic", 1.0) for _ in range(3)]
            + [_row("B", "no_haptic", 1.0) for _ in range(3)])


def _ptps(rows, key):
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(float(r[COL["PTP(mV)"]]))
    return out


# ── the shared rule ──────────────────────────────────────────────────────────

def test_group_key_for_is_the_inverse_of_split_group_key():
    cfg = _cfg()
    for key in CMAP:
        assert group_key_for(cfg, *split_group_key(cfg, key)) == key


def test_no_condition_is_the_stimulus_type():
    cfg = PipelineConfig()
    assert group_key_for(cfg, "A", "") == "A"
    assert group_key_for(cfg, "A", None) == "A"
    assert group_key_for(_cfg(), "A", "n/a") == "A"


# ── normalisation ────────────────────────────────────────────────────────────

def test_each_condition_is_normalised_to_its_own_reference():
    cfg, rows, log = _cfg(), _recording(), []

    def key(r):
        return group_key_for(cfg, r[COL["StimType"]], r[COL["Condition"]])

    apply_normalisation(rows, COL, _ptps(rows, key), cfg.reference_map,
                        plateau_tolerance=0, log_callback=log.append,
                        key_of=key)
    norm = {}
    for r in rows:
        if r[COL["StimType"]] == "B":
            norm.setdefault(r[COL["Condition"]], set()).add(r[COL["Normalised_PTP"]])
    assert norm == {"haptic": {0.5}, "no_haptic": {0.25}}
    assert not any(m.startswith("⚠️") for m in log)
    # A reference is never normalised itself.
    assert all(r[COL["Normalised_PTP"]] is None for r in rows
               if r[COL["StimType"]] == "A")


def test_the_old_lookup_is_now_reported_not_silent():
    # Without the group-key rule (the 1.4.9 behaviour) nothing matches; the
    # columns stay blank, but the log now says so for each group.
    cfg, rows, log = _cfg(), _recording(), []

    def key(r):
        return group_key_for(cfg, r[COL["StimType"]], r[COL["Condition"]])

    apply_normalisation(rows, COL, _ptps(rows, key), cfg.reference_map,
                        plateau_tolerance=0, log_callback=log.append)
    assert all(r[COL["Normalised_PTP"]] is None for r in rows)
    warned = [m for m in log if m.startswith("⚠️")]
    assert any("'B·haptic'" in m for m in warned)
    assert any("'B·no_haptic'" in m for m in warned)


def test_a_reference_with_no_trials_is_reported():
    log = []
    rows = [_row("B", "", 1.0)]
    apply_normalisation(rows, COL, {"B": [1.0]}, {"B": "M"},
                        plateau_tolerance=0, log_callback=log.append)
    assert any(m.startswith("⚠️") and "'M'" in m and "no trials" in m
               for m in log)


def test_unconditioned_recordings_behave_as_before():
    rows = [_row("A", "", v) for v in (2.0, 2.0)] + [_row("B", "", 1.0)]
    apply_normalisation(rows, COL, {"A": [2.0, 2.0], "B": [1.0]}, {"B": "A"},
                        plateau_tolerance=0, log_callback=lambda _: None)
    assert rows[-1][COL["Normalised_PTP"]] == 0.5


# ── EMG compensation ─────────────────────────────────────────────────────────

def test_compensation_fits_each_condition_separately():
    rng = np.random.default_rng(0)
    rows = []
    for cond, slope in (("haptic", 20.0), ("no_haptic", -5.0)):
        rms = rng.uniform(0.005, 0.02, 20)
        for x in rms:
            rows.append(_row("A", cond, 1.0 + slope * x + rng.normal(0, 0.02), x))
    apply_emg_compensation(rows, COL, extra_group_cols=("Condition",),
                           log_callback=lambda _: None)
    n = {r[COL["Condition"]]: r[COL["EMGComp_N"]] for r in rows}
    assert n == {"haptic": 20, "no_haptic": 20}
    slopes = {r[COL["Condition"]]: r[COL["EMGComp_Slope"]] for r in rows}
    assert slopes["haptic"] > 0 > slopes["no_haptic"]


# ── wired into the pipeline ──────────────────────────────────────────────────

def test_pipeline_uses_the_group_key_in_both_stages():
    src = (PKG / "pipeline.py").read_text(encoding="utf-8")
    assert src.count('extra_group_cols=("Condition",)') == 2
    assert src.count("key_of=_row_key") == 2
    assert "_st  = _row_key(_row)" in src
