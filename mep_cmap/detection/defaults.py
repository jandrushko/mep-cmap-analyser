"""
mep_cmap.detection.defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The single source of truth for detection parameter defaults.

Before this module the same default appeared in up to four places — the
``PipelineConfig`` dataclass, ``preferences.DEFAULTS``, the Tk variable
initialisers in ``app.py``, and the session-restore fallbacks in the same file
— and they were already out of step: ``PipelineConfig.onset_method`` defaulted
to ``peak_fraction`` while ``preferences`` defaulted to ``bigoni``. The GUI
masked it, because it always passes an explicit value, so the divergence only
surfaced for code driving ``PipelineConfig`` directly.

Every default is defined here once. Consumers read from these dicts rather than
restating literals, and ``tests/test_detection_defaults.py`` asserts that the
config, the preferences and this module agree on every shared key — so a future
change made in only one place fails the suite instead of shipping.

Keys are named for their ``PipelineConfig`` field. Three preference keys
predate that convention and cannot be renamed without invalidating every saved
``~/.mep_cmap/preferences.json``; ``PREF_KEY_ALIASES`` maps them.

  * DEFAULT_ONSET_METHOD
  * ONSET_DEFAULTS / OFFSET_DEFAULTS / DETECTION_DEFAULTS
  * PREF_KEY_ALIASES
  * pref_key_for / config_key_for
  * TK_BACKED_DETECTION_KEYS
  * prefs_detection_snapshot / config_detection_kwargs
"""

# The onset detector used when nothing else is specified.
#
# Kept at the derivative-based method: it makes no pre-stimulus baseline
# assumption, which is what a default has to cope with across resting and
# active paradigms. The envelope and CUSUM detectors measure better on clean
# synthetic data but inherit the baseline sensitivity of every threshold
# method, so promoting one of them is a decision for the validation work
# (Phase 4), not for a default.
DEFAULT_ONSET_METHOD = "bigoni"

# Superseded method keys. v1.3.3 called the median-across-methods detector
# "consensus"; the name was changed because it implies the agreed value is the
# correct one. Old keys resolve silently so saved sessions keep working.
METHOD_ALIASES = {"consensus": "methods_median"}

# Consensus members: fast, methodologically distinct, odd in number so the
# median is a member of the set rather than an average of two. ``bootstrap`` is
# excluded because it recomputes a 500-iteration bootstrap per call and would
# dominate the runtime; it can still be added explicitly.
DEFAULT_METHODS_MEDIAN_MEMBERS = (
    "bigoni",
    "bigoni_walkback",
    "rms_envelope",
    "cusum",
    "peak_fraction",
)

ONSET_DEFAULTS = {
    "onset_method": DEFAULT_ONSET_METHOD,

    # Shared across every detector
    "peak_fraction": 0.15,
    "min_peak_amplitude": 0.05,
    "slope_threshold": 0.08,

    # Bootstrap (legacy — frozen for reproducibility, see onset_bootstrap.py)
    "onset_bootstrap_crit": 1.96,
    "onset_bootstrap_n": 500,

    # Derivative-based (Bigoni et al. 2022) and its walkback variant
    "onset_bigoni_smooth_ms": 0.5,
    "onset_bigoni_min_run_ms": 0.5,
    "onset_bigoni_walkback_sd": 1.0,

    # RMS envelope
    "onset_env_window_ms": 5.0,
    "onset_env_criterion": 2.5,
    "onset_env_significance": 0.99,
    "onset_env_n_boot": 500,
    "onset_env_min_run_ms": 1.0,
    "onset_env_min_response_ms": 3.0,
    "onset_env_tkeo": False,
    "onset_env_causal": False,
    "onset_env_refine": True,
    "onset_env_refine_window_ms": 1.0,
    "onset_env_refine_sd": 2.5,
    "onset_env_refine_sustain_ms": 1.0,

    # CUSUM
    "onset_cusum_k": 0.5,
    "onset_cusum_h": 20.0,
    "onset_cusum_max_accum_ms": 10.0,
    "onset_cusum_min_response_ms": 3.0,
    "onset_cusum_tkeo": False,

    # Derivative-ratio (Boyles et al. 2026)
    # 2.5 ms — the width stated in the paper (Fig. 1 legend), which equals
    # its "B = 5 samples" at the 2 kHz used there.
    "onset_boyles_block_ms": 2.5,
    "onset_boyles_baseline_start_ms": 100.0,
    "onset_boyles_baseline_end_ms": 1.0,
    "onset_boyles_amplitude_gate": 1.1,
    "onset_boyles_peak_jitter_ms": 15.0,
    "onset_boyles_peak_window_length": 1.75,
    "onset_boyles_ratio_cutoff": 0.85,
    "onset_boyles_max_latency_ms": 35.0,
    "onset_boyles_deriv_check_ms": 2.0,
    "onset_boyles_deriv_check_duty": 0.75,
    "onset_boyles_base_deriv_sds": 1.5,
    "onset_boyles_deriv_window_length": 2.0,
    # Reproduce the reference MATLAB implementation exactly, including three
    # slips its own comments contradict. See detection/onset_boyles.py.
    "onset_boyles_literal": False,

    # Consensus and per-trial method agreement
    "onset_methods_median_members": list(DEFAULT_METHODS_MEDIAN_MEMBERS),
    # On by default. It runs every member detector on every trial, so onset
    # detection costs roughly the number of members -- but the consensus is
    # now the reported latency wherever it is available (see Latency_Source in
    # pipeline.LAT_COLS), not a review-triage aid, and a default of False
    # would mean the reported latency silently depended on a setting most
    # analysts never open.
    "onset_agreement": True,
}

OFFSET_DEFAULTS = {
    "mep_offset_enabled": True,
    "mep_offset_min_duration_ms": 5.0,
    # Raised from 60 ms, then again to 150. On real resting data a 60 ms cap
    # discarded responses rather than measuring them; at 100 ms a few still hit
    # the ceiling. At 150 every condition of a real recording was measured.
    "mep_offset_max_duration_ms": 150.0,
    "mep_offset_min_return_ms": 10.0,
    "mep_offset_env_window_ms": 5.0,
    "mep_offset_criterion": 2.5,
    # Fraction of the response's own peak envelope used as a floor under the
    # return threshold. NOW ZERO -- the floor is off by default.
    #
    # It was introduced to fix offset detection failing on 80 of 81 real
    # trials, but the cause there was a 60 ms duration cap; once that was
    # raised the baseline threshold alone found every trial. What the floor
    # actually does is shorten the answer, and it shortens it in proportion to
    # response size, so it truncates hardest on exactly the largest and
    # cleanest responses.
    #
    # Measured against an independent settle reference on a real M-wave
    # recording (mean error, negative = truncated early):
    #
    #     peak_frac      condition A      condition G
    #       0.12          -96.9 ms         -61.4 ms
    #       0.04          -77.0 ms         -45.5 ms
    #       0.00           -5.9 ms          +2.3 ms
    #
    # On a resting MEP recording, where responses are ~1 mV rather than ~9 mV,
    # 0.04 and 0.00 give identical offsets on every condition -- so removing
    # the floor costs nothing there and corrects tens of milliseconds here.
    #
    # It survives as a setting because a very quiet baseline can in principle
    # let a large response be chased into low-level drift, but it should be
    # raised deliberately after looking at the marker on the trace, not left on
    # by default.
    "mep_offset_peak_frac": 0.0,
}

PTP_ANCHOR_DEFAULTS = {
    # Anchor the PTP measurement window to each stimulus type's own median
    # onset, instead of using one window for every condition in the file.
    #
    # The PTP window is a single per-file setting while the latency profile is
    # per stimulus type, so a recording containing both M-waves and MEPs cannot
    # be served by one window. Measured on a real mixed file: with the default
    # 10-50 ms window, conditions whose M-wave onset was 4 ms had the first
    # 6 ms of every response excluded from the amplitude measurement -- which
    # for an M-wave is most of the response. Amplitude, not just latency, was
    # affected.
    #
    # Off by default so existing analyses reproduce; intended to become the
    # default once validated on real recordings.
    "ptp_anchor": False,
    # Window start = median onset - this. Small, so the window opens just
    # before the rise without admitting baseline.
    "ptp_anchor_pre_ms": 2.0,
    # Window length from the anchor. Generous enough for a slow or polyphasic
    # response; the user's PTP window end still applies as a hard ceiling, so
    # the window can never run past what was configured.
    "ptp_anchor_duration_ms": 40.0,
    # Minimum detected onsets in a condition before its median is trusted as an
    # anchor. Below this the file-wide window is used unchanged.
    "ptp_anchor_min_trials": 4,
}

DETECTION_DEFAULTS = dict(ONSET_DEFAULTS)
DETECTION_DEFAULTS.update(OFFSET_DEFAULTS)
DETECTION_DEFAULTS.update(PTP_ANCHOR_DEFAULTS)


# ── Legacy preference key names ───────────────────────────────────────────────
# config field name -> preference key stored on disk.
PREF_KEY_ALIASES = {
    "peak_fraction": "onset_peak_frac",
    "min_peak_amplitude": "onset_min_peak_amplitude",
    "slope_threshold": "onset_slope_threshold",
}

_CONFIG_KEY_FOR_PREF = {v: k for k, v in PREF_KEY_ALIASES.items()}


def pref_key_for(config_key):
    """Preference-file key corresponding to a PipelineConfig field name."""
    return PREF_KEY_ALIASES.get(config_key, config_key)


def config_key_for(pref_key):
    """PipelineConfig field name corresponding to a preference-file key."""
    return _CONFIG_KEY_FOR_PREF.get(pref_key, pref_key)


def as_pref_defaults():
    """DETECTION_DEFAULTS re-keyed for ``preferences.DEFAULTS``."""
    return {pref_key_for(k): v for k, v in DETECTION_DEFAULTS.items()}


def detector_params(cfg):
    """
    Extract every detection parameter from a config object or dict.

    Returns a plain dict keyed by config field name, falling back to the
    canonical default for anything the caller has not set. This is what gets
    handed to ``onset_methods_median``, whose adapters expect exactly these keys, so
    the member methods run with the same settings the individual detectors
    would have used.
    """
    is_map = isinstance(cfg, dict)
    out = {}
    for key, default in DETECTION_DEFAULTS.items():
        if is_map:
            out[key] = cfg.get(key, default)
        else:
            out[key] = getattr(cfg, key, default)
    return out


# ── GUI plumbing ──────────────────────────────────────────────────────────────
# These live here rather than in app.py because they are detection wiring, not
# interface code, and because app.py cannot be imported without a working
# matplotlib Tk backend -- which means anything defined there is untestable on
# a headless CI runner. The wiring is exactly what needs a test: a parameter
# that never reaches PipelineConfig produces a plausible number, not an error.

# Settings with a dedicated Tk variable, because Stage 1a can override them per
# file. Everything else is global and read straight from preferences.
TK_BACKED_DETECTION_KEYS = frozenset({
    "onset_method", "peak_fraction", "min_peak_amplitude", "slope_threshold",
    "onset_bootstrap_crit", "onset_bootstrap_n",
    "onset_bigoni_smooth_ms", "onset_bigoni_min_run_ms",
    "onset_bigoni_walkback_sd",
})


def prefs_detection_snapshot(prefs):
    """
    Detection settings without a Tk variable, read from ``prefs``.

    ``prefs`` is passed in rather than imported so this module stays free of a
    dependency on ``preferences``, which already depends on this one.
    """
    out = {}
    for key, default in DETECTION_DEFAULTS.items():
        if key in TK_BACKED_DETECTION_KEYS:
            continue
        out[key] = getattr(prefs, pref_key_for(key), default)
    return out


def config_detection_kwargs(params):
    """
    The same keys pulled back out of a params mapping, for ``PipelineConfig``.

    Excludes the Tk-backed keys, which the caller passes explicitly, so adding
    a setting here can never raise "got multiple values for argument".
    """
    return {k: params[k] for k in DETECTION_DEFAULTS
            if k not in TK_BACKED_DETECTION_KEYS and k in params}


# ── Migration of superseded defaults ─────────────────────────────────────────
# Raising a default does not reach anyone who has ever pressed Apply. The
# preferences dialog writes EVERY field on the tab, so a value the analyst
# never deliberately chose is stored verbatim and then shadows the new default
# forever.
#
# This is not hypothetical. `mep_offset_max_duration_ms` was raised from 60 to
# 100 ms after real recordings showed responses lasting ~54 ms, which left no
# room for the 10 ms return-to-baseline confirmation inside a 60 ms cap. On the
# machine where it mattered the stored 60 won, and offset detection succeeded
# on 1 trial out of 81 instead of 81 out of 81 -- with no error and no warning.
#
# A stored value is migrated only when it still equals the default it was
# saved under. Anything the analyst actually changed is left alone.

DETECTION_DEFAULTS_VERSION = 5

# Preference keys renamed between versions: old name -> new name. Aliasing was
# the alternative, but an alias leaves two names for one setting forever and
# splits the convention that a property is named for the key it reads. Moving
# the value once, on load, keeps a single canonical name.
RENAMED_PREF_KEYS = {
    # v1.3.3 called the median-across-methods detector "consensus".
    "onset_consensus_methods": "onset_methods_median_members",
}

# version introduced -> {key: the default value that version superseded}
SUPERSEDED_DEFAULTS = {
    2: {"mep_offset_max_duration_ms": 60.0},
    # The Preferences dialogue writes EVERY field on the tab, so a value the
    # analyst never chose is stored and thereafter shadows any later revision
    # of that default. Changing a default is therefore only half the change:
    # without an entry here it reaches new installations only, and the
    # machines that most need it -- the ones that hit the problem and opened
    # Preferences to look -- keep the old value.
    4: {"mep_offset_max_duration_ms": 100.0,
        "mep_offset_peak_frac": 0.12},
    # 0.04 still truncated large responses by tens of milliseconds; see the
    # table beside mep_offset_peak_frac above.
    5: {"mep_offset_peak_frac": 0.04},
}


def migrate_detection_defaults(data, stored_version=None):
    """Update untouched settings that were saved under a superseded default.

    Parameters
    ----------
    data : dict
        The preferences mapping, modified in place.
    stored_version : int or None
        The version recorded in the preferences FILE. This must be supplied by
        the caller and must not be read out of ``data``, because ``data`` is
        seeded from the shipped defaults before the file is merged over it --
        so it already carries the current version and every migration would
        report "already up to date" and do nothing. That is exactly what
        happened on the first attempt: the helper was correct, the caller
        handed it a dict that had been pre-stamped, and a stale 60 ms cap
        survived the upgrade untouched.

    Returns
    -------
    list of (key, old, new) describing what was migrated, for logging.
    """
    if stored_version is None:
        stored_version = data.get("detection_defaults_version", 1)
    stored_version = int(stored_version or 1)

    # Key renames run before the value migrations below, and regardless of the
    # stored version: a file written by an older release carries the old name
    # whatever else it contains.
    renamed = []
    for old_key, new_key in RENAMED_PREF_KEYS.items():
        if old_key in data:
            if new_key not in data:
                data[new_key] = data[old_key]
                renamed.append((old_key, data[old_key], new_key))
            data.pop(old_key, None)
    changed = list(renamed)
    if stored_version >= DETECTION_DEFAULTS_VERSION:
        data["detection_defaults_version"] = DETECTION_DEFAULTS_VERSION
        return changed

    for version in sorted(SUPERSEDED_DEFAULTS):
        if version <= stored_version:
            continue
        for key, old_default in SUPERSEDED_DEFAULTS[version].items():
            pref_key = pref_key_for(key)
            if pref_key not in data:
                continue
            new_default = DETECTION_DEFAULTS.get(key)
            try:
                unchanged = float(data[pref_key]) == float(old_default)
            except (TypeError, ValueError):
                unchanged = data[pref_key] == old_default
            if unchanged and new_default is not None:
                changed.append((pref_key, data[pref_key], new_default))
                data[pref_key] = new_default

    data["detection_defaults_version"] = DETECTION_DEFAULTS_VERSION
    return changed


def reset_detection_defaults(data):
    """Drop every stored detection setting so the canonical defaults apply."""
    removed = []
    for key in DETECTION_DEFAULTS:
        pref_key = pref_key_for(key)
        if pref_key in data:
            removed.append(pref_key)
            data.pop(pref_key, None)
    data["detection_defaults_version"] = DETECTION_DEFAULTS_VERSION
    return removed
