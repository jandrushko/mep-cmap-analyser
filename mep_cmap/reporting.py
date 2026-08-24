"""
mep_cmap.reporting
~~~~~~~~~~~~~~~~~~
The record of HOW a recording was analysed, written beside its results.

Why this exists
---------------
A methods section, a reporting checklist and a data-sharing submission all ask
the same question -- what was actually done -- and the answer is spread across
a PipelineConfig, a NIBS-BIDS sidecar, the recording's own header and the
results themselves. Reconstructing it afterwards means reading four things and
hoping they still agree.

So a run writes it down once, at the moment it is true.

Deliberately NOT limited to what TMS-RAT asks for. A record that answers one
checklist's questions has to be rewritten when the next one arrives, and
NIBS-DAS wants much of the same material. The cost of carrying more than any
one consumer needs is a slightly larger JSON per recording; the cost of
carrying less is a second pass over every dataset.

Scope: this module records FACTS ABOUT ONE RECORDING. It does not assess
reporting completeness, because that question is not answerable per file --
the TMS-RAT guidance assesses one protocol within one STUDY, and half its
quantitative items need a dispersion across participants that a single
recording cannot supply. Assessment happens at group level, from these
records. See ``tms_rat`` in the schema folder for the item mapping.
"""

from __future__ import annotations

import datetime
import json
import os

#: Bumped when the RECORD's shape changes, so a group-level reader can tell
#: what it is looking at. Independent of TOOL_VERSION: the tool changes far
#: more often than this file's layout.
#:
#: 2 — nibs_bids became a structured block rather than a flat mapping. v6.3
#:     puts 13 of the 25 required/recommended fields at PARAMETER SET scope,
#:     and a flat mapping has one slot for a fact that a paired-pulse protocol
#:     has two of: a test pulse at 120% rMT and a conditioning pulse at 80%
#:     are one key and two values. Flattening them loses the protocol.
REPORTING_SCHEMA_VERSION = "2"

#: The shape of the ``nibs_bids`` block, as built by
#: ``BidsifyState.nibs_record()``:
#:
#:     modality        "TMS" | "tES" | "TUS" | "PNS"
#:     fields          {schema key: value}   session- and file-scope, as filled
#:     parameter_sets  {set name: {"nibs_type", "position", "values": {...}}}
#:     code_sets       {stim code: set name}  this recording's assignment
#:
#: Keyed by SCHEMA FIELD KEY throughout, never by the v6.3 output name, so a
#: spec rename never invalidates a record that has already been written.
NIBS_RECORD_KEYS = ("modality", "fields", "parameter_sets", "code_sets")


#: Written next to the other per-recording outputs, under the report family.
SIDECAR_SUFFIX = "reporting.json"


def _schema_path(name):
    return os.path.join(os.path.dirname(__file__), "schema", name)


def load_tms_rat_mapping(version="1_0"):
    """The TMS-RAT item mapping, or None if the asset is absent.

    Returns None rather than raising: the mapping drives an optional report,
    and a missing asset must not stop an analysis from producing its
    measurements. The caller reports the absence.
    """
    try:
        with open(_schema_path(f"tms_rat_v{version}.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:                       # noqa: BLE001 — optional asset
        return None


def _round(v, n=6):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, n)          # f != f catches NaN


def _mean_sd(values):
    """Central tendency and dispersion, or None when there is nothing to say.

    Full reporting of a quantitative item needs BOTH, so they travel together
    and a single value reports its n as 1 rather than a zero SD, which would
    read as a measurement of no variability.

    Non-numeric entries are skipped rather than coerced. Trial tables carry
    sentinel strings in otherwise numeric columns -- Latency(ms) is written as
    "Not Detected" where no onset was found -- so float() on a column is never
    safe here.
    """
    vals = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:                          # f != f catches NaN
            vals.append(f)
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return {"mean": _round(mean), "sd": None, "n": n,
                "min": _round(mean), "max": _round(mean)}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return {"mean": _round(mean), "sd": _round(var ** 0.5), "n": n,
            "min": _round(min(vals)), "max": _round(max(vals))}


def analysis_facts(cfg, *, fs=None, emg_unit=None, source_format=None):
    """Everything about HOW the analysis was configured, from the config.

    Read from cfg rather than from the interface, because cfg is what the run
    used. A setting the analyst changed after starting is not what produced
    these results.
    """
    g = lambda k, d=None: getattr(cfg, k, d)      # noqa: E731

    filtering = {
        "sampling_rate_hz": _round(fs, 3),
        "amplitude_units": emg_unit,
        "bandpass_applied": bool(g("apply_bandpass", False)),
        # highpass/lowpass, not bp_low/bp_high. Read under the wrong names
        # these came back None, so a record could state that a bandpass was
        # applied and not say at what frequencies -- which is precisely the
        # Partial rating TMS-RAT J7 exists to catch, produced by the tool
        # meant to prevent it.
        "bandpass_low_hz": _round(g("highpass")),
        "bandpass_high_hz": _round(g("lowpass")),
        "bandpass_order": g("filter_order"),
        "bandpass_flexible": bool(g("flexible_bandpass", False)),
        "highpass_order": g("hp_order"),
        "lowpass_order": g("lp_order"),
        "notch_applied": bool(g("apply_notch", False)),
        "notch_hz": _round(g("notch_freq")),
        "notch_q": _round(g("notch_q")),
        "notch_harmonics": bool(g("filter_harmonics", False)),
        "mains_cancellation": bool(g("apply_humbug", False)),
        "mains_harmonics_modelled": g("humbug_harmonics"),
        # The guidance treats acquisition filtering and analysis filtering as
        # two separate facts and needs both for Full reporting of J7. Only the
        # second is knowable here: the first is set on the amplifier and the
        # recording does not carry it.
        "acquisition_filtering": None,
    }

    epoching = {
        "prestim_ms": _round(g("prestim_ms")),
        "pre_ms": _round(g("pre_ms")),
        "post_ms": _round(g("post_ms")),
        "window_map": dict(g("window_map", {}) or {}),
        "blanking_gap_ms": dict(g("gap_ms_map", {}) or {}),
        "event_delay_ms": dict(g("delay_ms_map", {}) or {}),
    }

    detection = {
        "onset_method": g("onset_method"),
        "onset_methods_median_members": list(
            g("onset_methods_median_members", []) or []),
        "min_peak_amplitude_mV": _round(g("min_peak_amplitude")),
        "latency_map": {k: list(v) for k, v in (g("latency_map", {}) or {}).items()},
        "amplitude_window_ms": [_round(g("ptp_start")), _round(g("ptp_end"))],
        "amplitude_window_anchored": bool(g("ptp_anchor", False)),
        "mep_offset_enabled": bool(g("mep_offset_enabled", False)),
        "outlier_review": bool(g("enable_outlier_review", False)),
        "outlier_threshold_z": _round(g("outlier_threshold")),
    }
    return {"filtering": filtering, "epoching": epoching,
            "detection": detection, "source_format": source_format}


def csp_facts(cfg):
    """The silent-period settings, as one block.

    Read through CspSettings rather than off cfg field by field, so this
    record cannot describe a different configuration from the one the
    detector ran under.
    """
    try:
        from .detection import CspSettings
        s = CspSettings.from_source(cfg)
        return {
            "types": sorted(getattr(cfg, "csp_types", None) or []),
            "min_silence_ms": s.min_silence_ms,
            "min_return_ms": s.min_return_ms,
            "criterion_sd": s.criterion,
            "significance": s.significance,
            "bootstrap_iterations": s.n_boot,
            "search_end_ms": s.search_end_ms,
            "max_offset_from_2nd_peak_ms": s.max_mep_offset_ms,
            "rms_window_ms": s.rms_window_ms,
            "search_start": "each trial's 2nd peak-to-peak landmark",
        }
    except Exception:                       # noqa: BLE001 — never fail a run
        return None


def condition_facts(rows, *, stim_col="StimType"):
    """Per-condition measured values, with dispersion.

    These are the items the tool can report Fully by construction: a
    quantitative item needs a central tendency AND a dispersion, and a
    computed column can always supply both.
    """
    if not rows:
        return {}
    by = {}
    for r in rows:
        by.setdefault(r.get(stim_col), []).append(r)

    out = {}
    for stim, rs in by.items():
        if stim is None:
            continue
        out[str(stim)] = {
            "n_trials": len(rs),
            "trial_to_trial_interval_s": _mean_sd(
                [r.get("Time_Since_Last_Stim(s)") for r in rs]),
            "ptp_mV": _mean_sd([r.get("PTP(mV)") for r in rs]),
            "latency_ms": _mean_sd([r.get("Latency(ms)") for r in rs]),
            "mep_duration_ms": _mean_sd([r.get("MEP_Duration(ms)") for r in rs]),
            "auc_mV_s": _mean_sd([r.get("AUC(mV*s)") for r in rs]),
            "csp_duration_ms": _mean_sd([r.get("cSP_Duration(ms)") for r in rs]),
            "prestim_rms_mV": _mean_sd([r.get("PreStimRMS") for r in rs]),
        }
    return out


def _analysed_ranges(cfg):
    """The time ranges this run was restricted to, or None for the whole file.

    None and [] are different answers and must not collapse into one: the
    first says the recording was analysed entire, the second would say a
    selection was made and was empty, which no run can produce. Written as
    lists of two floats so the record stays plain JSON.
    """
    ranges = getattr(cfg, "crop_ranges", None)
    if not ranges:
        start = getattr(cfg, "crop_start", None)
        end = getattr(cfg, "crop_end", None)
        if start is None or end is None:
            return None
        return [[float(start), float(end)]]
    out = []
    for pair in ranges:
        try:
            a, b = pair
            out.append([float(a), float(b)])
        except (TypeError, ValueError):
            continue
    return out or None


def build_record(cfg, *, name, channel=None, fs=None, emg_unit=None,
                 source_format=None, rows=None, nibs_record=None,
                 tool_version=None, detection_version=None):
    """Assemble the per-recording reporting record.

    ``nibs_record`` is the NIBS-BIDS description of this recording in the
    shape documented at ``NIBS_RECORD_KEYS``, not a flat mapping of fields.
    """
    facts = analysis_facts(cfg, fs=fs, emg_unit=emg_unit,
                           source_format=source_format)
    return {
        "schema": "mep-cmap-reporting",
        "schema_version": REPORTING_SCHEMA_VERSION,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        # Both stamps, because a study assembled from files analysed under
        # different detectors is not one study, and level 2 has to be able to
        # refuse rather than average across the difference.
        "tool_version": tool_version,
        "detection_version": detection_version,
        "recording": {"file": name, "channel": channel,
                      "source_format": source_format,
                      # Which part of the recording was analysed. Ranges are
                      # per channel, so two channels of one file can carry
                      # different trial counts for a reason nothing else in
                      # the record would explain; null means the whole
                      # recording, which is not the same fact as an empty list.
                      "analysed_ranges_s": _analysed_ranges(cfg)},
        "filtering": facts["filtering"],
        "epoching": facts["epoching"],
        "detection": facts["detection"],
        "silent_period": csp_facts(cfg),
        "conditions": condition_facts(rows or []),
        # As FOUND, not as required. What is absent is the point.
        "nibs_bids": dict(nibs_record or {}),
    }


def _filled(value):
    """A value that says something. Blank strings and empties do not."""
    if value in (None, "", [], {}):
        return False
    return str(value).strip() != ""


def _applies_to(field, modality):
    """Whether a schema field describes the modality that was delivered.

    A TMS study has no electrodes and no transducer. Reporting ElectrodeType
    and TransducerModel as unfilled is not a cautious warning, it is a wrong
    one: eight of the twenty-five required/recommended fields belong to tES,
    TUS or PNS, and naming them teaches the analyst that this list is noise.

    With no modality known, only the fields common to every modality can be
    judged. Under-reporting there is the honest failure -- the alternative is
    to pick a modality on the analyst's behalf and report against it.
    """
    mods = field.get("modalities") or []
    if "common" in mods:
        return True
    return bool(modality) and modality in mods


def missing_nibs_fields(nibs_record, *, levels=("required", "recommended")):
    """Schema fields at the given levels that this recording has not filled.

    Level 1 reports only what it KNOWS is absent -- a blank field in a schema
    it can read. It does not compute reporting coverage, because coverage is a
    property of a study and this is one recording; a per-file percentage would
    be a number that changes meaning as files are added.

    Scope decides where a field is looked for. Session- and file-scope fields
    live in ``fields``; parameter-set-scope fields live on each named set, and
    a set that does not carry one cannot be described -- so such a field counts
    as filled only when EVERY set this recording references carries it. Half a
    protocol described is a gap, not a value.
    """
    try:
        with open(_schema_path("nibs_bep037.json"), encoding="utf-8") as f:
            schema = json.load(f)
    except Exception:                       # noqa: BLE001 — optional asset
        return []

    rec = dict(nibs_record or {})
    fields = dict(rec.get("fields") or {})
    sets = dict(rec.get("parameter_sets") or {})
    code_sets = dict(rec.get("code_sets") or {})
    modality = rec.get("modality") or fields.get("StimulationModality") or ""

    # Only the sets some code in THIS recording references. A session-level set
    # belonging to another file describes stimulation that was never delivered
    # here, and judging this recording against it would report a gap that is
    # not one -- or hide one that is.
    referenced = {n for n in code_sets.values() if n}
    in_use = ([s for n, s in sets.items() if n in referenced] if referenced
              else list(sets.values()))

    out = []
    for fld in schema.get("fields", []):
        if fld.get("level") not in levels or not _applies_to(fld, modality):
            continue
        key = fld["key"]
        if fld.get("scope") == "parameter_set":
            if not in_use or not all(
                    _filled((s.get("values") or {}).get(key)) for s in in_use):
                out.append(key)
        elif not _filled(fields.get(key)):
            out.append(key)
    return out


def write_sidecar(record, out_dir, prefix):
    """Write the record beside the results. Returns the path, or None."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}_{SIDECAR_SUFFIX}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, ensure_ascii=False)
        return path
    except Exception:                       # noqa: BLE001 — never fail a run
        return None
