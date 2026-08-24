"""
mep_cmap.tms_rat
~~~~~~~~~~~~~~~~
Assessing a STUDY against the TMS-RAT reporting checklist.

Why this is level 2
-------------------
TMS-RAT assesses one protocol within one study, and half its quantitative items
are rated Full only when a central tendency AND a dispersion are reported. A
single recording cannot supply a dispersion across participants, so a per-file
assessment would rate those items Part for every study, however completely it
was reported. The assessment therefore reads the per-recording records that
mep_cmap.reporting wrote and does its work across them.

What this module does NOT do
----------------------------
It does not invent. An item the tool cannot know is Missing and says which fact
is absent; it is never rated from a plausible default. It does not collect
participant data either -- a participants.tsv is read if the study has one, and
those items are Missing if it does not.

Pure logic: no Tk, no file discovery, no workbook. The add-on does the I/O so
that these rules can be tested without a study on disk.
"""

from __future__ import annotations

import datetime

#: The rating vocabulary. Most items take the first three; gate items ("is this
#: section relevant?") take Yes/No, and some items may be Not Applicable.
FULL, PART, MISSING, NA = "Full", "Part", "Missing", "n/a"
YES, NO = "Yes", "No"

#: What the tool writes where it has nothing to say. The workbook uses "-" for
#: an empty comment, and matching it keeps a filled sheet indistinguishable in
#: shape from a hand-filled one.
BLANK = "-"


class AnalysisFault(Exception):
    """The records cannot honestly be pooled into one assessment."""


def _is_blank(v):
    return v in (None, "", [], {}) or str(v).strip() == ""


# ── agreement across records ─────────────────────────────────────────────────

def check_poolable(records):
    """Reasons these records are not one study's worth of analysis.

    Returned as messages rather than raised, so a caller can show all of them
    at once. ``assess`` raises on the fatal one.

    A study assembled from files analysed under different detectors is not one
    study: the same recording yields a different onset under a different
    algorithm, so a mean latency pooled across two detector versions is a
    number describing no analysis that was ever run. That is fatal. A differing
    TOOL_VERSION is not -- the tool changes for reasons that do not touch a
    measurement -- so it is reported and pooling continues.
    """
    fatal, notes = [], []
    dvs = {r.get("detection_version") for r in records}
    if len(dvs) > 1:
        fatal.append(
            "These recordings were analysed under different detector versions "
            "(" + ", ".join(sorted(str(d) for d in dvs)) + "). They cannot be "
            "pooled into one assessment: the same recording yields different "
            "onsets under different detectors, so a pooled mean would describe "
            "no analysis that was actually run. Re-analyse them under one "
            "version.")
    tvs = {r.get("tool_version") for r in records}
    if len(tvs) > 1:
        notes.append(
            "Analysed under more than one tool version ("
            + ", ".join(sorted(str(t) for t in tvs))
            + "). Pooled anyway: a tool version changes for reasons that need "
              "not touch a measurement. Worth checking before publication.")
    return fatal, notes


# ── the aggregate rules ──────────────────────────────────────────────────────

def agg_unique(values):
    """One value the whole study shares, or a divergence.

    Returns ``(value, divergent_values)``. A divergence here is an ANALYSIS
    fault, not a reporting one: two filter settings across twenty-four files
    means the files were not analysed the same way, and no single sentence in a
    methods section can be true of all of them. Reporting the majority value
    would produce a methods section that is false for the minority and give no
    sign that anything was wrong.
    """
    seen = [v for v in values if not _is_blank(v)]
    if not seen:
        return None, []
    uniq = []
    for v in seen:
        if v not in uniq:
            uniq.append(v)
    if len(uniq) == 1:
        return uniq[0], []
    return None, uniq


def agg_mean_sd(values):
    """Central tendency with dispersion, or None.

    Both or neither. A quantitative item is Full only when both are reported,
    so a mean travelling without an SD would be rated Full by a caller that
    only checked for a number.

    n < 2 returns the mean with sd None: one value has no dispersion, and a
    zero SD would read as a measurement of no variability.
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
        return {"mean": mean, "sd": None, "n": n}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return {"mean": mean, "sd": var ** 0.5, "n": n}


def agg_sum(values):
    total, any_seen = 0.0, False
    for v in values:
        try:
            total += float(v)
            any_seen = True
        except (TypeError, ValueError):
            continue
    return total if any_seen else None


def agg_list(values):
    """Every distinct value, in first-seen order."""
    out = []
    for v in values:
        if _is_blank(v):
            continue
        for item in (v if isinstance(v, (list, tuple, set)) else [v]):
            if not _is_blank(item) and item not in out:
                out.append(item)
    return out


AGGREGATORS = {"unique": agg_unique, "mean_sd": agg_mean_sd,
               "sum": agg_sum, "list": agg_list}


# ── pulling a fact out of the records ────────────────────────────────────────
#
# One function per PROVIDER name in the mapping, so that adding an item is a
# mapping edit plus one function, and an item whose provider has no function
# reports as a known gap rather than silently as Missing. The distinction
# matters: "the tool cannot know this" and "the tool forgot to look" read the
# same in an output sheet and are entirely different problems.

def _each_condition(records):
    """(record, stim_type, condition_block) for every condition in every file."""
    for r in records:
        for stim, block in (r.get("conditions") or {}).items():
            yield r, stim, block


def _filtering(records, key):
    return [(r.get("filtering") or {}).get(key) for r in records]


def _detection(records, key):
    return [(r.get("detection") or {}).get(key) for r in records]


def prov_sampling_and_filtering(records):
    """J7 -- sampling rate, filter type, cut-offs, and analysis filtering.

    Full needs all four. The first three are knowable; acquisition filtering is
    set on the amplifier and the recording does not carry it, so it is reported
    as a gap and the item cannot reach Full from this tool alone. Saying so is
    the point: J7 exists to catch exactly the paper that says "filtered" and
    not at what.

    A DIVERGENT value is a fault, not an absence. Discarding the divergence
    flag and letting agg_unique's None fall through would report 2000 Hz and
    4000 Hz across a study as "sampling rate not reported" -- which reads as a
    metadata gap the analyst should fill, when it is really two different
    analyses that cannot share one methods sentence.
    """
    found, divergent = {}, {}
    for label, key in (("sampling rate", "sampling_rate_hz"),
                       ("acquisition filtering", "acquisition_filtering")):
        value, diverged = agg_unique(_filtering(records, key))
        found[label] = value
        if diverged:
            divergent[label] = diverged

    applied, d_applied = agg_unique(_filtering(records, "bandpass_applied"))
    lo, d_lo = agg_unique(_filtering(records, "bandpass_low_hz"))
    hi, d_hi = agg_unique(_filtering(records, "bandpass_high_hz"))
    found["analysis filtering"] = (f"{lo}-{hi} Hz band-pass"
                                   if applied and lo and hi else None)
    for label, diverged in (("band-pass applied", d_applied),
                            ("band-pass low cut-off", d_lo),
                            ("band-pass high cut-off", d_hi)):
        if diverged:
            divergent[label] = diverged
    return found, divergent


def prov_trial_to_trial_interval(records):
    """G1 -- the interval between pulses, and its dispersion.

    The dispersion G1 asks for is BETWEEN TRIALS, not between files. Taking the
    per-file means and computing their SD answers a different question, and
    answers it misleadingly well: a study whose files all average 5 s reports
    SD 0, which reads as a perfectly regular train and says nothing about the
    intervals actually delivered. So the central tendency is pooled across
    files and the dispersion is the typical WITHIN-file SD.

    Returns the stats plus any condition whose SD exceeds its mean. That is not
    noise: an interval distribution with SD > mean is not summarised by a mean
    at all, which is what a blocked design looks like when the gap between
    blocks is counted as an inter-trial interval.
    """
    means, sds, suspect = [], [], []
    for _r, stim, block in _each_condition(records):
        stats = block.get("trial_to_trial_interval_s") or {}
        m, s = stats.get("mean"), stats.get("sd")
        if m is None:
            continue
        means.append(m)
        if s is not None:
            sds.append(s)
            if s > m:
                suspect.append((stim, m, s))
    if not means:
        return None, []
    pooled = agg_mean_sd(means)
    typical_sd = (sum(sds) / len(sds)) if sds else None
    return {"mean": pooled["mean"], "sd": typical_sd,
            "n": len(means), "sd_label": "typical within-file SD"}, suspect


def prov_number_of_pulses(records):
    return agg_sum([block.get("n_trials")
                    for _r, _s, block in _each_condition(records)])


def prov_evoked_response_raw(records):
    """G3 -- the evoked response amplitude, PER CONDITION.

    Never pooled across conditions. A recording with an M-wave on one code and
    an MEP on another gives 9.58 mV and 1.00 mV; their average, 2.61 mV, is an
    amplitude nothing produced, and a reader would take it for the study's MEP.
    The checklist's Full definition asks only for a central tendency and a
    dispersion, so a pooled number satisfies it on paper -- which is exactly
    why the pooling has to be refused here rather than left to the rating.

    With more than one recording the dispersion reported is BETWEEN recordings,
    which is the study-level dispersion the checklist is asking about. With one
    recording it is the within-file dispersion across trials, and the comment
    says which, because they are different quantities and a reader cannot tell
    them apart from a number.
    """
    per_stim = {}
    for _r, stim, block in _each_condition(records):
        stats = block.get("ptp_mV") or {}
        if stats.get("mean") is None:
            continue
        per_stim.setdefault(str(stim), []).append(stats)

    out = {}
    for stim, entries in sorted(per_stim.items()):
        if len(entries) > 1:
            pooled = agg_mean_sd([e["mean"] for e in entries])
            out[stim] = {"mean": pooled["mean"], "sd": pooled["sd"],
                         "n": len(entries), "basis": "between recordings"}
        else:
            e = entries[0]
            out[stim] = {"mean": e.get("mean"), "sd": e.get("sd"),
                         "n": e.get("n"), "basis": "within one recording"}
    return {"per_condition": out}


def prov_emg_measure(records):
    """J8 -- what was measured, over what interval, by what algorithm."""
    onset, _ = agg_unique(_detection(records, "onset_method"))
    win, _ = agg_unique([tuple(x) if isinstance(x, list) else x
                         for x in _detection(records, "amplitude_window_ms")])
    return {"feature measured": "peak-to-peak amplitude",
            "time interval": (f"{win[0]}-{win[1]} ms" if win else None),
            "extraction algorithm": onset}


PROVIDERS = {
    "sampling_and_filtering": prov_sampling_and_filtering,
    "trial_to_trial_interval": prov_trial_to_trial_interval,
    "number_of_pulses": prov_number_of_pulses,
    "evoked_response_raw": prov_evoked_response_raw,
    "emg_measure": prov_emg_measure,
}


# ── rating one item ──────────────────────────────────────────────────────────

def rate_nibs_item(item, nibs_records):
    """Rate an item whose facts live in the NIBS-BIDS descriptions.

    ``provider`` names the schema keys, joined by "+" when the item needs
    several facts and by "|" when any one of them will do. H2 is the second
    kind: a study reports a resting OR an active motor threshold, and demanding
    both would rate a correctly reported study Part.
    """
    prov = item.get("provider") or ""
    alternatives = "|" in prov
    keys = [k for k in prov.replace("|", "+").split("+") if k]
    found, absent = {}, []
    divergences = {}
    for key in keys:
        vals = []
        for nr in nibs_records:
            fields = nr.get("fields") or {}
            if not _is_blank(fields.get(key)):
                vals.append(fields[key])
            for s in (nr.get("parameter_sets") or {}).values():
                v = (s.get("values") or {}).get(key)
                if not _is_blank(v):
                    vals.append(v)
        value, divergent = agg_unique(vals)
        if divergent:
            # Recorded, and NOT collapsed to a majority. Two coil models across
            # a study is either two coils -- which the methods must say -- or a
            # typo, and picking one would hide both.
            divergences[key] = divergent
            found[key] = divergent
        elif value is not None:
            found[key] = value
        else:
            absent.append(key)

    if not found:
        return MISSING, absent, divergences, found
    if divergences:
        return PART, absent, divergences, found
    if alternatives or not absent:
        return FULL, [], divergences, found
    return PART, absent, divergences, found


def rate_full_requires(item, facts):
    """Rate a derived item against the facts its 'Full' definition names.

    ``full_requires`` is the checklist's own wording for what a Full rating
    needs. Rating against it rather than against "is there a number here"
    is what keeps a mean without a dispersion out of the Full column.
    """
    required = list(item.get("full_requires") or [])
    if not required:
        return (FULL if facts else MISSING), []
    absent = [k for k in required if _is_blank((facts or {}).get(k))]
    if len(absent) == len(required):
        return MISSING, absent
    return (FULL if not absent else PART), absent


def rate_quantitative(stats):
    """Full needs a central tendency AND a dispersion.

    n == 1 is Part, not Full. One participant has no dispersion, and this is
    the whole reason the assessment is a study-level question: a per-file
    assessment would hit this on every item and every study.
    """
    if not stats or stats.get("mean") is None:
        return MISSING, ["central tendency", "dispersion"]
    if stats.get("sd") is None:
        return PART, ["dispersion"]
    return FULL, []


# ── the assessment ───────────────────────────────────────────────────────────

#: OUTPUT sheet column order, verbatim. The sheet is the artifact a rater and a
#: reviewer both read, so the CSV matches it exactly rather than improving on it.
OUTPUT_COLUMNS = ["Study", "ID", "Version", "Rater", "Section", "Item",
                  "Required", "Rating", "Timestamp", "Comments"]


def assess(records, mapping, *, participants=None, study="", rater="",
           study_id="", protocol=None, now=None):
    """Rate every item in the mapping against these records.

    ``records`` are per-recording reporting sidecars; ``participants`` is the
    parsed participants.tsv, or None when the study has none.

    Raises AnalysisFault when the records cannot honestly be pooled.
    """
    fatal, notes = check_poolable(records)
    if fatal:
        raise AnalysisFault(" ".join(fatal))

    nibs_records = [r.get("nibs_bids") or {} for r in records]
    stamp = (now or datetime.datetime.now()).replace(microsecond=0).isoformat()
    relevance = mapping.get("section_relevance") or {}
    legend = mapping.get("relevance_legend") or {}

    results, faults = [], list(notes)
    for item in mapping.get("items") or []:
        source = item.get("source")
        rating, absent, comment = MISSING, [], ""

        if source == "out_of_scope":
            rating = MISSING
            comment = ("Not derivable from the recordings or their metadata: "
                       "this is a fact about the participants or the "
                       "experimental design, which the analysis never sees.")
        elif source == "gate":
            # Whether a section applies is a statement about the experiment,
            # not about the data. Answering it from the presence of a column
            # would let an incidental measurement switch on a whole section.
            rating = BLANK
            comment = "For the rater to answer."
        elif source == "participants":
            rating, absent, comment = _rate_participant_item(item, participants)
        elif source == "nibs_bids":
            rating, absent, divergences, found = rate_nibs_item(
                item, nibs_records)
            # State the values, not just the rating. A methods draft reads
            # these comments, and an item rated Full whose comment is blank
            # comes out of that draft as an unknown -- so the tool would print
            # "[TMS-RAT D1: Stimulator]" for a stimulator it knows the model of.
            if found:
                comment = "; ".join(
                    f"{k}: " + (", ".join(str(x) for x in v)
                                if isinstance(v, list) else str(v))
                    for k, v in found.items())
            for key, vals in divergences.items():
                shown = ", ".join(repr(str(v)) for v in vals)
                faults.append(
                    f"{item['id']} ({item['title']}): {key} differs across the "
                    f"study -- {shown}. These recordings were not all "
                    f"described the same way, so no single methods sentence is "
                    f"true of all of them. Rated {PART} rather than picking "
                    f"one.")
                comment = f"DIVERGENT {key}: {shown}"
        elif source == "nibs_gap":
            rating = MISSING
            comment = (f"[TMS-RAT {item['id']}: {item['title']}] — no "
                       f"NIBS-BIDS field carries this, so the tool cannot "
                       f"supply it. Add it to the methods by hand.")
        elif source == "derived":
            rating, absent, comment, more = _rate_derived_item(item, records)
            faults.extend(more)

        if absent and not comment:
            comment = "Not reported: " + ", ".join(str(a) for a in absent)

        results.append({
            "id": item["id"],
            "section": item["section"],
            "title": item["title"],
            "source": source,
            "input_row": item.get("input_row"),
            "rating": rating,
            "comment": comment or BLANK,
            "required": _required_for(item, relevance, legend, protocol),
            "timestamp": stamp,
            "study": study, "study_id": study_id, "rater": rater,
        })
    return results, faults


def _required_for(item, relevance, legend, protocol):
    """Whether this item's section is required for the protocol assessed.

    Read from the checklist's own relevance table rather than decided here.
    Without a protocol the question has no answer, and a default of "required"
    would inflate every denominator.
    """
    if not protocol:
        return BLANK
    mark = (relevance.get(item["section"]) or {}).get(protocol)
    return legend.get(mark, BLANK) if mark else BLANK


def _rate_participant_item(item, participants):
    """A1-A5, C2-C3: read from participants.tsv, or Missing.

    The tool does not collect participant data. Inventing it from a filename
    would be inventing a measurement, so its absence is reported as what it is.
    """
    if not participants:
        return MISSING, [], (
            f"[TMS-RAT {item['id']}: {item['title']}] — no participants.tsv "
            f"in the dataset. The tool does not collect participant data.")
    col = item.get("provider")
    values = [row.get(col) for row in participants if not _is_blank(row.get(col))]
    if not values:
        return MISSING, [], (
            f"[TMS-RAT {item['id']}: {item['title']}] — participants.tsv has "
            f"no {col!r} column.")
    if item.get("aggregate") == "mean_sd":
        rating, absent = rate_quantitative(agg_mean_sd(values))
        stats = agg_mean_sd(values)
        if stats and stats.get("mean") is not None:
            sd = stats.get("sd")
            return rating, absent, (
                f"{col}: mean {stats['mean']:.4g}"
                + (f", SD {sd:.4g}" if sd is not None else "")
                + f", n = {stats['n']}")
        return rating, absent, ""
    counts = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    return FULL, [], (f"{col}: "
                      + ", ".join(f"{k} n = {v}" for k, v in sorted(counts.items()))
                      + f" (total {len(values)})")


def _rate_derived_item(item, records):
    """An item the analysis itself can answer, if a provider exists for it."""
    fn = PROVIDERS.get(item.get("provider"))
    if fn is None:
        return MISSING, [], (
            f"[TMS-RAT {item['id']}: {item['title']}] — derivable in "
            f"principle but not yet wired to a provider."), []
    facts = fn(records)
    faults = []
    divergent = {}

    # A provider may return (facts, extra) when computing the fact also
    # reveals something wrong with it. ``extra`` is either a list of suspect
    # conditions or a mapping of divergent values.
    if isinstance(facts, tuple):
        facts, extra = facts
        if isinstance(extra, dict):
            divergent = extra
            for label, vals in extra.items():
                shown = ", ".join(repr(str(v)) for v in vals)
                faults.append(
                    f"{item['id']} ({item['title']}): {label} differs across "
                    f"the study -- {shown}. These recordings were not analysed "
                    f"the same way, so no single methods sentence is true of "
                    f"all of them. This is an analysis fault, not a missing "
                    f"field: re-analyse them consistently, or report them as "
                    f"separate datasets.")
        else:
            for stim, m, s in extra:
                faults.append(
                    f"{item['id']} ({item['title']}): condition {stim} has an "
                    f"inter-trial interval of mean {m:.4g} s with SD {s:.4g} s. "
                    f"An SD larger than the mean means this distribution is "
                    f"not summarised by a mean -- typically a blocked design "
                    f"where the gap between blocks is being counted as an "
                    f"inter-trial interval. Report a median and IQR, or split "
                    f"at the gaps.")

    # Reported per condition, because different stimulus types are different
    # measurements. Full only when EVERY condition carries both a central
    # tendency and a dispersion: one condition described and another not is a
    # partially reported item, not a complete one.
    if isinstance(facts, dict) and "per_condition" in facts:
        per = facts["per_condition"]
        if not per:
            return MISSING, ["central tendency", "dispersion"], "", faults
        parts, absent = [], []
        for stim, s in per.items():
            if s.get("mean") is None:
                absent.append(f"{stim}: central tendency")
                continue
            bit = f"{stim}: mean {s['mean']:.4g}"
            if s.get("sd") is None:
                absent.append(f"{stim}: dispersion")
            else:
                bit += f", SD {s['sd']:.4g} ({s.get('basis', '')})"
            bit += f", n = {s.get('n')}"
            parts.append(bit)
        rating = FULL if not absent else (PART if parts else MISSING)
        return rating, absent, "; ".join(parts), faults

    if isinstance(facts, dict) and "mean" in facts:
        rating, absent = rate_quantitative(facts)
        if faults:
            # The number exists but does not describe the thing it names.
            rating = PART
        sd = facts.get("sd")
        # Labelled, because a dispersion computed across FILES and one computed
        # across TRIALS are different quantities that look identical as numbers.
        label = facts.get("sd_label", "SD")
        comment = (f"mean {facts['mean']:.4g}"
                   + (f", {label} {sd:.4g}" if sd is not None else "")
                   + f", n = {facts['n']}") if facts.get("mean") is not None else ""
        if faults:
            comment += " — dispersion unreliable, see analysis faults"
        return rating, absent, comment, faults

    if isinstance(facts, dict):
        rating, absent = rate_full_requires(item, facts)
        stated = [f"{k}: {v}" for k, v in facts.items() if not _is_blank(v)]
        # A divergent fact is NOT an absent one, and must not be listed under
        # "Not reported": the analyst would go looking for a field to fill in
        # when the real problem is that the recordings disagree.
        for label, vals in divergent.items():
            absent = [a for a in absent if a != label]
            stated.append(f"DIVERGENT {label}: "
                          + ", ".join(repr(str(v)) for v in vals))
        if divergent:
            # Never Full (the study disagrees with itself) and never Missing
            # (the fact was reported, just inconsistently). Leaving it Missing
            # contradicts the comment sitting next to it, which names the
            # values found.
            rating = PART
        return rating, absent, "; ".join(stated), faults

    if isinstance(facts, list):
        return (FULL if facts else MISSING), [], ", ".join(str(f) for f in facts), faults

    if facts is None:
        return MISSING, [], "", faults
    if isinstance(facts, float) and facts.is_integer():
        # A count of pulses is a count. "240.0" invites the reader to wonder
        # what the fractional pulse was.
        return FULL, [], str(int(facts)), faults
    return FULL, [], str(facts), faults


def output_rows(results):
    """The assessment as OUTPUT-sheet rows."""
    return [{"Study": r["study"], "ID": r["study_id"], "Version": "1.0",
             "Rater": r["rater"] or BLANK, "Section": r["section"],
             "Item": r["id"][1:], "Required": r["required"],
             "Rating": r["rating"], "Timestamp": r["timestamp"],
             "Comments": r["comment"]}
            for r in results]


# ── participants.tsv ─────────────────────────────────────────────────────────

#: participants.tsv column name -> the mapping's provider name. BIDS names the
#: first two; the rest are conventional rather than specified, so several
#: spellings are accepted. Matching is case-insensitive.
PARTICIPANT_COLUMNS = {
    "age": "age",
    "sex": "sex",
    "gender": "sex",
    "handedness": "handedness",
    "hand": "handedness",
    "height": "height",
    "weight": "weight",
    "group": "groups",
    "population": "population",
}


def parse_participants_tsv(text):
    """Rows of a participants.tsv, keyed by PROVIDER name.

    BIDS writes "n/a" for a value that does not exist, and it must not become
    the string "n/a" in a count of handedness categories -- that would report a
    category nobody is in. Dropped here rather than in every consumer.

    Unknown columns are carried through under their own name, so a study whose
    extra column happens to match a future provider is not silently ignored.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split("\t")]
    keys = [PARTICIPANT_COLUMNS.get(h.lower(), h) for h in header]
    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("\t")]
        row = {}
        for k, v in zip(keys, cells):
            if v and v.lower() != "n/a":
                row[k] = v
        if row:
            rows.append(row)
    return rows


# ── the methods paragraph ────────────────────────────────────────────────────

def methods_markdown(results, faults, *, study="", protocol=None,
                     tool_version="", detection_version=""):
    """A methods section carrying an explicit marker wherever a fact is absent.

    The markers are the point. A generated paragraph that reads smoothly with
    the unknowable parts quietly omitted is worse than no paragraph at all: it
    looks finished, and the facts it cannot know are exactly the ones a
    reporting checklist exists to catch. So every gap is named in the text, in
    a form that is impossible to miss and trivial to search for before
    submission.
    """
    by_id = {r["id"]: r for r in results}

    def fact(item_id, fallback=None):
        r = by_id.get(item_id)
        if r and r["rating"] in (FULL, PART) and r["comment"] not in ("", BLANK):
            return r["comment"]
        title = r["title"] if r else item_id
        return fallback or f"[TMS-RAT {item_id}: {title}]"

    lines = [f"# Methods (draft) — {study or 'this study'}", ""]
    lines.append(
        "Generated by MEP-CMAP Analyser"
        + (f" {tool_version}" if tool_version else "")
        + (f", detection {detection_version}" if detection_version else "")
        + ". Every `[TMS-RAT ...]` marker below is a fact this tool could not "
          "know. Search for `[TMS-RAT` and replace each one before submitting.")
    lines += ["", "## Participants", "", fact("A1"), "", "## Stimulation", ""]
    lines.append("Stimulator: " + fact("D1"))
    lines.append("")
    lines.append("Coil: " + fact("D2"))
    lines.append("")
    lines.append("Intensity: " + fact("F1"))
    lines += ["", "## EMG recording", ""]
    lines.append("Electrodes: " + fact("J4"))
    lines.append("")
    lines.append("Sampling and filtering: " + fact("J7"))
    lines.append("")
    lines.append("Measurement: " + fact("J8"))
    lines += ["", "## Procedure", ""]
    lines.append("Pulses delivered: " + fact("G2"))
    lines.append("")
    lines.append("Inter-trial interval: " + fact("G1"))
    lines.append("")

    gaps = [r for r in results if r["rating"] == MISSING
            and r["source"] in ("nibs_gap", "participants", "derived")]
    if gaps:
        lines += ["## Facts this tool could not supply", ""]
        # The comment already opens with the marker, so the bullet does not
        # repeat it.
        lines += [f"- {r['comment']}" for r in gaps]
        lines.append("")

    if faults:
        lines += ["## Analysis faults found while assembling this", "",
                  "These are not reporting gaps. They are signs that the "
                  "recordings were not all analysed or described the same "
                  "way, and no single methods sentence is true of all of "
                  "them.", ""]
        lines += [f"- {f}" for f in faults]
        lines.append("")
    return "\n".join(lines)


def coverage(results):
    """How many items reached each rating, for the log line."""
    out = {}
    for r in results:
        out[r["rating"]] = out.get(r["rating"], 0) + 1
    return out
