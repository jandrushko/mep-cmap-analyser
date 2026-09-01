"""
mep_cmap.column_groups
~~~~~~~~~~~~~~~~~~~~~~
Which trial-level columns belong together, and which are never dropped.

``_trials.csv`` carries fifty-six columns. Most analyses use six of them. The
rest are not noise -- every one is there because some study needed it -- but a
table nobody can read across on one screen is a table whose columns get
selected in a spreadsheet by hand, which is where transcription errors come
from.

This module is to columns what ``results_layout`` is to folders: the single
place that says what goes where, so a metric added to the schema is assigned
deliberately rather than by whoever happens to touch the writer next.

NOTHING HERE NARROWS ``_trials.csv``. That file keeps every column, always.
Selection produces a SECOND file, ``_trials_selected.csv`` -- the TRIMMED
trials file, as the interface calls it -- beside it. The protected set below
is a complete join key -- File, Channel, StimType, Segment, Condition -- so
any column left out of the trimmed file can be recovered by merging it back
against the full one. That is what makes dropping a column safe: it is never
lost, only absent from one view.

LAT_COLS is not touched, read, or reordered by any of this. Narrowing happens
on a copy of the written frame, at write time.
"""

from __future__ import annotations

#: Added to the frame by ``_tag_channel`` AFTER the row builders have run, so
#: it is not in LAT_COLS and cannot be found there. Listed here because the
#: written frame is what this module partitions, not the row schema.
POST_BUILD_COLUMNS = ("Channel",)

#: Always written, whatever is selected. Shown ticked and disabled in the UI,
#: with the reason, because a control that cannot be changed and does not say
#: why reads as a bug.
#:
#: Together these identify a trial completely: File+Channel says which
#: recording and which muscle, StimType+Condition which experimental cell, and
#: Segment which trial within it. Drop any one and the narrow file can no
#: longer be merged back against the full one -- which is the property the
#: whole feature rests on.
PROTECTED = {
    "File":             "identifies the recording",
    "Channel":          "identifies the muscle",
    "StimType":         "identifies the experimental cell",
    "Stim_Label":       "the analyst's name for the stimulus type",
    "Segment":          "identifies the trial; the key rows are joined on",
    "Segment_Overall":  "trial order across the whole recording",
    "Limb":             "identifies the side recorded",
    "Measure":          "says what kind of response this is",
    "Outlier_Decision": "says whether the trial should be modelled at all",
    "Condition":        "identifies the experimental cell",
}

#: Selectable groups, in the order they are offered.
#:
#: Each entry is (key, label, columns). Grouped by what an analyst decides to
#: use TOGETHER: nobody keeps three of the four detrended columns, and nobody
#: keeps a QR-adjusted amplitude without the diagnostics that say whether the
#: fit was any good. Splitting a group finer would only offer choices that make
#: an output harder to interpret.
GROUPS = (
    ("timing", "Timing context", (
        "Stim_Time(s)",
        "Time_Since_Last_Stim(s)",
    )),
    ("amplitude", "Raw amplitude", (
        "PTP(mV)",
        "MEP_RMS(mV)",
        "AUC(mV*s)",
    )),
    ("onset", "Onset and latency", (
        "Latency(ms)",
        # Travels with the latency, as MEP_Offset_Source does with the offset:
        # a latency whose definition is not stated cannot be pooled with one
        # whose definition is.
        "Latency_Source",
        "MEP_Offset(ms)",
        "MEP_Duration(ms)",
        "MEP_Offset_Source",
    )),
    ("onset_agreement", "Onset method agreement", (
        "Onset_MethodsMedian(ms)",
        "Onset_Disagreement(ms)",
        "Onset_IQR(ms)",
        "Onset_Methods_N",
    )),
    ("csp", "Cortical silent period", (
        "cSP_Duration(ms)",
        "cSP_MEP_Offset(ms)",
        "cSP_EMG_Return(ms)",
        "cSP_MEP_Ratio(ms/mV)",
    )),
    ("prestim", "Pre-stimulus EMG", (
        "PreStimRMS",
        "PreStimPTP",
        "Z_PreStimRMS",
    )),
    ("prestim_normalised", "Amplitude per pre-stimulus EMG", (
        "PTP_per_PreStimRMS",
    )),
    ("within_file_z", "Within-file z-scores", (
        "Z_PTP_Within",
        "Z_PTP_Pooled",
    )),
    ("detrended", "Detrended amplitude", (
        "PTP_Detrended_WithinCond(mV)",
        "PTP_Detrended_WithinCond_Z",
        "PTP_Detrended_Session(mV)",
        "PTP_Detrended_Session_Z",
    )),
    ("reference", "Reference normalisation", (
        "Reference_Type",
        "Reference_Mean(mV)",
        "Reference_N",
        "Normalised_PTP",
        "Normalised_PTP_per_PreStimRMS",
    )),
    # Not splittable. The nine EMGComp_* columns are the record of how the
    # quantile regression behaved on this sample -- how many trials it fitted,
    # what it did to the PTP/RMS association, whether it fell back. An adjusted
    # amplitude reported without them cannot be assessed, only trusted.
    ("carson_qr", "EMG compensation (Carson QR)", (
        "Adjusted_PTP_QR(mV)",
        "Normalised_Adjusted_PTP_QR",
        "EMGComp_Method",
        "EMGComp_N",
        "EMGComp_Slope",
        "EMGComp_Intercept",
        "EMGComp_InterceptWeight",
        "EMGComp_Adjustment(mV)",
        "EMGComp_PseudoR2",
        "EMGComp_Rho_Pre",
        "EMGComp_Rho_Post",
    )),
    ("acquisition", "Acquisition flags", (
        "Clipped",
        "Units_Assumed",
    )),
    ("manual", "Manual annotation", (
        "Manual_Note",
    )),
)

#: {group key: group key it drags in}.
#:
#: The test for a dependency is NOT "could this be interpreted alone" -- the
#: narrow file can always be merged back against the full one, so nothing is
#: irrecoverable and most columns need no partner. It is whether the missing
#: column carries information that VARIES BETWEEN ROWS and silently changes
#: what the retained number means.
#:
#: Only one column pair meets that. reference_map is per stimulus type, so a
#: recording whose TMS conditions normalise to CSE and whose M-wave normalises
#: to Mmax puts ratios against two different denominators in one
#: Normalised_Adjusted_PTP_QR column. With Reference_Type present the column is
#: heterogeneous and says so; without it the column is heterogeneous and looks
#: uniform. Stack several sessions into the group file and the mixing is across
#: participants too.
#:
#: Everything else means the same thing on every row. Z_PTP_Within without
#: PTP(mV) is still a z-score; PTP_per_PreStimRMS without PreStimRMS is still
#: that ratio. Those get no edge, deliberately: pulling them in would make raw
#: amplitude effectively impossible to deselect, which is the case the feature
#: exists for.
#:
#: A table rather than one hardcoded pair because the next metric normalised
#: against something that varies per stimulus type will need the same rule.
DEPENDENCIES = {
    "carson_qr": "reference",
}

#: Why each dependency exists, for the log line when one is pulled in.
DEPENDENCY_REASONS = {
    ("carson_qr", "reference"):
        "Normalised_Adjusted_PTP_QR is a ratio to a reference mean that can "
        "differ between stimulus types; without Reference_Type the column "
        "mixes denominators without saying so",
}

GROUP_KEYS = tuple(k for k, _label, _cols in GROUPS)
GROUP_LABELS = {k: label for k, label, _cols in GROUPS}
GROUP_COLUMNS = {k: cols for k, _label, cols in GROUPS}


def columns_for(keys) -> list:
    """Every column belonging to the named groups, in schema order.

    Order comes from GROUPS rather than from the caller, so two sessions that
    selected the same groups produce the same column order however the boxes
    were ticked.
    """
    chosen = set(keys or ())
    out = []
    for key, _label, cols in GROUPS:
        if key in chosen:
            out.extend(cols)
    return out


def resolve(keys):
    """Expand a selection to include whatever its members depend on.

    Returns ``(resolved_keys, pulled_in)`` where ``pulled_in`` is a list of
    ``(dependent, required, reason)`` for anything added. The caller reports
    those: a selection that quietly grows is as confusing as one that quietly
    shrinks.

    Iterated to a fixed point rather than resolved in one pass, so a chain
    added later resolves without this needing to change.
    """
    resolved = set(keys or ())
    pulled = []
    changed = True
    while changed:
        changed = False
        for dependent in sorted(resolved):
            required = DEPENDENCIES.get(dependent)
            if required and required not in resolved:
                resolved.add(required)
                pulled.append((dependent, required,
                               DEPENDENCY_REASONS.get((dependent, required),
                                                      "")))
                changed = True
    return resolved, pulled


def select(all_columns, keys):
    """The columns a narrowed copy keeps, in the order the full file has them.

    ``all_columns`` is the written frame's own column list, so a column this
    module has never heard of cannot be dropped by accident: it is simply not
    in any group, and the coverage test fails rather than the file quietly
    losing it.
    """
    resolved, _pulled = resolve(keys)
    keep = set(PROTECTED) | set(columns_for(resolved))
    return [c for c in all_columns if c in keep]


def unassigned(all_columns) -> list:
    """Columns that are neither protected nor in any group.

    The reason the coverage test can be written at all. A metric appended to
    LAT_COLS without a group shows up here, which fails the test, rather than
    silently vanishing from every narrowed file ever written afterwards.
    """
    known = set(PROTECTED)
    for _key, _label, cols in GROUPS:
        known.update(cols)
    return [c for c in all_columns if c not in known]


def duplicated() -> list:
    """Columns claimed by more than one group, or by a group and PROTECTED.

    A column in two groups would be kept whenever either was selected, which
    makes the group list a description of what is offered rather than of what
    is written.
    """
    seen, dupes = {}, []
    for key, _label, cols in GROUPS:
        for c in cols:
            if c in seen or c in PROTECTED:
                dupes.append(c)
            seen[c] = key
    return dupes
