# MEP-CMAP Analyser

**Version 1.4.5 | August 2026**  
*Authors:* [*Justin W. Andrushko PhD*](https://orcid.org/0000-0003-2258-1689) · [*David A. Cunningham PhD*](https://orcid.org/0000-0003-2246-1548) *(*[*TMS Analysis ToolBox*](https://github.com/CunninghamLab/TMSAnalysisToolBox)*)*  —  *TMSMultiLab*

*Collaborators:* [*Nicholas Holmes PhD*](https://www.birmingham.ac.uk/staff/profiles/sportex/holmes-nick) *·* [*TMSMultiLab*](https://github.com/TMSMultiLab/TMSMultiLab/wiki)

[![PyPI version](https://badge.fury.io/py/mep-cmap-analyser.svg)](https://pypi.org/project/mep-cmap-analyser/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/jandrushko/mep-cmap-analyser/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**PyPI:** https://pypi.org/project/mep-cmap-analyser/  
**GitHub:** https://github.com/jandrushko/mep-cmap-analyser  
**Bug reports:** https://github.com/jandrushko/mep-cmap-analyser/issues

A BIDS-compliant, open-source, cross-platform desktop tool for processing, quantifying, and group-analysing TMS/EMG neurophysiology recordings. Built for researchers who need reproducible, auditable waveform analysis without writing custom scripts for every study — and extensible with community add-ons for analyses the core tool doesn't yet cover.

---

## Overview

MEP-CMAP Analyser is a GUI pipeline for EMG data collected with transcranial magnetic stimulation (TMS) and peripheral nerve stimulation (PNS) paradigms. It is organised around a neuroimaging-style **first-level / second-level** workflow:

* **Setup** — open a dataset, manage the file queue, and (optionally) convert raw recordings into a BIDS-compliant layout.
* **First Level: Single File** — process individual recordings: filter, segment trials around stimulation events, detect and quantify response features, review per trial, and run optional analysis add-ons.
* **Second Level: Group** — merge all processed sessions into a single, statistics-ready table for mixed-effects modelling, and run optional group-level add-ons.

Every setting, decision, and manual edit is saved in a sidecar JSON so analyses are fully reproducible and can be re-run or audited at any time.

The tool is not limited to any single measure or paradigm. It handles motor evoked potentials (MEPs), compound muscle action potentials (CMAPs), cortical silent periods (cSPs), M-wave recruitment curves, paired-pulse protocols such as SICI and ICF, and any other time-locked EMG response measurable by peak-to-peak amplitude, onset latency, or area under the curve. It operates on continuous recordings, pre-epoched trial stacks, and EMG bursts recorded without stimulation.

---

## What's New in 1.4.5

**Silent period values produced by earlier releases were wrong, in the short
direction. Re-run any analysis that reports them.** The detector, the review
window and the preview each measured the silent period slightly differently,
and two of the three truncated it. Details below; `DETECTION_VERSION` is now
`2026-modular-v5`, and it is stamped into every output so v4 and v5 results can
be told apart. Onset and MEP offset detection are untouched and reproduce v4
exactly.

### One silent-period detector, shared by the analysis, the preview and the review

Three places measured the silent period, and each built the detector's
arguments itself:

* the **Data Inspector** capped the search *end* at
  `second peak + Max offset from MEP 2nd peak`. That setting means the silent
  period must *start* within that distance of the response; capping the window
  with it truncated the thing being measured. At its default of 100 ms, no
  silent period longer than about 100 ms could be found during review, while
  the analysis reported the true duration for the same trial. A comment above
  the code asserted the cap was not applied there.
* the **preview** built its configuration without a single numeric cSP
  setting, so every one of them fell back to a default and the interface had no
  effect on it at all. Changing *Min return* from 40 ms to 2 ms produced a
  byte-identical preview.

All three now call one function with one settings object, and the search starts
at each trial's own second peak-to-peak landmark everywhere. A test fails if any
caller starts building the arguments by hand again.

**Search start** has been removed from the 1c panel. Nothing read it except the
preview, and that was the preview disagreeing with the analysis. It is not kept
as a floor: short-latency stimulus types reach their second peak well before the
old 40 ms default, so a floor would begin the search after a genuine early
silent period had already started.

### Min return is applied, and says when it cannot be

`Min return` sets how long EMG must stay back before the silence is called over.
It was carried through the interface, the configuration and every call, and
never read. Breakthrough EMG — a brief burst part way through an otherwise
complete silent period — therefore ended the measurement at the first burst.

It is now applied. A value shorter than the RMS smoothing window cannot be
enforced, because a moving-window RMS cannot rise and fall faster than its own
window; such a value is raised to the window and the trial says so rather than
appearing to work. This is easy to reach at high sampling rates, where the two
are set in milliseconds and compared in samples.

### The silent period is measured on every trial, not only reviewed ones

cSP columns were populated *only* from stored Data Inspector metadata, so a
trial nobody had opened carried no silent period and the column described review
history rather than the condition. On one 20-trial condition this showed as 9,
17 or 20 trials depending on how much clicking had happened.

The analysis now detects the silent period itself, through the same entry point
the Inspector uses. Stored markers still take precedence: a landmark placed or
checked by hand is a decision and detection never overrules it.

### Markers record who placed them

Auto-detected and hand-placed landmarks were stored identically, so nothing
could tell them apart. That mattered the moment the detector changed: markers
written by the previous version sat in the session and were reused verbatim,
and dropping them wholesale would have destroyed genuine manual edits alongside
them.

Each landmark now carries its provenance, and the segment records which detector
version produced its automatic markers. On opening a session written by an
earlier version, automatic markers are re-detected and manual edits are kept.

Sessions written before 1.4.5 carry no provenance, so their markers cannot be
attributed and are all re-detected. A manual edit made before this release will
need placing again; that is the cost of not silently reporting a stale
measurement.

### Rectified and RMS envelope overlays in the Data Inspector

Two display toggles, **Rectified (R)** and **Envelope (E)**, with `R` and `E`
flashing them on and off — an ambiguous offset is far easier to judge from an
overlay appearing and disappearing than from a static one.

The envelope is the one the detector thresholded, built with the same function
and the same window, drawn with the suppression threshold, the baseline mean and
the percentage between them. So a marker you disagree with can be checked
against the signal and the line that produced it, rather than against the raw
trace, which the detector never looks at.

It is drawn mirrored, as a band between `+env` and `−env`. One-sided it sits in
the positive half of the axis and collides with the response's positive peak,
which is where the peak-to-peak markers have to stay legible. Mirrored, a silent
period reads as the band pinching shut and reopening. The pre-stimulus window
the threshold was derived from is shaded, and the view widens to include it.

These are display only. Which trace a marker is measured on is a property of the
marker, not of what happens to be visible, so nothing you tick can change a
measurement.

### Smaller corrections

* **`cSP_Duration(ms)` is numeric.** It was written as `Not Marked` when a trial
  had no silent period, which made the whole column text while the three cSP
  columns beside it stayed numeric. `read.csv` typed one of the four
  differently and `mean()` on it returned `NA` without complaining. Blank now.
* **Area under the curve is computed whenever the response has a detected
  end**, whether that end came from a baseline return or from the start of a
  silent period. The window could previously only be closed by a baseline
  return, so trials whose silent period was detected rather than stored lost
  their AUC silently.
* **A truncated silent period says so.** When EMG has not returned for the
  required interval before the search window ends, the duration is reported as
  a lower bound with a message suggesting a longer search window, rather than
  as a measurement that happens to equal the window width.
* **`design_notch_sos` is now `design_notch_filters`.** It returns transfer
  function coefficients, not second-order sections, and the old name said
  otherwise. The old name still works.
* An unreachable duplicate of the detection module has been removed.

---

## What's New in 1.4.4

### Overlay a condition's trials beside the trial-by-trial view

**Preview detection** showed one trial at a time. That answers whether the
markers landed sensibly on *that* trial, and it cannot answer whether a setting
suits the condition, because the second question is about a distribution:
whether onsets cluster or scatter, whether the amplitude window contains the
response on most trials or only the large ones, whether one trial is unlike the
rest. No single trial shows any of that, and stepping through eighty turns a
distribution into a memory test.

The preview now opens one window with two halves. Above, every chosen trial of
the condition on shared axes, with

* the **amplitude window** the analysis resolved, anchored where anchoring
  applies,
* the **pre-stimulus window** the baseline measures are computed over, gap
  included,
* a strip beneath carrying one tick per **onset**, **offset** and
  **silent-period end**, so the spread of each is visible at a glance.

Below, the ordinary Data Inspector, read-only, showing one trial. Both are
driven by a single event-type control: as two windows with two dropdowns they
could disagree, and one panel showing a different stimulus type from the other
reads as the two contradicting each other rather than as controls out of step.

Trials are chosen from a list beside the plot. Selecting one moves the trial
view to it; selecting several is a request to compare them, so the view stays
where it is. Clicking a trace does the same. Left and Right step through trials
from anywhere in the window. A **Rectify** box rectifies the traces for display,
before the median is taken — a rectified average being the average of the
rectified trials rather than the rectified average of the raw ones, which differ
wherever trials disagree in sign.

Above sixty traces the drawing becomes a band of the per-sample minimum and
maximum with the median over it. A band rather than a subsample of trials,
because a subsample hides the outlier and the outlier is what an overlay is
read for.

Conditions cut to **different epochs are refused** rather than drawn together:
one time axis cannot describe two, and overlaying them would show a latency
difference that does not exist. The differing epochs are named rather than the
option quietly going missing.

### Four settings the preview was not reading

Each of these was silent. The preview reported a number, the number looked
plausible, and it had been derived without a setting that had been changed.

* The **blanking gap** and the **silent-period assignment** never reached it. A
  gap set to 50 ms had no visible effect anywhere in the preview, and the end of
  a response was found by return-to-baseline on stimulus types where a silent
  period defines it.
* The window drawn as the pre-stimulus baseline was **the wrong one of the two**
  the analysis cuts. The epoch carries its own lead-in, which the onset
  detectors threshold against; a *separate* segment ending a gap before the
  stimulus is what `PreStimRMS`, `PreStimPTP`, the outlier screen and the
  excitability compensation are computed from. They are different intervals
  whenever a gap is set.
* That window was also resolved **once, across every stimulus type in the
  file**, taking the largest gap — so one type's 20 ms gap displaced the shaded
  window on every other type in the recording. It is now per drawn type.
* **Silent-period detection could not run at all.** The detector now runs, and
  its own account of why it found nothing is reported along with the window it
  searched. A type with no silent period assigned is distinguished from one
  assigned and not found.

Offsets and silent periods are now found with the analysis's own detectors
before the preview opens, as onsets already were, so the trial view shows the
landmarks the run will produce rather than re-deriving them one trial at a time.
The silent period is found first, because the offset rule takes a detected
silent-period start as the end of the response: the two are one physical event.

Anything reported about detection is now counted on detection alone. The count
of pre-detected onsets had come to include trials seeded only with an offset or
a silent period, so a type with six onsets and twenty silent periods was
reported as twenty of twenty — a figure read to judge whether a setting is
working, and worse than none when inflated.

---

## What's New in 1.4.3

### Results grouped by output family

One recording writes nine files per channel, so a two-channel study left
eighteen loose files in one folder and a five-channel study forty-five, before
any add-on. Results are now filed under `trial-level/`, `summary/`,
`onset-methods/`, `segments/`, `add-ons/` and `report/`, grouped by what a file
**is** rather than by which channel produced it, so an analyst comparing a
measurement across channels has them side by side.

**Filenames do not change and nothing is moved.** A file has to be identifiable
from its name wherever it ends up, so the full BIDS prefix stays. Existing
studies keep the flat layout they were written with, both arrangements are read,
and a folder half in each state loads completely.

### An optional trimmed copy of the trial file

The trial-level table carries fifty-six columns. Most analyses use a handful,
and a table nobody can read across on one screen is a table whose columns get
selected in a spreadsheet by hand.

**Preferences ▸ Trial columns** enables a second file, `_trials_selected.csv`,
written beside the full one and holding a chosen subset at the same one row per
trial. `_trials.csv` is never affected and always carries every column.

Columns are chosen by group rather than individually — nobody keeps three of the
four detrended columns — and a group whose members cannot be read without
another pulls it in and says so in the log. Nothing is lost by using the trimmed
file: the columns that identify a trial are always kept, so it can be merged
back against the full one whenever a dropped column is wanted.

Off by default. The choice applies to every recording, and a single recording
can depart from it on tab 1c, where "use the preference" stays distinct from
"off for this recording" — otherwise switching the preference on later would
quietly switch a recording back on.

### The group analysis states what its table contains

**Second Level** can be built from either trial file. Where the trimmed file is
asked for, sessions that lack one, or that did not select the same columns, are
named and the build is refused rather than completed from whatever each session
happens to have: a table whose columns depend on which participants were
included is one where adding a participant silently changes the analysable
variables. Sessions are compared on the selection each recorded at analysis
time, not on their file headers, which cannot tell an analyst who chose to drop
a measure from a recording that had none.

Add-on outputs, previously joined unconditionally, can now be excluded or chosen
individually. The list offered is discovered from what is actually beside the
scanned sessions, so a third-party add-on appears without being known to the
tool, and one present in only some sessions says so before it is chosen. The
trial file source and the add-on choices are saved with the study design, since
a study rebuilt under different ones is a different table.

---

## What's New in 1.4.0

### Conditions

A recording's markers say what kind of stimulus fired, not what it was for.
Twenty pulses labelled `A` may be ten before an intervention and ten after, and
nothing in the file distinguishes them.

**Setup ▸ Conditions** assigns trials to named conditions. The table opens
populated from the recording's own events, so a file needing no conditions costs
one click through. Select a row to see its epochs drawn as an overlay, an
average, or both; select ten of twenty trials and split them into a new
condition. Several conditions can be drawn together in different colours, which
is how a split is checked before it is applied.

A condition is a second axis alongside the stimulus type, not a replacement.
`A` decides how a response is detected — its latency window, its muscle, whether
a silent period applies. `pre` and `post` decide what the trial means. The trial
file reports the two in separate columns, so a timepoint is a factor the group
analysis can model rather than a substring to be parsed out of a name.

Each condition may be given its own epoch, decided while looking at the
waveforms rather than typed blind on a settings tab. Epochs are held per
channel, since a hand muscle and a leg muscle want different windows, while the
conditions themselves are shared: a trial belongs to the same condition whichever
channel it is viewed on. Every edit is undoable.

Conditions are written to a BIDS `_events.tsv` beside the recording, with a
documented `condition` column, and read back on the next load. Not a private
format: an assignment should survive being read by something other than the
program that wrote it. Every event must belong to a condition or be explicitly
excluded, and an excluded trial is written as `n/a` rather than dropped, so the
file accounts for every event in the recording.

Conditions are carried into BIDS-ify, so a grouping made here describes the
converted recording too, and survives being written out and read back in.

### Stimulation described per protocol (NIBS-BIDS v6.3)

A recording can contain more than one kind of stimulation. A peripheral M-wave
on one stim code and a TMS MEP on another are two protocols, two intensities,
often two stimulators, and BIDS-ify had one intensity box and one modality for
the whole file — so it could describe neither.

Stimulation is now defined as a **parameter set**: named once per session, with
its own type (TMS, tES, TUS, or PNS), intensity, waveform and timing, and
optionally its own stimulator, coil or measured threshold where that protocol
differs from the session default. Each stim code in a recording then points at
one. Written once, referenced many times, which is also what stops a threshold
being retyped for every file and corrected in only some of them.

`PNS` matters here in particular: M-waves, H-reflexes and CMAPs are peripheral,
and before v6.3 of the proposal they had to be misdescribed as TMS, tES or TUS.

Dosing keeps its derivation. 120% of a resting motor threshold of 50 %MSO is
recorded as an intensity of 60 in %MSO, a reference of `rMT`, and a scaling of
1.2, with the measured threshold stated once in its own block. Encoding the
reference into the unit — the old `%RMT` — lost both the threshold and the
factor, leaving a number that cannot be compared with anyone else's.

Conversion writes the four-file structure the proposal defines: `*_nibs.tsv`
with one row per parameter set, `*_nibs.json` with the devices, the dosing
references, and a definition and unit for every column, `*_markers.tsv` with one
row per placement, and an `*_events.tsv` in which every delivery names the
protocol it was and where it was applied. Units are always declared rather than
inferred: 58 could be percent of maximum stimulator output or milliamps.

Where a condition changes the stimulus rather than only the meaning of a trial —
half of a code at 100 mA and half at 150 mA, or a recruitment curve — each part
takes its own parameter set and becomes its own row, which a per-code assignment
could not express. A file is not offered for conversion until every ticked code,
and every half of a split one, has been assigned.

A converted file can be re-opened and rewritten. Correcting an intensity or a
mis-assigned code no longer means starting again, and the existing output is
overwritten in place rather than deleted first.

Studies that do not use parameter sets still convert exactly as before, writing
the flat sidecar rather than an empty table.

**Point releases in the 1.4 series:** 1.4.1 kept the continuous integration suite in step with the project’s declared dependencies. 1.4.2 corrects the analysis window used by peak-to-peak amplitude, MEP offset and duration, and onset detection on any stimulus type given its own epoch window; re-detects Inspector landmarks when an event delay or an epoch changes under them; restores the TMSMultiLab mark in compiled builds; preserves channel units the quantities library cannot parse; and writes the stimulation description as `*_nibs.tsv`. Studies whose stimulus types all share one epoch window are unaffected by the window correction. 1.4.3 files results under folders named for what each file is rather than leaving them loose in one directory, adds an optional trimmed copy of the trial file for analyses that do not need all fifty-six columns, and lets the group analysis choose which trial file and which add-on outputs it is built from. Filenames are unchanged, nothing is moved, and both the flat and the foldered layouts are read, so existing studies are unaffected. 1.4.4 draws every chosen trial of a condition on one plot beside the trial-by-trial view, with the amplitude window, the pre-stimulus window and a strip of detected onsets, offsets and silent-period ends; and corrects four settings the preview was not reading, among them the blanking gap and the silent-period assignment, so that a preview now rehearses the run rather than approximating it. 1.4.5 gives the analysis, the preview and the Data Inspector one silent-period detector instead of three, applies the `Min return` setting that had never been read, measures the silent period on every trial rather than only reviewed ones, and records whether each landmark was detected or placed by hand so that a change of detector re-detects the former and keeps the latter. **Silent period values from 1.4.4 and earlier were wrong in the short direction and should be re-run.**

### Channel assignment for every format

The channel and event-marker dialogue used to run for Spike2 text exports
alone, so a LabChart export of six named channels was analysed on whichever came
first, with no route to Event sources at all. It now runs for every format, and
is not skipped when a file appears to offer no choice — a recording whose
embedded markers are wrong needs a threshold source, and that decision is only
reachable there. Reopen it from tab 1a or from **File ▸ Reassign channels** to
correct a choice without reloading.

### One session per recording

**Save Session** wrote wherever you pointed it, defaulting beside the raw data,
while the automatic save wrote a BIDS-named file under `derivatives/`. A
recording could carry two sessions that knew nothing of each other. There is now
one, in `derivatives/`; **File ▸ Save session copy** covers a named variant.

Save, Load, Preview and Run appear on every First Level tab rather than only on
1c, and moving between tabs saves — preparing a set of recordings before running
any of them is a workflow this tool supports. Run Analysis stays disabled until
the detection settings have been seen for that recording.

### Reading

**LabChart block exports** are recognised as pre-epoched. Each block is a trial
already cut about the stimulus, and nothing announced it: an over-long window ran
off the end of a block into the padding between them and then into the following
trial. Analysis windows are now clamped to what a block contains.

**Text files with no data rows are declined** when opened, rather than being
claimed as a Spike2 export and failing several steps later inside a parser.

> **The epoch window moved.** Pre- and post-stimulus extents are now set per
> stimulus type on tab 1a, not once for the file on tab 1c. A recording where
> every type should share a window behaves as before; a file mixing an M-wave
> with a cortical silent period no longer has to truncate one or carry
> unnecessary samples through every trial of the other. Tab 1c keeps the
> detectors' baseline, which genuinely is one setting for the recording.

**Preview detection** — try the current detection settings on chosen trials
before committing to a run. The preview reads its settings from the same
snapshot the analysis takes and cuts trials with the same epoching and event
delay, so what it shows is what the run will produce. The Data Inspector opens
read-only: markers are drawn where the detector puts them but cannot be
dragged, and nothing is saved.

**CED Signal** recordings are read directly from Signal's MATLAB export. Frames
are treated as pre-epoched trials and the frame state becomes the stimulus
type.

**Every column on tab 1a and field on tab 1c carries an explanation**, reached
by hovering or clicking the ⓘ beside it. Hovering shows it; clicking pins it
open.

**Event sources reach the analysis.** They previously populated the labels tab
and nothing else, so a configured threshold changed what was displayed while
the run went on reading the file's own markers. They are held per channel and
recorded in the derivative sidecar.

> **Reprocess if you use MEP offset, duration, AUC, or any file whose event
> markers are mistimed.** This release corrects three faults in offset detection
> and adds a correction for markers that do not coincide with the stimulus.
> Offsets and durations change on most recordings; AUC changes where it is
> bounded by the offset; latency changes only on stimulus types given an event
> delay. Peak-to-peak, cSP and normalisation are unchanged except where
> amplitude-window anchoring previously fell back to the file-wide window.
>
> **Check any analysis whose session was saved with File → Save session.** That
> path wrote thirteen fewer settings than the automatic save, among them the
> latency profiles, both muscle-group dropdowns, the onset method and every
> onset detector parameter. Reloading such a session restored the file and most
> of its settings but silently reverted those, so a subsequent run could use a
> latency window and a detector the analyst had not chosen. Sessions written by
> the automatic save — which is what runs after the Data Inspector closes — were
> always complete. Both now use one builder.

### Multiple channels in one analysis

The analysis now runs **once per selected channel**, in sequence, each pass using
that channel's own setup on tab 1a — labels, gaps, delays, cSP assignment,
references and latency profiles. An iSP recorded on one channel and a
contralateral MEP on another are different muscles under the same marker, and
they need different latency windows.

Channels are chosen in the **Channel Assignment** dialogue when a Spike2 file is
first opened, which is now a tick list rather than a single choice, or afterwards
from the **Analyse** button beside the channel dropdown. There is deliberately no
"primary" channel: every ticked channel is analysed identically, so a primary
would imply a hierarchy that does not exist. The first ticked is simply where
configuration starts. **File → Reassign channels…** reopens the dialogue for a
file already assigned.

**Confirm Setup** walks through the selected channels, returning to tab 1a on
each until all are confirmed, and the analysis refuses to start while any remain
unconfirmed. A channel never configured starts from defaults rather than
inheriting the previous one's table; **Copy this setup to all channels** is the
only way settings move between them.

Where more than one channel is analysed, output filenames carry a `channel-`
entity so the passes do not overwrite each other. A `Channel` column is written
to the trial and summary files either way, so a single-channel result can still
join a multi-channel dataset.

The pipeline itself is unchanged: it receives one channel's settings per run and
knows nothing of the others.

### Where stimulus events come from

Every reader exposed `extract_stim_times(path, marker_name)`, and every reader
meant something different by it: the event channel to read, the label to attach
to threshold detections, or nothing at all. The shared signature was a
coincidence of naming rather than an interface, so asking a reader for something
it did not already do was impossible — the one parameter that could have carried
the request already meant something else.

That is why LabChart MATLAB read its comment table and nothing else. Not a
missing feature so much as a missing question: the format carries comments,
digital inputs and fixed-interval sampling, and the tool could only ask for the
first.

Events now come from an explicit list of **sources**, of three kinds:

- **The file's own events** — comments, markers, annotations, event channels:
  whatever the format already carries.
- **A trigger channel** — a crossing on any analogue channel, with a level, an
  edge and a refractory period so that a pulse which rings is counted once.
- **Fixed interval** — for recordings triggered by something the file does not
  record. Nothing is detected here: the times are asserted, and no part of the
  recording can confirm them.

Several may be combined, each contributing its own stimulus type. Two sources
producing the *same* type is reported as an error rather than merged, since that
gives a trial count matching neither. Events from different sources falling
within a few milliseconds of each other are **kept and reported**: they may be
one stimulus logged twice, or two genuine stimuli in a paired-pulse protocol,
and nothing in the data distinguishes those — merging would silently halve a
paired-pulse trial count.

**Threshold and interval detection are format-independent.** Both need only a
waveform and a time base, which every reader already provides, so they are
written once and available everywhere. A reader has only to say which of its
channels are analogue.

`extract_stim_times` is unchanged and is what a file's-own-events source calls,
so a file configured the way every file was configured before runs through no
new code. That contract is checked on real recordings covering six formats, and
on files the tests build themselves for three of them, so it holds wherever the
suite runs rather than only where the sample recordings live.

### Choosing a level

A threshold level is not checkable by reading it. Two volts is right or wrong
depending on the trigger's amplitude, its baseline and whether the pulse rings,
none of which is visible from the box it was typed into.

**Event sources…**, on tab 1a and in the channel assignment dialogue, draws the
chosen channel with the level across it, every detected crossing marked, and a
count that updates as the level changes.

The trace is reduced by minimum and maximum per column rather than by
subsampling. A stimulus trigger is a one-sample spike: on a two-thousand-second
recording at five kilohertz, plain subsampling drew a flat line while the
detector found two hundred events. A preview that hides the pulses is worse than
none, because the level would then be set against a trace showing nothing of
what the detector sees.

### Choosing channels

The channel assignment dialogue offers a **tick list** rather than a single
choice, and appears every time a file is opened with the previous choices
pre-selected — it had appeared only when no saved assignment existed, so after
the first open the channel and trigger choices became invisible and
unchangeable, and the only way back was deleting the derivatives folder and the
sidecars by hand. Those two decisions determine what the whole analysis
measures. **File → Reassign channels…** discards a saved assignment explicitly.

There is deliberately no primary channel: every selected channel is analysed
identically with its own setup, so a primary would imply a hierarchy that does
not exist. The first selected is where configuration starts.

The dialogue runs for **every format**, and is not skipped when a file appears to
offer no choice: a recording whose embedded markers are wrong needs a threshold
source configured against a trigger channel, and that decision is only reachable
there. It had run for Spike2 text exports alone, so a LabChart export of six
named channels was analysed on whichever came first.

Reopen it from **Channel assignment…** on tab 1a, or from **File ▸ Reassign
channels…**, to correct a choice without reloading the recording.

### Files no reader can open

Format detection ended by assuming anything unrecognised was a Spike2 text
export, so a Word document, a configuration file or an ADInstruments `.adicht`
all failed somewhere downstream with a message naming the wrong format. A binary
file matching no known signature is now identified as unreadable, and where the
extension is recognised the message says what to do instead — for `.adicht`,
that LabChart can export text or `.adibin`, both of which this tool reads.

### Event delay: when the marker is not the stimulus

The event marker in a recording is not always the instant the stimulus fired. A
trigger written by software after the pulse, a stimulator delay setting, or a
different signal path for one block will all shift it, usually by a fixed amount.

The consequence is not a visibly wrong latency. It is an epoch whose zero is
wrong, so part of the response falls into the pre-stimulus window — and then
every measure defined relative to the baseline fails, each in a way that looks
like a separate fault. On one recording whose markers for a single condition were
2 ms late, the derivative-ratio detector returned no onsets at all from fifteen
trials, peak-to-peak was read from a shoulder rather than the peak, and the offset
landed part-way down a deflection. Those were diagnosed as three unrelated
problems before the common cause was found.

A **Delay (ms)** column in tab 1a applies a per-stimulus-type correction when
epoching, so everything measured from zero moves with it, including reported
latencies. **Detect delays** measures it from the stimulus artefact and fills the
column in.

The artefact is located by **peak slope, not peak amplitude** — it is the
steepest feature in the epoch but not always the largest, and on a supramaximal
M-wave an amplitude search returns the response instead. Two guards decide whether
a delay is proposed at all:

- **The spread across trials.** A genuine fixed delay measures a fraction of a
  millisecond in standard deviation; markers that truly jitter measured 3.9 ms on
  a real recording. Above a threshold the scan reports the spread and proposes
  nothing, because a single correction would then be wrong on every trial rather
  than right on average.
- **The width of the transient.** Without an artefact — a shielded rig, or one
  where it has been removed — the steepest feature is the response's own rising
  edge, and because that edge is consistent it would pass the spread test and be
  proposed with confidence. Across two real recordings the artefact measured
  0.4–0.6 ms wide at half its maximum against 4.6 ms for a response edge, so the
  two separate cleanly.

Nothing is applied silently: proposals populate the column for review, the value
and whether it was measured or typed are written to the BIDS sidecar, and changing
a delay marks affected files stale in the queue.

### Offset detection: three corrections

**The refinement no longer reports its own search boundary.** It scanned a fixed
neighbourhood for the first sustained quiet run; when the fine envelope was
already quiet at the start of that neighbourhood — the normal case, since a
centred coarse window places the crossing late — it returned the neighbourhood's
lower bound. The reported offset was then `crossing − envelope_window` on every
trial: a function of a smoothing setting rather than of the signal. It now walks
back to the last sample that was genuinely still elevated, which removes the
dependence on the search radius entirely. Across envelope windows of 3, 5 and
8 ms the offset now varies by 0.6 ms where it previously moved with the setting.

**The baseline is estimated robustly.** The envelope uses a centred window, so a
stimulus artefact smears backwards by half of it. The guard covered an artefact at
zero; one landing even slightly earlier reached past it, and a handful of
contaminated samples then dominated the mean and standard deviation. On a real
recording this raised the threshold seventy-three-fold against a neighbouring
condition, and the offset was reported while the response was still at a quarter
of its peak. The median and median absolute deviation are unmoved by a short
contaminated tail.

**The peak-fraction floor is off by default.** It was introduced to fix offset
detection failing on eighty of eighty-one trials — but the cause there was a 60 ms
duration cap, and once that was raised the baseline threshold alone found every
trial. What the floor does is shorten the answer, in proportion to response size,
so it truncates hardest on the largest and cleanest responses. Measured against an
independent settle reference on a real M-wave recording, it cut the offset by 45
to 97 ms depending on its value; at zero the same reference gave errors of −5.9
and +2.3 ms. On a resting MEP recording, where responses are an order of magnitude
smaller, the two settings give identical offsets.

**The return threshold now knows where the signal settles.** It was derived from
the pre-stimulus baseline alone, which assumes the signal comes back to where it
started. Measured across every condition of one session, the envelope floor late
in the epoch sat 1.3 to 2.0 times the pre-stimulus floor, so that was a target the
signal never reached and the offset landed wherever the envelope happened to dip.
The floor is now the larger of the two. Medians moved earlier in every condition
tested; trial-to-trial scatter improved in five of eight and worsened in two.

### Data Inspector

**MEP offset and duration** appear in the read-out, updating as markers are
dragged. Where a cortical silent period is detected, its start marker *is* the
offset marker — the two are one physical event, and two draggable markers for one
event can be moved apart. Otherwise the offset gets a marker of its own.

**The area-under-curve window is tied to the onset and offset** in both
directions, and is reconciled whenever a trial is drawn rather than only when a
marker is moved. It previously ended at a fixed 50 ms after onset and could
disagree with the results file. A background silent-period search that set the
window for every stimulus type — including resting recordings, where the concept
does not apply — no longer runs for types not assigned to cSP.

**The amplitude window now comes from the analysis.** The Inspector re-derived it
from the file-wide setting and knew nothing of anchoring, so with anchoring
enabled the review measured a different interval from the analysis.

**A failed onset detection is reported as such.** The marker fell back to the
stimulus, which read as "Latency: 0.0 ms" — and that index was returned by *Save
edits & close* as a manual override, silently turning a blank latency into a
measured zero. Nothing derived from a non-detection is now exported.

**Show event-type median** draws the condition's median waveform behind the trial:
the same waveform the derivative-ratio detector compares against, so what is shown
is what the algorithm saw.

### Amplitude window anchoring

When a stimulus type has too few detected onsets to anchor, the fallback is now
that type's own **latency profile** rather than the file-wide window. The
file-wide start is the very thing anchoring exists to replace — typically 10 ms,
which sits after the peak of an M-wave — so the condition already in trouble was
also the one whose amplitude was truncated. On a real recording the first phase of
a 3.8 mV response fell outside the window entirely and peak-to-peak was read from
a 2.1 mV shoulder.

### Selecting a data range

The crop dialogue now describes what a selection **contains**, not only its time
bounds: how many events of each stimulus type, and where they sit in the file's
own numbering.

```
Selection: 2 ranges · A: 45 events (#1–45 of 90) · C: 30 events (#1–30 of 30)
```

Indices are per stimulus type, matching the Data Inspector, and discontinuous
selections are reported as they are rather than collapsed to their outer bounds.

### Smaller changes

- The **sampling rate and amplitude unit** are reported when a file is opened.
  Both were read automatically but neither was shown until the analysis ran, so
  opening a file gave no way to confirm what had been detected.
- **A missing latency profile is reported rather than invented.** Onset detection
  fell back to a hardcoded 10–50 ms, and every detector bounds its result by the
  minimum, so a stimulus type with no profile returned exactly 10.00 ms on every
  trial with a between-trial SD of zero — a window edge reported as a
  physiological latency. Both the analysis and the Inspector now bound onsets by
  the amplitude window and say so. The maps are per channel, so this was reached
  most easily by previewing a channel that had never been set up.
- **The read-back check no longer fails correct conversions.** EDF stores
  integers, so the written RMS moves by less than one quantisation step; the
  tolerance is now that bound rather than a fixed percentage. It reported a
  mismatch most readily on a channel carrying a stimulus artefact, where the step
  is coarse relative to the EMG being measured.
- **BIDS-ify no longer offers its own output back as input.** Conversion writes
  into the folder being scanned, so the worklist grew by one file after every
  conversion and the duplicates reached the analysis queue as extra recordings
  for the same participant.
- **The participant entered in Study Metadata is used.** A recording whose
  filename carries no `sub-` was converted to `sub-unknown`, contradicting what
  had been typed and the folder its derivatives were already in.
- **Reset & reprocess from scratch** now deletes the session it means to. It was
  removing the pre-derivatives filename, which nothing has written since sessions
  moved, so inspector edits, PTP markers, silent-period boundaries and exclusions
  all survived a reset that reported success.
- **Horizontal scrollbars work** in the Second Level table and the BIDS-ify
  worklist, with Shift+wheel. Columns were stretched to fill the width, so there
  was never anything to scroll to.
- **Second Level follows the derivatives folder** chosen in Setup instead of
  reading it once, before one had been chosen.
- The **filter preview** accepts every format the readers handle. It carried its
  own list of extensions that had not been updated when EDF, BrainVision, MATLAB,
  AcqKnowledge and CSV support was added, so a `.mat` file silently failed to load
  there and the preview asked for a sampling rate the file had already declared.
  The M-wave reference file dialogue had drifted the same way.
- **Tab 1a's guidance** now states what the Gap setting does to the background
  window: with a 10 ms gap and a 100 ms pre-stimulus window it runs from −110 to
  −10 ms, a full 100 ms shifted back rather than 90 ms.
- Quitting no longer prints Tk errors about invalid command names. Callbacks that
  reschedule themselves were left queued against an interpreter that was being
  torn down. The window close button now follows the same path as **File → Exit**.
- A **Latency window that contradicts its muscle group** is reported. A saved
  window wins over the profile, because a typed value must not be overwritten, but
  the only previous symptom of the two disagreeing was onsets pinning at the
  bottom of a profile the tab was no longer showing.
- **Marker edits no longer cross between channels.** A marker position is an
  index into one channel's waveform and means nothing applied to another. In a
  multi-channel run the second channel inherited the first's, which produced
  *negative* peak-to-peak values where the stored maximum sample sat below the
  stored minimum, and reported every offset as manually set when none had been.
  Edits are now held per channel through the analysis, the Data Inspector and
  the session file. A session written before this carries one unattributed map;
  it is used for single-channel review, where it unambiguously belongs to the
  channel on screen, and ignored in a multi-channel run rather than guessed at.
- **A negative peak-to-peak can no longer be written.** It is a magnitude. The
  value propagated into the normalisation and z-score columns as though it
  meant something.
- **The Data Inspector no longer raises when a key is pressed as it closes.**
  Arrow-key navigation redraws the window, and events queued before the window
  was destroyed still arrive; in a multi-channel run the window closes twice as
  often, so this was easy to hit.

### New onset detection method: derivative ratio (Boyles et al. 2026)

Working backwards from the first peak of the response, each candidate sample is
scored by the ratio of the mean absolute derivative *ahead* of it to the mean
absolute derivative *behind* it; the onset is the earliest sample still reaching a
set fraction of the maximum ratio, subject to three slope and latency gates. It is
methodologically independent of everything else here — not a threshold crossing,
not a run length in the derivative, not a cumulative change point — which is what
makes it useful as a member method. Ported from the MATLAB reference
implementation in the [TMSMultiLab library](https://github.com/TMSMultiLab/TMSMultiLab).
See [MEP Onset Detection](#mep-onset-detection).

**Detectors can now receive a condition average.** The derivative-ratio method
needs a grand-mean waveform to reject trials whose peak falls far from the
expected latency. The analysis supplies the outlier-screened median waveform for
each stimulus type — the same waveform onset anchoring already used, now computed
whether or not anchoring is enabled. No other method uses it, and none changes
behaviour as a result. The Data Inspector supplies the same waveform, so review
and analysis apply the same gate.

**Three corrections to the reference implementation, each switchable.** The
published MATLAB code contains three details that its own comments contradict.
They are corrected by default and reproduced exactly under *Reproduce the
published implementation literally*:

1. The slope comparison window is fixed in **samples**, not milliseconds. The
   reference computes the correct 5 ms width in samples and then never uses it,
   indexing with the literal default `5` instead — so the window is 2.5 ms at
   2 kHz, 1 ms at 5 kHz, 0.5 ms at 10 kHz. Measured on a real recording, literal
   detection fell from 18 of 20 trials at 1 kHz to 11 of 20 at 5 kHz while the
   corrected version held at 18 of 20; at 1 kHz the two agree exactly, since
   5 samples *is* 5 ms there.
2. The amplitude gate compares the response's peak-to-peak against the baseline
   **maximum** rather than the baseline peak-to-peak, making it roughly half as
   strict as its name implies.
3. The peak-jitter gate compares the trial's **largest** peak against the
   condition average's **first** peak. On a biphasic response whose second phase
   is larger, those differ by the peak-to-trough interval on every trial.

**A limitation of the derivative-ratio method.** All of its gates are stated in
absolute derivatives, so it needs the response to be spectrally *richer* than the
baseline, not merely larger. Measured at the published window on synthetic data,
detection of a smooth response fell from 19 of 20 trials at a 0.012 mV baseline to
0 of 20 at 0.020 mV, while a response with harmonic content still gave 16 of 20;
the RMS envelope method scored 20 of 20 throughout. Quiet resting recordings sit
well inside the working range — on a real recording with a 0.0034 mV pre-stimulus
RMS it found 17–18 of 20 trials per condition — but a noisy baseline or heavy
low-pass filtering will disable it, and it fails by returning nothing rather than
a wrong latency. It is also bounded by the first peak, so the reported onset can
never precede it.

The method is **off by default and not among the default member methods**: it has
the most parameters of any detector here, two of them scaled by the trial's own
peak-to-trough interval, its published validation was on three participants, and
adding it would make the member method count even — turning the median into an
average of two.

**Renamed: "Consensus" is now "Median across methods".** The former name implied
that the agreed value was the correct one, which is precisely what the method's
own output cautions against. `Onset_Consensus(ms)` becomes
`Onset_MethodsMedian(ms)`. Sessions and preference files written by the
previous release continue to work: the former method key still resolves, and
the renamed preference is carried across on first load.

---

## What's New in 1.3.3

> **Important: reprocess if you use onset latency or AUC.** This release fixes
> two faults that silently produced plausible-looking wrong numbers rather than
> failing. Onset latencies were clipped by the amplitude-measurement window, and
> MEP offset (and therefore AUC at rest) frequently failed to detect at all.
> Amplitude, cSP and normalisation measures are unaffected.

**Onset detection no longer depends on the amplitude window.** The onset search
window was taken from the PTP window, which is a single per-file setting, while
the latency profile is per stimulus type. A stimulus type whose profile began
before the PTP window start had its onsets pinned to the window edge — measured
on a deltoid-like case with a true onset of 8.9 ms, a profile of 8–16 ms and the
default 10–50 ms window, every trial returned exactly 10.00 ms with a
between-trial SD of zero. Implausibly consistent latencies are the signature,
which makes this considerably more dangerous than a detector that returns
nothing. The search window is now derived from the latency profile, widened by
the PTP window, and the log warns whenever onsets collapse onto a search bound.

**Four new onset detection methods**, bringing the total to seven: an RMS
envelope detector with SD-scaled threshold, a CUSUM change-point detector,
optional Teager–Kaiser preconditioning, and a method that takes the median
across several detectors. See [MEP Onset Detection](#mep-onset-detection).

**MEP offset and duration** are now detected and reported, giving `MEP_Offset(ms)`
and `MEP_Duration(ms)`. Where a cortical silent period is detected its start *is*
the end of the MEP and is reported as the offset; where there is none, the return
to baseline is detected instead. `MEP_Offset_Source` records which rule applied.
This also gives **AUC a principled endpoint in resting-state recordings**, where
there is no silent period to close the integration window and the endpoint
previously had to be dragged by hand.

**PTP window anchoring per stimulus type** *(opt-in)*. The amplitude window is one
setting for the whole file, so a recording containing both M-waves and MEPs
cannot be measured correctly by a single window. On a real mixed recording the
M-wave conditions had the first 6 ms of every response excluded from the
amplitude measurement, understating it by around 20%. Enable **PTP Window
Anchoring** in Preferences → Detection to give each stimulus type a window placed
on its own median onset; the 1c window end still applies as a ceiling.

**Onset method agreement and comparison outputs.** With *Compare methods on every
trial* enabled, every member method runs on every trial and the spread between
them is reported as `Onset_Disagreement(ms)` — a direct triage signal for which
trials need manual review. The individual latencies are also written to two CSVs
and five figure types, including Bland–Altman against a leave-one-out median.
See [Onset Method Comparison](#onset-method-comparison).

**Analysis and review now use identical detection.** The Data Inspector carried
its own copy of the onset dispatch. It knew only four methods, so selecting a
newer one silently fell through to peak-fraction, and it forwarded no amplitude
gate, peak fraction or slope threshold — meaning a re-detection during review
could use a different algorithm, and different settings, than the analysis it was
reviewing. Both paths now share one implementation.

**Preferences carry forward raised defaults.** The Preferences dialog writes every
field on the tab, so a value you never deliberately chose was stored and then
shadowed any later change to that default. Detection settings are now
version-stamped and migrated on load, but only where the stored value is still
the default it superseded — anything you actually changed is left alone. A
**Restore detection defaults** button covers the rest.

**Smaller fixes.** The amplitude window fields in 1c are relabelled and now state
plainly that they do not constrain onset detection. `min_peak_amplitude`,
`peak_fraction` and `slope_threshold` are honoured consistently across the
pipeline and the Inspector. Detection defaults are defined in exactly one place,
with a test that fails if any consumer drifts from it.

---

## What's New in 1.3.2

> **Important: reprocess existing derivatives.** This release fixes a
> column-alignment fault that affected every trial-level CSV written by earlier
> versions. Six columns held the wrong measurement and one was always empty. Any
> analysis that used the detrended amplitudes or the pooled z-score should be
> re-run after reprocessing. Peak-to-peak amplitude, latency, AUC, cSP measures
> and the raw waveforms were never affected.

* **Fixed: trial-level columns were shifted by one place.** The per-trial rows
  are built as positional lists, and the code that filled the z-score and
  detrended fields addressed them by literal index. Those literals were one
  lower than the column they named, so `Z_PTP_Pooled` received the detrended
  amplitude in millivolts, `PTP_Detrended_WithinCond(mV)` received a z-score,
  `PTP_Detrended_WithinCond_Z` received the session-detrended amplitude,
  `PTP_Detrended_Session(mV)` received a z-score, `PTP_Detrended_Session_Z` was
  never written at all, and `Z_PreStimRMS` was overwritten with the
  within-condition PTP z-score. The shift is easy to spot in an affected file:
  columns labelled `(mV)` hold values on a z-score scale and vice versa. Every
  such write now resolves its position from the column list by name, so the
  layout and the writers cannot disagree again.
* **New measure: MEP RMS.** Root-mean-square amplitude over the same window as
  peak-to-peak, written as `MEP_RMS(mV)` with matching summary columns.
  Peak-to-peak is determined by two extreme samples and is therefore sensitive
  to a single spike, while RMS integrates the whole response, so a broad
  low-amplitude MEP and a narrow spiky one of equal peak-to-peak amplitude are
  no longer indistinguishable.
* **Fixed: two definitions of pre-stimulus RMS.** `detection.quantification`
  is documented as the single source of truth for scalar trial metrics, but
  nothing in the codebase actually called it, and its `compute_prestim_rms` did
  not remove the DC offset while the pipeline's own version did. The two
  disagreed by around ten percent on a segment carrying a modest offset, so an
  add-on that followed the documentation got a value that silently contradicted
  the `PreStimRMS` column. `compute_prestim_rms` now takes `demean=True`,
  matching the pipeline and therefore the CSV, and the pipeline delegates to it
  so there is one implementation. The per-trial PTP, RMS and AUC that reach the
  trial CSV are now computed through these shared functions as well. Trial
  values are unchanged; only the standalone helper's default moved, and it moved
  onto the correct value. `compute_rms` is exported alongside the others.
* **New format: pre-epoched MATLAB trial stacks.** Files whose trials are
  already cut around the stimulus are read directly, rather than requiring a
  continuous recording with a trigger channel. The analysis windows are clamped
  to the extent the file actually contains, with the reduction reported in the
  log, and a warning is raised when the available post-stimulus window is too
  short for cortical silent period detection. The amplitude unit is confirmed
  once and saved to a sidecar when the file does not declare one.
* **Variability and reliability add-ons** — two new built-in add-ons quantifying how much a measure varies from trial to trial, and what that means for study design. **variability** (first level) reports the coefficient of variation four ways with confidence intervals, robust and log-scale z scores, the precision of a condition mean and how many trials would tighten it, autoregressive structure and within-session drift, single-trial limits of agreement, the typical error and RMSE family, contrasts between conditions, and correlations among trial-level measures. **variability_group** (second level) decomposes variance into between-participant, between-session, and trial-level components, then turns that into the reliability, SEM, and MDC95 of a measurement averaged over any number of trials and sessions.
* **The dispersion family, not just the CV** — MAD and IQR are reported alongside the coefficient of variation, each scaled so all three estimate the same quantity under normality and can be read against one another. Skewed amplitudes with occasional very large trials do not honour the assumptions the CV rests on, so the robust alternatives are there to be compared rather than assumed away.
* **Raw or log scale, measured rather than assumed** — the group add-on regresses log(SD) on log(mean) across recordings. A slope near zero means additive noise and the SD is the meaningful summary; a slope near one means noise scales with amplitude, so the CV is appropriate and the log scale is the natural one to analyse on. The verdict is read off the confidence interval, and reports honestly when the dataset cannot distinguish the two.
* **Does the answer depend on the outliers, or on one trial?** — every dispersion metric is reported with and without robust-z outliers, side by side with the percentage change, and a leave-one-out jackknife shows how far each single trial moves the estimate. A CV that collapses when one trial is dropped is describing that trial as much as the series, which a summary statistic cannot reveal on its own.
* **Do the metrics agree?** — the group add-on ranks recordings by each dispersion metric and reports whether the ranking survives the choice. If it does, the choice is a convention; if it does not, it is a finding that needs justifying.
* **Reliability separated from change** — the group add-on reads your Second Level design and classifies each factor by whether it varies within a participant. Between-participant factors (group, arm) can be split on, so between-session reliability survives inside each stratum; within-participant factors (timepoint) label the session axis instead, because splitting on one would leave a single session per participant. A session pair straddling an intervention is reported as measuring **change**, not test-retest reliability, since its limits reflect measurement error plus whatever really changed.
* **Figure captions** — figures now save a plain-language caption beside them, filled with that recording's own numbers, explaining what each panel shows and flagging traps only when the data triggers them (a drift that inflates the CV, or a typical error inflated by trial-to-trial alternation).
* **An unusable MNE-Python install is now treated as absent.** MNE loads its
  submodules lazily, so importing it can succeed while its readers cannot load
  at all, most often a version mismatch against SciPy. Such an install was
  previously reported as available and then failed at the moment a file was
  opened. It is now detected up front, so the formats it would have handled are
  simply not claimed and every native reader is unaffected.
* **Dropdown settings for add-ons** — `ADDON_SETTINGS` gained `choices` and `choices_from`. A measure to analyse is chosen from a list read from the columns your dataset actually contains rather than typed by hand, and settings declared `"type": "bool"` render as a checkbox instead of a text box expecting the word `True`.

## What's New in 1.3

* **Temporal MEP decomposition add-on** — splits each MEP into successive bins from onset (2 ms by default) and aggregates them into an **early** and a **late** phase, following the approach used to dissociate fast-conducting corticospinal from slower, polysynaptic cortico-reticulospinal transmission. The analysis window is clamped to the detected MEP offset so the late phase cannot run into the silent period, background EMG is subtracted over each window, and per-condition diagnostic figures show the onset-aligned mean trace and bin profile. Bin width, window length, and the early/late boundary are all settings — the bin profile is the primary output and the boundary is a derived convenience, not a baked-in assumption.
* **Add-on results reach the group table** — Second Level ▸ Group Analysis now joins per-trial add-on outputs onto the merged table automatically. Any add-on writing `File`, `StimType`, and `Segment` alongside its measurements is picked up with no configuration. The join is additive: core measurements are never overwritten, and a column whose name clashes with an existing one is namespaced after its add-on (for example `mepfeatx_Latency(ms)`) rather than dropped. A sidecar whose keys match no trials is skipped with a note instead of appending a block of empty columns.
* **MEPFeatX outputs are now joinable** — the add-on emits a 1-based `Segment` key matching the core per-trial table, and reports the source file name in `File` rather than the BIDS prefix. The 0-based `Trial` index is retained.

## What's New in 1.2

* **Extensible add-ons framework** — drop-in Python modules that run post-hoc on saved results and write their own new files, at two scopes: **single-file** (first level) and **group-level** (second level). Ships with a faithful port of **MEPFeatX** (morphological MEP features), a rectified-area example, and a group-summary example.
* **Second Level add-ons tab** — group-level add-ons operate on the merged group table.
* **Reorganised interface** — a clearer two-level tab structure (Setup / First Level / Second Level) with a persistent active-file header and step-by-step first-level sub-tabs (1a–1d + Add-ons).
* **Check for updates** — Settings → Check for updates queries GitHub Releases and offers an assisted update (pip upgrade, or a link to the download page for compiled builds).
* **BIDS-ify** — convert non-BIDS recordings into a BIDS-compliant `rawdata/` layout with shared, per-file editable stimulation metadata (NIBS BEP037).
* **Broader format support** — added BIOPAC AcqKnowledge (`.acq` and `.mat`), Brainsight neuronavigation exports, BrainVision, and LabChart MATLAB exports.
* **Cross-platform polish** — readable coloured action buttons on macOS, Windows, and Linux, and consistent font scaling across the interface.

**Point releases in the 1.2 series:** EDF/BDF files (including BIDS-ify output) now load correctly, plus release-pipeline and repository cleanup.

---

## The Interface

```
Setup                    First Level: Single File        Second Level: Group
├── Dataset              ├── 1a  Labels & Analysis Setup   ├── Group Analysis (LME)
├── Conditions           ├── 1b  Data Filtering            └── Add-ons
└── BIDS-ify             ├── 1c  Feature Detection Setup
                         ├── 1d  Normalisation (optional)
                         └── Add-ons
                         [ Save · Load · Preview · Run ]
```

The active file, channel, and event marker are shown in a persistent header above
the First-Level sub-tabs, and Save Session, Load Session, Preview detection and
Run Analysis in a footer below them, so both stay reachable as you move between
steps. Run is disabled until the detection settings have been seen for the
recording.

Opening a file lands on **Setup ▸ Conditions**: what a stimulus type is *for* is
decided before how its response is detected.

---

## Features at a Glance

### Data Ingestion and Format Support

The tool auto-detects the file format on open (by extension, binary signature, or header sniff) and dispatches to the correct reader. Supported formats:

#### Spike-2 native (`.smr`) — *requires `neo`*

Native Spike-2 binary files read directly via the [Neo](https://neo.readthedocs.io) library — no text-export step. On first open a dialog identifies the EMG channel and stim/trigger channel; the choice is saved to a sidecar (`.smr_config.json`) and not asked again. DigMark marker codes (A, B, C, …) are decoded from the event channel and each appears as a separate stimulus type. Neo is installed automatically with `pip install mep-cmap-analyser`.

#### Spike-2 text export (`.txt`)

Exported via **File → Export → Text**. Waveform channels are read along with DigMark event timestamps; any number of stimulus types and marker codes are supported. I/O is accelerated by the compiled Rust extension (`mep_cmap_io`).

#### LabChart text export (`.txt`)

Auto-detected from the `Interval=` header. Each recording block is treated as a pre-aligned trial; no trigger channel is required. Rust-accelerated.

A block export is a **pre-epoched** recording: each block is a trial already cut about the stimulus, its time column running from a negative value to a positive one and restarting for the next. The stored extent is reported, so analysis and viewing windows are clamped to what a block actually contains rather than running past its end into the padding between blocks and then into the following trial.

#### LabChart MATLAB export (`.mat`)

LabChart's MATLAB export is detected by its signature variables and read natively — no LabChart installation required.

#### ADInstruments CFWB binary (`.adibin`)

Exported from LabChart via **File → Export → ADInstruments Binary**. The CFWB format is parsed natively in Rust. Stimulation times are derived from a trigger/TTL channel auto-detected by name (`stim`, `trig`, `ttl`). Rust-accelerated.

#### BIOPAC AcqKnowledge (`.acq`)

Native BIOPAC AcqKnowledge acquisition files, read via `bioread`. Channels and event markers are decoded directly from the file.

#### BIOPAC AcqKnowledge MATLAB export (`.mat`)

AcqKnowledge's MATLAB export, detected by its signature variables and read without a BIOPAC installation.

#### Pre-epoched MATLAB trial stack (`.mat`)

Recognised by its per-trial structure rather than a continuous time series:
trials are already cut around the stimulus, so no trigger channel is needed and
the stimulus sits at a fixed sample index within every epoch. Because the file
contains data only inside its own epochs, the pre- and post-stimulus analysis
windows are clamped to what actually exists and the reduction is reported in the
log; a window that cannot support cortical silent period detection is flagged
rather than silently producing unreliable values. Where the export does not
declare an amplitude unit, it is confirmed once on first open and saved to a
sidecar.

#### Brainsight neuronavigation export (`.txt`)

Brainsight TMS neuronavigation session exports are recognised by their header signature, with stimulation events taken from the navigation record.

#### BrainVision (`.vhdr` / `.vmrk` / `.eeg`)

The BrainVision Core Data Format, resolved via the sibling `.vhdr` header and `.vmrk` markers.

#### KinEMG / NI-DAQ CSV

NI-DAQ / KinEMG CSV exports; sampling rate and channel names (`Dev1/ai0`, …) are read from the file header.

#### Generic Format Wizard (`.txt`, `.csv`)

For any other tabular text file (tab, space, or comma delimited). A one-time, four-step wizard configures **column-wise** layouts (rows = samples, columns = channels) or **row-wise** layouts (rows = channels, e.g. Delsys Trigno with one continuous TTL row and one EMG row), auto-detecting header lines, channel-name rows, and embedded sampling-rate metadata. Per-channel roles (EMG / Stim-Trigger / Ignore) are assigned once and saved to a sidecar; subsequent opens load instantly.

#### Format detection summary

|Input|Format|Stim time source|Rust accelerated|
|-|-|-|-|
|`.smr`|Spike-2 native (Neo)|DigMark / event channel|No (Neo)|
|`.txt`|Spike-2 text export|DigMark timestamps|Yes|
|`.txt`|LabChart text export|Interval resets|Yes|
|`.mat`|LabChart MATLAB export|Comments / event channel|No|
|`.adibin`|ADInstruments CFWB binary|TTL channel (auto-detected)|Yes|
|`.acq`|BIOPAC AcqKnowledge|Event markers|No|
|`.mat`|BIOPAC AcqKnowledge (MATLAB)|Event markers|No|
|`.mat`|Pre-epoched MATLAB trial stack|Fixed index within each epoch|No|
|`.txt`|Brainsight neuronavigation|Navigation events|No|
|`.vhdr`|BrainVision|`.vmrk` markers|No|
|`.csv` / `.txt`|KinEMG / NI-DAQ CSV|Trigger channel (optional)|No|
|`.txt` / `.csv`|Generic TSV (wizard)|Trigger channel / TTL row|Yes|

New formats are easy to add: a single `formats/<name>.py` module with three public functions, plus two lines in `io.py`.

### Signal Processing

* Bandpass filter (default 20–450 Hz, adjustable) using Butterworth or Chebyshev Type I designs, with independent highpass and lowpass orders
* Notch filter at any frequency (e.g. 50 or 60 Hz) with adjustable Q factor
* Humbug-style mains-noise canceller with configurable harmonic count
* Real-time filter preview showing frequency response and a wavelet time-frequency decomposition of the raw signal
* A **Confirm filter settings** step advances the guided first-level workflow from filtering to feature detection

### Trial Segmentation

* Pre- and post-stimulus windows set **per stimulus type** on tab 1a, or per condition in the Conditions tab; a type left alone uses the file-wide default
* **Conditions**: trials assigned to named groups when the recording cannot distinguish them — two timepoints, three intensities, a block design — written to a BIDS `_events.tsv` and analysed as separate groups while `StimType` and `Condition` stay separate columns in the trial file
* Windows are clamped to what a pre-epoched recording contains, and the shortening is reported rather than applied silently
* Per stimulus-type gap parameter to skip the TMS artefact period before onset search
* Multi-stimulus support within a single recording: every marker/event label gets its own settings, colour, and output columns
* Stim times sourced from DigMark timestamps (Spike-2), interval resets (LabChart), TTL/trigger rising edges (CFWB, generic TTL rows), event markers (BIOPAC, BrainVision, Brainsight), or manual entry

### Response Quantification

|Measure|Description|
|-|-|
|**PTP amplitude (mV)**|Peak-to-peak amplitude within the user-specified MEP/CMAP window|
|**MEP RMS (mV)**|Root-mean-square amplitude over the same window as PTP — integrates the whole response rather than two extreme samples|
|**Onset latency (ms)**|MEP onset relative to stimulus (see onset detection methods)|
|**AUC (mV·s)**|Area under the rectified EMG from onset to the end of the response — the cSP start where one is detected, otherwise the detected offset, or a user-defined window via drag selector|
|**MEP offset (ms)**|End of the evoked response relative to stimulus (see [MEP Offset and Duration](#mep-offset-and-duration))|
|**MEP duration (ms)**|Offset minus onset|
|**Onset disagreement (ms)**|Spread between onset detection methods on that trial — a triage signal for manual review|
|**cSP duration (ms)**|Cortical silent period, from EMG suppression onset to EMG return|
|**cSP MEP offset (ms)**|Time from stimulus to start of cSP|
|**cSP EMG return (ms)**|Time from stimulus to EMG recovery after cSP|
|**cSP/MEP ratio (ms/mV)**|cSP duration divided by MEP PTP amplitude (Orth & Rothwell, 2004 [5])|
|**Normalised PTP**|PTP as a fraction of an Mmax or single-pulse reference mean|
|**Paired-pulse ratio**|Conditioned / reference amplitude for SICI, ICF, or any custom pairing|
|**Z-score (within / pooled)**|Standardised amplitude within each stimulus type, and pooled across conditions|
|**Detrended PTP — within condition (mV)**|Linearly detrended amplitude within each stimulus type, removing condition-specific drift|
|**Detrended PTP — session (mV)**|Linearly detrended using a single trend across all trials in chronological order (captures fatigue/potentiation)|
|**Overall trial number**|Chronological trial index across all stimulus types, by stimulus timestamp order|
|**Stimulus time (s)**|Absolute timestamp of each stimulus (seconds from recording start)|
|**Inter-stimulus interval (s)**|Time since the immediately preceding stimulus (any type) — a useful covariate for variable ISIs|
|**Trial-to-trial variability**|Coefficient of variation, typical error, limits of agreement, drift and serial dependence per condition — via the `variability` add-on|
|**Reliability (ICC, SEM, MDC95)**|Variance components and the reliability of an averaged measurement across participants and sessions — via the `variability_group` add-on|

### MEP Onset Detection

Eight detection methods are available. The global default is set in **Settings →
Preferences → Detection** and can be overridden per file in 1a without affecting
the preference. All methods share the same physiological latency bounds (see
[Physiological Latency Profiles](#physiological-latency-profiles)) and return
`None` rather than a floor value when no confident onset is found, so ambiguous
trials are flagged rather than silently mislabelled.

The **latency profile governs onset detection**, not the amplitude window. The
search window is derived from each stimulus type's profile, widened by the
amplitude window, and floored at the artefact blanking period. When most onsets
in a condition land on a search bound the log says so — that pattern means the
profile is wrong for the muscle, and the resulting latencies are a window edge
rather than a measurement.

**Derivative-based — Bigoni et al. 2022 (default)** — onset is the start of the
longest sustained positive-derivative run on the rising edge, with optional
Savitzky–Golay pre-smoothing. It assumes nothing about background EMG level, so
it suits both resting and active-contraction paradigms and biphasic waveforms of
either polarity. Follows Bigoni et al. [6], adapted for variable sampling rates
and muscle-group windows.

**Derivative-based + walkback** — as above, then walks the onset back to the
point of departure from baseline. Use when the plain method lands mid-rise.

**RMS envelope + SD threshold** — a moving-window RMS envelope against a
threshold of *baseline mean + k × SD*, with the minimum duration calibrated
against the chance distribution of above-threshold run lengths rather than fixed
by hand. The envelope crossing is treated as a coarse anchor and the onset is
re-detected on a much shorter window, so the result is largely insensitive to the
smoothing width — the usual criticism of this method class. Most precise on a
quiet baseline; like every threshold method it degrades when background EMG is
high.

**CUSUM change-point** — accumulates the running excess over the baseline mean
and reports the point at which the mean *changed*, rather than the point at which
the signal crossed a level. The change point is estimated by backtracking to
where the accumulator was last zero, so it does not inherit the delay between the
true change and the moment enough evidence had accrued. Tolerant of raised
background EMG.

**Teager–Kaiser preconditioning** *(option on the envelope and CUSUM methods)* —
applies the Teager–Kaiser energy operator before detection, amplifying components
that are both large and fast-changing. Sharpens the contrast of the transition
rather than the amplitude, which helps most at low signal-to-noise ratio.

**Derivative ratio — Boyles et al. 2026** — scores each candidate sample by the
ratio of the mean absolute derivative after it to the mean absolute derivative
before it, working back from the first peak, and takes the earliest sample still
reaching a set fraction of the maximum ratio. Three gates follow: a latency
ceiling, a requirement that most of the first few forward derivatives exceed the
baseline mean derivative, and a requirement that the following window be clearly
steeper than baseline. Requires a condition average, which the analysis supplies.
Independent of the other methods in what it measures, which is its value in a
consensus. Note that its gates are stated in absolute derivatives, so heavily
low-pass filtered data will defeat it, and that the search is bounded by the
first peak so the onset can never precede it. Three details of the published
implementation are corrected by default; see
[What's New in 1.4.0](#whats-new-in-140) and the *Reproduce the published
implementation literally* option.

**Median across methods** — runs several detectors and reports the median of
those that find an onset. The median is not a verdict on which method is right;
it is the middle value, chosen because it resists one stray member. Its main
value is that the spread between members is reported as
`Onset_Disagreement(ms)`, which flags the trials worth reviewing by hand.
Members are chosen in Preferences.

**Peak-fraction** — finds the largest positive and negative peaks, then scans
back from the dominant peak to where the signal first crosses a fraction of it
(default 15%), with a minimum-amplitude guard. Best on clean, high-amplitude
responses with a near-silent baseline.

**Bootstrap threshold** *(legacy)* — retained so analyses run on v1.3.x and
earlier reproduce exactly. Its threshold is clipped to a multiple of the baseline
mean, which overrides the SD scaling and places onsets systematically early;
prefer the RMS envelope method for new work.

#### Amplitude and duration guards

The envelope and CUSUM detectors apply a width guard: a candidate onset is
rejected unless the response stays elevated for a minimum duration. Smoothing
widens a single-sample artefact — a cable transient, an electrical spike — into
an excursion as wide as the smoothing window, which then satisfies any shorter
run-length criterion. Amplitude- and energy-based detectors need this guard;
derivative-based ones largely do not.

### MEP Offset and Duration

`MEP_Offset(ms)` marks the end of the evoked response and `MEP_Duration(ms)` is
the interval from onset to offset. One precedence rule is applied, and
`MEP_Offset_Source` records which branch fired, so no value's provenance has to
be inferred:

|Condition|`MEP_Offset(ms)`|`MEP_Offset_Source`|
|-|-|-|
|Manual marker set in the Inspector|the manual value|`manual`|
|cSP enabled for this stimulus type and detected|cSP start|`csp_start`|
|Otherwise|envelope return to baseline|`envelope`|
|No onset, or no confident return found|blank|`none`|

During voluntary contraction the end of the MEP and the start of the silent
period are the same physical event, so they are reported as the same number
rather than as two near-duplicate estimates. `cSP_MEP_Offset(ms)` is retained
unchanged for backward compatibility; prefer `MEP_Offset(ms)` in new analyses,
since it is also populated at rest.

The return threshold is the larger of a baseline-derived level and a fraction of
the response's own peak envelope (default 12%). A purely baseline-derived
threshold is an absolute level, and on a quiet resting recording it is a very low
one: real EMG does not settle back to resting-quiet within tens of milliseconds
after a large response, so such a threshold fails *worse* the cleaner the trial.
Scaling the criterion to the response removes that.

Detecting the offset also gives **AUC an endpoint in resting-state recordings**.
Where there is no silent period to close the integration window, AUC now runs
from onset to the detected offset instead of requiring the endpoint to be dragged
by hand.

### PTP Window Anchoring

*Off by default.* The amplitude window in 1c is one setting for the whole file,
but each stimulus type has its own latency profile. A recording containing both
M-waves and MEPs cannot be measured correctly by a single window: with a 10 ms
start, an M-wave beginning at 4 ms has most of its response excluded from the
amplitude measurement, and an M-wave's entire biphasic deflection lasts only
5–15 ms.

With anchoring enabled, each stimulus type gets an amplitude window placed on its
own **median detected onset** — not per trial, which would make amplitude a
function of onset-detection error and leave trials without an onset with no
amplitude at all. The condition median keeps the window identical for every trial
in the condition, so within-condition amplitudes stay comparable. The 1c window
end still applies as a ceiling, and stimulus types with too few detected onsets
fall back to the file-wide window. The window chosen for each condition is
printed to the log rather than changing silently.

### Onset Method Comparison

With **Compare methods on every trial** enabled, every member method runs on
every trial regardless of which method is selected — so a method choice can be
justified while still running the one you trust. Beyond the per-trial agreement
columns, this writes:

- `<prefix>_onset_methods.csv` — long format, one row per trial × method,
  including rows where a method failed, so detection rate stays recoverable
- `<prefix>_onset_method_summary.csv` — per stimulus type × method: detection
  rate, latency statistics, bias and 95% limits of agreement
- `figures/<prefix>_onset_methods_figures/` — per condition, where each method's
  onset lands on the actual waveform with per-trial onsets beneath; latency by
  method across the file; Bland–Altman of each method against the others; and the
  distribution of disagreement

Bland–Altman uses a **leave-one-out median**: the method under test is
excluded from the median it is compared against. Comparing a method with a
composite that contains it is a part-whole comparison, which drags the bias
toward zero and narrows the limits, flattering every method. On real data this
understated the limits of agreement by around a quarter.

> Agreement is not accuracy. Detectors that share an assumption can be wrong
> together, and the two derivative variants are not independent — the walkback
> starts from the plain method's answer. Read the spread as the practical
> consequence of a method choice; establishing which method is *correct* requires
> ground truth, not agreement.

### Cortical Silent Period (cSP) Detection

A bootstrap method on the RMS envelope. A suppression threshold is estimated from the pre-stimulus baseline, and the search runs from each trial's own second peak-to-peak landmark — the end of that trial's response — to a configurable end. Onset is the first sustained suppression below threshold; offset is the first sustained *return* of EMG, so breakthrough activity part way through a silent period does not end the measurement. Configurable criteria include minimum silence duration (default 25 ms), minimum EMG-return duration (default 40 ms, and never shorter than the RMS window), bootstrap criterion (default 1.96 SD), significance level (default 99th percentile), search-window end, and the maximum distance between the response and the *start* of the silent period.

The analysis measures the silent period on every trial of an enabled stimulus type; the Data Inspector is for reviewing and overriding it, not for producing it. cSP detection can be enabled or disabled per stimulus type and overridden per trial. Where EMG has not returned before the search window ends, the duration is reported as a lower bound rather than as a measurement. The 1c Feature Detection Setup tab exposes all of these with inline guidance.

### M-wave Normalisation and Mmax

A separate Mmax file can be designated containing M-waves across a range of intensities. The plateau region is detected robustly for three scenarios: a full recruitment curve (averages the plateau within a tolerance band, default ±10%), a few supramaximal pulses (averages the largest similar-amplitude cluster), or a single M-wave (used directly). Normalised PTP is then reported for all MEP trials as a fraction of Mmax.

For designs where the background excitability of spinal motoneurones varies across trials, for example active-contraction paradigms, the tool also compensates evoked-potential magnitude for pre-stimulus excitability by quantile regression, following the method of Carson (2026) [9], as an alternative or complement to Mmax normalisation.

### EMG Excitability Compensation (Carson 2026)

MEP amplitude covaries positively with the level of background EMG in the period immediately preceding the pulse, over a range far below any conventional rejection threshold. Rather than discarding trials, the amplitude is regressed on pre-stimulus r.m.s. EMG by median quantile regression within each sample (one participant, one stimulus type, one intensity, one block), and each trial's residual is re-expressed relative to an uncertainty-weighted reference value. The reference blends the regression intercept with the median of the fitted values, weighted by the relative standard error of the fitted ordinate at each point, so where no association is present the adjustment vanishes.

The implementation is verified against the author's own reference code and example data (`annotated_QR_example_code.R`, Zenodo 20037178); `tests/test_carson_compensation.py` locks the slope, intercept, intercept weighting and reference value to his published values.

Reported per trial:

|Column|Description|
|-|-|
|`Adjusted_PTP_QR(mV)`|Excitability-compensated PTP|
|`Normalised_Adjusted_PTP_QR`|Adjusted PTP as a fraction of the reference mean|
|`EMGComp_Slope`, `EMGComp_Intercept`|Fitted relationship, in PTP units per PreStimRMS unit|
|`EMGComp_InterceptWeight`|Wi, the weight given to the intercept in the reference value|
|`EMGComp_Adjustment(mV)`|reference minus median(fitted): the shift applied to the sample|
|`EMGComp_N`, `EMGComp_Method`|Trials in the fit, and the backend or fallback reason|
|`EMGComp_PseudoR2`|Koenker-Machado pseudo-R-squared|
|`EMGComp_Rho_Pre`, `EMGComp_Rho_Post`|Spearman rho with pre-stimulus RMS before and after adjustment; the second should be near zero|

Two points worth knowing when reading the output. First, adjusted amplitudes **larger** than unadjusted are expected, not a fault: within any sample the low-RMS trials always shift upward, and a whole sample shifts upward whenever `EMGComp_Slope` is negative (74 of Carson's 182 participants). Second, the per-trial shift is exactly `reference - (intercept + slope * PreStimRMS)`, so plotting `Adjusted_PTP_QR(mV) - PTP(mV)` against `PreStimRMS` must give a straight line with slope `-EMGComp_Slope`. That is the quickest check that a suspicious file is behaving correctly. Before comparing adjusted values across conditions, check that the slopes and intercepts are comparable, as the paper recommends.

Background EMG is quantified as the r.m.s. of a `prestim_ms` window (default 100 ms) ending `rms_guard_ms` before the pulse (default 3 ms, matching the paper, and widened automatically if a stimulus type needs a longer artefact gap). The window's DC offset is removed first, since an offset is not motoneurone activity and carrying it into the r.m.s. adds between-trial variance that masks the association the method exists to remove.

Compensation is skipped for M-wave runs, which are direct muscle responses rather than spinally mediated, and typically span multiple intensities. Trials marked Removed or Excluded are left out of the fit and receive no compensation values; trials flagged by the z-screen but kept by the reviewer stay in, since retaining datapoints is one of the stated benefits of the method.

### Paired-Pulse Protocols

Any stimulus type can be designated as a conditioned stimulus and paired with a reference in 1a. Conditioned/reference ratios (e.g. SICI at 2–6 ms ISI, ICF at 10–15 ms ISI) are produced as a standard output column, with multiple reference assignments supported within a single file.

### Outlier Detection and Review

* Z-score flagging on PTP amplitude and RMS with a configurable threshold (default ±1.96)
* Interactive review dialog showing the flagged waveform in context, with include / exclude / note options
* Decisions persist across reruns — reviewed trials are not re-presented — and are recorded in the trial CSV's `Outlier_Decision` column

### Data Inspector

Per-trial interactive review with a zoomed trial view plus a wider context window; draggable onset, cSP-start, and cSP-end markers; a drag-to-select AUC window; per-trial notes; and keyboard navigation.

Optional **Rectified** and **RMS envelope** overlays (`R` and `E`) draw the signal the detector actually thresholded, with its suppression threshold, the baseline it was derived from, and the percentage between them — so a marker can be judged against what produced it rather than against the raw trace. They are display only and cannot change a measurement: which trace a marker is measured on is a property of the marker, not of what is visible.

All edits are saved to the session JSON and applied on every subsequent run without re-review. Each landmark records whether it was detected or placed by hand, and each segment records which detector version produced its automatic markers; when the detector changes, automatic markers are re-detected and manual edits are kept.

### Add-ons (Extensible Analyses)

Add-ons are optional, drop-in Python modules that run **after** processing, read the saved results, and write **their own new files** — they never modify core outputs. They come in two scopes:

* **First-level (single-file) add-ons** run on each recording's saved waveform bundle (`<prefix>_segments.npz`) and appear on the **First Level → Add-ons** tab.
* **Second-level (group-level) add-ons** run on the merged group table (`group_level_LME_ready.csv`) and appear on the **Second Level → Add-ons** tab.

Built-in add-ons:

|Add-on|Scope|What it does|
|-|-|-|
|**mepfeatx**|single-file|A faithful port of MEPFeatX (Nguyen et al. 2025 [10]): morphological MEP features — amplitude, latency, AUC, waveform thickness, number of turns and phases, duration, and the two dominant peaks (T1/T2) — with per-trial and per-condition diagnostic figures and a transparent rejection reason for every trial it can't quantify|
|**rectified_area**|single-file|Rectified area under each MEP over the analysis window (a minimal example)|
|**temporal_decomposition**|single-file|Splits each MEP into successive time bins from onset and aggregates them into early and late phases, with baseline EMG correction and per-condition diagnostic figures|
|**variability**|single-file|Trial-to-trial variability per stimulus type: coefficient of variation with confidence intervals, robust and log-scale z scores, precision of the condition mean, autoregressive structure and drift, single-trial limits of agreement, typical error and the RMSE family, how many trials an average needs, contrasts between conditions, and correlations among trial-level measures. Emits a per-trial sidecar that joins into the group table|
|**variability_group**|group-level|Dataset-level reliability: variance components (between participant, between session, trial to trial), the reliability / SEM / MDC95 of a measurement averaged over k trials and m sessions, ICC(1,1), ICC(2,1), ICC(3,1) and their k-forms, and session-to-session agreement labelled as reliability or as change. Optionally split by between-participant design factors|
|**group_summary**|group-level|Per-condition mean, SD, and N of every metric across the group (a minimal example)|

Add-ons can declare their own settings, which render as controls in the add-on's box — a checkbox, a numeric field, or a dropdown, according to what the add-on declares (for example, MEPFeatX exposes a tunable noise-gate ratio for handling MEPs recorded during voluntary contraction, and the variability add-ons offer a measure picked from the columns your dataset actually contains). Point the tool at your own add-ons folder in **Settings → Preferences → Add-ons**; place first-level add-ons in a `single_file/` subfolder and group-level add-ons in a `group_level/` subfolder. See [Writing Add-ons](#writing-add-ons).

### BIDS-ify

The **Setup → BIDS-ify** tab converts non-BIDS recordings into a BIDS-compliant `rawdata/` layout, and describes the stimulation that produced them following the NIBS-BIDS proposal (BEP037, v6.3).

Stimulation is described **per protocol, not per file**. You define a stimulation parameter set once per session — its type (TMS, tES, TUS or PNS), intensity, dosing reference and scaling, and optionally its own stimulator, coil or measured threshold where that protocol differs — and then say which stim code in each recording used it. A recording containing a peripheral M-wave on one code and a TMS MEP on another is therefore described properly, with two rows and two devices, rather than being forced onto a single intensity and a single modality.

Dosing keeps its derivation: 120% of a resting motor threshold of 50 %MSO is recorded as an intensity of 60 in %MSO, a reference of `rMT` and a scaling of 1.2, with the measured threshold stated once. The delivered dose and how it was arrived at both survive, which is what makes intensities comparable across sites.

Conversion writes the four-file v6.3 structure alongside the recording: `*_nibs.tsv` with one row per parameter set, `*_nibs.json` holding the devices, dosing references and column definitions with their units, `*_markers.tsv` with one row per placement, and an `*_events.tsv` whose every delivery names the protocol it was and where it was applied.

Conditions assigned in the **Conditions** tab are carried through. Where a condition changes the stimulus rather than only the meaning of a trial — half of a code at 100 mA and half at 150 mA, or a recruitment curve — each part takes its own parameter set and becomes its own row. Files are reviewed, accepted and converted from a persistent, status-coloured worklist, and a converted file can be re-opened and rewritten if something needs correcting.

### Session Persistence and Reproducibility

Every setting the user touches — filter parameters, time windows, onset method, latency maps, cSP thresholds, normalisation references, Inspector edits, outlier decisions, analysis options — is saved in a per-file session JSON alongside the derivatives. Reloading restores the exact state; changing a setting and re-running produces a clean new result without losing manual review work. File paths in session JSONs are stored relative to the study root for portability across machines and cloud sync.

### Dataset Queue

* Open a study folder or individual files; the tool auto-detects BIDS `rawdata/` and `derivatives/` subfolders
* A persistent queue tracks status (Not Started, In Progress, Needs Review, Complete, Stale)
* Excluded files are remembered and not re-added on refresh; a right-click menu restores them, marks files for reprocessing, or opens the derivatives folder
* Process a highlighted recording with **Run selected**
* Queue state is saved to `dataset_session.json`

### Check for Updates

**Settings → Check for updates** queries GitHub Releases in the background, compares the latest version to the one you're running, and — if you're behind — shows the release notes and offers an assisted update: a `pip install --upgrade` for source/pip installs, or a link to the download page for compiled builds. It fails gracefully offline and falls back to version tags if no formal release is published.

---

## Supported Use Cases

The tool handles any paradigm where a time-locked EMG response is expected within a defined post-stimulus window, including but not limited to:

* **TMS MEP studies** — single-pulse, paired-pulse (SICI, ICF, LICI, SAI), or multi-intensity recruitment curves; any accessible muscle
* **Peripheral nerve stimulation CMAPs** — M-wave recruitment curves for Mmax determination or peripheral conduction
* **Corticospinal excitability assays** — resting and active MEP series, pre/post intervention, crossover and parallel designs
* **TMS-EMG silent period studies** — cSP duration, cSP/MEP ratio, and derived inhibitory indices
* **MEP waveform morphology** — via the MEPFeatX add-on (turns, phases, thickness, T1/T2)
* **Re-analysis of pre-epoched datasets** — trial stacks exported from another pipeline or downloaded from a public repository, where no continuous recording is available
* **Voluntary EMG bursts** — files with no stimulation events can be loaded for waveform inspection, RMS quantification, and trial-level output

---

## Installation

### Option 1: pip (recommended for most users)

```bash
pip install mep-cmap-analyser
mep-cmap
```

Python 3.9 or later is required. Tkinter must be available:

* **Windows / macOS** — included with standard Python installers
* **Linux (Ubuntu / Debian)** — `sudo apt install python3-tk`

### Option 2: Compiled binaries (no Python required)

Pre-built builds for each platform are on the [Releases page](https://github.com/jandrushko/mep-cmap-analyser/releases). Download, unzip, and run.

|Platform|File|
|-|-|
|Windows|`MEP-CMAP_Analyser_Windows.zip`|
|macOS (Apple Silicon)|`MEP-CMAP_Analyser_Mac_apple-silicon.zip`|
|macOS (Intel)|`MEP-CMAP_Analyser_Mac_intel.zip`|
|Linux|`MEP-CMAP_Analyser_Linux.tar.gz`|

Apple Silicon Macs are M1 and later; choose the Intel build for anything older.
If you pick the wrong one macOS reports a "bad CPU type in executable" error —
download the other file rather than anything else being wrong.

#### First launch on macOS

The macOS app is **not notarized by Apple**, so the first time you open it
macOS will refuse and may claim the app is *damaged*. Nothing is damaged: this
is Gatekeeper blocking any app downloaded from the internet that has not been
through Apple's paid notarization service. Two ways past it, both one-time:

* **Right-click** (or Control-click) the app, choose **Open**, then **Open**
  again in the dialog. This records your consent and normal double-clicking
  works from then on.
* Or clear the download quarantine flag from Terminal:

  ```bash
  xattr -dr com.apple.quarantine "MEP-CMAP Analyser.app"
  ```

If neither works, macOS may have quarantined the zip's contents on extraction —
move the app to `/Applications` first, then repeat.

Running from source (Option 3) or installing from PyPI (Option 1) avoids
Gatekeeper entirely and is the simplest route on macOS.

### Option 3: Run from source

```bash
git clone https://github.com/jandrushko/mep-cmap-analyser.git
cd mep-cmap-analyser
pip install -r requirements.txt
python -m mep_cmap
```

---

## Workflow

### 1. Setup → Dataset

Open a study folder or an individual recording. The tool auto-detects a BIDS layout (`rawdata/` beside `derivatives/`) or sets up a derivatives folder in the standard location. Files appear in the queue with their status; double-click any file to load it, or highlight one and click **Run selected**. Opening an unrecognised format launches the Format Wizard for a one-time configuration.

### 2. Setup → BIDS-ify *(optional)*

If your data isn't yet in BIDS, use BIDS-ify to set shared stimulation metadata and convert the ready files into a `rawdata/` tree.

### 3. First Level → 1a Labels & Analysis Setup

For each stimulus type in the recording, configure the display label and colour, the artefact gap (ms), whether to run cSP detection, the stimulus category and target muscle (which set physiological latency bounds), and any normalisation/paired-pulse reference pairing. Click **✔ Confirm Setup** when ready. Settings carry over between files.

### 4. First Level → 1b Data Filtering

Set bandpass, notch, and Humbug options, preview the filter, then click **✔ Confirm filter settings → Feature Detection**.

### 5. First Level → 1c Feature Detection Setup

Set time windows, onset-detection parameters, cSP settings, outlier detection, and analysis options, then click **▶ Run Analysis**. The tool extracts trials, quantifies all measures, flags outliers, optionally runs the Data Inspector, and writes results to the derivatives folder. Reloading a processed file offers to reuse the saved crop range, pick a new one, or use the full file, with all prior edits restored.

### 6. First Level → 1d Normalisation *(optional)* and Add-ons *(optional)*

Normalise processed results against a reference file, and/or run first-level add-ons (e.g. MEPFeatX) on the saved bundles.

### 7. Second Level → Group Analysis (LME)

The tool scans the derivatives folder and lists completed sessions. Assign study-design columns (Group, Condition, Timepoint, or any custom factor), configure stim roles, select sessions to include, and click **▶ Build group analysis file** to produce `group_level_LME_ready.csv`.

### 8. Second Level → Add-ons *(optional)*

Run group-level add-ons (e.g. group_summary) on the merged group table.

---

## Output Files

Results are written to a `derivatives/` folder beside the raw data, following BIDS derivative conventions:

```
study/
├── rawdata/
│   └── sub-001/ses-01/sub-001_ses-01_recording.txt
└── derivatives/
    ├── dataset_session.json               ← file queue and processing status
    ├── study_design.json                  ← Second-Level design configuration
    ├── group_level_LME_ready.csv          ← merged group output (Second Level)
    ├── group_level_LME_ready_*.csv        ← group-level add-on outputs
    └── sub-001/
        └── ses-01/
            ├── sub-001_ses-01_session.json           ← full session state
            ├── results/
            │   ├── sub-001_ses-01_<StimType>_trials.csv   ← one per stim type
            │   ├── sub-001_ses-01_..._segments.npz        ← waveform bundle (add-on input)
            │   ├── sub-001_ses-01_..._variability.csv     ← per-trial add-on output (joins into the group table)
            │   ├── sub-001_ses-01_..._onset_methods.csv   ← per trial × method (agreement enabled)
            │   ├── sub-001_ses-01_..._onset_method_summary.csv
            │   └── ...                                    ← add-on outputs, e.g. *_mepfeatx.csv
            └── figures/                                   ← add-on figures, each saved with a *_caption.txt
                └── sub-001_ses-01_onset_methods_figures/  ← onset method comparison figures
```

### Trial-level CSV columns

Each `<prefix>_<StimType>_trials.csv` contains the full column set below. Which
columns are *populated* depends on what you enabled — cSP columns fill only when
cSP detection is on, normalisation columns when a reference file is set, and the
excitability-compensation block when that option is run.

|Column(s)|Description|
|-|-|
|`File`, `StimType`, `Stim_Label`|Recording identifier and stimulus-type code / display label|
|`Segment`, `Segment_Overall`|Trial index within the condition, and chronological index across all conditions|
|`Stim_Time(s)`, `Time_Since_Last_Stim(s)`|Absolute stimulus time and inter-stimulus interval|
|`Limb`|Limb identifier (from filename or entered)|
|`PTP(mV)`, `MEP_RMS(mV)`, `Latency(ms)`, `AUC(mV*s)`|Core response: peak-to-peak amplitude, RMS over the same window, onset latency (`Not Marked` when unresolved), and area under the rectified EMG. `MEP_RMS(mV)` is a window statistic, so a manual peak-marker adjustment in the Inspector changes `PTP(mV)` but not the RMS|
|`Measure`|Optional auxiliary / manual measurement (blank unless used)|
|`cSP_Duration(ms)`, `cSP_MEP_Offset(ms)`, `cSP_EMG_Return(ms)`, `cSP_MEP_Ratio(ms/mV)`|Silent-period duration (`Not Marked` when absent), stimulus→cSP-start, stimulus→EMG-return, and cSP ÷ PTP ratio (Orth & Rothwell, 2004 [5])|
|`MEP_Offset(ms)`, `MEP_Duration(ms)`, `MEP_Offset_Source`|End of the evoked response, its duration from onset, and which rule produced the offset (`manual` / `csp_start` / `envelope` / `none`). Where a silent period is detected, `MEP_Offset(ms)` and `cSP_MEP_Offset(ms)` carry the same value by design — they are the same physical event|
|`Onset_MethodsMedian(ms)`, `Onset_Disagreement(ms)`, `Onset_IQR(ms)`, `Onset_Methods_N`|Populated only when *Compare methods on every trial* is enabled: the median across onset detection methods, the max–min spread, the interquartile range (robust to one stray method), and how many methods found an onset. High disagreement flags a trial for review; it does not mean the reported onset is wrong|
|`PreStimRMS`, `PreStimPTP`, `PTP_per_PreStimRMS`, `Z_PreStimRMS`|Pre-stimulus baseline EMG: RMS, peak-to-peak, PTP-per-RMS, and standardised RMS|
|`Z_PTP_Within`, `Z_PTP_Pooled`|PTP z-scores within each condition and pooled across conditions|
|`PTP_Detrended_WithinCond(mV)` + `_Z`, `PTP_Detrended_Session(mV)` + `_Z`|Amplitude detrended within condition and across the whole session (fatigue / potentiation), each with its z-score|
|`Reference_Type`, `Reference_Mean(mV)`, `Reference_N`, `Normalised_PTP`, `Normalised_PTP_per_PreStimRMS`|Mmax / reference normalisation: reference used, its mean and N, and the normalised amplitudes|
|`Adjusted_PTP_QR(mV)`, `Normalised_Adjusted_PTP_QR`|**Excitability-compensated PTP** — adjusted for spinal motoneurone excitability by quantile regression on pre-stimulus EMG (Carson, 2026 [9]), raw and reference-normalised|
|`EMGComp_Method`, `EMGComp_N`, `EMGComp_Slope`, `EMGComp_Intercept`, `EMGComp_InterceptWeight`, `EMGComp_Adjustment(mV)`, `EMGComp_PseudoR2`, `EMGComp_Rho_Post`|Excitability-compensation fit: method / status, N, regression coefficients and intercept weighting, per-trial adjustment, and diagnostics (pseudo-R² and residual PTP–EMG correlation)|
|`Outlier_Decision`, `Manual_Note`|Review outcome (`Not flagged` or your include / exclude decision) and free-text annotation|

### Group-level LME-ready CSV

Every trial-level column from every included session, prefixed with design columns — `participant_id`, `session`, `task`, `timepoint`, `Stim_Role`, and any custom between/within-subject factors defined at the second level. Output is at the trial level (outliers retained with their Z-scores as covariates rather than pre-excluded), so the analyst keeps full control of trial-level modelling. Per-trial add-on outputs are joined on automatically (see [Writing Add-ons](#writing-add-ons)), so columns such as the variability add-on's robust z scores arrive alongside the core measurements and can be used as covariates. This file is also the input for group-level add-ons.

---

## Physiological Latency Profiles

The derivative-based (Bigoni), bootstrap, and peak-fraction onset detectors all search within a per-muscle physiological window. Defaults assume contralateral cortical stimulation with active facilitation (resting latencies are typically 1–3 ms longer). Windows can be overridden per stimulus type in 1a, and the global defaults edited in **Settings → Preferences → Latency Profiles**.

|Stimulus type / Muscle target|Window (ms)|Reference(s)|
|-|-|-|
|TMS → deltoid / trapezius|8–16|[1], [2]|
|TMS → biceps / triceps brachii|12–20|[1], [2]|
|TMS → trunk / external oblique|12–22|[3]|
|TMS → hand / FDI / APB / ADM|18–28|[4], [1]|
|TMS → forearm (FCR / ECR)|16–26|[1]|
|TMS → vastus lateralis / quad|18–30|[1], [4]|
|TMS → hamstrings|18–32|[1]|
|TMS → tibialis anterior / leg|28–45|[4], [1]|
|PNS → upper limb (M-wave)|2–12|[1]|
|PNS → lower limb (M-wave)|4–18|[1]|

**Notes.** Latency scales positively with height and age, particularly for lower-limb muscles. The lower bound excludes the TMS artefact; the upper bound captures the ±2 SD range of normative cohort data while avoiding late oligosynaptic MEPs. The trunk window is anchored to the contralateral onset latency of 15.8 ± 1.4 ms reported by Miyano et al. [3].

---

## Writing Add-ons

An add-on is a small Python module exposing an `ADDON_NAME`, an optional description/version/author, an optional `ADDON_SCOPE` (`"single_file"` or `"group_level"`), optional `ADDON_SETTINGS`, and a `run(context)` function. It reads from the context and writes **new** files into `context.results_dir`.

**First-level (`single_file`)** add-ons receive a context with the per-trial waveforms grouped by stimulus type, sampling rate, unit, a stimulus-aligned time axis, the per-trial table, the analysis config, and output paths:

```python
ADDON_NAME  = "my_addon"
ADDON_SCOPE = "single_file"

def run(context):
    import os, numpy as np
    rows = []
    for stim_type, stack in context.segments.items():   # stack: (n_trials, n_samples)
        for i, trace in enumerate(stack):
            rows.append((stim_type, i, float(np.ptp(trace))))
    out = os.path.join(context.results_dir, f"{context.bids_prefix}_my_addon.csv")
    # ... write `rows` to `out` ...
    context.log(f"my_addon → {os.path.basename(out)}")
    return [out]
```

**Second-level (`group_level`)** add-ons receive `context.group_table` (the merged group DataFrame), with `design_columns` / `metric_columns` split out for convenience:

```python
ADDON_NAME  = "my_group_addon"
ADDON_SCOPE = "group_level"

def run(context):
    import os
    summary = context.group_table.groupby("StimType")[context.metric_columns].mean()
    out = os.path.join(context.results_dir, f"{context.bids_prefix}_my_group_addon.csv")
    summary.to_csv(out)
    context.log(f"my_group_addon → {os.path.basename(out)}")
    return [out]
```

### Add-on settings

An add-on may declare `ADDON_SETTINGS`, a list of dictionaries that render as controls in the add-on's box. Each value is passed into `context.config` under its `key` when the add-on runs.

|Field|Purpose|
|-|-|
|`key`|Config key the value arrives under. Prefix it with the add-on name to avoid collisions|
|`label`|Text shown beside the control|
|`help`|Longer explanation shown under the control|
|`type`|`str`, `int`, `float`, or `bool`. A `bool` renders as a checkbox|
|`default`|Value used when the user has not touched the control|
|`min`, `max`|Advisory bounds for numeric settings|
|`choices`|Fixed list of valid options, rendered as a read-only dropdown|
|`choices_from`|`"trial_columns"` or `"group_columns"` — populate the dropdown from the columns your dataset actually contains. Stays editable, so an unlisted column can still be typed, and falls back to `choices` when no dataset is open|

```python
ADDON_SETTINGS = [
    {
        "key": "my_addon_metric",
        "label": "Measure to analyse",
        "help": "Trial-level column to quantify.",
        "type": "str",
        "default": "PTP(mV)",
        "choices": ["PTP(mV)", "AUC(mV*s)"],   # fallback list
        "choices_from": "trial_columns",        # real columns when a dataset is open
    },
    {
        "key": "my_addon_plot",
        "label": "Write figures",
        "type": "bool",
        "default": False,
    },
]
```

An unrecognised `type` falls back to a plain text box, so an add-on written for a newer version still loads on an older build.

### Computing the same measurements the pipeline does

Add-ons and external scripts should quantify amplitudes through the shared
helpers rather than reimplementing them, so their numbers agree with the trial
CSVs:

```python
from mep_cmap.detection import (compute_ptp, compute_rms, compute_auc,
                                compute_prestim_rms, compute_prestim_ptp)

ptp = compute_ptp(segment, start_idx, end_idx)      # peak-to-peak in a window
rms = compute_rms(segment, start_idx, end_idx)      # RMS over the same window
auc = compute_auc(segment, onset_idx, end_idx, fs)  # area under rectified EMG
base = compute_prestim_rms(prestim_segment)          # DC offset removed
```

`compute_prestim_rms` removes the DC offset by default, which is what the
`PreStimRMS` column contains and what the Carson (2026) compensation expects.
Pass `demean=False` only if the raw, offset-inclusive r.m.s. is specifically
what you want.

### Getting per-trial results into the group table

Second Level ▸ Group Analysis left-joins any CSV sitting beside `<prefix>_trials.csv` that carries the join keys `StimType` and `Segment` (plus `File` when present). `Segment` is 1-based, matching the core trial table, and `File` must be the source file name as it appears there rather than the BIDS prefix.

The join is additive and safe: a column whose name already exists in the core table arrives namespaced after its add-on rather than overwriting it, and the merge is validated one-to-one, so a sidecar must hold exactly one row per `(File, StimType, Segment)`. A sidecar whose keys match no trials is skipped with a note instead of appending empty columns.

Output that is not per trial — a per-condition summary, a contrast table — simply omits `Segment` and is left alone by the join. Avoid ending such a file's name with `_summary.csv`, which is reserved for core outputs.

Put your modules in the matching subfolder (`single_file/` or `group_level/`) of the add-ons folder set in **Settings → Preferences → Add-ons**. The built-in add-ons are good starting templates: `rectified_area` and `group_summary` are deliberately minimal, `mepfeatx` and `temporal_decomposition` show per-trial output that joins into the group table, and `variability` shows settings, dropdowns, figures, and captions.

---

## Building from Source

```bash
# Windows
python build_windows.py

# macOS
python3 build_mac.py

# Linux
python3 -m venv venv_linux && source venv_linux/bin/activate
pip install -r requirements.txt
python3 build_linux.py
```

The build scripts create a local virtual environment, compile the Rust I/O extension (`mep_cmap_io`) if a Rust toolchain is present, and run PyInstaller with the platform spec. The bundled add-ons and BIDS schema ship automatically.

---

## Dependencies

|Package|Purpose|
|-|-|
|`numpy`|Numerical arrays and signal operations|
|`scipy`|Filtering, statistics, interpolation, signal processing|
|`pandas`|CSV I/O and data manipulation|
|`matplotlib`|Waveform plotting, interactive figures, add-on figures|
|`statsmodels`|Regression utilities for excitability compensation (Carson, 2026 [9])|
|`PyWavelets`|Wavelet time-frequency display in filter preview|
|`Pillow`|Image handling for splash screen and icons|
|`neo`|Native Spike-2 `.smr` reading|
|`pyedflib`|EDF/BDF handling for BIDS-ify|
|`bioread`|BIOPAC AcqKnowledge `.acq` reading|
|`tkinter`|GUI (bundled with standard Python)|

The optional Rust extension `mep_cmap_io` provides accelerated I/O for the Spike-2 text, LabChart text, Generic TSV, and CFWB binary formats; all formats fall back to pure Python if it is unavailable.

---

## Citation

If you use MEP-CMAP Analyser in published research, please cite:

> Justin W. Andrushko. (2026). jandrushko/mep-cmap-analyser: MEP-CMAP Analyser (Version v1.4.5) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21810844
> https://github.com/jandrushko/mep-cmap-analyser

---

## References

[1] Groppa, S., Oliviero, A., Eisen, A., Quartarone, A., Cohen, L.G., Mall, V., Kaelin-Lang, A., Mima, T., Rossi, S., Thickbroom, G.W., Rossini, P.M., Ziemann, U., Valls-Solé, J., & Siebner, H.R. (2012). A practical guide to diagnostic transcranial magnetic stimulation: Report of an IFCN committee. *Clinical Neurophysiology*, 123(5), 858–882. https://doi.org/10.1016/j.clinph.2012.01.010

[2] Colebatch, J.G., Rothwell, J.C., Day, B.L., Thompson, P.D., & Marsden, C.D. (1990). Cortical outflow to proximal arm muscles in man. *Brain*, 113(6), 1843–1856. https://doi.org/10.1093/brain/113.6.1843

[3] Miyano, R., Shirota, Y., Kodama, S., Toda, T., & Hamada, M. (2026). Ipsilateral and contralateral cortical control of the external oblique muscles revealed by TMS. *Clinical Neurophysiology*, 181, 2111400. https://doi.org/10.1016/j.clinph.2025.2111400

[4] Cantone, M., Lanza, G., Fisicaro, F., Bella, R., Ferri, R., Pennisi, G., Waterstraat, G., & Pennisi, M. (2023). Sex-specific reference values for total, central, and peripheral latency of motor evoked potentials from a large cohort. *Frontiers in Human Neuroscience*, 17, 1152204. https://doi.org/10.3389/fnhum.2023.1152204

[5] Orth, M., & Rothwell, J.C. (2004). The cortical silent period: intrinsic variability and relation to the waveform of the transcranial magnetic stimulation pulse. *Clinical Neurophysiology*, 115(5), 1076–1082. https://doi.org/10.1016/j.clinph.2003.12.005

[6] Bigoni, C., Cadic-Melchior, A., Vassiliadis, P., Morishita, T., & Hummel, F.C. (2022). An automatized method to determine latencies of motor-evoked potentials under physiological and pathophysiological conditions. *Journal of Neural Engineering*, 19(2), 024002. https://doi.org/10.1088/1741-2552/ac636c

[7] Hupfeld, K.E., Swanson, C.W., Fling, B.W., & Seidler, R.D. (2021). TMS-induced silent periods: A review of methods and call for consistency. *Journal of Neuroscience Methods*, 346, 108950. https://doi.org/10.1016/j.jneumeth.2020.108950

[8] Rossini, P.M., et al. (2015). Non-invasive electrical and magnetic stimulation of the brain, spinal cord, roots and peripheral nerves: Basic principles and procedures for routine clinical and research application. *Clinical Neurophysiology*, 126(6), 1071–1107. https://doi.org/10.1016/j.clinph.2015.02.001

[9] Carson, R.G. (2026). A method of compensating for the excitability of spinal motoneurones when estimating the magnitude of potentials evoked in skeletal muscles. *The Journal of Physiology*, 604, 5731–5757. https://doi.org/10.1113/JP290979

[10] Nguyen, T.D., et al. (2025). MEPFeatX: feature extraction for motor evoked potentials. *Frontiers in Neuroscience*, 18, 1415257. https://doi.org/10.3389/fnins.2024.1415257

[11] Hopkins, W.G. (2000). Measures of reliability in sports medicine and science. *Sports Medicine*, 30(1), 1–15. https://doi.org/10.2165/00007256-200030010-00001

[12] Shrout, P.E., & Fleiss, J.L. (1979). Intraclass correlations: uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428. https://doi.org/10.1037/0033-2909.86.2.420

[13] McGraw, K.O., & Wong, S.P. (1996). Forming inferences about some intraclass correlation coefficients. *Psychological Methods*, 1(1), 30–46. https://doi.org/10.1037/1082-989X.1.1.30

[14] Bland, J.M., & Altman, D.G. (1999). Measuring agreement in method comparison studies. *Statistical Methods in Medical Research*, 8(2), 135–160. https://doi.org/10.1177/096228029900800204

[15] Vangel, M.G. (1996). Confidence intervals for a normal coefficient of variation. *The American Statistician*, 50(1), 21–26. https://doi.org/10.1080/00031305.1996.10473537

---

## License

GNU General Public License v3.0 or later — see
[LICENSE](https://github.com/jandrushko/mep-cmap-analyser/blob/main/LICENSE)
for the full text.

**Versions 1.3.3 and earlier were released under the MIT Licence and remain
so.** That grant is irrevocable: anyone who obtained those releases keeps the
rights they were given under them. From version 1.4.0 the project is GPL-3,
which was necessary in order to incorporate work derived from the
[TMS Analysis ToolBox](https://github.com/CunninghamLab/TMSAnalysisToolBox)
(Cunningham, Zhang & Cahn, 2021), a GPL-3 project whose terms require
derivative works to carry the same licence. See [NOTICE](https://github.com/jandrushko/mep-cmap-analyser/blob/main/NOTICE)
for the licence history and third-party attribution.

In practice this means software incorporating MEP-CMAP Analyser must also be
distributed under GPL-3. Using the application to analyse data, and publishing
results obtained with it, are unaffected.

