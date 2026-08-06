# MEP-CMAP Analyser

**Version 1.2.8 | August 2026**  
*Author:* [*Justin Andrushko PhD, Northumbria University*](https://researchportal.northumbria.ac.uk/en/persons/justin-w-andrushko/)

*Collaborators:* [*David Cunningham PhD*](https://fescenter.org/team/investigators/cunningham-david-phd/) *(*[*TMS Analysis ToolBox*](https://github.com/CunninghamLab/TMSAnalysisToolBox)*) ·* [*Nicholas Holmes PhD*](https://www.birmingham.ac.uk/staff/profiles/sportex/holmes-nick) *·* [*TMSMultiLab*](https://github.com/TMSMultiLab/TMSMultiLab/wiki)

[!\[PyPI version](https://badge.fury.io/py/mep-cmap-analyser.svg)](https://pypi.org/project/mep-cmap-analyser/)
[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jandrushko/mep-cmap-analyser/blob/main/LICENSE)
[!\[Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**PyPI:** https://pypi.org/project/mep-cmap-analyser/  
**GitHub:** https://github.com/jandrushko/mep-cmap-analyser  
**Bug reports:** https://github.com/jandrushko/mep-cmap-analyser/issues

A BIDS-compliant, open-source, cross-platform desktop tool for processing, quantifying, and group-analysing TMS/EMG neurophysiology recordings. Built for researchers who need reproducible, auditable waveform analysis without writing custom scripts for every study — and extensible with community add-ons for analyses the core tool doesn't yet cover.

\---

## Overview

MEP-CMAP Analyser is a GUI pipeline for EMG data collected with transcranial magnetic stimulation (TMS) and peripheral nerve stimulation (PNS) paradigms. It is organised around a neuroimaging-style **first-level / second-level** workflow:

* **Setup** — open a dataset, manage the file queue, and (optionally) convert raw recordings into a BIDS-compliant layout.
* **First Level: Single File** — process individual recordings: filter, segment trials around stimulation events, detect and quantify response features, review per trial, and run optional analysis add-ons.
* **Second Level: Group** — merge all processed sessions into a single, statistics-ready table for mixed-effects modelling, and run optional group-level add-ons.

Every setting, decision, and manual edit is saved in a sidecar JSON so analyses are fully reproducible and can be re-run or audited at any time.

The tool is not limited to any single measure or paradigm. It handles motor evoked potentials (MEPs), compound muscle action potentials (CMAPs), cortical silent periods (cSPs), M-wave recruitment curves, paired-pulse protocols such as SICI and ICF, and any other time-locked EMG response measurable by peak-to-peak amplitude, onset latency, or area under the curve. It operates on continuous recordings, pre-epoched trial stacks, and EMG bursts recorded without stimulation.

\---

## What's New in 1.2

* **Extensible add-ons framework** — drop-in Python modules that run post-hoc on saved results and write their own new files, at two scopes: **single-file** (first level) and **group-level** (second level). Ships with a faithful port of **MEPFeatX** (morphological MEP features), a rectified-area example, and a group-summary example.
* **Second Level add-ons tab** — group-level add-ons operate on the merged group table.
* **Reorganised interface** — a clearer two-level tab structure (Setup / First Level / Second Level) with a persistent active-file header and step-by-step first-level sub-tabs (1a–1d + Add-ons).
* **Check for updates** — Settings → Check for updates queries GitHub Releases and offers an assisted update (pip upgrade, or a link to the download page for compiled builds).
* **BIDS-ify** — convert non-BIDS recordings into a BIDS-compliant `rawdata/` layout with shared, per-file editable stimulation metadata (NIBS BEP037).
* **Broader format support** — added BIOPAC AcqKnowledge (`.acq` and `.mat`), Brainsight neuronavigation exports, BrainVision, and LabChart MATLAB exports.
* **Cross-platform polish** — readable coloured action buttons on macOS, Windows, and Linux, and consistent font scaling across the interface.

**Point releases (1.2.1–1.2.8):** EDF/BDF files (including BIDS-ify output) now load correctly, plus release-pipeline and repository cleanup.

\---

## The Interface

```
Setup                         First Level: Single File                  Second Level: Group
├── Dataset                   ├── 1a  Labels \\\\\\\& Analysis Setup      ├── Group Analysis (LME)
└── BIDS-ify                  ├── 1b  Data Filtering                  └── Add-ons
                               ├── 1c  Feature Detection Setup
                               │       (+ ▶ Run Analysis)
                               ├── 1d  Normalisation (optional)
                               └── Add-ons
```

The active file, channel, and event marker are shown in a persistent header above the First-Level sub-tabs, so context stays visible as you move between steps.

\---

## Features at a Glance

### Data Ingestion and Format Support

The tool auto-detects the file format on open (by extension, binary signature, or header sniff) and dispatches to the correct reader. Supported formats:

#### Spike-2 native (`.smr`) — *requires `neo`*

Native Spike-2 binary files read directly via the [Neo](https://neo.readthedocs.io) library — no text-export step. On first open a dialog identifies the EMG channel and stim/trigger channel; the choice is saved to a sidecar (`.smr\\\\\\\_config.json`) and not asked again. DigMark marker codes (A, B, C, …) are decoded from the event channel and each appears as a separate stimulus type. Neo is installed automatically with `pip install mep-cmap-analyser`.

#### Spike-2 text export (`.txt`)

Exported via **File → Export → Text**. Waveform channels are read along with DigMark event timestamps; any number of stimulus types and marker codes are supported. I/O is accelerated by the compiled Rust extension (`mep\\\\\\\_cmap\\\\\\\_io`).

#### LabChart text export (`.txt`)

Auto-detected from the `Interval=` header. Each recording block is treated as a pre-aligned trial; no trigger channel is required. Rust-accelerated.

#### LabChart MATLAB export (`.mat`)

LabChart's MATLAB export is detected by its signature variables and read natively — no LabChart installation required.

#### ADInstruments CFWB binary (`.adibin`)

Exported from LabChart via **File → Export → ADInstruments Binary**. The CFWB format is parsed natively in Rust. Stimulation times are derived from a trigger/TTL channel auto-detected by name (`stim`, `trig`, `ttl`). Rust-accelerated.

#### BIOPAC AcqKnowledge (`.acq`)

Native BIOPAC AcqKnowledge acquisition files, read via `bioread`. Channels and event markers are decoded directly from the file.

#### BIOPAC AcqKnowledge MATLAB export (`.mat`)

AcqKnowledge's MATLAB export, detected by its signature variables and read without a BIOPAC installation.

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

* Configurable pre-stimulus and post-stimulus windows (ms)
* Per stimulus-type gap parameter to skip the TMS artefact period before onset search
* Multi-stimulus support within a single recording: every marker/event label gets its own settings, colour, and output columns
* Stim times sourced from DigMark timestamps (Spike-2), interval resets (LabChart), TTL/trigger rising edges (CFWB, generic TTL rows), event markers (BIOPAC, BrainVision, Brainsight), or manual entry

### Response Quantification

|Measure|Description|
|-|-|
|**PTP amplitude (mV)**|Peak-to-peak amplitude within the user-specified MEP/CMAP window|
|**Onset latency (ms)**|MEP onset relative to stimulus (see onset detection methods)|
|**AUC (mV·s)**|Area under the rectified EMG from onset to cSP start (or a user-defined window via drag selector)|
|**cSP duration (ms)**|Cortical silent period, from EMG suppression onset to EMG return|
|**cSP MEP offset (ms)**|Time from stimulus to start of cSP|
|**cSP EMG return (ms)**|Time from stimulus to EMG recovery after cSP|
|**cSP/MEP ratio (ms/mV)**|cSP duration divided by MEP PTP amplitude (Orth \& Rothwell, 2004 \[5])|
|**Normalised PTP**|PTP as a fraction of an Mmax or single-pulse reference mean|
|**Paired-pulse ratio**|Conditioned / reference amplitude for SICI, ICF, or any custom pairing|
|**Z-score (within / pooled)**|Standardised amplitude within each stimulus type, and pooled across conditions|
|**Detrended PTP — within condition (mV)**|Linearly detrended amplitude within each stimulus type, removing condition-specific drift|
|**Detrended PTP — session (mV)**|Linearly detrended using a single trend across all trials in chronological order (captures fatigue/potentiation)|
|**Overall trial number**|Chronological trial index across all stimulus types, by stimulus timestamp order|
|**Stimulus time (s)**|Absolute timestamp of each stimulus (seconds from recording start)|
|**Inter-stimulus interval (s)**|Time since the immediately preceding stimulus (any type) — a useful covariate for variable ISIs|

### MEP Onset Detection

Three detection methods are available. The global default is set in **Settings → Preferences → Detection** and can be overridden per file in 1a without affecting the preference. All methods share the same physiological latency bounds (see [Physiological Latency Profiles](#physiological-latency-profiles)) and return `None` rather than a floor value when no confident onset is found, so ambiguous trials are flagged rather than silently mislabelled.

**Derivative-based method — Bigoni et al. 2022 (default)** — identifies onset as the start of the longest sustained positive-derivative run on the MEP rising edge, with optional Savitzky-Golay pre-smoothing. It makes no assumptions about background EMG level, so it is robust for both resting and active-contraction paradigms and for biphasic waveforms of either polarity. Follows Bigoni et al. \[6] with adaptations for variable sampling rates and muscle-group windows. Tuneable: smoothing window (default 2 ms) and minimum positive-run length (default 1 ms).

**Peak-fraction method** — finds the largest positive and negative peaks, then scans backward from the dominant peak to the point where the signal first crosses a fraction of that peak (default 15%), with a minimum-amplitude guard. Best on clean, high-amplitude MEPs with a near-silent baseline.

**Bootstrap threshold method** — estimates a noise threshold from the pre-stimulus baseline via a bootstrap distribution, then scans backward within a physiologically plausible latency window. More sensitive on low-amplitude signals. Latency windows are per stimulus type with published normative defaults.

### Cortical Silent Period (cSP) Detection

A vectorised bootstrap method: a silence threshold is estimated from the pre-stimulus baseline and a search runs from a configurable offset after MEP onset. Configurable criteria include minimum silence duration (default 25 ms), minimum EMG-return window, bootstrap criterion (default 1.96 SD), significance level (default 99th percentile), search-window end, and maximum MEP-to-cSP offset. cSP detection can be enabled/disabled per stimulus type and overridden per trial in the Data Inspector. The 1c Feature Detection Setup tab exposes all of these with inline guidance.

### M-wave Normalisation and Mmax

A separate Mmax file can be designated containing M-waves across a range of intensities. The plateau region is detected robustly for three scenarios: a full recruitment curve (averages the plateau within a tolerance band, default ±10%), a few supramaximal pulses (averages the largest similar-amplitude cluster), or a single M-wave (used directly). Normalised PTP is then reported for all MEP trials as a fraction of Mmax.

For designs where the background excitability of spinal motoneurones varies across trials, for example active-contraction paradigms, the tool also compensates evoked-potential magnitude for pre-stimulus excitability by quantile regression, following the method of Carson (2026) \[9], as an alternative or complement to Mmax normalisation.

### EMG Excitability Compensation (Carson 2026)

MEP amplitude covaries positively with the level of background EMG in the period immediately preceding the pulse, over a range far below any conventional rejection threshold. Rather than discarding trials, the amplitude is regressed on pre-stimulus r.m.s. EMG by median quantile regression within each sample (one participant, one stimulus type, one intensity, one block), and each trial's residual is re-expressed relative to an uncertainty-weighted reference value. The reference blends the regression intercept with the median of the fitted values, weighted by the relative standard error of the fitted ordinate at each point, so where no association is present the adjustment vanishes.

The implementation is verified against the author's own reference code and example data (`annotated\_QR\_example\_code.R`, Zenodo 20037178); `tests/test\_carson\_compensation.py` locks the slope, intercept, intercept weighting and reference value to his published values.

Reported per trial:

|Column|Description|
|-|-|
|`Adjusted\_PTP\_QR(mV)`|Excitability-compensated PTP|
|`Normalised\_Adjusted\_PTP\_QR`|Adjusted PTP as a fraction of the reference mean|
|`EMGComp\_Slope`, `EMGComp\_Intercept`|Fitted relationship, in PTP units per PreStimRMS unit|
|`EMGComp\_InterceptWeight`|Wi, the weight given to the intercept in the reference value|
|`EMGComp\_Adjustment(mV)`|reference minus median(fitted): the shift applied to the sample|
|`EMGComp\_N`, `EMGComp\_Method`|Trials in the fit, and the backend or fallback reason|
|`EMGComp\_PseudoR2`|Koenker-Machado pseudo-R-squared|
|`EMGComp\_Rho\_Pre`, `EMGComp\_Rho\_Post`|Spearman rho with pre-stimulus RMS before and after adjustment; the second should be near zero|

Two points worth knowing when reading the output. First, adjusted amplitudes **larger** than unadjusted are expected, not a fault: within any sample the low-RMS trials always shift upward, and a whole sample shifts upward whenever `EMGComp\_Slope` is negative (74 of Carson's 182 participants). Second, the per-trial shift is exactly `reference - (intercept + slope \* PreStimRMS)`, so plotting `Adjusted\_PTP\_QR(mV) - PTP(mV)` against `PreStimRMS` must give a straight line with slope `-EMGComp\_Slope`. That is the quickest check that a suspicious file is behaving correctly. Before comparing adjusted values across conditions, check that the slopes and intercepts are comparable, as the paper recommends.

Background EMG is quantified as the r.m.s. of a `prestim\_ms` window (default 100 ms) ending `rms\_guard\_ms` before the pulse (default 3 ms, matching the paper, and widened automatically if a stimulus type needs a longer artefact gap). The window's DC offset is removed first, since an offset is not motoneurone activity and carrying it into the r.m.s. adds between-trial variance that masks the association the method exists to remove.

Compensation is skipped for M-wave runs, which are direct muscle responses rather than spinally mediated, and typically span multiple intensities. Trials marked Removed or Excluded are left out of the fit and receive no compensation values; trials flagged by the z-screen but kept by the reviewer stay in, since retaining datapoints is one of the stated benefits of the method.

### Paired-Pulse Protocols

Any stimulus type can be designated as a conditioned stimulus and paired with a reference in 1a. Conditioned/reference ratios (e.g. SICI at 2–6 ms ISI, ICF at 10–15 ms ISI) are produced as a standard output column, with multiple reference assignments supported within a single file.

### Outlier Detection and Review

* Z-score flagging on PTP amplitude and RMS with a configurable threshold (default ±1.96)
* Interactive review dialog showing the flagged waveform in context, with include / exclude / note options
* Decisions persist across reruns — reviewed trials are not re-presented — and are recorded in the trial CSV's `Outlier\\\\\\\_Decision` column

### Data Inspector

Per-trial interactive review with a zoomed trial view plus a wider context window; draggable onset, cSP-start, and cSP-end markers; a drag-to-select AUC window; per-trial notes; and keyboard navigation. All edits are saved to the session JSON and applied on every subsequent run without re-review.

### Add-ons (Extensible Analyses)

Add-ons are optional, drop-in Python modules that run **after** processing, read the saved results, and write **their own new files** — they never modify core outputs. They come in two scopes:

* **First-level (single-file) add-ons** run on each recording's saved waveform bundle (`<prefix>\\\\\\\_segments.npz`) and appear on the **First Level → Add-ons** tab.
* **Second-level (group-level) add-ons** run on the merged group table (`group\\\\\\\_level\\\\\\\_LME\\\\\\\_ready.csv`) and appear on the **Second Level → Add-ons** tab.

Built-in add-ons:

|Add-on|Scope|What it does|
|-|-|-|
|**mepfeatx**|single-file|A faithful port of MEPFeatX (Nguyen et al. 2025 \[10]): morphological MEP features — amplitude, latency, AUC, waveform thickness, number of turns and phases, duration, and the two dominant peaks (T1/T2) — with per-trial and per-condition diagnostic figures and a transparent rejection reason for every trial it can't quantify|
|**rectified\_area**|single-file|Rectified area under each MEP over the analysis window (a minimal example)|
|**group\_summary**|group-level|Per-condition mean, SD, and N of every metric across the group (a minimal example)|

Add-ons can declare their own settings, which render as controls in the add-on's box (for example, MEPFeatX exposes a tunable noise-gate ratio for handling MEPs recorded during voluntary contraction). Point the tool at your own add-ons folder in **Settings → Preferences → Add-ons**; place first-level add-ons in a `single\\\\\\\_file/` subfolder and group-level add-ons in a `group\\\\\\\_level/` subfolder. See [Writing Add-ons](#writing-add-ons).

### BIDS-ify

The **Setup → BIDS-ify** tab converts non-BIDS recordings into a BIDS-compliant `rawdata/` layout. Shared stimulation metadata (following the NIBS BEP037 proposal) is edited once and applied to every file, with per-file overrides where needed. Files are reviewed, accepted, and converted from a persistent, status-coloured worklist, so a whole study can be brought into BIDS in one pass.

### Session Persistence and Reproducibility

Every setting the user touches — filter parameters, time windows, onset method, latency maps, cSP thresholds, normalisation references, Inspector edits, outlier decisions, analysis options — is saved in a per-file session JSON alongside the derivatives. Reloading restores the exact state; changing a setting and re-running produces a clean new result without losing manual review work. File paths in session JSONs are stored relative to the study root for portability across machines and cloud sync.

### Dataset Queue

* Open a study folder or individual files; the tool auto-detects BIDS `rawdata/` and `derivatives/` subfolders
* A persistent queue tracks status (Not Started, In Progress, Needs Review, Complete, Stale)
* Excluded files are remembered and not re-added on refresh; a right-click menu restores them, marks files for reprocessing, or opens the derivatives folder
* Process a highlighted recording with **Run selected**
* Queue state is saved to `dataset\\\\\\\_session.json`

### Check for Updates

**Settings → Check for updates** queries GitHub Releases in the background, compares the latest version to the one you're running, and — if you're behind — shows the release notes and offers an assisted update: a `pip install --upgrade` for source/pip installs, or a link to the download page for compiled builds. It fails gracefully offline and falls back to version tags if no formal release is published.

\---

## Supported Use Cases

The tool handles any paradigm where a time-locked EMG response is expected within a defined post-stimulus window, including but not limited to:

* **TMS MEP studies** — single-pulse, paired-pulse (SICI, ICF, LICI, SAI), or multi-intensity recruitment curves; any accessible muscle
* **Peripheral nerve stimulation CMAPs** — M-wave recruitment curves for Mmax determination or peripheral conduction
* **Corticospinal excitability assays** — resting and active MEP series, pre/post intervention, crossover and parallel designs
* **TMS-EMG silent period studies** — cSP duration, cSP/MEP ratio, and derived inhibitory indices
* **MEP waveform morphology** — via the MEPFeatX add-on (turns, phases, thickness, T1/T2)
* **Voluntary EMG bursts** — files with no stimulation events can be loaded for waveform inspection, RMS quantification, and trial-level output

\---

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
|Windows|`MEP-CMAP\\\\\\\_Analyser\\\\\\\_Windows.zip`|
|macOS|`MEP-CMAP\\\\\\\_Analyser\\\\\\\_Mac.zip`|
|Linux|`MEP-CMAP\\\\\\\_Analyser\\\\\\\_Linux.tar.gz`|

### Option 3: Run from source

```bash
git clone https://github.com/jandrushko/mep-cmap-analyser.git
cd mep-cmap-analyser
pip install -r requirements.txt
python -m mep\\\\\\\_cmap
```

\---

## Workflow

### 1\. Setup → Dataset

Open a study folder or an individual recording. The tool auto-detects a BIDS layout (`rawdata/` beside `derivatives/`) or sets up a derivatives folder in the standard location. Files appear in the queue with their status; double-click any file to load it, or highlight one and click **Run selected**. Opening an unrecognised format launches the Format Wizard for a one-time configuration.

### 2\. Setup → BIDS-ify *(optional)*

If your data isn't yet in BIDS, use BIDS-ify to set shared stimulation metadata and convert the ready files into a `rawdata/` tree.

### 3\. First Level → 1a Labels \& Analysis Setup

For each stimulus type in the recording, configure the display label and colour, the artefact gap (ms), whether to run cSP detection, the stimulus category and target muscle (which set physiological latency bounds), and any normalisation/paired-pulse reference pairing. Click **✔ Confirm Setup** when ready. Settings carry over between files.

### 4\. First Level → 1b Data Filtering

Set bandpass, notch, and Humbug options, preview the filter, then click **✔ Confirm filter settings → Feature Detection**.

### 5\. First Level → 1c Feature Detection Setup

Set time windows, onset-detection parameters, cSP settings, outlier detection, and analysis options, then click **▶ Run Analysis**. The tool extracts trials, quantifies all measures, flags outliers, optionally runs the Data Inspector, and writes results to the derivatives folder. Reloading a processed file offers to reuse the saved crop range, pick a new one, or use the full file, with all prior edits restored.

### 6\. First Level → 1d Normalisation *(optional)* and Add-ons *(optional)*

Normalise processed results against a reference file, and/or run first-level add-ons (e.g. MEPFeatX) on the saved bundles.

### 7\. Second Level → Group Analysis (LME)

The tool scans the derivatives folder and lists completed sessions. Assign study-design columns (Group, Condition, Timepoint, or any custom factor), configure stim roles, select sessions to include, and click **▶ Build group analysis file** to produce `group\\\\\\\_level\\\\\\\_LME\\\\\\\_ready.csv`.

### 8\. Second Level → Add-ons *(optional)*

Run group-level add-ons (e.g. group\_summary) on the merged group table.

\---

## Output Files

Results are written to a `derivatives/` folder beside the raw data, following BIDS derivative conventions:

```
study/
├── rawdata/
│   └── sub-001/ses-01/sub-001\\\\\\\_ses-01\\\\\\\_recording.txt
└── derivatives/
    ├── dataset\\\\\\\_session.json               ← file queue and processing status
    ├── study\\\\\\\_design.json                  ← Second-Level design configuration
    ├── group\\\\\\\_level\\\\\\\_LME\\\\\\\_ready.csv          ← merged group output (Second Level)
    └── sub-001/
        └── ses-01/
            ├── sub-001\\\\\\\_ses-01\\\\\\\_session.json           ← full session state
            ├── results/
            │   ├── sub-001\\\\\\\_ses-01\\\\\\\_<StimType>\\\\\\\_trials.csv   ← one per stim type
            │   ├── sub-001\\\\\\\_ses-01\\\\\\\_...\\\\\\\_segments.npz        ← waveform bundle (add-on input)
            │   └── ...                                    ← add-on outputs, e.g. \\\\\\\*\\\\\\\_mepfeatx.csv
            └── figures/                                   ← add-on figures, e.g. MEPFeatX plots
```

### Trial-level CSV columns

Each `<prefix>\\\\\\\_<StimType>\\\\\\\_trials.csv` contains the full column set below. Which
columns are *populated* depends on what you enabled — cSP columns fill only when
cSP detection is on, normalisation columns when a reference file is set, and the
excitability-compensation block when that option is run.

|Column(s)|Description|
|-|-|
|`File`, `StimType`, `Stim\\\\\\\_Label`|Recording identifier and stimulus-type code / display label|
|`Segment`, `Segment\\\\\\\_Overall`|Trial index within the condition, and chronological index across all conditions|
|`Stim\\\\\\\_Time(s)`, `Time\\\\\\\_Since\\\\\\\_Last\\\\\\\_Stim(s)`|Absolute stimulus time and inter-stimulus interval|
|`Limb`|Limb identifier (from filename or entered)|
|`PTP(mV)`, `Latency(ms)`, `AUC(mV\\\\\\\*s)`|Core response: peak-to-peak amplitude, onset latency (`Not Marked` when unresolved), and area under the rectified EMG|
|`Measure`|Optional auxiliary / manual measurement (blank unless used)|
|`cSP\\\\\\\_Duration(ms)`, `cSP\\\\\\\_MEP\\\\\\\_Offset(ms)`, `cSP\\\\\\\_EMG\\\\\\\_Return(ms)`, `cSP\\\\\\\_MEP\\\\\\\_Ratio(ms/mV)`|Silent-period duration (`Not Marked` when absent), stimulus→cSP-start, stimulus→EMG-return, and cSP ÷ PTP ratio (Orth \& Rothwell, 2004 \[5])|
|`PreStimRMS`, `PreStimPTP`, `PTP\\\\\\\_per\\\\\\\_PreStimRMS`, `Z\\\\\\\_PreStimRMS`|Pre-stimulus baseline EMG: RMS, peak-to-peak, PTP-per-RMS, and standardised RMS|
|`Z\\\\\\\_PTP\\\\\\\_Within`, `Z\\\\\\\_PTP\\\\\\\_Pooled`|PTP z-scores within each condition and pooled across conditions|
|`PTP\\\\\\\_Detrended\\\\\\\_WithinCond(mV)` + `\\\\\\\_Z`, `PTP\\\\\\\_Detrended\\\\\\\_Session(mV)` + `\\\\\\\_Z`|Amplitude detrended within condition and across the whole session (fatigue / potentiation), each with its z-score|
|`Reference\\\\\\\_Type`, `Reference\\\\\\\_Mean(mV)`, `Reference\\\\\\\_N`, `Normalised\\\\\\\_PTP`, `Normalised\\\\\\\_PTP\\\\\\\_per\\\\\\\_PreStimRMS`|Mmax / reference normalisation: reference used, its mean and N, and the normalised amplitudes|
|`Adjusted\\\\\\\_PTP\\\\\\\_QR(mV)`, `Normalised\\\\\\\_Adjusted\\\\\\\_PTP\\\\\\\_QR`|**Excitability-compensated PTP** — adjusted for spinal motoneurone excitability by quantile regression on pre-stimulus EMG (Carson, 2026 \[9]), raw and reference-normalised|
|`EMGComp\\\\\\\_Method`, `EMGComp\\\\\\\_N`, `EMGComp\\\\\\\_Slope`, `EMGComp\\\\\\\_Intercept`, `EMGComp\\\\\\\_InterceptWeight`, `EMGComp\\\\\\\_Adjustment(mV)`, `EMGComp\\\\\\\_PseudoR2`, `EMGComp\\\\\\\_Rho\\\\\\\_Post`|Excitability-compensation fit: method / status, N, regression coefficients and intercept weighting, per-trial adjustment, and diagnostics (pseudo-R² and residual PTP–EMG correlation)|
|`Outlier\\\\\\\_Decision`, `Manual\\\\\\\_Note`|Review outcome (`Not flagged` or your include / exclude decision) and free-text annotation|

### Group-level LME-ready CSV

Every trial-level column from every included session, prefixed with design columns — `participant\\\\\\\_id`, `session`, `task`, `timepoint`, `Stim\\\\\\\_Role`, and any custom between/within-subject factors defined at the second level. Output is at the trial level (outliers retained with their Z-scores as covariates rather than pre-excluded), so the analyst keeps full control of trial-level modelling. This file is also the input for group-level add-ons.

\---

## Physiological Latency Profiles

The derivative-based (Bigoni), bootstrap, and peak-fraction onset detectors all search within a per-muscle physiological window. Defaults assume contralateral cortical stimulation with active facilitation (resting latencies are typically 1–3 ms longer). Windows can be overridden per stimulus type in 1a, and the global defaults edited in **Settings → Preferences → Latency Profiles**.

|Stimulus type / Muscle target|Window (ms)|Reference(s)|
|-|-|-|
|TMS → deltoid / trapezius|8–16|\[1], \[2]|
|TMS → biceps / triceps brachii|12–20|\[1], \[2]|
|TMS → trunk / external oblique|12–22|\[3]|
|TMS → hand / FDI / APB / ADM|18–28|\[4], \[1]|
|TMS → forearm (FCR / ECR)|16–26|\[1]|
|TMS → vastus lateralis / quad|18–30|\[1], \[4]|
|TMS → hamstrings|18–32|\[1]|
|TMS → tibialis anterior / leg|28–45|\[4], \[1]|
|PNS → upper limb (M-wave)|2–12|\[1]|
|PNS → lower limb (M-wave)|4–18|\[1]|

**Notes.** Latency scales positively with height and age, particularly for lower-limb muscles. The lower bound excludes the TMS artefact; the upper bound captures the ±2 SD range of normative cohort data while avoiding late oligosynaptic MEPs. The trunk window is anchored to the contralateral onset latency of 15.8 ± 1.4 ms reported by Miyano et al. \[3].

\---

## Writing Add-ons

An add-on is a small Python module exposing an `ADDON\\\\\\\_NAME`, an optional description/version/author, an optional `ADDON\\\\\\\_SCOPE` (`"single\\\\\\\_file"` or `"group\\\\\\\_level"`), optional `ADDON\\\\\\\_SETTINGS`, and a `run(context)` function. It reads from the context and writes **new** files into `context.results\\\\\\\_dir`.

**First-level (`single\\\\\\\_file`)** add-ons receive a context with the per-trial waveforms grouped by stimulus type, sampling rate, unit, a stimulus-aligned time axis, the per-trial table, the analysis config, and output paths:

```python
ADDON\\\\\\\_NAME  = "my\\\\\\\_addon"
ADDON\\\\\\\_SCOPE = "single\\\\\\\_file"

def run(context):
    import os, numpy as np
    rows = \\\\\\\[]
    for stim\\\\\\\_type, stack in context.segments.items():   # stack: (n\\\\\\\_trials, n\\\\\\\_samples)
        for i, trace in enumerate(stack):
            rows.append((stim\\\\\\\_type, i, float(np.ptp(trace))))
    out = os.path.join(context.results\\\\\\\_dir, f"{context.bids\\\\\\\_prefix}\\\\\\\_my\\\\\\\_addon.csv")
    # ... write `rows` to `out` ...
    context.log(f"my\\\\\\\_addon → {os.path.basename(out)}")
    return \\\\\\\[out]
```

**Second-level (`group\\\\\\\_level`)** add-ons receive `context.group\\\\\\\_table` (the merged group DataFrame), with `design\\\\\\\_columns` / `metric\\\\\\\_columns` split out for convenience:

```python
ADDON\\\\\\\_NAME  = "my\\\\\\\_group\\\\\\\_addon"
ADDON\\\\\\\_SCOPE = "group\\\\\\\_level"

def run(context):
    import os
    summary = context.group\\\\\\\_table.groupby("StimType")\\\\\\\[context.metric\\\\\\\_columns].mean()
    out = os.path.join(context.results\\\\\\\_dir, f"{context.bids\\\\\\\_prefix}\\\\\\\_my\\\\\\\_group\\\\\\\_addon.csv")
    summary.to\\\\\\\_csv(out)
    context.log(f"my\\\\\\\_group\\\\\\\_addon → {os.path.basename(out)}")
    return \\\\\\\[out]
```

Put your modules in the matching subfolder (`single\\\\\\\_file/` or `group\\\\\\\_level/`) of the add-ons folder set in **Settings → Preferences → Add-ons**. The built-in add-ons (`mepfeatx`, `rectified\\\\\\\_area`, `group\\\\\\\_summary`) are good starting templates.

\---

## Building from Source

```bash
# Windows
python build\\\\\\\_windows.py

# macOS
python3 build\\\\\\\_mac.py

# Linux
python3 -m venv venv\\\\\\\_linux \\\\\\\&\\\\\\\& source venv\\\\\\\_linux/bin/activate
pip install -r requirements.txt
python3 build\\\\\\\_linux.py
```

The build scripts create a local virtual environment, compile the Rust I/O extension (`mep\\\\\\\_cmap\\\\\\\_io`) if a Rust toolchain is present, and run PyInstaller with the platform spec. The bundled add-ons and BIDS schema ship automatically.

\---

## Dependencies

|Package|Purpose|
|-|-|
|`numpy`|Numerical arrays and signal operations|
|`scipy`|Filtering, statistics, interpolation, signal processing|
|`pandas`|CSV I/O and data manipulation|
|`matplotlib`|Waveform plotting, interactive figures, add-on figures|
|`statsmodels`|Regression utilities for excitability compensation (Carson, 2026 \[9])|
|`PyWavelets`|Wavelet time-frequency display in filter preview|
|`Pillow`|Image handling for splash screen and icons|
|`neo`|Native Spike-2 `.smr` reading|
|`pyedflib`|EDF/BDF handling for BIDS-ify|
|`bioread`|BIOPAC AcqKnowledge `.acq` reading|
|`tkinter`|GUI (bundled with standard Python)|

The optional Rust extension `mep\\\\\\\_cmap\\\\\\\_io` provides accelerated I/O for the Spike-2 text, LabChart text, Generic TSV, and CFWB binary formats; all formats fall back to pure Python if it is unavailable.

\---

## Citation

If you use MEP-CMAP Analyser in published research, please cite:

> Justin W. Andrushko. (2026). jandrushko/mep-cmap-analyser: MEP-CMAP Analyser (Version v1.2.8) \[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21810844
> Northumbria University. https://github.com/jandrushko/mep-cmap-analyser

\---

## References

\[1] Groppa, S., Oliviero, A., Eisen, A., Quartarone, A., Cohen, L.G., Mall, V., Kaelin-Lang, A., Mima, T., Rossi, S., Thickbroom, G.W., Rossini, P.M., Ziemann, U., Valls-Solé, J., \& Siebner, H.R. (2012). A practical guide to diagnostic transcranial magnetic stimulation: Report of an IFCN committee. *Clinical Neurophysiology*, 123(5), 858–882. https://doi.org/10.1016/j.clinph.2012.01.010

\[2] Colebatch, J.G., Rothwell, J.C., Day, B.L., Thompson, P.D., \& Marsden, C.D. (1990). Cortical outflow to proximal arm muscles in man. *Brain*, 113(6), 1843–1856. https://doi.org/10.1093/brain/113.6.1843

\[3] Miyano, R., Shirota, Y., Kodama, S., Toda, T., \& Hamada, M. (2026). Ipsilateral and contralateral cortical control of the external oblique muscles revealed by TMS. *Clinical Neurophysiology*, 181, 2111400. https://doi.org/10.1016/j.clinph.2025.2111400

\[4] Cantone, M., Lanza, G., Fisicaro, F., Bella, R., Ferri, R., Pennisi, G., Waterstraat, G., \& Pennisi, M. (2023). Sex-specific reference values for total, central, and peripheral latency of motor evoked potentials from a large cohort. *Frontiers in Human Neuroscience*, 17, 1152204. https://doi.org/10.3389/fnhum.2023.1152204

\[5] Orth, M., \& Rothwell, J.C. (2004). The cortical silent period: intrinsic variability and relation to the waveform of the transcranial magnetic stimulation pulse. *Clinical Neurophysiology*, 115(5), 1076–1082. https://doi.org/10.1016/j.clinph.2003.12.005

\[6] Bigoni, C., Cadic-Melchior, A., Vassiliadis, P., Morishita, T., \& Hummel, F.C. (2022). An automatized method to determine latencies of motor-evoked potentials under physiological and pathophysiological conditions. *Journal of Neural Engineering*, 19(2), 024002. https://doi.org/10.1088/1741-2552/ac636c

\[7] Hupfeld, K.E., Swanson, C.W., Fling, B.W., \& Seidler, R.D. (2021). TMS-induced silent periods: A review of methods and call for consistency. *Journal of Neuroscience Methods*, 346, 108950. https://doi.org/10.1016/j.jneumeth.2020.108950

\[8] Rossini, P.M., et al. (2015). Non-invasive electrical and magnetic stimulation of the brain, spinal cord, roots and peripheral nerves: Basic principles and procedures for routine clinical and research application. *Clinical Neurophysiology*, 126(6), 1071–1107. https://doi.org/10.1016/j.clinph.2015.02.001

\[9] Carson, R.G. (2026). A method of compensating for the excitability of spinal motoneurones when estimating the magnitude of potentials evoked in skeletal muscles. *The Journal of Physiology*, 604, 5731–5757. https://doi.org/10.1113/JP290979

\[10] Nguyen, T.D., et al. (2025). MEPFeatX: feature extraction for motor evoked potentials. *Frontiers in Neuroscience*, 18, 1415257. https://doi.org/10.3389/fnins.2024.1415257

\---

## License

MIT License — see [LICENSE](https://github.com/jandrushko/mep-cmap-analyser/blob/main/LICENSE) for details.

