# MEP-CMAP Analyser

**Version 0.9.6 | May 2026**  
*Author: Justin Andrushko PhD, Northumbria University*

[![PyPI version](https://badge.fury.io/py/mep-cmap-analyser.svg)](https://pypi.org/project/mep-cmap-analyser/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jandrushko/mep-cmap-analyser/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**PyPI:** https://pypi.org/project/mep-cmap-analyser/  
**GitHub:** https://github.com/jandrushko/mep-cmap-analyser  
**Bug reports:** https://github.com/jandrushko/mep-cmap-analyser/issues

A BIDS-compliant, open-source desktop tool for processing, quantifying, and group-analysing TMS/EMG neurophysiology recordings. Built for researchers who need reproducible, auditable waveform analysis without writing custom scripts for every study.

---

## Overview

MEP-CMAP Analyser is a two-stage GUI pipeline for EMG data collected with transcranial magnetic stimulation (TMS) and peripheral nerve stimulation (PNS) paradigms. Stage 1 processes individual recordings — filtering, segmenting trials around stimulation events, detecting and quantifying response features, and allowing per-trial human review. Stage 2 merges all processed sessions into a single, statistics-ready output file. Every setting, decision, and manual edit is saved in a sidecar JSON so analyses are fully reproducible and can be re-run or audited at any time.

The tool is not limited to any single measure or paradigm. It handles motor evoked potentials (MEPs), compound muscle action potentials (CMAPs), cortical silent periods (cSPs), M-wave recruitment curves, paired-pulse protocols such as SICI and ICF, and any other time-locked EMG response measurable by peak-to-peak amplitude, onset latency, or area under the curve. It operates on continuous recordings, pre-epoched trial stacks, and EMG bursts recorded without stimulation.

---

## Features at a Glance

### Data Ingestion and Format Support

- **Spike-2 text export** — waveform channels plus DigMark event timestamps; multi-channel recordings with any number of stimulus types
- **LabChart text export** — auto-detected from `Interval=` header; each recording block treated as a pre-aligned trial; no trigger channel required
- **KinEMG / NI-DAQ CSV export** — auto-detected from `Author,KinEMG` header; sampling rate and channel names (`Dev1/ai0` etc.) parsed directly from the file header
- **Generic Format Wizard** — for any other tabular text file (tab, space, or comma delimited); a one-time, four-step wizard configures:
  - **Column-wise** layouts (rows = time samples, columns = channels) — typical for most DAQ exports and continuous recordings
  - **Row-wise** layouts (rows = channels, columns = time samples) — used by Delsys Trigno and similar systems, where one row is a continuous TTL trigger signal and another row is the EMG recording
  - Automatic detection of non-numeric header lines, channel name rows, and embedded sampling rate metadata so the wizard pre-fills as much as possible before the user clicks through
  - Per-channel role assignment: EMG, Stim/Trigger, or Ignore
  - Configuration saved as a sidecar JSON; subsequent opens load instantly without re-running the wizard

### Signal Processing

- Bandpass filter (default 20–450 Hz, adjustable) using Butterworth or Chebyshev Type I designs, with independent highpass and lowpass orders
- Notch filter at any frequency (e.g. 50 or 60 Hz) with adjustable Q factor
- Humbug-style mains noise canceller with configurable harmonic count
- Flexible bandpass mode for independent control of highpass and lowpass cutoff orders
- Real-time filter preview showing frequency response and wavelet time-frequency decomposition of the raw signal

### Trial Segmentation

- Configurable pre-stimulus and post-stimulus windows (ms)
- Per stimulus-type gap parameter to skip the TMS artefact period before onset search
- Multi-stimulus support within a single recording: every marker/event label gets its own settings, colour, and output column
- Stim times sourced from: DigMark timestamps (Spike-2), interval resets (LabChart), TTL/trigger channel rising edges (generic TTL rows), or manually entered

### Response Quantification

| Measure | Description |
|---|---|
| **PTP amplitude (mV)** | Peak-to-peak amplitude within the user-specified MEP/CMAP window |
| **Onset latency (ms)** | MEP onset relative to stimulus, using peak-anchored backward scan or bootstrap threshold detection |
| **AUC (mV·s)** | Area under the rectified EMG from onset to cSP start (or user-defined window via drag selector) |
| **cSP duration (ms)** | Duration of the cortical silent period, from EMG suppression onset to EMG return |
| **cSP MEP offset (ms)** | Time from stimulus to start of cSP |
| **cSP EMG return (ms)** | Time from stimulus to EMG recovery after cSP |
| **MEP/cSP ratio** | PTP amplitude divided by cSP duration (Orth & Rothwell, 2004) |
| **Normalised PTP** | PTP expressed as a fraction of an Mmax reference or single-pulse reference mean |
| **Paired-pulse ratio** | Conditioned / reference amplitude for SICI, ICF, or any custom pairing |
| **Z-score (within type)** | Standardised amplitude within each stimulus type |
| **Z-score (pooled)** | Standardised amplitude across all conditions in the file |
| **Detrended PTP (mV)** | Linearly detrended amplitude to remove slow amplitude drift |

### MEP Onset Detection

Two detection methods are available and switchable per file:

**Peak-fraction method (default)** — finds the largest positive and negative peaks in the MEP window, then scans backward from the dominant peak to find where the signal first crosses a fraction of that peak (configurable, default 15%). A minimum peak amplitude threshold guards against noise false-positives.

**Bootstrap threshold method** — estimates a noise threshold from the pre-stimulus baseline using a bootstrap distribution, then scans forward within a physiologically plausible latency window to find the first sample exceeding the criterion. Latency windows are defined per stimulus type and have built-in defaults based on published normative data:

| Stimulus / Muscle Target | Latency Window |
|---|---|
| TMS → hand / FDI | 18–30 ms |
| TMS → vastus lateralis | 18–35 ms |
| TMS → lower leg | 28–45 ms |
| PNS → upper limb | 2–12 ms |
| PNS → lower limb | 4–18 ms |

Custom windows can be set for any stim type label.

### Cortical Silent Period (cSP) Detection

cSP detection uses a vectorised bootstrap method: a silence threshold is estimated from the pre-stimulus baseline and a search is conducted from a configurable offset after MEP onset. Detection criteria are configurable:

- Minimum silence duration (default 25 ms)
- Minimum EMG return window (default 40 ms)
- Bootstrap criterion (default 1.96 SD)
- Statistical significance level (default 99th percentile)
- Maximum search window end (default 400 ms)
- Maximum MEP-to-cSP offset (default 100 ms)

cSP detection can be enabled or disabled per stimulus type and overridden per trial in the Data Inspector.

### M-wave Normalisation and Mmax

A separate Mmax file can be designated containing M-wave responses across a range of stimulus intensities. The tool automatically detects the plateau region using a robust algorithm that handles three real-world scenarios:

- **Full recruitment curve** — finds and averages the plateau region within a configurable tolerance band (default ±10%)
- **A few supramaximal pulses** — averages the largest cluster of similar amplitudes
- **Single M-wave** — uses that value directly

Normalised PTP is then reported for all MEP trials as a fraction of Mmax.

### Paired-Pulse Protocols

Any stimulus type can be designated as a conditioned stimulus and paired with a reference in Stage 1a. The tool computes conditioned/reference amplitude ratios (e.g. SICI at 2–6 ms ISI, ICF at 10–15 ms ISI) as a standard output column. Multiple reference assignments can be configured within a single file for complex designs.

### Outlier Detection and Review

- Z-score flagging on PTP amplitude and RMS with a configurable threshold (default ±1.96)
- Interactive outlier review dialog showing the flagged waveform in context with the option to include, exclude, or note the trial
- Outlier decisions persist across reruns — previously reviewed trials are not re-presented
- All decisions are written to a separate outlier log CSV

### Data Inspector

Per-trial interactive review with:

- Zoomed trial view plus a wider context window (configurable width, default ±3 s around stim)
- Draggable onset marker
- Draggable cSP start and end markers
- Drag-to-select AUC window with a toggle button
- Per-trial annotation notes
- Keyboard navigation (next/previous trial, next stim type)
- All edits saved to the session JSON and applied on every subsequent run without re-review

### Session Persistence and Reproducibility

Every setting the user touches — filter parameters, time windows, onset detection method, latency maps, cSP thresholds, normalisation references, Inspector edits, outlier decisions, analysis options — is saved in a per-file session JSON alongside the derivatives. Reloading a file restores the exact state. Changing a filter or threshold and re-running produces a clean new result without losing the manual review work.

### Dataset Queue

- Load a study folder or individual files; the tool auto-detects BIDS `rawdata/` and `derivatives/` subfolders
- A persistent file queue tracks processing status: Not Started, In Progress, Needs Review, Complete, or Stale
- Previously excluded files are remembered and not re-added on refresh
- Right-click menu to restore excluded files, mark files for reprocessing, or open the derivatives folder
- Queue state saved to `dataset_session.json`

### Group Analysis (Stage 2)

Once any number of sessions are processed, Stage 2 merges them into a single LME-ready CSV:

- Scans the derivatives folder to discover all completed sessions
- User assigns study design columns (Group, Condition, Timepoint, or any custom factor)
- Stim type roles are assigned globally (Reference, Conditioned, M-wave, etc.)
- Output includes all trial-level data with Z-scores, detrended values, normalised amplitudes, and study design columns — ready for linear mixed effects models in R, Python, or SPSS

---

## Supported Use Cases

The tool is general enough to handle any paradigm where a time-locked EMG response is expected within a defined post-stimulus window. Examples include but are not limited to:

- **TMS MEP studies** — single-pulse, paired-pulse (SICI, ICF, LICI, SAI), or multi-intensity recruitment curves; cortical and cerebellar targets; any accessible muscle
- **Peripheral nerve stimulation CMAPs** — M-wave recruitment curves for Mmax determination or peripheral motor nerve conduction
- **Corticospinal excitability assays** — resting and active MEP series, pre/post intervention, crossover and parallel group designs
- **TMS-EMG silent period studies** — cSP duration, MEP/cSP ratio, and derived inhibitory indices
- **Voluntary EMG bursts** — files with no stimulation events can be loaded for waveform inspection, RMS quantification, and trial-level output even without stim-triggered segmentation

---

## Installation

### Option 1: pip (recommended for most users)

```bash
pip install mep-cmap-analyser
mep-cmap
```

Python 3.9 or later is required. Tkinter must be available on your system:

- **Windows / macOS** — included with standard Python installers
- **Linux (Ubuntu / Debian)** — `sudo apt install python3-tk`

### Option 2: Compiled binaries (no Python required)

Pre-built executables for all platforms are available on the [Releases page](https://github.com/jandrushko/mep-cmap-analyser/releases). Download, unzip, and run — no installation or Python knowledge needed.

| Platform | File |
|---|---|
| Windows | `MEP-CMAP_Analyser_Windows.zip` |
| macOS | `MEP-CMAP_Analyser_Mac.zip` |
| Linux | `MEP-CMAP_Analyser_Linux.tar.gz` |

### Option 3: Run from source

```bash
git clone https://github.com/jandrushko/mep-cmap-analyser.git
cd mep-cmap-analyser
pip install -r requirements.txt
python -m mep_cmap
```

---

## Workflow

### Step 1 — Dataset Setup

Open a study folder or an individual recording. The tool auto-detects whether the folder follows BIDS conventions (`rawdata/` present beside `derivatives/`) or sets up a new derivatives folder in the standard location. Files appear in the queue with their current status. Double-click any file to load it, or click **Run all unprocessed** to process the queue in sequence.

If you open a file in a format the tool has not seen before, the Format Wizard launches automatically to guide you through a one-time configuration.

### Step 2 — Stage 1a: Labels and Analysis Setup

For each stimulus type present in the recording, configure:

- Display label and colour
- Gap (ms) — samples to skip after the artefact before MEP onset search begins
- Whether to run cSP detection on this stim type
- Stimulus category and target muscle (sets physiological latency bounds)
- Normalisation reference pairing (for paired-pulse or Mmax-relative output)

Settings are preserved between files so you rarely need to re-enter them for a new session.

### Step 3 — Stage 1b: Processing

Set filters, time windows, onset detection parameters, and cSP settings. Run the analysis. The tool extracts trials, quantifies all measures, detects outliers, presents flagged trials for review (if enabled), runs the Data Inspector for manual review (optional), and writes results to the derivatives folder.

Reloading a previously processed file offers three options: reuse the saved crop range, select a new range, or use the full file. All prior edits and exclusions are restored automatically.

### Step 4 — Stage 2: Group Analysis

Navigate to the Stage 2 tab. The tool scans the derivatives folder and lists all completed sessions. Assign study design variables, configure stim roles, select sessions to include, and click **Build group analysis file** to produce the merged LME-ready CSV.

---

## Input Formats

### Spike-2 text export (`.txt`)

Waveform channels (EMG, force, additional analogue inputs) plus DigMark event timestamps. Multiple stimulus types are distinguished by their DigMark codes or keyboard labels. Files should follow BIDS naming for automatic metadata parsing, though this is not required:

```
sub-001_ses-01_task-resting_limb-left_recording.txt
```

### LabChart text export (`.txt`)

Auto-detected by the `Interval=` header. Each recording block is treated as a pre-aligned trial and no trigger channel is required. Both single-file sessions with multiple stimulus blocks and multi-file sessions (one file per condition) are supported.

### KinEMG / NI-DAQ CSV (`.csv`)

Auto-detected by `Author,KinEMG` in the first line. Sampling rate is read from the `Sample Clock Rate` row; NI-DAQ channel names (`Dev1/ai0`, `Dev1/ai1`, etc.) are read from the channel-name row. If a trigger channel is present in the data it can be assigned via the Format Wizard. If no trigger channel is present, analyses proceed on the continuous waveform.

### Generic tab/space/comma delimited files

Any numeric tabular file that is not recognised as one of the above formats opens the Format Wizard on first use. The wizard auto-detects:

- Non-numeric header lines to skip (including embedded metadata such as `Sample Clock Rate,2000.00`)
- Channel name rows, pre-populated as signal name defaults
- Sampling rate from metadata lines matching common patterns

The four wizard steps are:

1. **Data preview and layout** — confirm delimiter, set skip rows, choose column-wise or row-wise orientation
2. **Time axis** — select the time column/row if present, or enter the sampling rate manually
3. **Channel definition** — assign a name, role (EMG / Stim/Trigger / Ignore), and unit to each channel; mini waveform thumbnails and auto-suggested roles assist this step
4. **Summary and save** — review the configuration and save it as a sidecar JSON

Subsequent opens of the same file read the sidecar directly — no wizard interaction needed.

**Row-wise files (e.g. Delsys Trigno):** Files where each row is a channel and each column is a time sample are fully supported. A common pattern is row 0 = continuous TTL trigger signal (~5 V pulses), row 1 = continuous EMG recording. The tool detects TTL rising edges to find stimulus times, then epochs the EMG row around each event. The -0.75 V startup transient common to Delsys Trigno recordings is handled robustly and does not interfere with trigger detection.

---

## Output Files

Results are written to a `derivatives/` folder beside the raw data, following BIDS derivative conventions:

```
study/
├── rawdata/
│   └── sub-001/ses-01/sub-001_ses-01_recording.txt
└── derivatives/
    ├── dataset_session.json               ← file queue and processing status
    ├── study_design.json                  ← Stage 2 design configuration
    ├── group_level_LME_ready.csv          ← merged group output (Stage 2)
    └── sub-001/
        └── ses-01/
            ├── sub-001_ses-01_session.json            ← full session state
            └── results/
                ├── sub-001_ses-01_All_stims_trial_summary.csv
                ├── sub-001_ses-01_All_stims_trial_summary.json
                ├── sub-001_ses-01_ptp_results.csv
                ├── sub-001_ses-01_ptp_results_with_outliers.csv
                └── sub-001_ses-01_<StimType>_trials.csv   ← one per stim type
```

### Trial-level CSV columns

| Column | Description |
|---|---|
| `participant_id` | File or BIDS subject identifier |
| `stim_type` | Stimulus type label as configured in Stage 1a |
| `stim_label` | Custom display label |
| `trial` | Trial index (1-based) |
| `limb` | Limb identifier parsed from filename or entered manually |
| `measure` | Measure label |
| `PTP(mV)` | MEP / CMAP peak-to-peak amplitude |
| `Latency(ms)` | Response onset latency relative to stimulus |
| `AUC(mV·s)` | Area under the rectified EMG curve |
| `cSP_Duration(ms)` | Cortical silent period duration |
| `cSP_MEP_Offset(ms)` | Time from stimulus to cSP onset |
| `cSP_EMG_Return(ms)` | Time from stimulus to EMG return after cSP |
| `MEP_cSP_Ratio` | PTP / cSP duration |
| `Normalised_PTP` | PTP expressed as a proportion of the reference |
| `Reference_Type` | How the reference was computed (Mmax, single-pulse mean, etc.) |
| `Reference_Mean(mV)` | Reference amplitude used |
| `Z_PTP_Within` | Z-score within this stim type |
| `Z_PTP_Pooled` | Z-score pooled across all stim types |
| `PTP_Detrended(mV)` | Linearly detrended PTP amplitude |
| `Outlier_Decision` | Include / Exclude / Reviewed |
| `Manual_Note` | Annotation from the Data Inspector |

### Group-level LME-ready CSV

All trial-level columns from every processed session, plus:

| Column | Description |
|---|---|
| `session` | Session label |
| `task` | Task label (if assigned) |
| `timepoint` | Timepoint label (if assigned) |
| `Stim_Role` | Role assigned in Stage 2 (Reference, Conditioned, M-wave, etc.) |
| `[custom columns]` | Any user-defined between/within-subject factors from Stage 2 |

The output is at the trial level with outlier Z-scores included as covariates, rather than pre-excluding outliers, so that the analyst retains full control of trial-level modelling decisions.

---

## Building from Source

```bash
# Windows
python build_windows.py

# Linux
python3 -m venv venv_linux && source venv_linux/bin/activate
pip install -r requirements.txt
python3 build_linux.py

# macOS
python3 build_mac.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays and signal operations |
| `scipy` | Filtering, statistics, signal processing |
| `pandas` | CSV I/O and data manipulation |
| `matplotlib` | Waveform plotting and interactive figures |
| `pywt` | Wavelet time-frequency display in filter preview |
| `tkinter` | GUI (bundled with standard Python) |

---

## Citation

If you use MEP-CMAP Analyser in published research, please cite:

> Andrushko, J.W. (2026). MEP-CMAP Analyser (Version 0.9.6) [Software].
> Northumbria University. https://github.com/jandrushko/mep-cmap-analyser

---

## References

- Orth, M., & Rothwell, J.C. (2004). The cortical silent period: intrinsic variability and relation to the waveform of the transcranial magnetic stimulation pulse. *Clinical Neurophysiology*, 115(5), 1076–1082.
- Hupfeld, K.E., Swanson, C.W., Fling, B.W., & Seidler, R.D. (2021). TMS-induced silent periods: A review of methods and call for consistency. *Journal of Neuroscience Methods*, 346, 108950.
- Rossini, P.M., et al. (2015). Non-invasive electrical and magnetic stimulation of the brain, spinal cord, roots and peripheral nerves: Basic principles and procedures for routine clinical and research application. *Clinical Neurophysiology*, 126(6), 1071–1107.

---

## License

MIT License — see [LICENSE](https://github.com/jandrushko/mep-cmap-analyser/blob/main/LICENSE) for details.
