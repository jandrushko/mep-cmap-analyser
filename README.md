# MEP-CMAP Analyser

**Version 0.9.3 | May 2026**  
*Author: Justin Andrushko PhD, Northumbria University*

[![PyPI version](https://badge.fury.io/py/mep-cmap-analyser.svg)](https://pypi.org/project/mep-cmap-analyser/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A BIDS-compliant, open-source tool for processing and quantifying TMS-evoked motor evoked potentials (MEPs) and cortical silent periods (cSPs) from Spike-2 and LabChart EMG recordings.

---

## Features

- **Dataset Setup** — multi-file session management with a persistent file queue, BIDS folder auto-detection, processing status tracking, and excluded-file management across sessions
- **BIDS-compliant** derivatives output with automatic participant/session naming
- **Per-trial MEP quantification** — PTP amplitude, onset latency, AUC, cSP duration, paired-pulse ratios
- **Physiologically-bounded MEP onset detection** — peak-anchored backward scan with per-stim-type latency profiles derived from published normative data (TMS: hand/FDI 18–30 ms, vastus lateralis 18–35 ms, leg 28–45 ms; peripheral nerve: upper limb 2–12 ms, lower limb 4–18 ms)
- **Cortical silent period (cSP) detection** using a vectorised bootstrap threshold method with MEP-anchored search cap
- **Auto AUC** — onset-to-cSP start computed automatically for all event types; user can override via drag selector
- **Auto SP and AUC for unreviewed segments** — silent period and AUC computed in the pipeline for any segment not manually reviewed in the Data Inspector
- **M-wave normalisation** with plateau-based Mmax detection
- **Paired-pulse ratios** (SICI, ICF, etc.) with flexible per-stim reference assignment in Stage 1a
- **Data Inspector** — interactive per-trial review with draggable markers, zoom toolbar, AUC selector, silent period annotation; all edits persist across reruns
- **Filter preview** — real-time frequency response and wavelet time-frequency display
- **Outlier detection** with interactive review and Z-score thresholding
- **Multi-channel support** — additional EMG and force channels for visual inspection
- **Full session persistence** — every adjustable setting (filters, onset detection, CSP, latency profiles, normalisation, inspector edits, analysis options) is saved and restored automatically per file
- **Group analysis** — Stage 2 merges all processed sessions into a single LME-ready CSV with study design columns (between/within-subject factors, stim roles)
- **LME-ready trial-level CSV output** with Z-scores, detrended values, normalised PTP, and pooled statistics
- **Cross-platform** — Windows, macOS, and Linux supported

---

## Installation

### Option 1: pip (recommended for Python users)

```bash
pip install mep-cmap-analyser
mep-cmap
```

Python 3.9+ required. Tkinter must be available:

- **Windows / macOS**: included with standard Python
- **Linux (Ubuntu/Debian)**: `sudo apt install python3-tk`

### Option 2: Compiled binaries (no Python required)

Download the latest release for your platform from the [Releases page](https://github.com/justinAndrushko/mep-cmap-analyser/releases):

| Platform | File |
|----------|------|
| Windows  | `MEP-CMAP_Analyser_Windows.zip` |
| macOS    | `MEP-CMAP_Analyser_Mac.zip` |
| Linux    | `MEP-CMAP_Analyser_Linux.tar.gz` |

Unzip and run the executable — no installation required.

### Option 3: Run from source

```bash
git clone https://github.com/justinAndrushko/mep-cmap-analyser.git
cd mep-cmap-analyser
pip install -r requirements.txt
python -m mep_cmap
```

---

## Quick Start

```bash
# Launch the GUI
mep-cmap

# Or from Python
from mep_cmap import run_app
run_app()
```

### Scripted pipeline (no GUI)

```python
from mep_cmap import run_pipeline, PipelineConfig

run_pipeline(
    input_path  = "path/to/recording.txt",
    pre_ms      = 20,
    post_ms     = 400,
    ptp_start   = 10,
    ptp_end     = 50,
    prestim_ms  = 100,
    marker_name = "Keyboard",
)
```

---

## Workflow

The tool is organised into four tabs:

### Dataset Setup

Load a study folder or individual files. The tool auto-detects `rawdata/` and `derivatives/` subfolders (BIDS structure) or accepts manual folder selection. The derivatives folder is always placed beside `rawdata/`, never inside it. A persistent file queue tracks processing status across sessions — files are marked Not started, In progress, Needs review, Complete, or Stale.

- Double-click any file to load it
- Use **Run all unprocessed** to batch process
- Use **Refresh** to detect newly added files (previously excluded files are remembered and not re-added)
- Right-click → **Show excluded files** to restore previously removed files
- Queue state is saved to `dataset_session.json` in the derivatives folder

### Stage 1a — Labels & Analysis Setup

Appears automatically after a file is loaded. Configure per-stim-type settings:

- Label, colour, gap (ms), CSP detection toggle
- Stimulation type and muscle group (sets physiological latency bounds for onset detection)
- Internal normalisation references and plateau tolerance for Mmax detection

Confirming setup auto-switches to Stage 1b. Settings persist between files.

### Stage 1b — Single File Processing

Filter settings (bandpass, notch, mains noise canceller), time window and MEP onset detection parameters, CSP detection settings, outlier review, and analysis options. All settings are saved per file and restored on reload.

When reloading a previously processed file:

- Option to reuse the saved data range, select a new range, or use the whole file
- Previously adjusted marker positions, notes, exclusions, and AUC windows are restored and carried through the re-analysis

### Stage 2 — Group Analysis & LME Setup

Scan the derivatives folder to discover all processed sessions. Assign study design columns (e.g. Group, Condition, Timepoint), configure stim type roles (Reference, Conditioned, M-wave), and click **Build group analysis file** to merge all selected sessions into a single `group_level_LME_ready.csv`.

---

## Input Formats

### Spike-2 text export (`.txt`)

Files contain waveform channels (EMG, force, etc.) and DigMark event timestamps. Files should follow BIDS naming conventions for automatic metadata parsing:

```
sub-001_ses-01_limb-left_recording.txt
```

### LabChart text export (`.txt`)

LabChart exports are auto-detected by their `Interval=` header. Each recording block is treated as a trial — no DigMark channel is required as LabChart pre-aligns each block to the stimulation event.

Both single-file recordings with multiple stimulus types and multi-file sessions (one file per condition) are supported.

---

## Output

Results are saved to a `derivatives/` folder beside the raw data folder, following BIDS structure:

```
study/
├── rawdata/
│   └── sub-001/ses-01/...
└── derivatives/
    ├── dataset_session.json               ← queue state, processing status, excluded files
    ├── group_level_LME_ready.csv          ← merged group-level output (Stage 2)
    └── sub-001/
        └── ses-01/
            ├── results/
            │   ├── sub-001_ses-01_limb-left_All_stims_trial_summary.csv
            │   ├── sub-001_ses-01_limb-left_All_stims_trial_summary.json
            │   ├── sub-001_ses-01_limb-left_ptp_results.csv
            │   └── sub-001_ses-01_limb-left_ptp_results_with_outliers.csv
            ├── figures/
            └── sub-001_ses-01_limb-left_session.json  ← per-file session state
```

### Trial-level CSV columns

| Column | Description |
|--------|-------------|
| `PTP(mV)` | MEP peak-to-peak amplitude |
| `Latency(ms)` | MEP onset latency |
| `cSP_Duration(ms)` | Cortical silent period duration |
| `cSP_MEP_Offset(ms)` | Time from stim to cSP onset |
| `cSP_EMG_Return(ms)` | Time from stim to EMG return after cSP |
| `MEP_cSP_Ratio` | PTP / cSP duration (Orth & Rothwell, 2004) |
| `AUC(mV*s)` | Area under the rectified EMG curve (onset to cSP start) |
| `Normalised_PTP` | PTP / reference mean (Mmax or single-pulse reference) |
| `Reference_Type` | How the reference mean was computed |
| `Reference_Mean(mV)` | Reference amplitude used for normalisation |
| `Z_PTP_Within` | Z-score within stim type |
| `Z_PTP_Pooled` | Z-score pooled across all conditions |
| `PTP_Detrended(mV)` | Linearly detrended PTP amplitude |
| `Outlier_Decision` | Include / Exclude / Reviewed |
| `Manual_Note` | Annotator note from Data Inspector |

### Group-level CSV additional columns

| Column | Description |
|--------|-------------|
| `participant_id` | BIDS participant identifier |
| `session` | Session label |
| `task` | Task label (if assigned) |
| `timepoint` | Timepoint label (if assigned) |
| `Limb` | Limb identifier parsed from filename |
| `Stim_Role` | Role assigned in Stage 2 (Reference, Conditioned, M-wave, etc.) |
| `[custom columns]` | User-defined between/within-subject factors |

---

## Building from Source (Developers)

```bash
# Windows
python build_windows.py

# Linux
python3 -m venv venv_linux
source venv_linux/bin/activate
pip install -r requirements.txt
python3 build_linux.py

# macOS
python3 build_mac.py
```

---

## Citation

If you use this tool in your research, please cite:

> Andrushko, J.W. (2026). MEP-CMAP Analyser (Version 0.9.3) [Software].
> Northumbria University. https://doi.org/10.5281/zenodo.XXXXXXX

---

## References

- Orth, M., & Rothwell, J.C. (2004). The cortical silent period: intrinsic variability and relation to the waveform of the transcranial magnetic stimulation pulse. *Clinical Neurophysiology*, 115(5), 1076–1082.
- Hupfeld, K.E., Swanson, C.W., Fling, B.W., & Seidler, R.D. (2021). TMS-induced silent periods: A review of methods and call for consistency. *Journal of Neuroscience Methods*, 346, 108950.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
