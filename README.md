# MEP-CMAP Analyser

**Version 0.9.1 | May 2026**  
*Author: Justin Andrushko PhD, Northumbria University*

A BIDS-compliant, open-source tool for processing and quantifying TMS-evoked motor evoked potentials (MEPs) and cortical silent periods (cSPs) from Spike-2 EMG recordings.

\---

## Features

* **Dataset Setup** — multi-file session management with a persistent file queue, BIDS folder auto-detection, processing status tracking, and excluded-file management across sessions
* **BIDS-compliant** derivatives output with automatic participant/session naming
* **Per-trial MEP quantification** — PTP amplitude, onset latency, AUC, cSP duration, paired-pulse ratios
* **Physiologically-bounded MEP onset detection** — peak-anchored backward scan with per-stim-type latency profiles derived from published normative data (TMS: hand/FDI 18–30 ms, vastus lateralis 18–35 ms, leg 28–45 ms; peripheral nerve: upper limb 2–12 ms, lower limb 4–18 ms)
* **Cortical silent period (cSP) detection** using a vectorised bootstrap threshold method with MEP-anchored search cap
* **Auto AUC** — onset-to-cSP start computed automatically for all event types; user can override via drag selector
* **Auto SP and AUC for unreviewed segments** — silent period and AUC computed in the pipeline for any segment not manually reviewed in the Data Inspector
* **M-wave normalisation** with plateau-based Mmax detection
* **Paired-pulse ratios** (SICI, ICF, etc.) with flexible per-stim reference assignment in Stage 1a
* **Data Inspector** — interactive per-trial review with draggable markers, zoom toolbar, AUC selector, silent period annotation; all edits persist across reruns
* **Filter preview** — real-time frequency response and wavelet time-frequency display
* **Outlier detection** with interactive review and Z-score thresholding
* **Multi-channel support** — additional EMG and force channels for visual inspection
* **Full session persistence** — every adjustable setting (filters, onset detection, CSP, latency profiles, normalisation, inspector edits, analysis options) is saved and restored automatically per file
* **Group analysis** — Stage 2 merges all processed sessions into a single LME-ready CSV with study design columns (between/within-subject factors, stim roles)
* **LME-ready trial-level CSV output** with Z-scores, detrended values, normalised PTP, and pooled statistics
* **Cross-platform** — Windows, macOS, and Linux supported

\---

## Workflow

The tool is organised into four tabs:

### Dataset Setup

Load a study folder or individual files. The tool auto-detects `rawdata/` and `derivatives/` subfolders (BIDS structure) or accepts manual folder selection. The derivatives folder is always placed beside `rawdata/`, never inside it. A persistent file queue tracks processing status across sessions — files are marked Not started, In progress, Needs review, Complete, or Stale.

* Double-click any file to load it
* Use **Run all unprocessed** to batch process
* Use **Refresh** to detect newly added files (previously excluded files are remembered and not re-added)
* Right-click → **Show excluded files** to restore previously removed files
* Queue state is saved to `dataset\\\_session.json` in the derivatives folder

### Stage 1a — Labels \& Analysis Setup

Appears automatically after a file is loaded. Configure per-stim-type settings:

* Label, colour, gap (ms), CSP detection toggle
* Stimulation type and muscle group (sets physiological latency bounds for onset detection)
* Internal normalisation references and plateau tolerance for Mmax detection

Confirming setup auto-switches to Stage 1b. Settings persist between files.

### Stage 1b — Single File Processing

Filter settings (bandpass, notch, mains noise canceller), time window and MEP onset detection parameters, CSP detection settings, outlier review, and analysis options. All settings are saved per file and restored on reload.

When reloading a previously processed file:

* Option to reuse the saved data range, select a new range, or use the whole file
* Previously adjusted marker positions, notes, exclusions, and AUC windows are restored and carried through the re-analysis

### Stage 2 — Group Analysis LME Setup

Scan the derivatives folder to discover all processed sessions. Assign study design columns (e.g. Group, Condition, Timepoint), configure stim type roles (Reference, Conditioned, M-wave), and click **Build group analysis file** to merge all selected sessions into a single `group\\\_level\\\_LME\\\_ready.csv`.

\---

## Installation

### Option 1: pip (recommended for Python users)

```bash
pip install mep-cmap-analyser
mep-cmap
```

Python 3.9+ required. Tkinter must be available:

* **Windows/Mac**: included with standard Python
* **Linux (Ubuntu/Debian)**: `sudo apt install python3-tk`

### Option 2: Compiled binaries (no Python required)

Download the latest release for your platform from the [Releases page](https://github.com/justinAndrushko/mep-cmap-analyser/releases):

|Platform|File|
|-|-|
|Windows|`MEP-CMAP\\\_Analyser\\\_Windows.zip`|
|macOS|`MEP-CMAP\\\_Analyser\\\_Mac.zip`|
|Linux|`MEP-CMAP\\\_Analyser\\\_Linux.tar.gz`|

Unzip and run the executable — no installation required.

### Option 3: Run from source

```bash
git clone https://github.com/justinAndrushko/mep-cmap-analyser.git
cd mep-cmap-analyser
pip install -r requirements.txt
python MEP\\\_CMAP\\\_Analyser.py
```

\---

## Quick start

```bash
# Launch the GUI
mep-cmap

# Or from Python
from mep\\\_cmap import run\\\_app
run\\\_app()
```

### Scripted pipeline (no GUI)

```python
from mep\\\_cmap import run\\\_pipeline, PipelineConfig

run\\\_pipeline(
    input\\\_path   = "path/to/recording.txt",
    pre\\\_ms       = 20,
    post\\\_ms      = 400,
    ptp\\\_start    = 10,
    ptp\\\_end      = 50,
    prestim\\\_ms   = 100,
    marker\\\_name  = "Keyboard",
)
```

\---

## Input format

The tool reads **Spike-2 text exports** (`.txt`) containing:

* Waveform channels (EMG, force, etc.)
* DigMark event timestamps

Files should follow BIDS naming conventions for automatic metadata parsing:

```
sub-001\\\_ses-01\\\_limb-left\\\_recording.txt
```

Single-file recordings with multiple stimulus types and multi-file sessions (one file per condition) are both supported.

\---

## Output

Results are saved to a `derivatives/` folder beside (not inside) the raw data folder, following BIDS structure:

```
study/
├── rawdata/
│   └── sub-001/ses-01/...
└── derivatives/
    ├── dataset\\\_session.json               ← queue state, processing status, excluded files
    ├── group\\\_level\\\_LME\\\_ready.csv          ← merged group-level output (Stage 2)
    └── sub-001/
        └── ses-01/
            ├── results/
            │   ├── sub-001\\\_ses-01\\\_limb-left\\\_All\\\_stims\\\_trial\\\_summary.csv
            │   ├── sub-001\\\_ses-01\\\_limb-left\\\_All\\\_stims\\\_trial\\\_summary.json
            │   ├── sub-001\\\_ses-01\\\_limb-left\\\_ptp\\\_results.csv
            │   └── sub-001\\\_ses-01\\\_limb-left\\\_ptp\\\_results\\\_with\\\_outliers.csv
            ├── figures/
            └── sub-001\\\_ses-01\\\_limb-left\\\_session.json  ← per-file session state
```

### Trial-level CSV columns

|Column|Description|
|-|-|
|`PTP(mV)`|MEP peak-to-peak amplitude|
|`Latency(ms)`|MEP onset latency|
|`cSP\\\_Duration(ms)`|Cortical silent period duration|
|`cSP\\\_MEP\\\_Offset(ms)`|Time from stim to cSP onset|
|`cSP\\\_EMG\\\_Return(ms)`|Time from stim to EMG return after cSP|
|`MEP\\\_cSP\\\_Ratio`|PTP / cSP duration (Orth \& Rothwell, 2004)|
|`AUC(mV\\\*s)`|Area under the rectified EMG curve (onset to cSP start)|
|`Normalised\\\_PTP`|PTP / reference mean (Mmax or single-pulse reference)|
|`Reference\\\_Type`|How the reference mean was computed|
|`Reference\\\_Mean(mV)`|Reference amplitude used for normalisation|
|`Z\\\_PTP\\\_Within`|Z-score within stim type|
|`Z\\\_PTP\\\_Pooled`|Z-score pooled across all conditions|
|`PTP\\\_Detrended(mV)`|Linearly detrended PTP amplitude|
|`Outlier\\\_Decision`|Include / Exclude / Reviewed|
|`Manual\\\_Note`|Annotator note from Data Inspector|

### Group-level CSV additional columns

|Column|Description|
|-|-|
|`participant\\\_id`|BIDS participant identifier|
|`session`|Session label|
|`task`|Task label (if assigned)|
|`timepoint`|Timepoint label (if assigned)|
|`Limb`|Limb identifier parsed from filename|
|`Stim\\\_Role`|Role assigned in Stage 2 (Reference, Conditioned, M-wave, etc.)|
|`\\\[custom columns]`|User-defined between/within-subject factors|

\---

## Building from source (developers)

```bash
# Windows
python build\\\_windows.py

# Linux
python3 -m venv venv\\\_linux
source venv\\\_linux/bin/activate
pip install -r requirements.txt
python3 build\\\_linux.py

# macOS
python3 build\\\_mac.py
```

\---

## Citation

If you use this tool in your research, please cite:

> Andrushko, J.W. (2026). MEP-CMAP Analyser (Version 0.9.1) \\\[Software].
> Northumbria University. https://doi.org/10.5281/zenodo.XXXXXXX

\---

## References

* Orth, M., \& Rothwell, J.C. (2004). The cortical silent period: intrinsic variability and relation to the waveform of the transcranial magnetic stimulation pulse. *Clinical Neurophysiology*, 115(5), 1076–1082.
* Hupfeld, K.E., Swanson, C.W., Fling, B.W., \& Seidler, R.D. (2021). TMS-induced silent periods: A review of methods and call for consistency. *Journal of Neuroscience Methods*, 346, 108950.

\---

## License

MIT License — see [LICENSE](LICENSE) for details.

