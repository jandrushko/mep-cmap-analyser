# MEP-CMAP Analyser

**Version 0.8.4 | May 2026**  
*Author: Justin Andrushko PhD, Northumbria University*

A BIDS-compliant, open-source tool for processing and quantifying TMS-evoked motor evoked potentials (MEPs) and cortical silent periods (cSPs) from Spike-2 EMG recordings.

---

## Features

- **BIDS-compliant** derivatives output with automatic participant/session naming
- **Per-trial MEP quantification** — PTP amplitude, onset latency, AUC
- **Cortical silent period (cSP) detection** using a bootstrap threshold method
- **M-wave normalisation** with plateau-based Mmax detection
- **Paired-pulse ratios** (SICI, ICF, etc.) with flexible reference assignment
- **Data Inspector** — interactive per-trial review with draggable markers
- **Filter preview** — real-time frequency response and wavelet time-frequency display
- **Outlier detection** with interactive review
- **Multi-channel support** — additional EMG and force channels for visual inspection
- **Session save/load** — resume analysis across sessions
- **LME-ready trial-level CSV output** with Z-scores, detrended values, and pooled statistics

---

## Installation

### Option 1: pip (recommended for Python users)

```bash
pip install mep-cmap-analyser
mep-cmap
```

Python 3.9+ required. Tkinter must be available:
- **Windows/Mac**: included with standard Python
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
python MEP_CMAP_Analyser.py
```

---

## Quick start

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
    input_path   = "path/to/recording.txt",
    pre_ms       = 20,
    post_ms      = 400,
    ptp_start    = 10,
    ptp_end      = 50,
    prestim_ms   = 100,
    bootstrap_iter = 1000,
    marker_name  = "Keyboard",
)
```

---

## Input format

The tool reads **Spike-2 text exports** (`.txt`) containing:
- Waveform channels (EMG, force, etc.)
- DigMark event timestamps

Files should follow BIDS naming conventions for automatic metadata parsing:
```
sub-001_ses-01_limb-left_recording.txt
```

---

## Output

Results are saved to a `derivatives/` folder alongside the input file, following BIDS structure:

```
derivatives/
└── sub-001/
    └── ses-01/
        ├── sub-001_ses-01_limb-left_All_stims_trial_summary.csv
        ├── sub-001_ses-01_limb-left_ptp_results.csv
        ├── sub-001_ses-01_limb-left_bootstrap_comparisons.csv
        └── figures/
```

### Trial-level CSV columns

| Column | Description |
|--------|-------------|
| `PTP(mV)` | MEP peak-to-peak amplitude |
| `Latency(ms)` | MEP onset latency |
| `cSP_Duration(ms)` | Cortical silent period duration |
| `cSP_MEP_Offset(ms)` | Time of MEP offset (cSP onset) relative to stim |
| `cSP_EMG_Return(ms)` | Time of EMG return (cSP offset) relative to stim |
| `MEP_cSP_Ratio` | PTP / cSP duration (Orth & Rothwell, 2004) |
| `Normalised_PTP` | PTP / reference mean (Mmax or SP reference) |
| `Z_PTP_Pooled` | Z-score pooled across all conditions |
| `PTP_Detrended(mV)` | Linearly detrended PTP amplitude |

---

## Building from source (developers)

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

> Andrushko, J.W. (2026). MEP-CMAP Analyser (Version 0.8.4) [Software].
> Northumbria University. https://doi.org/10.5281/zenodo.XXXXXXX

---

## References

- Orth, M., & Rothwell, J.C. (2004). The cortical silent period: intrinsic variability and relation to the waveform of the transcranial magnetic stimulation pulse. *Clinical Neurophysiology*, 115(5), 1076–1082.
- Hupfeld, K.E., Swanson, C.W., Fling, B.W., & Seidler, R.D. (2021). TMS-induced silent periods: A review of methods and call for consistency. *Journal of Neuroscience Methods*, 346, 108950.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
