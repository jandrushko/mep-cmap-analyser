"""
MEP_CMAP_Analyser.py
~~~~~~~~~~~~~~~~~~~~
Backward-compatible entry point.  All logic now lives in the mep_cmap package.
Run via:
    python MEP_CMAP_Analyser.py
    python -m mep_cmap
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from mep_cmap import run_app

if __name__ == "__main__":
    run_app()
