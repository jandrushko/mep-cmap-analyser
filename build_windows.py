#!/usr/bin/env python3
"""
build_windows.py
~~~~~~~~~~~~~~~~
Build script for MEP-CMAP Analyser (Windows, package version).
Run from the folder that contains mep_cmap/ and launcher.py.

No admin privileges needed — uses a local virtual environment.

Usage
-----
First run (creates venv):
    python build_windows.py

Subsequent builds (activate venv first):
    venv_windows\\Scripts\\activate
    python build_windows.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run(cmd, description):
    print(f"🔄  {description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌  Failed: {description}")
        return False
    return True


def main():
    print("🔨  Building MEP-CMAP Analyser for Windows...")
    print()

    # ── Check virtual environment ─────────────────────────────────────────────
    # ── Ensure we are running inside the local venv ─────────────────────────
    venv_python = Path("venv_windows") / "Scripts" / "python.exe"

    # If the venv doesn't exist yet, create it and re-launch inside it.
    if not venv_python.exists():
        print("⚠️   Virtual environment not found — creating one...")
        if not run(f'"{sys.executable}" -m venv venv_windows', "Creating venv"):
            return False
        print("✅  Virtual environment created!")
        print()

    # If we're not already running from the venv, re-launch ourselves inside it.
    if sys.executable.lower() != str(venv_python.resolve()).lower():
        print(f"🔄  Re-launching inside venv...")
        import subprocess as _sp
        result = _sp.run([str(venv_python)] + sys.argv)
        sys.exit(result.returncode)

    print("✅  Running in virtual environment")
    print()

    # ── Install dependencies ──────────────────────────────────────────────────
    run(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip")
    if not run(f'"{sys.executable}" -m pip install -r requirements.txt',
               "Installing dependencies"):
        return False

    # ── Verify mep_cmap package is present ───────────────────────────────────
    if not Path("mep_cmap").is_dir():
        print("❌  mep_cmap/ folder not found in current directory.")
        print("    Run this script from the folder containing mep_cmap/")
        return False
    print("✅  mep_cmap/ package found")

    # ── Clean previous build ──────────────────────────────────────────────────
    print()
    print("🧹  Cleaning previous build...")
    for path in ["build", "dist", "__pycache__"]:
        if Path(path).exists():
            shutil.rmtree(path)

    # ── Check for icon ────────────────────────────────────────────────────────
    if not Path("MEP.ico").exists():
        print("⚠️   MEP.ico not found — app will use default icon")
        print("    Edit MEP_CMAP_Windows.spec and remove the icon= line to suppress this")

    # ── Build ─────────────────────────────────────────────────────────────────
    print()
    print("📦  Building with PyInstaller (this takes a few minutes)...")
    print()

    if not run("pyinstaller MEP_CMAP_Windows.spec --clean", "Building application"):
        print()
        print("❌  Build failed — common causes:")
        print("    • PyInstaller not installed: pip install pyinstaller")
        print("    • Missing dependency: check requirements.txt")
        print("    • mep_cmap/ not in current directory")
        return False

    # ── Check output ──────────────────────────────────────────────────────────
    exe = Path('dist') / 'MEP-CMAP Analyser' / 'MEP-CMAP Analyser.exe'
    if exe.exists():
        size_mb = sum(f.stat().st_size for f in
                      Path('dist/MEP-CMAP Analyser').rglob('*')
                      if f.is_file()) / 1_048_576
        print()
        print("✅  Build successful!")
        print()
        print(f"📁  Output folder : dist\\MEP-CMAP Analyser\\")
        print(f"🚀  Executable    : {exe}")
        print(f"📦  Total size    : {size_mb:.0f} MB")
        print()
        print("To test:")
        print('    .\\dist\\"MEP-CMAP Analyser"\\"MEP-CMAP Analyser.exe"')
        print()
        print("To distribute:")
        print('    Zip the entire "dist\\MEP-CMAP Analyser" folder:')
        print('    Compress-Archive -Path "dist\\MEP-CMAP Analyser"'
              ' -DestinationPath MEP-CMAP_Analyser_Windows.zip')
        return True
    else:
        print("❌  Build failed — executable not created")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️   Build cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌  Unexpected error: {e}")
        sys.exit(1)
