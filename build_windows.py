#!/usr/bin/env python3
"""
build_windows.py
~~~~~~~~~~~~~~~~
Build script for MEP-CMAP Analyser (Windows, package version).
Run from the folder that contains mep_cmap/ and launcher.py.

No admin privileges needed — uses a local virtual environment.

Prerequisites
-------------
  * Python 3.9+ (python.org installer)
  * Rust toolchain  https://rustup.rs  (one-time install)

Usage
-----
    python build_windows.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

VENV = Path("venv_windows")
VENV_PY = VENV / "Scripts" / "python.exe"


def run(cmd, description):
    print(f"🔄  {description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌  Failed: {description}")
        return False
    return True


def pip(*args):
    """Run pip via the venv Python — avoids broken .exe launchers entirely."""
    return f'"{VENV_PY}" -m pip {" ".join(args)}'


def pyinstaller(*args):
    """Run pyinstaller via the venv Python."""
    return f'"{VENV_PY}" -m PyInstaller {" ".join(args)}'


def maturin(*args):
    """Run maturin via the venv Python."""
    return f'"{VENV_PY}" -m maturin {" ".join(args)}'


def check_rust():
    result = subprocess.run("cargo --version", shell=True,
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅  Rust found: {result.stdout.strip()}")
        return True
    print("❌  Rust not found.")
    print("    Install from https://rustup.rs then re-run this script.")
    return False


def ensure_venv():
    """
    Always recreate the venv from the Python running this script.
    This guarantees the venv's internal paths match the current machine —
    stale venvs from a previous install location are silently replaced.
    """
    if VENV.exists():
        # Quick sanity-check: try python --version via -m to detect a broken venv
        probe = subprocess.run(f'"{VENV_PY}" --version',
                               shell=True, capture_output=True)
        if probe.returncode != 0:
            print("⚠️   Existing venv is broken — deleting and recreating...")
            shutil.rmtree(VENV)
        else:
            # Venv Python itself works. Check pip works too.
            probe2 = subprocess.run(f'"{VENV_PY}" -m pip --version',
                                    shell=True, capture_output=True)
            if probe2.returncode != 0:
                print("⚠️   Venv pip is broken — deleting and recreating...")
                shutil.rmtree(VENV)
            else:
                print("✅  Virtual environment OK")
                return True

    if not VENV.exists():
        print("⚠️   Creating virtual environment...")
        if not run(f'"{sys.executable}" -m venv "{VENV}"', "Creating venv"):
            return False
        print("✅  Virtual environment created!")

    return True


def build_rust_extension():
    """
    Compile mep_cmap_io via  maturin build --release  then install the
    resulting wheel into the venv.  Uses python -m maturin so no broken
    .exe launcher is involved.
    """
    rust_dir = Path("mep_cmap_io")
    if not rust_dir.is_dir():
        print("⚠️   mep_cmap_io/ Rust crate not found — skipping.")
        print("     File loading will work correctly but may be slower.")
        return True  # non-fatal

    print("🦀  Compiling Rust I/O extension (mep_cmap_io)...")

    if not run(pip("install maturin -q"), "Installing maturin"):
        print("⚠️   Could not install maturin — continuing with Python fallback.")
        return True  # non-fatal

    wheel_dir = Path("mep_cmap_io") / "target" / "wheels"
    result = subprocess.run(
        maturin(f'build --release --manifest-path mep_cmap_io\\Cargo.toml '
                f'--out "{wheel_dir}"'),
        shell=True, cwd=str(Path.cwd()))
    if result.returncode != 0:
        print("⚠️   Rust build failed — continuing with Python fallback.")
        return True  # non-fatal

    wheels = list(wheel_dir.glob("mep_cmap_io-*.whl"))
    if not wheels:
        print("⚠️   Wheel not found after build — continuing with Python fallback.")
        return True  # non-fatal

    wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    if not run(pip(f'install "{wheel}" --force-reinstall -q'),
               f"Installing {wheel.name}"):
        print("⚠️   Wheel install failed — continuing with Python fallback.")
        return True  # non-fatal

    print(f"✅  mep_cmap_io compiled and installed ({wheel.name})")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔨  Building MEP-CMAP Analyser for Windows...")
    print()

    if not ensure_venv():
        return False

    if not check_rust():
        return False
    print()

    run(pip("install --upgrade pip -q"), "Upgrading pip")
    if not run(pip("install -r requirements.txt"), "Installing dependencies"):
        return False

    print()
    if not build_rust_extension():
        return False
    print()

    if not Path("mep_cmap").is_dir():
        print("❌  mep_cmap/ folder not found.")
        print("    Run this script from the folder containing mep_cmap/")
        return False
    print("✅  mep_cmap/ package found")

    print()
    print("🧹  Cleaning previous build...")
    for path in ["build", "dist", "__pycache__"]:
        if Path(path).exists():
            shutil.rmtree(path)

    if not Path("MEP.ico").exists():
        print("⚠️   MEP.ico not found — app will use default icon")

    print()
    print("📦  Building with PyInstaller (this takes a few minutes)...")
    print()

    if not run(pyinstaller("MEP_CMAP_Windows.spec --clean"),
               "Building application"):
        print()
        print("❌  Build failed — common causes:")
        print("    * Missing dependency in requirements.txt")
        print("    * mep_cmap/ not in current directory")
        return False

    # Search for any .exe in dist/ — the spec may include a version number in
    # the name (e.g. "MEP-CMAP Analyser v0.9.0.exe"), so don't hardcode it.
    dist_root = Path('dist')
    exe_candidates = list(dist_root.rglob('*.exe')) if dist_root.exists() else []
    exe = exe_candidates[0] if exe_candidates else None
    dist_folder = exe.parent if exe else dist_root / 'MEP-CMAP Analyser'

    if exe and exe.exists():
        size_mb = sum(f.stat().st_size for f in dist_folder.rglob('*')
                      if f.is_file()) / 1_048_576
        print()
        print("✅  Build successful!")
        print()
        print(f"📁  Output folder : {dist_folder}")
        print(f"🚀  Executable    : {exe}")
        print(f"📦  Total size    : {size_mb:.0f} MB")
        print()
        print("To test:")
        print(f'    .\\"dist\\{dist_folder.name}\\{exe.name}"')
        print()
        print("To distribute:")
        print(f'    Compress-Archive -Path "dist\\{dist_folder.name}"'
              f' -DestinationPath MEP-CMAP_Analyser_Windows.zip')
        return True
    else:
        print("❌  Build failed — no .exe found in dist/")
        print("    Check the PyInstaller output above for errors.")
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
