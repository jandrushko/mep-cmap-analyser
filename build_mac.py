#!/usr/bin/env python3
"""
build_mac.py — MEP-CMAP Analyser (macOS)

Prerequisites
-------------
  xcode-select --install
  brew install python-tk
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust

Usage
-----
    python3 -m venv venv_mac
    source venv_mac/bin/activate
    pip install -r requirements.txt
    python3 build_mac.py
"""

import os, sys, subprocess, shutil
from pathlib import Path

def run(cmd, desc):
    print(f"🔄  {desc}...")
    return subprocess.run(cmd, shell=True).returncode == 0

def check_rust():
    result = subprocess.run("cargo --version", shell=True,
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅  Rust found: {result.stdout.strip()}")
        return True
    print("❌  Rust not found.")
    print("    Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
    print("    Then restart your shell and re-run this script.")
    return False

def build_rust_extension():
    """Compile mep_cmap_io into the active venv using maturin develop."""
    rust_dir = Path("mep_cmap_io")
    if not rust_dir.is_dir():
        print("❌  mep_cmap_io/ Rust crate not found.")
        return False

    print("🦀  Compiling Rust I/O extension (mep_cmap_io)...")
    if not run(f'"{sys.executable}" -m pip install maturin -q',
               "Installing maturin"):
        return False

    result = subprocess.run(
        f'maturin develop --release --manifest-path mep_cmap_io/Cargo.toml',
        shell=True, cwd=str(Path.cwd()))
    if result.returncode != 0:
        print("⚠️   Rust extension build failed — continuing with Python fallback.")
        print("    File loading will work correctly but may be slower.")
        return True   # non-fatal
    print("✅  mep_cmap_io compiled and installed")
    return True

def main():
    print("🔨  Building MEP-CMAP Analyser for Mac...\n")

    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("⚠️   Activate the virtual environment first:\n")
        print("        python3 -m venv venv_mac")
        print("        source venv_mac/bin/activate")
        print("        pip install -r requirements.txt")
        print("        python3 build_mac.py\n")
        return False

    print("✅  Virtual environment active\n")

    # ── Check Rust ────────────────────────────────────────────────────────────
    if not check_rust():
        return False
    print()

    if not run(f'"{sys.executable}" -m pip install --upgrade pip -q', "Upgrading pip"):
        return False
    if not run(f'"{sys.executable}" -m pip install -r requirements.txt', "Installing deps"):
        return False

    # ── Compile Rust extension ────────────────────────────────────────────────
    print()
    if not build_rust_extension():
        return False
    print()

    if not Path("mep_cmap").is_dir():
        print("❌  mep_cmap/ not found — run from the project root"); return False

    print("🧹  Cleaning...")
    for p in ["build","dist","__pycache__"]:
        if Path(p).exists(): shutil.rmtree(p)

    print("\n📦  Running PyInstaller...")
    if not run("pyinstaller MEP_CMAP_Mac.spec --clean", "Building"):
        print("\n💡  If failed: brew install python-tk")
        return False

    app = Path("dist/MEP-CMAP Analyser.app")
    exe = Path("dist/MEP-CMAP Analyser/MEP-CMAP Analyser")
    output = app if app.exists() else exe
    if output.exists():
        mb = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())/1e6
        print(f"\n✅  Done!  ({mb:.0f} MB)")
        print(f"🚀  {output}")
        print('\nTo distribute:')
        print('    zip -r MEP-CMAP_Analyser_Mac.zip "dist/MEP-CMAP Analyser"')
        return True
    print("❌  Executable not created"); return False

if __name__ == "__main__":
    try: sys.exit(0 if main() else 1)
    except KeyboardInterrupt: print("\n⚠️  Cancelled"); sys.exit(1)
