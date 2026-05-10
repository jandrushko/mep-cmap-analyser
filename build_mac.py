#!/usr/bin/env python3
"""
build_mac.py — MEP-CMAP Analyser (Mac)

Prerequisites:
    sudo apt install python3 python3-venv python3-tk binutils

Usage:
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

    if not run(f'"{sys.executable}" -m pip install --upgrade pip -q', "Upgrading pip"):
        return False
    if not run(f'"{sys.executable}" -m pip install -r requirements.txt', "Installing deps"):
        return False
    if not Path("mep_cmap").is_dir():
        print("❌  mep_cmap/ not found — run from the project root"); return False

    print("\n🧹  Cleaning...")
    for p in ["build","dist","__pycache__"]:
        if Path(p).exists(): shutil.rmtree(p)

    print("\n📦  Running PyInstaller...")
    if not run("pyinstaller MEP_CMAP_Mac.spec --clean", "Building"):
        print("\n💡  If failed: sudo apt install python3-tk tk-dev binutils")
        return False

    exe = Path("dist/MEP-CMAP Analyser/MEP-CMAP Analyser")
    if exe.exists():
        exe.chmod(0o755)
        mb = sum(f.stat().st_size for f in Path("dist/MEP-CMAP Analyser").rglob("*") if f.is_file())/1e6
        print(f"\n✅  Done!  ({mb:.0f} MB)")
        print(f"🚀  {exe}")
        print('\nTo distribute:')
        print('    tar -czf MEP-CMAP_Analyser_Mac.tar.gz -C dist "MEP-CMAP Analyser"')
        return True
    print("❌  Executable not created"); return False

if __name__ == "__main__":
    try: sys.exit(0 if main() else 1)
    except KeyboardInterrupt: print("\n⚠️  Cancelled"); sys.exit(1)
