# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MEP-CMAP Analyser (Windows, package version)
Entry point: launcher.py  →  mep_cmap package
Uses --onedir (fast startup, reliable)
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

block_cipher = None

# ── Data files ────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('pywt')
datas += collect_data_files('statsmodels')   # EMG excitability compensation (QR)

# Include the entire mep_cmap package as data so all .py modules are bundled
datas += [('mep_cmap', 'mep_cmap')]
# BIDS-ify (BEP037) schema asset - explicit, in case the whole-package
# data bundling above ever changes. The app hard-depends on this JSON.
datas += [('mep_cmap/schema/nibs_bep037.json', 'mep_cmap/schema')]
# Built-in add-ons (mepfeatx, rectified_area, __init__) - explicit so they
# ship even if the whole-package bundling above ever changes. The add-on
# loader discovers these by file path at runtime.
datas += [('mep_cmap/add_ons', 'mep_cmap/add_ons')]

# ── Hidden imports ────────────────────────────────────────────────────────────
hiddenimports = []

# mep_cmap package — all submodules
hiddenimports += collect_submodules('mep_cmap')

# Scientific stack
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('pandas')
hiddenimports += collect_submodules('pywt')
hiddenimports += collect_submodules('mpl_toolkits')
hiddenimports += collect_submodules('pyedflib')   # BIDS-ify EDF/BDF writer (C ext)
hiddenimports += collect_submodules('statsmodels') # EMG excitability compensation (QR)
hiddenimports += collect_submodules('patsy')       # statsmodels formula dependency

# Explicit extras that PyInstaller sometimes misses
hiddenimports += [
    'scipy.signal',
    'scipy.signal.windows',
    'scipy.stats',
    'scipy.optimize',
    'scipy.fft',
    'scipy.interpolate',   # MEPFeatX add-on: CubicSpline upsampling
    'statsmodels.api',
    'statsmodels.regression.quantile_regression',
    'statsmodels.tools',
    'patsy',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
    'mpl_toolkits.axes_grid1',
    'mpl_toolkits.axes_grid1.axes_divider',
    'PIL._tkinter_finder',
    'pywt',
    'tkinter',
    'tkinter.ttk',
    'tkinter.scrolledtext',
    'tkinter.font',
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ['launcher.py', 'splash_screen.py'],
    pathex=['.'],               # current directory (contains mep_cmap/)
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # --onedir (faster startup than --onefile)
    name='MEP-CMAP Analyser v1.2.8',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,              # no console window for GUI app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='MEP.ico',             # replace with your .ico file, or remove this line
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MEP-CMAP Analyser v1.2.8',
)
