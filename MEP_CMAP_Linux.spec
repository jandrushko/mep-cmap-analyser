# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MEP-CMAP Analyser (Linux)
Produces a --onedir binary (fast startup, reliable)
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('pywt')
datas += [('mep_cmap', 'mep_cmap')]

hiddenimports = []
hiddenimports += collect_submodules('mep_cmap')
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('pandas')
hiddenimports += collect_submodules('pywt')
hiddenimports += collect_submodules('mpl_toolkits')
hiddenimports += [
    'scipy.signal',
    'scipy.signal.windows',
    'scipy.stats',
    'scipy.optimize',
    'scipy.fft',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
    'mpl_toolkits.axes_grid1',
    'PIL._tkinter_finder',
    'pywt',
    'tkinter',
    'tkinter.ttk',
    'tkinter.scrolledtext',
    'tkinter.font',
]

a = Analysis(
    ['launcher.py', 'splash_screen.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MEP-CMAP Analyser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MEP-CMAP Analyser',
)
