# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MEP-CMAP Analyser (macOS)
Produces a .app bundle via --onedir approach (reliable, fast startup)
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Data files ────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('pywt')
datas += collect_data_files('statsmodels')   # EMG excitability compensation (QR)
datas += [('mep_cmap', 'mep_cmap')]   # bundle the entire package
# BIDS-ify (BEP037) schema asset - explicit, in case the whole-package
# data bundling above ever changes. The app hard-depends on this JSON.
datas += [('mep_cmap/schema/nibs_bep037.json', 'mep_cmap/schema')]
datas += [('mep_cmap/assets', 'mep_cmap/assets')]   # TMSMultiLab mark
# Built-in add-ons (mepfeatx, rectified_area, __init__) - explicit so they
# ship even if the whole-package bundling above ever changes. The add-on
# loader discovers these by file path at runtime.
datas += [('mep_cmap/add_ons', 'mep_cmap/add_ons')]

# ── Hidden imports ────────────────────────────────────────────────────────────
hiddenimports = []
hiddenimports += collect_submodules('mep_cmap')
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('pandas')
hiddenimports += collect_submodules('pywt')
hiddenimports += collect_submodules('mpl_toolkits')
hiddenimports += collect_submodules('pyedflib')   # BIDS-ify EDF/BDF writer (C ext)
hiddenimports += collect_submodules('statsmodels') # EMG excitability compensation (QR)
hiddenimports += collect_submodules('h5py')       # CED Signal MATLAB v7.3 exports (HDF5)
hiddenimports += collect_submodules('patsy')       # statsmodels formula dependency
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
    # UPX is DISABLED on macOS deliberately. It rewrites Mach-O headers, which
    # corrupts the binary on some systems and invalidates any code signature
    # applied afterwards — including the ad-hoc signature the release workflow
    # adds. Do not re-enable it here.
    upx=False,
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
    upx=False,                # see the note on the EXE block above
    upx_exclude=[],
    name='MEP-CMAP Analyser',
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────
app = BUNDLE(
    coll,
    name='MEP-CMAP Analyser.app',
    # Intentionally .ico, not .icns. Convention says macOS wants .icns, but
    # MEP.icns produces NO icon on macOS in practice while MEP.ico renders
    # correctly (verified on a real Mac). Leave this alone.
    icon='MEP.ico',
    # Reverse-DNS of a domain the author controls. Changing this makes
    # macOS treat the app as a new one, so preferences and any codesigning
    # identity keyed to the old identifier do not carry over -- harmless
    # here because no signed release has used it.
    bundle_identifier='io.github.jandrushko.mep-cmap-analyser',
    info_plist={
        'CFBundleName':              'MEP-CMAP Analyser',
        'CFBundleDisplayName':       'MEP-CMAP Analyser',
        'CFBundleVersion':           '1.4.5',
        'CFBundleShortVersionString':'1.4.5',
        'NSHighResolutionCapable':   True,
        'NSRequiresAquaSystemAppearance': False,  # supports dark mode
    },
)
