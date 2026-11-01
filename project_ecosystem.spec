# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Project Ecosystem.

Bundles the PySide6 GUI app into a single executable. PyInstaller's
built-in PySide6 hook collects the Qt plugins and libraries we need;
we add explicit hidden imports for the modules our app uses and
exclude the heavy Qt sub-modules we don't (WebEngine, Pdf, Quick3D,
etc.) to keep the bundle under ~80 MB.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect the PySide6 Qt plugins, translations and data files.
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all("PySide6")
shiboken6_datas, shiboken6_binaries, shiboken6_hiddenimports = collect_all("shiboken6")

# Explicit hidden imports for our own package (PyInstaller's analysis
# sometimes misses these because they're imported dynamically inside
# functions or via string paths).
hiddenimports = [
    "ecosystem",
    "ecosystem.app",
    "ecosystem.core",
    "ecosystem.core.genome",
    "ecosystem.core.resource",
    "ecosystem.core.organism",
    "ecosystem.core.species",
    "ecosystem.core.world",
    "ecosystem.core.weather",
    "ecosystem.core.simulation",
    "ecosystem.core.journal",
    "ecosystem.persistence",
    "ecosystem.gui",
    "ecosystem.gui.main_window",
    "ecosystem.gui.world_view",
    "ecosystem.gui.panels",
    "ecosystem.gui.theme",
] + pyside6_hiddenimports + shiboken6_hiddenimports

# Qt sub-modules we don't use -> exclude to shrink the bundle.
excluded = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQuick3D",
    "PySide6.QtQuick",
    "PySide6.QtQml",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia",
    "PySide6.QtNetwork",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtXml",
]

# Heavy Qt sub-modules & libraries we don't use -> filter out to shrink
# the bundle from ~270 MB to ~90 MB. We keep: QtCore, QtGui, QtWidgets,
# QtOpenGL (needed for rendering), shiboken6, and the xcb/offscreen
# platform plugins.
_EXCLUDE_PATTERNS = [
    "webengine", "pdf", "quick3d", "qml", "multimedia",
    "qtquick", "qtcharts", "qtgraphs", "qt3d", "qtvirtualkeyboard",
    "avcodec", "avformat", "avutil", "swscale", "swresample",
    "assimp", "sqldrivers", "sqlite", "mysql", "psql", "odbc",
    "virtualkeyboard", "graphical",
    "qt5compat", "qtvkbpinyinplugin", "qtvkbopenwnnplugin",
    "test", "designer", "help", "uitools",
]

filtered_binaries = []
for b in pyside6_binaries + shiboken6_binaries:
    dest = b[1]
    if any(p in dest.lower() for p in _EXCLUDE_PATTERNS):
        continue
    filtered_binaries.append(b)

filtered_datas = []
for d in pyside6_datas + shiboken6_datas:
    dest = d[1]
    if any(p in dest.lower() for p in _EXCLUDE_PATTERNS):
        continue
    filtered_datas.append(d)

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=filtered_binaries,
    datas=filtered_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="project-ecosystem",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
