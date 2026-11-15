# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from setuptools_scm import get_version

# Refresh the version even when an editable install predates the current commit.
get_version(
    root=SPECPATH,
    relative_to=Path(SPECPATH) / 'pyproject.toml',
    version_file='llm_conductor/_version.py',
    dist_name='llm-conductor',
)

a = Analysis(
    ['llm-conductor.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='llm-conductor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
