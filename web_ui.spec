# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Locate directories to package
templates_path = os.path.abspath('templates')
static_path = os.path.abspath('static')

a = Analysis(
    ['web_ui.py'],
    pathex=[],
    binaries=[],
    datas=[
        (templates_path, 'templates'),
        (static_path, 'static')
    ],
    hiddenimports=[
        'flask',
        'numpy',
        'rawpy',
        'tifffile',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    exclude_binaries=True,
    name='film-convert-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
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
    name='backend',
)
