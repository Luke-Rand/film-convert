# -*- mode: python ; coding: utf-8 -*-
import sys
import os

import glob

block_cipher = None

# Locate directories to package
templates_path = os.path.abspath('src/templates')
static_path = os.path.abspath('src/static')
icc_path = os.path.abspath('src/icc_profiles')

datas = [
    (templates_path, 'templates'),
    (static_path, 'static'),
    (icc_path, 'icc_profiles')
]

# Locate gphoto2 camlibs and iolibs if present
for p in ['/opt/homebrew/lib', '/usr/local/lib']:
    for m in glob.glob(os.path.join(p, 'libgphoto2', '*')):
        if os.path.isdir(m) and os.path.exists(os.path.join(m, 'ptp2.so')):
            datas.append((m, os.path.join('gphoto2', 'camlibs', os.path.basename(m))))
    for m in glob.glob(os.path.join(p, 'libgphoto2_port', '*')):
        if os.path.isdir(m) and os.path.exists(os.path.join(m, 'usb1.so')):
            datas.append((m, os.path.join('gphoto2', 'iolibs', os.path.basename(m))))

a = Analysis(
    ['web_ui.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'flask',
        'numpy',
        'scipy',
        'scipy.fft',
        'rawpy',
        'tifffile',
        'piexif',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'multiprocessing',
        'concurrent.futures',
        'batch_worker'
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
