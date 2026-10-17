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

binaries = []

# 1. Package macOS bundled relocatable libgphoto2 if present
mac_vendor_gphoto2 = os.path.abspath('src/vendor/gphoto2_mac')
if os.path.isdir(mac_vendor_gphoto2):
    camlibs_dir = os.path.join(mac_vendor_gphoto2, 'camlibs')
    iolibs_dir = os.path.join(mac_vendor_gphoto2, 'iolibs')
    lib_dir = os.path.join(mac_vendor_gphoto2, 'lib')
    if os.path.isdir(camlibs_dir):
        datas.append((camlibs_dir, os.path.join('gphoto2', 'camlibs')))
    if os.path.isdir(iolibs_dir):
        datas.append((iolibs_dir, os.path.join('gphoto2', 'iolibs')))
    if os.path.isdir(lib_dir):
        datas.append((lib_dir, os.path.join('gphoto2', 'lib')))
        for dylib in glob.glob(os.path.join(lib_dir, '*.dylib')):
            binaries.append((dylib, '.'))
else:
    # Fallback to local development Homebrew paths on macOS
    for p in ['/opt/homebrew/lib', '/usr/local/lib']:
        for m in glob.glob(os.path.join(p, 'libgphoto2', '*')):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'ptp2.so')):
                datas.append((m, os.path.join('gphoto2', 'camlibs', os.path.basename(m))))
        for m in glob.glob(os.path.join(p, 'libgphoto2_port', '*')):
            if os.path.isdir(m) and os.path.exists(os.path.join(m, 'usb1.so')):
                datas.append((m, os.path.join('gphoto2', 'iolibs', os.path.basename(m))))

# 2. Package Windows native camera bridge helper if present
win_vendor_bridge = os.path.abspath('src/vendor/camera_bridge_win')
if os.path.isdir(win_vendor_bridge):
    datas.append((win_vendor_bridge, 'camera_bridge'))

hiddenimports = [
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
    'PIL.ImageFont',
    'PIL.PdfImagePlugin',
    'multiprocessing',
    'concurrent.futures',
    'batch_worker',
    'contact_sheet',
    'pydantic',
    'pydantic_core',
    'schemas',
    'camera_backends',
    'camera_backends.base',
    'camera_backends.simulated',
    'camera_backends.windows_bridge',
    'camera_backends.gphoto2_backend',
    'waitress'
]

# 3. Package python-gphoto2 bindings and bundled libgphoto2 drivers (Linux / macOS)
try:
    import gphoto2 as _gp
    from PyInstaller.utils.hooks import collect_submodules
    hiddenimports.extend(collect_submodules('gphoto2'))
    gp_pkg_dir = os.path.dirname(_gp.__file__)
    gp_lib_dir = os.path.join(gp_pkg_dir, 'libgphoto2')
    if os.path.isdir(gp_lib_dir):
        datas.append((gp_lib_dir, os.path.join('gphoto2', 'libgphoto2')))
    gp_libs_sibling = os.path.join(os.path.dirname(gp_pkg_dir), 'gphoto2.libs')
    if os.path.isdir(gp_libs_sibling):
        datas.append((gp_libs_sibling, 'gphoto2.libs'))
except ImportError:
    pass

a = Analysis(
    ['web_ui.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
