#!/usr/bin/env python3
"""
Bundle and relocate libgphoto2, its dependencies, camlibs, and iolibs for macOS.
This script produces a fully self-contained directory with relocatable @rpath
references, ensuring the packaged FilmConvert app runs on Macs without Homebrew.
"""

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

def run_cmd(cmd, check=True):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()

def find_homebrew_prefix():
    # Check environment variable first
    if 'HOMEBREW_PREFIX' in os.environ and os.path.exists(os.environ['HOMEBREW_PREFIX']):
        return os.environ['HOMEBREW_PREFIX']
    for candidate in ['/opt/homebrew', '/usr/local']:
        if os.path.exists(candidate):
            return candidate
    return None

def get_dylib_dependencies(path):
    """Parses otool -L output to find non-system dylib dependencies."""
    output = run_cmd(['otool', '-L', str(path)])
    deps = []
    lines = output.splitlines()
    for line in lines[1:]:  # First line is binary path
        line = line.strip()
        if not line:
            continue
        dep = line.split(' (compatibility')[0].strip()
        # Filter out system frameworks / dylibs
        if dep.startswith('/usr/lib') or dep.startswith('/System/Library'):
            continue
        deps.append(dep)
    return deps

def relocate_binary(binary_path, target_lib_dir):
    """Updates install names and dependencies to @rpath using install_name_tool."""
    base_name = os.path.basename(binary_path)
    # Set ID to @rpath/filename
    try:
        run_cmd(['install_name_tool', '-id', f'@rpath/{base_name}', str(binary_path)], check=False)
    except Exception:
        pass

    # Ensure @rpath contains @loader_path and relative lib paths
    run_cmd(['install_name_tool', '-add_rpath', '@loader_path', str(binary_path)], check=False)
    run_cmd(['install_name_tool', '-add_rpath', '@loader_path/../lib', str(binary_path)], check=False)
    run_cmd(['install_name_tool', '-add_rpath', '@loader_path/../../lib', str(binary_path)], check=False)

    deps = get_dylib_dependencies(binary_path)
    for dep in deps:
        if dep.startswith('@rpath'):
            continue
        dep_base = os.path.basename(dep)
        run_cmd(['install_name_tool', '-change', dep, f'@rpath/{dep_base}', str(binary_path)], check=False)

    # Re-sign with ad-hoc signature
    run_cmd(['codesign', '-f', '-s', '-', str(binary_path)], check=False)

def bundle_macos_gphoto2(dest_dir=None):
    if sys.platform != 'darwin':
        print("[bundle_macos_gphoto2] Warning: Not running on macOS; skipping bundle generation.")
        return False

    prefix = find_homebrew_prefix()
    if not prefix:
        raise RuntimeError("Could not locate Homebrew installation at /opt/homebrew or /usr/local")

    repo_root = Path(__file__).resolve().parent.parent
    if dest_dir is None:
        dest_dir = repo_root / 'src' / 'vendor' / 'gphoto2_mac'
    else:
        dest_dir = Path(dest_dir)

    print(f"Bundling libgphoto2 from Homebrew prefix: {prefix}")
    print(f"Destination: {dest_dir}")

    lib_dest = dest_dir / 'lib'
    camlibs_dest = dest_dir / 'camlibs'
    iolibs_dest = dest_dir / 'iolibs'

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    lib_dest.mkdir(parents=True, exist_ok=True)
    camlibs_dest.mkdir(parents=True, exist_ok=True)
    iolibs_dest.mkdir(parents=True, exist_ok=True)

    brew_lib = Path(prefix) / 'lib'

    # Find required dylibs
    dylibs_to_copy = set()
    patterns = [
        'libgphoto2.*.dylib',
        'libgphoto2_port.*.dylib',
        'libusb-1.0.*.dylib',
        'libexif.*.dylib',
        'libltdl.*.dylib'
    ]
    for pat in patterns:
        for match in brew_lib.glob(pat):
            if match.is_symlink():
                continue
            dylibs_to_copy.add(match)

    if not dylibs_to_copy:
        raise RuntimeError(f"No libgphoto2 dylibs found under {brew_lib}")

    copied_dylibs = []
    for dylib in dylibs_to_copy:
        dest = lib_dest / dylib.name
        shutil.copy2(dylib, dest)
        copied_dylibs.append(dest)
        # Also create clean unversioned symlink if needed (e.g. libgphoto2.dylib -> libgphoto2.6.dylib)
        parts = dylib.name.split('.')
        clean_name = f"{parts[0]}.dylib"
        clean_link = lib_dest / clean_name
        if not clean_link.exists():
            clean_link.symlink_to(dylib.name)

    # Find and copy camlibs (specifically ptp2 and canon)
    camlib_roots = list(brew_lib.glob('libgphoto2/*'))
    copied_camlibs = []
    for c_root in camlib_roots:
        if c_root.is_dir():
            for so_file in c_root.glob('*.so'):
                dest = camlibs_dest / so_file.name
                shutil.copy2(so_file, dest)
                copied_camlibs.append(dest)

    # Find and copy iolibs (specifically usb1 and disk)
    iolib_roots = list(brew_lib.glob('libgphoto2_port/*'))
    copied_iolibs = []
    for i_root in iolib_roots:
        if i_root.is_dir():
            for so_file in i_root.glob('*.so'):
                dest = iolibs_dest / so_file.name
                shutil.copy2(so_file, dest)
                copied_iolibs.append(dest)

    print(f"Copied {len(copied_dylibs)} dylibs, {len(copied_camlibs)} camlibs, {len(copied_iolibs)} iolibs.")

    # Relocate all copied binaries
    all_binaries = copied_dylibs + copied_camlibs + copied_iolibs
    for b in all_binaries:
        relocate_binary(b, lib_dest)

    print("[bundle_macos_gphoto2] Successfully bundled and relocated libgphoto2 for macOS.")
    return True

if __name__ == '__main__':
    try:
        bundle_macos_gphoto2()
    except Exception as e:
        print(f"Error bundling libgphoto2: {e}", file=sys.stderr)
        sys.exit(1)
