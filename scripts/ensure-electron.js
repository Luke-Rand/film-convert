#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawnSync } = require('child_process');

function getPlatformPath() {
  const platform = process.env.npm_config_platform || os.platform();
  switch (platform) {
    case 'mas':
    case 'darwin':
      return 'Electron.app/Contents/MacOS/Electron';
    case 'freebsd':
    case 'openbsd':
    case 'linux':
      return 'electron';
    case 'win32':
      return 'electron.exe';
    default:
      return 'electron';
  }
}

async function main() {
  const electronDir = path.join(__dirname, '..', 'node_modules', 'electron');
  if (!fs.existsSync(electronDir)) {
    return;
  }

  const pkgPath = path.join(electronDir, 'package.json');
  if (!fs.existsSync(pkgPath)) {
    return;
  }
  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf-8'));
  const version = pkg.version;
  const platformPath = getPlatformPath();

  const distDir = path.join(electronDir, 'dist');
  const pathFile = path.join(electronDir, 'path.txt');
  const targetBinary = path.join(distDir, platformPath);

  if (fs.existsSync(targetBinary) && fs.existsSync(pathFile)) {
    const recordedPath = fs.readFileSync(pathFile, 'utf-8').trim();
    if (recordedPath === platformPath) {
      return;
    }
  }

  console.log('[ensure-electron] Electron binary missing or incomplete. Extracting...');

  const { downloadArtifact } = require('@electron/get');
  const zipPath = await downloadArtifact({
    version,
    artifactName: 'electron',
    platform: process.env.npm_config_platform || process.platform,
    arch: process.env.npm_config_arch || process.arch
  });

  if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
  }

  let extracted = false;
  const unzipRes = spawnSync('unzip', ['-q', '-o', zipPath, '-d', distDir], { stdio: 'inherit' });
  if (unzipRes.status === 0) {
    extracted = true;
  } else {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const pyRes = spawnSync(pythonCmd, ['-m', 'zipfile', '-e', zipPath, distDir], { stdio: 'inherit' });
    if (pyRes.status === 0) {
      extracted = true;
    }
  }

  if (!extracted) {
    throw new Error(`Failed to extract Electron zip from ${zipPath}`);
  }

  if (process.platform !== 'win32') {
    try {
      fs.chmodSync(targetBinary, 0o755);
    } catch (e) {
      // ignore
    }
  }

  fs.writeFileSync(pathFile, platformPath, 'utf-8');

  const distVersionPath = path.join(distDir, 'version');
  if (!fs.existsSync(distVersionPath)) {
    fs.writeFileSync(distVersionPath, `v${version}`, 'utf-8');
  }

  console.log('[ensure-electron] Electron binary successfully installed at', targetBinary);
}

main().catch(err => {
  console.error('[ensure-electron] Error:', err);
  process.exit(1);
});
