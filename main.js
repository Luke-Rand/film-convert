const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const net = require('net');

let mainWindow = null;
let pythonProcess = null;
let flaskPort = null;

// Determine if we are running in packaged production mode or development
const isPackaged = app.isPackaged;

// Find a free port dynamically to prevent conflicts
function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close(() => {
        resolve(port);
      });
    });
  });
}

// Start the Python Web UI process
function startPythonBackend(port) {
  let pythonBin = '';
  let args = [];

  if (isPackaged) {
    // Path to the compiled PyInstaller binary inside extraResources
    // In production, pyinstaller creates a directory or executable named 'film-convert-backend'
    const binaryName = process.platform === 'win32' ? 'film-convert-backend.exe' : 'film-convert-backend';
    pythonBin = path.join(process.resourcesPath, 'backend', binaryName);
  } else {
    // In development, run python from the local virtual environment
    const venvBinDir = process.platform === 'win32' ? 'Scripts' : 'bin';
    pythonBin = path.join(__dirname, '.venv', venvBinDir, process.platform === 'win32' ? 'python.exe' : 'python');
    args = [path.join(__dirname, 'src', 'web_ui.py')];
  }

  console.log(`Starting Python backend: ${pythonBin} with args:`, args);

  // Set environment variables for host and port
  const env = {
    ...process.env,
    HOST: '127.0.0.1',
    PORT: port.toString(),
    PYTHONUNBUFFERED: '1'
  };

  pythonProcess = spawn(pythonBin, args, {
    cwd: isPackaged ? process.resourcesPath : __dirname,
    env: env
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python stdout]: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python stderr]: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python backend exited with code ${code}`);
    if (code !== 0 && code !== null) {
      dialog.showErrorBox(
        'Backend Process Terminated',
        `The Python backend exited unexpectedly with code ${code}. Please check system dependencies.`
      );
    }
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python process:', err);
    dialog.showErrorBox(
      'Failed to Start Backend',
      `Could not spawn the Python backend: ${err.message}`
    );
  });
}

// Poll the Flask server until it is fully responsive
async function waitForBackend(port, retries = 50, delay = 200) {
  const url = `http://127.0.0.1:${port}/api/camera/status`;
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        console.log(`Backend is ready on port ${port} after ${i + 1} checks.`);
        return true;
      }
    } catch (e) {
      // Ignore connection failures while booting
    }
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
  throw new Error('Flask server failed to respond within timeout limit.');
}

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    title: 'FilmConvert Desktop',
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Configure WebSerial handlers on the active window session
  const ses = mainWindow.webContents.session;

  ses.setPermissionCheckHandler((webContents, permission) => {
    if (permission === 'serial') {
      return true;
    }
    return false;
  });

  ses.setDevicePermissionHandler((details) => {
    if (details.deviceType === 'serial') {
      return true;
    }
    return false;
  });

  ses.on('select-serial-port', (event, portList, webContents, callback) => {
    // Prevent default selection behavior
    event.preventDefault();

    if (portList && portList.length > 0) {
      if (portList.length === 1) {
        // Auto-select if there is exactly one serial device connected
        console.log(`Auto-selected only serial port available: ${portList[0].portName}`);
        callback(portList[0].portId);
      } else {
        // Prompt user with a native Electron dialog if multiple options exist
        const portButtons = portList.map(p => `${p.portName} (${p.displayName || 'Unknown Device'})`);
        portButtons.push('Cancel');

        dialog.showMessageBox(mainWindow, {
          type: 'question',
          buttons: portButtons,
          title: 'Select Serial Port',
          message: 'Choose a serial device to connect to Scanlight:',
          cancelId: portButtons.length - 1
        }).then(({ response }) => {
          if (response === portButtons.length - 1) {
            callback(''); // Cancelled
          } else {
            console.log(`User selected serial port: ${portList[response].portName}`);
            callback(portList[response].portId);
          }
        });
      }
    } else {
      // Alert user if no serial ports were detected
      dialog.showMessageBox(mainWindow, {
        type: 'warning',
        buttons: ['OK'],
        title: 'No Serial Devices Found',
        message: 'Could not find any connected serial ports. Make sure your Scanlight is powered and connected to USB.'
      });
      callback('');
    }
  });

  // Load the Python Flask application URL directly
  mainWindow.loadURL(`http://127.0.0.1:${port}`);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App lifecycle management
app.whenReady().then(async () => {
  try {
    flaskPort = await findFreePort();
    console.log(`Selected dynamic port for Flask backend: ${flaskPort}`);
    
    startPythonBackend(flaskPort);
    
    // Wait for python backend to be ready before showing window
    await waitForBackend(flaskPort);
    
    createWindow(flaskPort);
  } catch (err) {
    console.error('App startup failed:', err);
    dialog.showErrorBox(
      'Startup Error',
      `An error occurred during application startup: ${err.message}`
    );
    app.quit();
  }
});

// Clean up processes on close
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  if (pythonProcess) {
    console.log('Terminating Python backend subprocess...');
    pythonProcess.kill('SIGTERM');
  }
});
