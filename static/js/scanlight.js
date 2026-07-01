// WebSerial Big Scanlight Protocol and UI Controller
// Ports the original Vue controls into the FilmConvert web UI context

class ScanlightProtocol {
  constructor() {
    this.UART_BAUD_RATE = 115200;
    this.PACKET_START = 254;

    // Host-to-device packets
    this.PKT_H2D_SET_COLOR = 0;
    this.PKT_H2D_GET_DEFAULT_RGB = 1;
    this.PKT_H2D_GET_FW_VERSION = 2;
    this.PKT_H2D_SHUTTER_PULSE = 3;
    this.PKT_H2D_DFU_MODE = 4;
    this.PKT_H2D_SET_TRIM = 5;
    this.PKT_H2D_GET_TRIM = 6;

    // Device-to-host packets
    this.PKT_D2H_ACK = 0;
    this.PKT_D2H_LED_TEMP = 1;
    this.PKT_D2H_VBUS = 2;
    this.PKT_D2H_FW_VERSION = 3;
    this.PKT_D2H_DEFAULT_RGB = 4;
    this.PKT_D2H_TRIM = 5;

    this.PACKET_MAX_LENGTH = 128;

    this.incomingPacketHeader = 0;
    this.incomingPacketBuffer = new ArrayBuffer(this.PACKET_MAX_LENGTH);
    this.incomingPacket = new Uint8Array(this.incomingPacketBuffer);
    this.packetIndex = 0;
    this.packetEnd = 0;

    this.callbacks = {};
    this.port = null;
    this.writer = null;
    this.reader = null;
    this.keepReading = false;
  }

  async readUntilClosed() {
    this.keepReading = true;
    while (this.port && this.port.readable && this.keepReading) {
      this.reader = this.port.readable.getReader();
      try {
        while (true) {
          const { value, done } = await this.reader.read();
          if (done || !this.keepReading) {
            break;
          }
          for (let i = 0; i < value.length; i++) {
            if (this.packetIndex == 0) {
              if (value[i] == this.PACKET_START) this.packetIndex++;
              else continue;
            } else if (this.packetIndex == 1) {
              this.incomingPacketHeader = value[i];
              this.packetIndex++;
            } else if (this.packetIndex == 2) {
              this.packetEnd = value[i] + 2;
              this.packetIndex++;
            } else if (this.packetIndex >= 3 && this.packetIndex < this.packetEnd && this.packetIndex - 3 < this.PACKET_MAX_LENGTH - 1) {
              this.incomingPacket[this.packetIndex - 3] = value[i];
              this.packetIndex++;
            } else if (this.packetIndex == this.packetEnd) {
              this.incomingPacket[this.packetIndex - 3] = value[i];
              this.incomingPacket[this.packetIndex - 2] = 0;
              this.handlePacket();
              this.packetIndex = 0;
            } else {
              this.packetIndex = 0;
            }
          }
        }
      } catch (error) {
        console.error("Serial read error:", error);
        if (window.appendLogLine) {
          window.appendLogLine(`[Scanlight Error] Serial error: ${error.message}`);
        }
      } finally {
        this.reader.releaseLock();
      }
    }
  }

  handlePacket() {
    if (this.incomingPacketHeader in this.callbacks) {
      // Create a copy of the buffer to avoid overwritten data
      const packetCopy = new Uint8Array(this.incomingPacketBuffer.slice(0));
      this.callbacks[this.incomingPacketHeader](
        this.incomingPacketHeader,
        packetCopy,
        new DataView(packetCopy.buffer)
      );
    }
  }

  async sendPacket(header, packetData) {
    if (!this.writer) return;
    const data = new Uint8Array(packetData.length + 3);
    data[0] = this.PACKET_START;
    data[1] = header;
    data[2] = packetData.length;
    for (let i = 3; i < packetData.length + 3; i++) {
      data[i] = packetData[i - 3];
    }
    await this.writer.write(data);
  }

  async connect() {
    this.port = await navigator.serial.requestPort();
    await this.port.open({ baudRate: this.UART_BAUD_RATE });
    this.writer = this.port.writable.getWriter();
    // Run the reader in background asynchronously
    this.readUntilClosed();
  }

  async disconnect() {
    this.keepReading = false;
    if (this.reader) {
      await this.reader.cancel();
    }
    if (this.writer) {
      this.writer.releaseLock();
      this.writer = null;
    }
    if (this.port) {
      await this.port.close();
      this.port = null;
    }
  }

  addCallback(header, callback) {
    this.callbacks[header] = callback;
  }
}

// Config mappings
const ScanlightConfig = {
  USBVBUSThreshold5V: 4000,
  USBVBUSThreshold9V: 8000,
  OverTemperatureThresholdMdegc: 77000,
  FWVersionStrings: {
    0: "v1.0.0",
  },
  LatestFWVersionID: 0,
  HWVersionStrings: {
    0: "big scanlight v1",
    1: "scanlight v4",
  },
  SequenceRGB: [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
  ],
  SequenceRGBIR: [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
  ],
  SequenceNWIR: [
    [1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
  ],
  SequenceBWIR: [
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
  ],
};

class ScanlightUIController {
  constructor() {
    this.protocol = new ScanlightProtocol();
    this.connected = false;
    this.red = 255;
    this.green = 255;
    this.blue = 255;
    this.trimR = 0;
    this.trimG = 0;
    this.trimB = 0;
    this.trimW = 0;
    this.enabledChannels = [0, 0, 0, 0, 0]; // [R, G, B, W, IR]
    this.presets = [];
    this.selectedPresetName = "";
    this.shutterPulseLength = 0.1;
    this.postShutterDelay = 1.0;
    this.inputVoltageMv = 0;
    this.ledTemperatureMdegc = 0;
    this.fwUpdateAvailable = false;
    this.fwVersionString = "Unknown";
    this.hwVersionString = "Unknown";
    
    this.isSmallHW = false;
    this.isSequenceRunning = false;
    this.postSequenceLight = "off";
    this.triggerMethod = "scanlight";
    this._isCalibrating = false;
  }

  get isCalibrating() {
    return this._isCalibrating;
  }

  set isCalibrating(val) {
    this._isCalibrating = val;
    const btn = document.getElementById("btn-sl-calibrate");
    if (btn) {
      if (val) {
        btn.innerHTML = "⏹️ Cancel Calibration";
        btn.style.backgroundColor = "rgba(239, 68, 68, 0.12)";
        btn.style.borderColor = "rgba(239, 68, 68, 0.3)";
        btn.style.color = "#f87171";
      } else {
        btn.innerHTML = "⚖️ Auto-Calibrate RGB Exposures";
        btn.style.backgroundColor = "rgba(14, 165, 233, 0.08)";
        btn.style.borderColor = "rgba(14, 165, 233, 0.25)";
        btn.style.color = "var(--accent-blue)";
      }
    }
  }

  init() {
    this.checkBrowserSupport();
    this.loadPresetsFromStorage();
    this.bindEvents();
    this.updateControlsState();
    this.initializeCustomSpinners();
    
    // Sync calibration button state
    this.isCalibrating = false;
    
    // Bind calibration event hook
    window.onTripletMeansReceived = (data) => this.handleCalibrationData(data);
  }

  initializeCustomSpinners() {
    document.querySelectorAll(".num-input").forEach(input => {
      // Avoid double wrapping if initialized twice
      if (input.parentNode.classList.contains("num-input-wrapper")) return;

      const wrapper = document.createElement("div");
      wrapper.className = "num-input-wrapper";
      
      input.parentNode.insertBefore(wrapper, input);
      wrapper.appendChild(input);

      const spin = document.createElement("div");
      spin.className = "num-input-spin";

      const upBtn = document.createElement("button");
      upBtn.type = "button";
      upBtn.className = "spin-up";
      upBtn.innerHTML = `<svg viewBox="0 0 24 24" width="10" height="10" stroke="currentColor" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg>`;

      const downBtn = document.createElement("button");
      downBtn.type = "button";
      downBtn.className = "spin-down";
      downBtn.innerHTML = `<svg viewBox="0 0 24 24" width="10" height="10" stroke="currentColor" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>`;

      spin.appendChild(upBtn);
      spin.appendChild(downBtn);
      wrapper.appendChild(spin);

      const step = parseFloat(input.getAttribute("step")) || 1;
      const min = input.getAttribute("min") !== null ? parseFloat(input.getAttribute("min")) : -Infinity;
      const max = input.getAttribute("max") !== null ? parseFloat(input.getAttribute("max")) : Infinity;

      const changeVal = (delta) => {
        if (input.disabled || input.classList.contains("disabled-input")) return;
        
        let val = parseFloat(input.value);
        if (isNaN(val)) val = min !== -Infinity ? min : 0;
        let newVal = val + delta;

        // Resolve floating point precision issues
        const decimals = (step.toString().split('.')[1] || '').length;
        if (decimals > 0) {
          newVal = parseFloat(newVal.toFixed(decimals));
        }

        newVal = Math.min(Math.max(newVal, min), max);
        input.value = newVal;

        // Dispatch input & change events for reactive bindings
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      };

      let holdTimeout = null;
      let holdInterval = null;
      let holdTriggered = false;

      const startHold = (delta) => {
        stopHold();
        if (input.disabled || input.classList.contains("disabled-input")) return;
        holdTriggered = false;

        holdTimeout = setTimeout(() => {
          holdTriggered = true;
          holdInterval = setInterval(() => {
            if (input.disabled || input.classList.contains("disabled-input")) {
              stopHold();
              return;
            }
            changeVal(delta);
          }, 60);
        }, 350);
      };

      const stopHold = () => {
        if (holdTimeout) {
          clearTimeout(holdTimeout);
          holdTimeout = null;
        }
        if (holdInterval) {
          clearInterval(holdInterval);
          holdInterval = null;
        }
      };

      upBtn.tabIndex = "-1";
      downBtn.tabIndex = "-1";

      // Mouse Events
      upBtn.addEventListener("mousedown", (e) => {
        if (e.button === 0) startHold(step);
      });
      upBtn.addEventListener("mouseup", stopHold);
      upBtn.addEventListener("mouseleave", stopHold);

      downBtn.addEventListener("mousedown", (e) => {
        if (e.button === 0) startHold(-step);
      });
      downBtn.addEventListener("mouseup", stopHold);
      downBtn.addEventListener("mouseleave", stopHold);

      // Touch Events
      upBtn.addEventListener("touchstart", () => startHold(step), { passive: true });
      upBtn.addEventListener("touchend", stopHold);
      upBtn.addEventListener("touchcancel", stopHold);

      downBtn.addEventListener("touchstart", () => startHold(-step), { passive: true });
      downBtn.addEventListener("touchend", stopHold);
      downBtn.addEventListener("touchcancel", stopHold);

      // Click event fallback / absorption
      upBtn.addEventListener("click", (e) => {
        e.preventDefault();
        if (holdTriggered) {
          holdTriggered = false;
          return;
        }
        changeVal(step);
      });

      downBtn.addEventListener("click", (e) => {
        e.preventDefault();
        if (holdTriggered) {
          holdTriggered = false;
          return;
        }
        changeVal(-step);
      });
    });
  }

  checkBrowserSupport() {
    const isSupported = 'serial' in navigator;
    if (!isSupported) {
      this.log("[Scanlight] WebSerial API is not supported on this browser. Please use Chrome/Edge.");
      const banner = document.getElementById("scanlight-unsupported-banner");
      if (banner) {
        banner.style.display = "block";
      }
      const connectBtn = document.getElementById("scanlight-connect-btn");
      if (connectBtn) {
        connectBtn.disabled = true;
      }
    }
  }

  log(msg) {
    if (window.appendLogLine) {
      window.appendLogLine(msg);
    } else {
      console.log(msg);
    }
  }

  bindEvents() {
    // Connection
    document.getElementById("scanlight-connect-btn")?.addEventListener("click", () => this.toggleConnection());
    document.getElementById("scanlight-dfu-btn")?.addEventListener("click", () => this.enterDFUMode());

    // Color Sliders & Inputs
    const sliders = ["red", "green", "blue"];
    sliders.forEach(c => {
      const slider = document.getElementById(`scanlight-${c}-slider`);
      const valInput = document.getElementById(`scanlight-${c}-val`);
      
      slider?.addEventListener("input", (e) => {
        this.isCalibrating = false;
        this[c] = parseInt(e.target.value);
        if (valInput) valInput.value = this[c];
        this.updateColor();
      });
      
      valInput?.addEventListener("input", (e) => {
        this.isCalibrating = false;
        let val = parseInt(e.target.value);
        if (isNaN(val)) return;
        val = Math.min(Math.max(val, 0), 255);
        this[c] = val;
        if (slider) slider.value = val;
        this.updateColor();
      });
      
      valInput?.addEventListener("change", (e) => {
        this.isCalibrating = false;
        let val = parseInt(e.target.value);
        if (isNaN(val)) val = 255;
        val = Math.min(Math.max(val, 0), 255);
        this[c] = val;
        e.target.value = val;
        if (slider) slider.value = val;
        this.updateColor();
      });
    });

    // Quick presets channel triggers
    document.getElementById("btn-sl-rgb")?.addEventListener("click", () => { this.isCalibrating = false; this.setEnabledChannels([1, 1, 1, 0, 0]); });
    document.getElementById("btn-sl-white")?.addEventListener("click", () => { this.isCalibrating = false; this.setEnabledChannels([0, 0, 0, 1, 0]); });
    document.getElementById("btn-sl-off")?.addEventListener("click", () => { this.isCalibrating = false; this.setEnabledChannels([0, 0, 0, 0, 0]); });
    document.getElementById("btn-sl-r")?.addEventListener("click", () => { this.isCalibrating = false; this.setEnabledChannels([1, 0, 0, 0, 0]); });
    document.getElementById("btn-sl-g")?.addEventListener("click", () => { this.isCalibrating = false; this.setEnabledChannels([0, 1, 0, 0, 0]); });
    document.getElementById("btn-sl-b")?.addEventListener("click", () => { this.isCalibrating = false; this.setEnabledChannels([0, 0, 1, 0, 0]); });
    document.getElementById("btn-sl-ir")?.addEventListener("click", () => { this.isCalibrating = false; this.setEnabledChannels([0, 0, 0, 0, 1]); });

    // Preset options
    document.getElementById("scanlight-preset-select")?.addEventListener("change", (e) => {
      this.isCalibrating = false;
      this.selectedPresetName = e.target.value;
    });
    document.getElementById("btn-sl-preset-load")?.addEventListener("click", () => { this.isCalibrating = false; this.loadPreset(); });
    document.getElementById("btn-sl-preset-create")?.addEventListener("click", () => this.createPreset());
    document.getElementById("btn-sl-preset-rename")?.addEventListener("click", () => this.renamePreset());
    document.getElementById("btn-sl-preset-delete")?.addEventListener("click", () => this.deletePreset());
    document.getElementById("btn-sl-default-write")?.addEventListener("click", () => this.writeDefaultDialog());
    document.getElementById("btn-sl-default-load")?.addEventListener("click", () => this.loadDefault());

    // Brightness Trimming dialog
    document.getElementById("btn-sl-trim-open")?.addEventListener("click", () => this.openTrimModal());
    document.getElementById("btn-sl-trim-cancel")?.addEventListener("click", () => this.closeTrimModal());
    document.getElementById("btn-sl-trim-ok")?.addEventListener("click", () => this.saveTrimValues());

    // Trim sliders
    ["trimR", "trimG", "trimB", "trimW"].forEach(t => {
      document.getElementById(`scanlight-${t}-val`)?.addEventListener("change", (e) => {
        let val = parseInt(e.target.value);
        if (isNaN(val)) val = 0;
        val = Math.min(Math.max(val, -127), 127);
        this[t] = val;
        e.target.value = val;
      });
    });

    // Automation Parameters
    document.getElementById("scanlight-shutter-pulse")?.addEventListener("change", (e) => {
      let val = parseFloat(e.target.value);
      if (isNaN(val)) val = 0.1;
      this.shutterPulseLength = Math.min(Math.max(val, 0.01), 0.5);
      e.target.value = this.shutterPulseLength.toFixed(2);
    });

    document.getElementById("scanlight-shutter-delay")?.addEventListener("change", (e) => {
      let val = parseFloat(e.target.value);
      if (isNaN(val)) val = 1.0;
      this.postShutterDelay = Math.min(Math.max(val, 0.1), 12.75);
      e.target.value = this.postShutterDelay.toFixed(2);
    });

    document.getElementById("scanlight-post-seq-light")?.addEventListener("change", (e) => {
      this.postSequenceLight = e.target.value;
    });

    document.getElementById("scanlight-trigger-method")?.addEventListener("change", (e) => {
      this.triggerMethod = e.target.value;
    });

    // Sequences
    document.getElementById("btn-seq-rgb")?.addEventListener("click", () => this.runSequence("SequenceRGB"));
    document.getElementById("btn-seq-rgbir")?.addEventListener("click", () => this.runSequence("SequenceRGBIR"));
    document.getElementById("btn-seq-nwir")?.addEventListener("click", () => this.runSequence("SequenceNWIR"));
    document.getElementById("btn-seq-bwir")?.addEventListener("click", () => this.runSequence("SequenceBWIR"));
    document.getElementById("btn-seq-shutter-test")?.addEventListener("click", () => this.shutterTest());
    document.getElementById("btn-sl-calibrate")?.addEventListener("click", () => {
      if (this.isCalibrating) {
        this.cancelCalibration();
      } else {
        this.startCalibration();
      }
    });
  }

  async toggleConnection() {
    this.isCalibrating = false;
    if (this.connected) {
      this.log("[Scanlight] Disconnecting from device...");
      await this.protocol.disconnect();
      this.connected = false;
      this.updateControlsState();
      this.log("[Scanlight] Disconnected successfully.");
    } else {
      this.log("[Scanlight] Prompting for serial connection...");
      try {
        await this.protocol.connect();
        
        // Add protocol packet handlers
        this.protocol.addCallback(this.protocol.PKT_D2H_FW_VERSION, (h, r, d) => this.onFWVersion(h, r, d));
        this.protocol.addCallback(this.protocol.PKT_D2H_LED_TEMP, (h, r, d) => this.onLEDTemp(h, r, d));
        this.protocol.addCallback(this.protocol.PKT_D2H_VBUS, (h, r, d) => this.onVbus(h, r, d));
        this.protocol.addCallback(this.protocol.PKT_D2H_DEFAULT_RGB, (h, r, d) => this.onDefaultRGB(h, r, d));
        this.protocol.addCallback(this.protocol.PKT_D2H_TRIM, (h, r, d) => this.onTrimValues(h, r, d));

        // Handshake: Request version
        await this.protocol.sendPacket(this.protocol.PKT_H2D_GET_FW_VERSION, []);
        
        // Safety timeout if device doesn't respond
        setTimeout(() => {
          if (!this.connected) {
            this.log("[Scanlight] Handshake timeout. Retrying...");
            this.protocol.disconnect();
            this.connected = false;
            this.updateControlsState();
          }
        }, 2000);
      } catch (err) {
        this.log(`[Scanlight Error] Connection failed: ${err.message}`);
        this.connected = false;
        this.updateControlsState();
      }
    }
  }

  onFWVersion(header, rawData, dataView) {
    const versionIdWord = dataView.getUint32(0);
    const fwVersion = versionIdWord & 0xFFFF;
    const hwVersion = (versionIdWord >> 16) & 0xFFFF;
    
    this.fwVersionString = ScanlightConfig.FWVersionStrings[fwVersion] || `Unknown (${fwVersion})`;
    this.hwVersionString = ScanlightConfig.HWVersionStrings[hwVersion] || `Unknown (${hwVersion})`;
    
    this.isSmallHW = (hwVersion === 1); // scanlight v4
    
    if (fwVersion < ScanlightConfig.LatestFWVersionID) {
      this.fwUpdateAvailable = true;
      this.log(`[Scanlight] Firmware update available! Download at jackw01's releases.`);
    }

    this.connected = true;
    this.log(`[Scanlight] Connected to: ${this.hwVersionString} (FW: ${this.fwVersionString})`);
    
    // Request initial presets and trimming parameters
    this.protocol.sendPacket(this.protocol.PKT_H2D_GET_DEFAULT_RGB, []);
    this.protocol.sendPacket(this.protocol.PKT_H2D_GET_TRIM, []);

    this.updateControlsState();
  }

  onDefaultRGB(header, rawData, dataView) {
    this.red = dataView.getUint8(0);
    this.green = dataView.getUint8(1);
    this.blue = dataView.getUint8(2);
    
    // Sync values to inputs
    document.getElementById("scanlight-red-slider").value = this.red;
    document.getElementById("scanlight-red-val").value = this.red;
    document.getElementById("scanlight-green-slider").value = this.green;
    document.getElementById("scanlight-green-val").value = this.green;
    document.getElementById("scanlight-blue-slider").value = this.blue;
    document.getElementById("scanlight-blue-val").value = this.blue;

    this.updateControlsState();
    this.updateColor();
  }

  onTrimValues(header, rawData, dataView) {
    this.trimR = dataView.getInt8(0);
    this.trimG = dataView.getInt8(1);
    this.trimB = dataView.getInt8(2);
    this.trimW = dataView.getInt8(3);
    
    document.getElementById("scanlight-trimR-val").value = this.trimR;
    document.getElementById("scanlight-trimG-val").value = this.trimG;
    document.getElementById("scanlight-trimB-val").value = this.trimB;
    document.getElementById("scanlight-trimW-val").value = this.trimW;
  }

  onLEDTemp(header, rawData, dataView) {
    this.ledTemperatureMdegc = dataView.getInt32(0);
    const tempC = (this.ledTemperatureMdegc / 1000).toFixed(1);
    const label = document.getElementById("scanlight-lbl-temp");
    if (label) {
      label.textContent = `${tempC}°C`;
    }
    
    const warning = document.getElementById("scanlight-temp-warning");
    if (warning) {
      if (this.ledTemperatureMdegc > ScanlightConfig.OverTemperatureThresholdMdegc) {
        warning.style.display = "block";
      } else {
        warning.style.display = "none";
      }
    }
  }

  onVbus(header, rawData, dataView) {
    this.inputVoltageMv = dataView.getInt32(0);
    const voltageV = (this.inputVoltageMv / 1000).toFixed(2);
    const label = document.getElementById("scanlight-lbl-vbus");
    if (label) {
      label.textContent = `${voltageV}V`;
    }

    // Toggle VBUS Warnings
    const powerWarning = document.getElementById("scanlight-power-warning");
    const vbusThreshold = (this.hwVersionString === "scanlight v4") ? ScanlightConfig.USBVBUSThreshold5V : ScanlightConfig.USBVBUSThreshold9V;
    
    // Disable controls if voltage is too low
    const controlsContainer = document.getElementById("scanlight-active-controls");
    const lowPowerBanner = document.getElementById("scanlight-low-power-banner");
    
    if (this.inputVoltageMv <= ScanlightConfig.USBVBUSThreshold5V) {
      if (controlsContainer) controlsContainer.classList.add("disabled");
      if (lowPowerBanner) lowPowerBanner.style.display = "block";
    } else {
      if (controlsContainer) controlsContainer.classList.remove("disabled");
      if (lowPowerBanner) lowPowerBanner.style.display = "none";
    }
    
    // Warn if operating in reduced power state (voltage between 5V and 9V for big scanlight)
    if (powerWarning) {
      if (this.hwVersionString === "big scanlight v1" && this.inputVoltageMv > ScanlightConfig.USBVBUSThreshold5V && this.inputVoltageMv < ScanlightConfig.USBVBUSThreshold9V) {
        powerWarning.style.display = "block";
      } else {
        powerWarning.style.display = "none";
      }
    }
  }

  setEnabledChannels(ch) {
    this.enabledChannels = ch;
    
    // De-select quick presets styling and highlight the clicked button
    const mapBtnId = {
      "[1,1,1,0,0]": "btn-sl-rgb",
      "[0,0,0,1,0]": "btn-sl-white",
      "[0,0,0,0,0]": "btn-sl-off",
      "[1,0,0,0,0]": "btn-sl-r",
      "[0,1,0,0,0]": "btn-sl-g",
      "[0,0,1,0,0]": "btn-sl-b",
      "[0,0,0,0,1]": "btn-sl-ir"
    };

    const targetKey = JSON.stringify(ch);
    Object.keys(mapBtnId).forEach(key => {
      const btn = document.getElementById(mapBtnId[key]);
      if (btn) {
        if (key === targetKey) {
          btn.classList.add("btn-highlight");
        } else {
          btn.classList.remove("btn-highlight");
        }
      }
    });

    this.updateColor();
  }

  updateColor() {
    if (!this.connected) return;
    
    // Format payload: [Red, Green, Blue, White, IR, SavePresetFlag]
    const newColor = [
      this.red * this.enabledChannels[0],
      this.green * this.enabledChannels[1],
      this.blue * this.enabledChannels[2],
      255 * this.enabledChannels[3],
      255 * this.enabledChannels[4],
      0, // save preset flag
    ];
    
    this.protocol.sendPacket(
      this.protocol.PKT_H2D_SET_COLOR, 
      newColor.map(x => Math.floor(x) & 0xff)
    );

    // Update color glow in UI
    const colorGlow = document.getElementById("scanlight-color-glow");
    if (colorGlow) {
      const r = Math.floor(this.red * this.enabledChannels[0]);
      const g = Math.floor(this.green * this.enabledChannels[1]);
      const b = Math.floor(this.blue * this.enabledChannels[2]);
      const w = this.enabledChannels[3] ? 255 : 0;
      const ir = this.enabledChannels[4] ? 255 : 0;

      if (r || g || b) {
        colorGlow.style.background = `rgba(${r}, ${g}, ${b}, 0.5)`;
        colorGlow.style.boxShadow = `0 0 25px rgba(${r}, ${g}, ${b}, 0.6)`;
      } else if (w) {
        colorGlow.style.background = `rgba(255, 255, 255, 0.7)`;
        colorGlow.style.boxShadow = `0 0 25px rgba(255, 255, 255, 0.8)`;
      } else if (ir) {
        colorGlow.style.background = `rgba(163, 24, 108, 0.4)`;
        colorGlow.style.boxShadow = `0 0 25px rgba(163, 24, 108, 0.5)`;
      } else {
        colorGlow.style.background = `transparent`;
        colorGlow.style.boxShadow = `none`;
      }
    }
  }

  async runSequence(sequence, isCalibration = false) {
    if (!this.connected || this.isSequenceRunning) return;
    
    if (!isCalibration) {
      this.isCalibrating = false;
    }
    
    this.isSequenceRunning = true;
    this.disableControlTriggers(true);
    
    const seqData = ScanlightConfig[sequence];
    this.log(`[Scanlight] Starting automated sequence: ${sequence} (${seqData.length} steps)`);

    try {
      for (let i = 0; i < seqData.length; i++) {
        if (!this.connected || !this.isSequenceRunning) break;
        
        this.log(`[Scanlight] Step ${i + 1}/${seqData.length}: Setting channel mask [${seqData[i]}]`);
        this.setEnabledChannels(seqData[i]);
        
        // Wait 150ms for LEDs to stabilize before triggering shutter
        await new Promise(r => setTimeout(r, 150));
        
        if (this.triggerMethod === "usb") {
          this.log(`[Scanlight] Step ${i + 1}/${seqData.length}: Triggering camera capture via USB...`);
          try {
            const res = await fetch('/api/camera/capture', { method: 'POST' });
            const captureResult = await res.json();
            if (!captureResult.success) {
              throw new Error(captureResult.message || "Unknown capture error");
            }
            this.log(`[Scanlight] Step ${i + 1}/${seqData.length}: USB Capture completed. Path: ${captureResult.path}`);
          } catch (err) {
            this.log(`[Scanlight Error] Shutter trigger via USB failed: ${err.message}`);
            throw err;
          }
          const delayMs = this.postShutterDelay * 1000;
          await new Promise(r => setTimeout(r, delayMs));
        } else {
          this.log(`[Scanlight] Step ${i + 1}/${seqData.length}: Triggering camera shutter via Scanlight hardware port...`);
          const pulse10ms = Math.min(Math.max(Math.round(this.shutterPulseLength * 100), 1), 255);
          await this.protocol.sendPacket(this.protocol.PKT_H2D_SHUTTER_PULSE, [pulse10ms]);
          const delayMs = (this.shutterPulseLength + this.postShutterDelay) * 1000;
          await new Promise(r => setTimeout(r, delayMs));
        }
      }
    } catch (err) {
      this.log(`[Scanlight Error] Automation aborted: ${err.message}`);
    } finally {
      this.isSequenceRunning = false;
      this.disableControlTriggers(false);
      if (this.connected) {
        if (this.postSequenceLight === "rgb") {
          this.setEnabledChannels([1, 1, 1, 0, 0]);
        } else if (this.postSequenceLight === "white") {
          this.setEnabledChannels([0, 0, 0, 1, 0]);
        } else {
          this.setEnabledChannels([0, 0, 0, 0, 0]);
        }
      } else {
        this.setEnabledChannels([0, 0, 0, 0, 0]);
      }
      this.log("[Scanlight] Automated sequence finished.");
    }
  }

  async shutterTest() {
    if (!this.connected) return;
    if (this.triggerMethod === "usb") {
      this.log("[Scanlight] Triggering test camera capture via USB...");
      try {
        const res = await fetch('/api/camera/capture', { method: 'POST' });
        const captureResult = await res.json();
        if (captureResult.success) {
          this.log(`[Scanlight] Shutter test success: Captured ${captureResult.path}`);
        } else {
          this.log(`[Scanlight Error] Shutter test failed: ${captureResult.message}`);
        }
      } catch (e) {
        this.log(`[Scanlight Error] Shutter test network error: ${e.message}`);
      }
    } else {
      this.log("[Scanlight] Triggering test camera shutter pulse via Scanlight hardware port...");
      const pulse10ms = Math.min(Math.max(Math.round(this.shutterPulseLength * 100), 1), 255);
      await this.protocol.sendPacket(this.protocol.PKT_H2D_SHUTTER_PULSE, [pulse10ms]);
    }
  }

  async startCalibration() {
    if (!this.connected) {
      alert("Scanlight is not connected. Please connect the device first.");
      return;
    }
    
    const dot = document.getElementById('summary-status-dot');
    const isMonitoring = (window.systemStatus === 'monitoring') || (dot && dot.classList.contains('monitoring'));
    if (!isMonitoring) {
      alert("Folder monitoring is not active. Please start the monitoring session in the 'Live Scanner' tab before calibrating.");
      this.log("[Scanlight] Calibration aborted: Monitoring session is not active.");
      return;
    }
    
    if (!confirm("Start Auto-Calibration? This will run a test RGB exposure capture at reference power (150) to measure exposure levels and calibrate the optimal light balance.")) {
      return;
    }
    
    this.log("[Scanlight] Starting Auto-Calibration...");
    this.isCalibrating = true;
    
    // Set colors to reference power 150
    this.red = 150;
    this.green = 150;
    this.blue = 150;
    
    document.getElementById("scanlight-red-slider").value = 150;
    document.getElementById("scanlight-red-val").value = 150;
    document.getElementById("scanlight-green-slider").value = 150;
    document.getElementById("scanlight-green-val").value = 150;
    document.getElementById("scanlight-blue-slider").value = 150;
    document.getElementById("scanlight-blue-val").value = 150;
    
    // Update color glow and light configuration
    this.updateColor();
    
    // Run the RGB sequence
    try {
      await this.runSequence("SequenceRGB", true);
      this.log("[Scanlight] Calibration sequence complete. Waiting for image analysis results...");
    } catch (err) {
      this.isCalibrating = false;
      this.log(`[Scanlight Error] Calibration sequence failed: ${err.message}`);
    }
  }

  cancelCalibration() {
    this.log("[Scanlight] Calibration canceled by user.");
    this.isCalibrating = false;
    this.isSequenceRunning = false;
    this.disableControlTriggers(false);
    
    if (this.connected) {
      if (this.postSequenceLight === "rgb") {
        this.setEnabledChannels([1, 1, 1, 0, 0]);
      } else if (this.postSequenceLight === "white") {
        this.setEnabledChannels([0, 0, 0, 1, 0]);
      } else {
        this.setEnabledChannels([0, 0, 0, 0, 0]);
      }
    }
    
    alert("Calibration canceled.");
  }

  handleCalibrationData(data) {
    if (!this.isCalibrating) return;
    
    this.log(`[Scanlight] Received exposure means from composite analysis: R=${data.r_mean.toFixed(0)}, G=${data.g_mean.toFixed(0)}, B=${data.b_mean.toFixed(0)}`);
    
    const r_mean = data.r_mean;
    const g_mean = data.g_mean;
    const b_mean = data.b_mean;
    
    // Check if means are valid
    if (r_mean <= 0 || g_mean <= 0 || b_mean <= 0) {
      this.log("[Scanlight Error] Invalid channel means received. Cannot calibrate.");
      alert("Calibration failed: One or more channel exposure means are zero. Please ensure your camera is taking pictures and files are being processed.");
      this.isCalibrating = false;
      return;
    }
    
    const targetExposure = 55000;
    const currentPower = 150;
    
    // Proportional scaling: targetPower = currentPower * (targetExposure / mean)
    let r_target = currentPower * (targetExposure / r_mean);
    let g_target = currentPower * (targetExposure / g_mean);
    let b_target = currentPower * (targetExposure / b_mean);
    
    this.log(`[Scanlight] Calculated raw targets: R=${r_target.toFixed(1)}, G=${g_target.toFixed(1)}, B=${b_target.toFixed(1)}`);
    
    // Handling saturation:
    // If any channel target exceeds 255, cap the highest channel at 255
    // and scale down other channels proportionally to maintain white balance.
    const maxTarget = Math.max(r_target, g_target, b_target);
    if (maxTarget > 255) {
      const scale = 255 / maxTarget;
      r_target *= scale;
      g_target *= scale;
      b_target *= scale;
      this.log(`[Scanlight Warning] Camera exposure is too low (underexposed). Even at maximum LED power (255), the target channel exposure of 55,000 could not be reached. Consider increasing camera exposure time, opening the lens aperture, or increasing ISO to avoid noise.`);
      this.log(`[Scanlight] Exposure targets capped. Scaling channels down by factor of ${scale.toFixed(3)} to preserve color balance.`);
    }
    
    // Constrain results between 0 and 255, rounding to integers
    this.red = Math.min(Math.max(Math.round(r_target), 0), 255);
    this.green = Math.min(Math.max(Math.round(g_target), 0), 255);
    this.blue = Math.min(Math.max(Math.round(b_target), 0), 255);
    
    // Update inputs and sliders in UI
    document.getElementById("scanlight-red-slider").value = this.red;
    document.getElementById("scanlight-red-val").value = this.red;
    document.getElementById("scanlight-green-slider").value = this.green;
    document.getElementById("scanlight-green-val").value = this.green;
    document.getElementById("scanlight-blue-slider").value = this.blue;
    document.getElementById("scanlight-blue-val").value = this.blue;
    
    // Reset calibration state
    this.isCalibrating = false;
    
    // Turn the light back on in RGB mode
    this.setEnabledChannels([1, 1, 1, 0, 0]);
    this.updateColor();
    
    this.log(`[Scanlight] Calibration complete! Optimal RGB values set to: R=${this.red}, G=${this.green}, B=${this.blue}`);
    
    // Alert the user
    alert(`Calibration complete!\nOptimal RGB values set to:\nRed: ${this.red}\nGreen: ${this.green}\nBlue: ${this.blue}`);
  }

  async loadDefault() {
    if (!this.connected) return;
    this.log("[Scanlight] Requesting default RGB startup configuration...");
    await this.protocol.sendPacket(this.protocol.PKT_H2D_GET_DEFAULT_RGB, []);
  }

  writeDefaultDialog() {
    if (!this.connected) return;
    if (confirm("Store current RGB values? The current RGB settings will be saved to nonvolatile flash memory and used as power-on defaults.")) {
      this.log("[Scanlight] Storing current configuration to flash memory...");
      const savePacket = [this.red, this.green, this.blue, 0, 0, 1]; // save preset flag set to 1
      this.protocol.sendPacket(this.protocol.PKT_H2D_SET_COLOR, savePacket);
    }
  }

  openTrimModal() {
    const modal = document.getElementById("scanlight-trim-modal");
    if (modal) modal.classList.add("active");
  }

  closeTrimModal() {
    const modal = document.getElementById("scanlight-trim-modal");
    if (modal) modal.classList.remove("active");
  }

  async saveTrimValues() {
    if (!this.connected) return;
    this.log("[Scanlight] Saving brightness trimming configuration to nonvolatile memory...");
    const trimPacket = [
      this.trimR & 0xff,
      this.trimG & 0xff,
      this.trimB & 0xff,
      this.trimW & 0xff
    ];
    await this.protocol.sendPacket(this.protocol.PKT_H2D_SET_TRIM, trimPacket);
    this.closeTrimModal();
  }

  async enterDFUMode() {
    if (!this.connected) return;
    if (confirm("WARNING: Put Scanlight in DFU upgrade mode? The USB Serial connection will close. Proceed?")) {
      this.log("[Scanlight] Rebooting device into firmware upgrade mode (DFU)...");
      await this.protocol.sendPacket(this.protocol.PKT_H2D_DFU_MODE, []);
      this.protocol.disconnect();
      this.connected = false;
      this.updateControlsState();
    }
  }

  // Presets Handlers (localStorage)
  loadPresetsFromStorage() {
    const stored = localStorage.getItem("rgb_presets");
    this.presets = stored ? JSON.parse(stored) : [];
    this.renderPresetsDropdown();
  }

  savePresetsToStorage() {
    localStorage.setItem("rgb_presets", JSON.stringify(this.presets));
    this.renderPresetsDropdown();
  }

  renderPresetsDropdown() {
    const select = document.getElementById("scanlight-preset-select");
    if (!select) return;
    
    select.innerHTML = '<option value="">-- Choose Preset --</option>';
    this.presets.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
      select.appendChild(opt);
    });
    
    if (this.selectedPresetName) {
      select.value = this.selectedPresetName;
    }
  }

  createPreset() {
    const name = prompt("Enter a name for the new preset:");
    if (!name) return;
    
    // Check duplicate
    if (this.presets.some(p => p.name.toLowerCase() === name.toLowerCase())) {
      alert("A preset with this name already exists.");
      return;
    }
    
    this.presets.push({ name, red: this.red, green: this.green, blue: this.blue });
    this.selectedPresetName = name;
    this.savePresetsToStorage();
    this.log(`[Scanlight] Saved preset: "${name}"`);
  }

  renamePreset() {
    if (!this.selectedPresetName) return;
    const idx = this.presets.findIndex(p => p.name === this.selectedPresetName);
    if (idx === -1) return;
    
    const newName = prompt("Enter a new name:", this.presets[idx].name);
    if (!newName || newName === this.selectedPresetName) return;
    
    this.presets[idx].name = newName;
    this.selectedPresetName = newName;
    this.savePresetsToStorage();
    this.log(`[Scanlight] Renamed preset to: "${newName}"`);
  }

  deletePreset() {
    if (!this.selectedPresetName) return;
    if (!confirm(`Delete preset "${this.selectedPresetName}"?`)) return;
    
    this.presets = this.presets.filter(p => p.name !== this.selectedPresetName);
    this.selectedPresetName = "";
    this.savePresetsToStorage();
    this.log("[Scanlight] Preset deleted.");
  }

  loadPreset() {
    if (!this.selectedPresetName) return;
    const p = this.presets.find(preset => preset.name === this.selectedPresetName);
    if (!p) return;
    
    this.red = p.red;
    this.green = p.green;
    this.blue = p.blue;
    
    document.getElementById("scanlight-red-slider").value = this.red;
    document.getElementById("scanlight-red-val").value = this.red;
    document.getElementById("scanlight-green-slider").value = this.green;
    document.getElementById("scanlight-green-val").value = this.green;
    document.getElementById("scanlight-blue-slider").value = this.blue;
    document.getElementById("scanlight-blue-val").value = this.blue;
    
    this.log(`[Scanlight] Loaded preset: "${p.name}"`);
    this.updateColor();
  }

  // UI state layout manipulation
  updateControlsState() {
    const wrapper = document.getElementById("panel-scanlight");
    if (!wrapper) return;
    
    const connectBtn = document.getElementById("scanlight-connect-btn");
    
    if (this.connected) {
      if (connectBtn) {
        connectBtn.textContent = "🔌 Disconnect Scanlight";
        connectBtn.className = "btn btn-primary monitoring-active";
      }
      wrapper.classList.remove("disconnected-state");
      wrapper.classList.add("connected-state");
      
      // Update info fields
      document.getElementById("scanlight-lbl-info").style.display = "block";
      document.getElementById("scanlight-lbl-noinfo").style.display = "none";
      document.getElementById("scanlight-lbl-hw").textContent = this.hwVersionString;
      document.getElementById("scanlight-lbl-fw").textContent = this.fwVersionString;
      
      // Toggle IR quick button and layout elements if scanlight v4
      const btnIR = document.getElementById("btn-sl-ir");
      if (btnIR) {
        btnIR.style.display = this.isSmallHW ? "none" : "flex";
      }
      const btnSeqRGBIR = document.getElementById("btn-seq-rgbir");
      if (btnSeqRGBIR) btnSeqRGBIR.style.display = this.isSmallHW ? "none" : "inline-flex";
      const btnSeqNWIR = document.getElementById("btn-seq-nwir");
      if (btnSeqNWIR) btnSeqNWIR.style.display = this.isSmallHW ? "none" : "inline-flex";
      const btnSeqBWIR = document.getElementById("btn-seq-bwir");
      if (btnSeqBWIR) btnSeqBWIR.style.display = this.isSmallHW ? "none" : "inline-flex";
      
      const trimBtn = document.getElementById("btn-sl-trim-open");
      if (trimBtn) trimBtn.style.display = this.isSmallHW ? "none" : "inline-block";
    } else {
      if (connectBtn) {
        connectBtn.textContent = "🔌 Connect Big Scanlight";
        connectBtn.className = "btn btn-primary";
      }
      wrapper.classList.add("disconnected-state");
      wrapper.classList.remove("connected-state");
      
      document.getElementById("scanlight-lbl-info").style.display = "none";
      document.getElementById("scanlight-lbl-noinfo").style.display = "block";
      document.getElementById("scanlight-lbl-temp").textContent = "--";
      document.getElementById("scanlight-lbl-vbus").textContent = "--";
      document.getElementById("scanlight-temp-warning").style.display = "none";
      document.getElementById("scanlight-power-warning").style.display = "none";
      document.getElementById("scanlight-low-power-banner").style.display = "none";

      const colorGlow = document.getElementById("scanlight-color-glow");
      if (colorGlow) {
        colorGlow.style.background = `transparent`;
        colorGlow.style.boxShadow = `none`;
      }
    }
  }

  disableControlTriggers(disable) {
    document.querySelectorAll(".disable-on-sequence").forEach(el => {
      el.disabled = disable;
      if (disable) {
        el.classList.add("disabled-input");
      } else {
        el.classList.remove("disabled-input");
      }
    });

    // Handle calibration button override
    const btn = document.getElementById("btn-sl-calibrate");
    if (btn) {
      if (disable) {
        if (this.isCalibrating) {
          btn.disabled = false;
          btn.classList.remove("disabled-input");
        } else {
          btn.disabled = true;
          btn.classList.add("disabled-input");
        }
      } else {
        btn.disabled = false;
        btn.classList.remove("disabled-input");
      }
    }
  }
}

// Instantiate global scanlight controller and expose on window for app.js
const scanlightController = new ScanlightUIController();
window.scanlightController = scanlightController;


document.addEventListener("DOMContentLoaded", () => {
  scanlightController.init();
});
