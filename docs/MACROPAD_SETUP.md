# Megalodon Triple Knob Macro Pad (KB16) Setup for FilmConvert

This guide details the VIA keymap layout tailored specifically for the **DOIO Megalodon KB16-01 Triple Knob Macro Pad** to streamline high-throughput film digitizing and scanning workflows.

---

## 🎛️ Physical Layout & Control Mapping

```
┌───────────────────────────────────────┬──────────────┐
│  [ Tab 1 ]   [ Tab 2 ]   [ Tab 3 ]   [ Tab 4 ] │  (•) KNOB 1  │ <- Zoom / View
│  Live Scan     Batch      Gallery    Scanlight │              │
├───────────────────────────────────────┼──────────────┤
│  [  Live  ]  [ Fullscr ] [ Rotate ]  [ Margin  ]│  (•) KNOB 2  │ <- Fine Focus
│    Feed        Toggle      180°       Overlay  │  [ OLED Disp ]│
├───────────────────────────────────────┼──────────────┤
│  [ Rebate ]  [ Auto-Tune][ Peaking ] [ Histo   ]│              │
│  Eyedropper   ETTR LEDs   Toggle     Channels  │  (•) KNOB 3  │ <- Coarse Focus
├───────────────────────────────────────┤   (BIG KNOB) │    & Shutter
│  [ Auto   ]  [ HotFolder][ Help ? ]  [ Escape  ]│              │
│  RGB Seq       Monitor   Shortcuts    / Cancel │              │
└───────────────────────────────────────┴──────────────┘
```

---

## ⚙️ Key & Knob Functions

### 🔘 Rotary Encoders (Knobs)

| Control | Action | Function | Keycode |
| :--- | :--- | :--- | :--- |
| **Knob 1 (Top Small)** | **Rotate Left (CCW)** | Zoom 1x (Full View) | `1` (`KC_1`) |
| | **Rotate Right (CW)** | Zoom 3x (Digital Loupe) | `2` (`KC_2`) |
| | **Press** | Toggle Fullscreen Live View | `F` (`KC_F`) |
| **Knob 2 (Mid Small)** | **Rotate Left (CCW)** | Focus Near (Fine Step 1) | `[` (`KC_LBRC`) |
| | **Rotate Right (CW)** | Focus Far (Fine Step 1) | `]` (`KC_RBRC`) |
| | **Press** | Trigger Autofocus Cycle | `Shift + A` / `U` (`KC_U`) |
| **Knob 3 (Bottom Big)** | **Rotate Left (CCW)** | Focus Near (Coarse Step 3) | `Shift + [` (`S(KC_LBRC)`) |
| | **Rotate Right (CW)** | Focus Far (Coarse Step 3) | `Shift + ]` (`S(KC_RBRC)`) |
| | **Press** | 📸 **Capture RAW Frame** | `Space` (`KC_SPC`) |

---

### ⌨️ 4×4 Matrix Keys

#### Row 1: Workspace Tab Switching
- **Key (0,0)**: Switch to Live Scanner Tab (`Alt + 1` / `A(KC_1)`)
- **Key (0,1)**: Switch to Batch Processor Tab (`Alt + 2` / `A(KC_2)`)
- **Key (0,2)**: Switch to Scan Gallery Tab (`Alt + 3` / `A(KC_3)`)
- **Key (0,3)**: Switch to Scanlight Controller Tab (`Alt + 4` / `A(KC_4)`)

#### Row 2: Live View & Framing Controls
- **Key (1,0)**: Toggle Live Video Feed (`V` / `KC_V`)
- **Key (1,1)**: Maximize / Fullscreen Live View (`F` / `KC_F`)
- **Key (1,2)**: Rotate Live View 180° (`R` / `KC_R`)
- **Key (1,3)**: Toggle Margin Exclusion Overlay (`O` / `KC_O`)

#### Row 3: Light & Focus Calibration
- **Key (2,0)**: Film Rebate Eyedropper Base Picker (`E` / `KC_E`)
- **Key (2,1)**: Auto-Tune Optimal LEDs to ETTR Target (`T` / `KC_T`)
- **Key (2,2)**: Toggle Focus Peaking Overlay (`P` / `KC_P`)
- **Key (2,3)**: Cycle Histogram Channels (`H` / `KC_H`)

#### Row 4: Capture Triggers & Shortcuts Help
- **Key (3,0)**: Auto R → G → B Capture Sequence (`A` / `KC_A`)
- **Key (3,1)**: Toggle Hot Folder Monitor (`M` / `KC_M`)
- **Key (3,2)**: Keyboard Shortcuts Help Modal (`?` / `S(KC_SLSH)`)
- **Key (3,3)**: Escape / Cancel / Close Modal (`Esc` / `KC_ESC`)

---

## 🚀 How to Load into VIA

1. Connect your Megalodon KB16 macropad via USB.
2. Open [usevia.app](https://usevia.app) (Chrome, Edge, Opera or WebHID-compatible browser) or open the VIA Desktop app.
3. Authorize the device when prompted.
4. Click the **Save + Load** tab (floppy disk icon in VIA).
5. Click **Load Layout** and select [`docs/megalodon_kb16_via_config.json`](./megalodon_kb16_via_config.json).
6. Test keys in the **Key Tester** tab.
