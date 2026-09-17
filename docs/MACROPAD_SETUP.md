# Megalodon Triple Knob Macro Pad (KB16) Setup for FilmConvert

This guide details the optimized VIA keymap layout tailored specifically for the **DOIO Megalodon KB16-01 Triple Knob Macro Pad** for rapid film scanning and exposure tuning.

---

## 🎛️ Physical Layout & Control Mapping

```
┌───────────────────────────────────────┬──────────────┐
│  [ Tab 1 ]   [ Tab 2 ]   [ Tab 3 ]   [ Monitor ]│  (•) KNOB 1  │ <- Shutter Speed
│  Live Scan     Batch      Gallery    Hot Folder│              │    CCW: Slower (-)
├───────────────────────────────────────┼──────────────┤    CW:  Faster (+)
│  [  Live  ]  [ Rotate ]  [ Margin  ] [ Rebate  ]│  (•) KNOB 2  │ <- Fine Focus
│    Feed        180°       Overlay    Eyedropper│  [ OLED Disp ]│    CCW: Near [ / CW: Far ]
├───────────────────────────────────────┼──────────────┤    Press: 1. Autofocus (U)
│  [★Auto-AF]  [★Auto-LED] [★Auto-RGB] [ Capture ]│              │
│  1. Autofocus 2.Auto-Tune 3.Auto Seq   4.RAW   │  (•) KNOB 3  │ <- Coarse Focus
├───────────────────────────────────────┤   (BIG KNOB) │    CCW: Coarse Near (Shift+[)
│  [ Shutter-] [ Shutter+] [ Help ? ]  [ Escape  ]│              │    CW:  Coarse Far  (Shift+])
│    Slower      Faster    Shortcuts    / Cancel │              │    Press: Shutter (Space)
└───────────────────────────────────────┴──────────────┘
```

---

## ⚙️ Key & Knob Functions

### 🔘 Rotary Encoders (Knobs)

| Control | Action | Workflow Function | Keycode |
| :--- | :--- | :--- | :--- |
| **Knob 1 (Top Small)** | **Rotate Left (CCW)** | Slower Shutter Speed (Step Down) | `-` (`KC_MINS`) |
| | **Rotate Right (CW)** | Faster Shutter Speed (Step Up) | `+` (`KC_EQL`) |
| | **Press** | Toggle Fullscreen Live View | `F` (`KC_F`) |
| **Knob 2 (Mid Small)** | **Rotate Left (CCW)** | Focus Near (Fine Step 1) | `[` (`KC_LBRC`) |
| | **Rotate Right (CW)** | Focus Far (Fine Step 1) | `]` (`KC_RBRC`) |
| | **Press** | 🎯 **1. Trigger Autofocus** | `Shift + A` / `U` (`KC_U`) |
| **Knob 3 (Bottom Big)** | **Rotate Left (CCW)** | Focus Near (Coarse Step 3) | `Shift + [` (`S(KC_LBRC)`) |
| | **Rotate Right (CW)** | Focus Far (Coarse Step 3) | `Shift + ]` (`S(KC_RBRC)`) |
| | **Press** | 📸 **4. Capture RAW Frame** | `Space` (`KC_SPC`) |

---

### ⌨️ 4×4 Matrix Keys

#### Row 1: Workspace Navigation & Hot Folder
- **Key (0,0)**: Switch to Live Scanner Tab (`Alt + 1` / `A(KC_1)`)
- **Key (0,1)**: Switch to Batch Processor Tab (`Alt + 2` / `A(KC_2)`)
- **Key (0,2)**: Switch to Scan Gallery Tab (`Alt + 3` / `A(KC_3)`)
- **Key (0,3)**: Toggle Hot Folder Monitor (`M` / `KC_M`)

#### Row 2: Live View & Framing Controls
- **Key (1,0)**: Toggle Live Video Feed (`V` / `KC_V`)
- **Key (1,1)**: Rotate Live View 180° (`R` / `KC_R`)
- **Key (1,2)**: Toggle Margin Exclusion Overlay (`O` / `KC_O`)
- **Key (1,3)**: Film Rebate Eyedropper Base Picker (`E` / `KC_E`)

#### Row 3: 🔥 The Frequent Auto Workflow Trio + Shutter
- **Key (2,0)**: 🎯 **1. Autofocus** (`Shift + A` / `KC_U`)
- **Key (2,1)**: ⚡ **2. Auto-Tune Optimal LEDs (ETTR)** (`T` / `KC_T`)
- **Key (2,2)**: ▶ **3. Auto R → G → B Triplet Sequence** (`A` / `KC_A`)
- **Key (2,3)**: 📸 **4. Capture Single RAW Frame** (`Space` / `KC_SPC`)

#### Row 4: Shutter Speed Steps & Help
- **Key (3,0)**: Shutter Speed Slower (`-` / `KC_MINS`)
- **Key (3,1)**: Shutter Speed Faster (`+` / `KC_EQL`)
- **Key (3,2)**: Keyboard Shortcuts Help Modal (`?` / `S(KC_SLSH)`)
- **Key (3,3)**: Escape / Cancel / Close Modal (`Esc` / `KC_ESC`)

---

## 🚀 How to Load into VIA

1. Connect your Megalodon KB16 macropad via USB.
2. Open [usevia.app](https://usevia.app) in your browser.
3. Click the **Save + Load** tab (floppy disk icon in VIA).
4. Click **Load Layout** and select either:
   - `~/Downloads/kb16_01.layout.json`
   - `~/Downloads/filmconvert_kb16_layout.json`
   - or `docs/megalodon_kb16_via_config.json`
5. It will load instantly into the macropad without errors.

---

## ⌨️ Full FilmConvert Keyboard Shortcuts Reference

For operators using standard computer keyboards or programming custom stream decks:

| Key Binding | Action / Feature | Notes |
| :--- | :--- | :--- |
| `Space` or `C` | **Capture RAW Frame** | Immediate high-res RAW exposure & download |
| `U` or `Shift + A` | **Trigger Autofocus** | Camera lens autofocus sequence |
| `T` | **Auto-Tune LEDs (ETTR)** | Calibrates LED brightness for maximum dynamic range |
| `A` | **Auto R-G-B Triplet Sequence** | Sequences Red → Green → Blue captures |
| `[` / `]` | **Fine Focus Step (Near / Far)** | Step 1 micro-focus movement |
| `{` / `}` *(Shift + `[` / `]`)* | **Coarse Focus Step (Near / Far)** | Step 3 coarse focus movement |
| `-` / `_` / `,` | **Shutter Speed Slower** | Step camera exposure time down |
| `=` / `+` / `.` | **Shutter Speed Faster** | Step camera exposure time up |
| `P` | **Toggle Focus Peaking** | High-contrast edge detection overlay |
| `H` | **Cycle Histogram Channel** | Cycles RGB, Red, Green, Blue, or Luminance |
| `E` | **Film Rebate Eyedropper** | Toggle interactive film base mask sampling |
| `O` | **Toggle Margins Overlay** | Display histogram border margin exclusion boxes |
| `R` | **Rotate Live View 180°** | Inverts viewfinder orientation |
| `V` or `L` | **Toggle Live View Feed** | Start/stop camera sensor preview stream |
| `F` | **Toggle Fullscreen** | Expand Live View to full display window |
| `M` | **Toggle Hot Folder Monitor** | Start/stop automatic capture watcher |
| `Alt + 1` | **Live Scanner Tab** | Switch to main camera & scan controls |
| `Alt + 2` | **Batch Processor Tab** | Switch to offline folder batch processing |
| `Alt + 3` | **Scan Gallery Tab** | Switch to positives gallery & contact sheets |
| `Alt + 4` | **Scanlight Controller Tab** | Switch to hardware LED controls |
| `?` or `Shift + /` | **Shortcuts Help Modal** | Show interactive keyboard shortcuts overlay |
| `Esc` | **Cancel / Close Modal** | Closes overlays, modals, and eyedropper |

