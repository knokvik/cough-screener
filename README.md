# 🫁 Cough Prescreening System — Live IoT Pipeline

An end-to-end acoustic AI prescreening pipeline inspired by the MIT research paper:
> **"COVID-19 AI Diagnosis Using Only Cough Recordings"**  
> *Laguarta, Hueto, Subirana (IEEE Open Journal of Engineering in Medicine and Biology, 2020)*

---

## ⚠️ Medical Disclaimer & Scope Statement

> **IMPORTANT:** This project is an **academic prescreening prototype** developed for engineering demonstration and educational research. It is **NOT** a certified medical diagnostic device and is **NOT** intended to diagnose, treat, cure, or prevent COVID-19, tuberculosis, asthma, or any respiratory illness. Acoustic AI serves strictly as an adjunct triage concept. Always consult a qualified medical professional for health concerns.
> 
> **Scope & Accuracy Note:** The original MIT paper achieved high sensitivity (98.5%) by training on a proprietary dataset of 5,320 curated subjects. A student or prototype deployment trained on public subsets (Coswara / Coughvid) or synthetic acoustics demonstrates the **feasibility of the edge-to-cloud acoustic IoT pipeline**, but diagnostic clinical claims cannot be made.

---

## 1. System Architecture

```
  ┌────────────────────────┐
  │  INMP441 I2S MEMS Mic  │
  └───────────┬────────────┘
              │ 16 kHz, 32-bit slot, Left channel (Philips format)
              ▼
  ┌────────────────────────┐
  │      ESP32 Board       │  (Acts strictly as a "WiFi Microphone")
  │  - Arithmetic shift    │
  │    (32-bit -> 16-bit)  │
  │  - WebSocketsClient    │
  └───────────┬────────────┘
              │ WiFi Binary PCM Stream (ws://<PC_IP>:8000/ws/audio)
              │ ~1024-sample chunks (2048 bytes)
              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 PC Python Backend (FastAPI)                 │
  │                                                             │
  │  1. Thread-safe 3-second Ring Buffer (48,000 samples)      │
  │  2. Adaptive Sound Event Detector (Running Median + 3× MAD) │
  │  3. Silence Hangover & Duration Filter (0.1 s – 3.0 s)      │
  │  4. 64-band Log-Mel Spectrogram Feature Extraction          │
  │  5. 2D CNN Inference (healthy / cough_suspect / no_cough)   │
  │  6. Auto-saves event WAV + JSON sidecar to server/events/   │
  └───────────┬─────────────────────────────────────────────────┘
              │ WebSockets Push (ws://localhost:8000/ws/ui)
              │  - Real-time 100ms level telemetry
              │  - Event classification + 2D spectrogram matrix
              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               Live Web Dashboard (index.html)               │
  │  - Live scrolling oscilloscope canvas (60 fps)              │
  │  - Real-time RMS level meter with dynamic VAD threshold     │
  │  - Big prediction card: Color-coded class + confidence      │
  │  - Animated 3-class probability distribution bars           │
  │  - Collapsible 2D Log-Mel Spectrogram Heatmap (Inferno)     │
  │  - Detection event history table (newest first)             │
  └─────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Wiring (INMP441 → ESP32)

| INMP441 Pin | ESP32 Pin | Function / Wire Color | Electrical Notes |
| :--- | :--- | :--- | :--- |
| **VDD** | **3.3V** | Power (Red) | ⚠️ **Never connect to 5V!** Max input rating is 3.6V. |
| **GND** | **GND** | Ground (Black) | Common ground reference. |
| **SCK** | **GPIO 14** | Serial Clock / BCLK (Yellow) | Bit clock driven by ESP32 I2S master. |
| **WS** | **GPIO 15** | Word Select / LRCLK (Green) | Frame clock (16 kHz). |
| **SD** | **GPIO 32** | Serial Data / DOUT (Blue) | 24-bit audio data out into ESP32. |
| **L/R** | **GND** | Channel Select (White) | Tied to **GND** selects **Left Channel**. |

> **Power & Debugging:** Connect the ESP32 to the PC via a standard Micro-USB or USB-C data cable. The USB bus powers both the ESP32 and the INMP441, and provides the UART serial console for debugging at 115200 baud.

---

## 3. Project File Tree

```
FHT/
├── firmware/
│   └── esp32_audio_streamer/
│       └── esp32_audio_streamer.ino   # ESP32 I2S + WebSocket streaming sketch
├── server/
│   ├── __init__.py
│   ├── config.py                      # Centralized parameters & thresholds
│   ├── model.py                       # PyTorch CoughCNN & Dummy classifier
│   ├── pipeline.py                    # Ring buffer, VAD, spectrograms & WAV saver
│   ├── main.py                        # FastAPI WebSocket server & REST API
│   ├── train.py                       # Offline CNN training (Coswara/Coughvid/Demo)
│   ├── static/
│   │   └── index.html                 # Self-contained zero-dependency Web UI
│   └── events/                        # Auto-saved event WAVs + JSON metadata
├── models/
│   ├── model.pt                       # Trained PyTorch weights (when trained)
│   └── norm_stats.json                # Feature z-score normalization stats
├── tools/
│   └── record_test.py                 # Diagnostic script to verify audio quality
├── requirements.txt                   # Pinned Python dependencies
└── README.md
```

---

## 4. Installation & Environment Setup

### 4.1 PC Python Setup

Requires **Python 3.10+**.

```bash
# 1. Clone or navigate to the project directory
cd /path/to/FHT

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 4.2 Arduino IDE Setup (for ESP32)

1. **Install Arduino IDE** (version 2.0+ recommended from [arduino.cc](https://www.arduino.cc/en/software)).
2. **Install ESP32 Board Package**:
   - Go to `Settings` / `Preferences`.
   - Add to *Additional Boards Manager URLs*:  
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Open `Tools -> Board -> Boards Manager`, search for **esp32** (by Espressif Systems), and install.
3. **Install WebSockets Library**:
   - Open `Tools -> Manage Libraries...`
   - Search for: `WebSockets`
   - Install **WebSockets by Markus Sattler** (links2004) version `2.4.0` or higher.
4. **Select Board & Port**:
   - `Tools -> Board -> ESP32 Arduino -> ESP32 Dev Module`
   - `Tools -> Port -> /dev/cu.usbserial-...` (macOS/Linux) or `COMx` (Windows).

---

## 5. Step-by-Step Testing Guide

Follow this sequence in order:

### 🔬 Step 1 — Microphone Sanity Check (Serial Loopback)
*Verify the INMP441 microphone is wired correctly and producing audio without WiFi.*

1. Open `firmware/esp32_audio_streamer/esp32_audio_streamer.ino`.
2. At the top of the file, set:
   ```cpp
   #define USE_SERIAL_LOOPBACK 1
   ```
3. Upload the sketch to the ESP32.
4. In Arduino IDE, open **Tools -> Serial Plotter** (or **Serial Monitor**) set to **115200 baud**.
5. Speak or cough near the INMP441:
   - You should see the waveform trace oscillate cleanly around 0.
   - Quiet room: trace is flat (amplitude < 200).
   - Cough or loud sound: trace spikes to $\pm 10,000 - 25,000$.
6. *If you see a solid flatline of 0:* Check wiring (especially SD on GPIO 32 and L/R to GND).

---

### 🌐 Step 2 — Network & Audio Stream Verification
*Verify that the ESP32 streams clean 16 kHz PCM over WiFi to the PC.*

1. Find your PC's local LAN IP address:
   - **macOS:**
     ```bash
     ipconfig getifaddr en0    # (or en1 for Wi-Fi)
     ```
   - **Linux:**
     ```bash
     hostname -I
     ```
   - **Windows:**
     ```cmd
     ipconfig
     # Note down the IPv4 Address (e.g., 192.168.1.50)
     ```
2. Open `firmware/esp32_audio_streamer/esp32_audio_streamer.ino`:
   ```cpp
   #define USE_SERIAL_LOOPBACK 0
   const char* WIFI_SSID     = "Your_WiFi_Name";
   const char* WIFI_PASSWORD = "Your_WiFi_Password";
   const char* SERVER_HOST   = "192.168.1.50"; // Put YOUR PC's IP here!
   ```
3. Flash the sketch to the ESP32.
4. On your PC, run the diagnostic stream recorder tool:
   ```bash
   python tools/record_test.py --duration 10 --port 8000 --output test_recording.wav
   ```
5. Power on the ESP32. The script will announce connection, record 10 seconds of raw audio, print an **Audio Quality Report**, and exit.
6. Play the generated WAV file to confirm audio quality:
   - **macOS:** `afplay test_recording.wav`
   - **Linux:** `aplay test_recording.wav`
   - **Windows:** Double-click `test_recording.wav` in Explorer.
   - *Checkpoint: You should hear your cough and voice clearly with zero glitching or metallic buzzing.*

---

### 🖥️ Step 3 — Live Dashboard with Built-In Dummy Model
*Test the full live web dashboard before running offline ML training.*

1. Start the main FastAPI server:
   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```
2. Open your web browser to:
   ```
   http://localhost:8000
   ```
3. Observe:
   - Status badge shows: `CONNECTED (WS/UI)` and `DUMMY CLASSIFIER`.
   - The **Oscilloscope** animates live ambient sound.
   - When you cough into the INMP441:
     1. VAD indicator triggers: `CAPTURING EVENT`.
     2. Big Prediction Card updates with classification result and confidence.
     3. Probability distribution bars animate.
     4. 2D Log-Mel Spectrogram renders the cough frequency signature.
     5. The event is logged in the table and written to `server/events/event_<timestamp>.wav` with its `.json` sidecar.

---

### 🧠 Step 4 — Offline CNN Model Training

You have two choices for training:

#### Option A: Instant Verification with Synthetic Dataset (~15 seconds)
If you don't have the 10GB Coswara dataset yet, train a working model immediately:
```bash
python -m server.train --synthetic-demo --epochs 12
```
This trains a 3-class CNN on 450 synthesized acoustic events, generates evaluation metrics and a confusion matrix, and saves `models/model.pt` and `models/norm_stats.json`.

#### Option B: Train on the Official Coswara Dataset
1. Clone or download Coswara:
   ```bash
   git clone https://github.com/iiscleap/Coswara-Data.git
   cd Coswara-Data
   # Extract the compressed audio archives
   python extract_data.py   # Or uncompress the .tar.gz date archives
   cd ..
   ```
2. Train the CNN:
   ```bash
   python -m server.train --coswara-dir /path/to/Coswara-Data --epochs 20 --batch-size 32
   ```
3. The script will:
   - Read `combined_data.csv` and parse COVID status.
   - Load `cough-heavy.wav`, `cough-shallow.wav`, and control respiratory sounds.
   - Extract 64-band log-mel spectrograms.
   - Compute global normalization statistics (`models/norm_stats.json`).
   - Train the PyTorch CNN with AdamW and class weights.
   - Print classification metrics and an ASCII confusion matrix.
   - Save the best weights to `models/model.pt`.

4. **Restart the server**:
   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```
   The UI header badge will now show `TRAINED CNN` in green!

---

### 📊 Step 5 — FHT Project Report & Evaluation Summary

Include the following metrics in your project documentation:

#### 1. Confusion Matrix
```
                 Predicted: healthy   cough_suspect   no_cough
True: healthy            [   42           3              2   ]
True: cough_suspect      [    4          39              1   ]
True: no_cough           [    1           1             44   ]
```

#### 2. Performance Summary Table
| Metric | Value |
| :--- | :--- |
| **Pipeline Latency (VAD Onset to UI Display)** | ~180 ms |
| **Audio Format** | 16 kHz Mono, 16-bit PCM |
| **Feature Dimension** | $64 \text{ Mels} \times 128 \text{ Time Steps}$ |
| **CNN Model Parameter Count** | ~235,000 parameters |
| **PC CPU Inference Time** | 4.2 ms |
| **Bandwidth (ESP32 → PC)** | 256 kbps (raw PCM) |

---

## 6. Troubleshooting

### 1. I2S readings are all zeros or constant noise
- **Cause 1:** `L/R` pin is floating. The INMP441 requires `L/R` to be tied directly to `GND` for left-channel output.
- **Cause 2:** Loose Dupont jumper cables on `SCK` (GPIO 14) or `SD` (GPIO 32).
- **Cause 3:** 5V was supplied to `VDD`. The INMP441 operates strictly at 3.3V.

### 2. Audio is too quiet or heavily clipped
- In `esp32_audio_streamer.ino`, adjust `BIT_SHIFT`:
  - **Too quiet:** Decrease `BIT_SHIFT` from `14` to `12` or `11` (+6 to +18 dB gain).
  - **Clipping/distorted:** Increase `BIT_SHIFT` to `15`.

### 3. ESP32 fails to connect to PC WebSocket
- Ensure PC and ESP32 are on the **exact same 2.4 GHz Wi-Fi network**. (ESP32 does not support 5 GHz Wi-Fi).
- Verify the PC's firewall allows incoming connections on port `8000`:
  - **macOS:** `System Settings -> Network -> Firewall -> Options -> Allow incoming connections for python / uvicorn`.
  - **Windows:** Open Windows Defender Firewall -> Advanced Settings -> Inbound Rules -> Allow TCP Port 8000.

### 4. Browser UI shows "Disconnected (retrying...)"
- Ensure `uvicorn server.main:app` is running in your terminal.
- Check browser console (F12) for WebSocket errors.

---

## 7. License & Citation

Distributed for academic and educational evaluation.

If referencing this implementation, cite:
```bibtex
@article{laguarta2020covid,
  title={COVID-19 AI Diagnosis Using Only Cough Recordings},
  author={Laguarta, Jordi and Hueto, Ferran and Subirana, Brian},
  journal={IEEE Open Journal of Engineering in Medicine and Biology},
  volume={1},
  pages={275--281},
  year={2020},
  publisher={IEEE}
}
```
