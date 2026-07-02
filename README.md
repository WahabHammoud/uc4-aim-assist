# UC4 Aim Assist

Real-time AI aim assist for **Uncharted 4 Multiplayer** on PS5,
streamed to PC via Chiaki.

## What it does
- Detects enemies in real time using YOLOv8 computer vision
- Shows exactly ONE stable red bounding box on the active target
- Never shows boxes on teammates or the player's own character
- Gently assists the right stick via virtual DS4 controller

## How it works
Chiaki stream → YOLOv8 detection → ByteTrack → TargetLock → PID → ViGEm

## Requirements
- Windows 10/11
- Python 3.10+
- NVIDIA GPU recommended
- Chiaki (PS5 streaming)
- ViGEm Bus Driver

## Installation
pip install -r requirements.txt

## Run
python main.py
python main.py --debug

## Run tests
pytest tests/ -v

## Architecture
EnemyDetector → EnemyClassifier → ObjectFilter →
ByteTrack → TargetLock → DualAxisPID → VirtualGamepad

## Results
- mAP50: 0.726
- 110/110 tests passing
- Max consecutive lock: 9.92 seconds

---

---

## Architecture

```
Chiaki window (screen)
        │
        ▼ mss capture (threaded)
┌─────────────────────────────────────────────────────────────┐
│                    InferencePipeline                        │
│                                                             │
│  Frame ──► EnemyDetector (YOLOv8 TensorRT ~3 ms)           │
│             │                                               │
│             ▼                                               │
│         EnemyClassifier (HSV marker colour ~0.5 ms)         │
│          • red marker  → enemy                              │
│          • blue marker → teammate (REJECT)                  │
│          • no marker   → motion filter (frame-diff)         │
│             │                                               │
│             ▼                                               │
│         ObjectFilter (HUD zones, aspect ratio, area)        │
│             │                                               │
│             ▼                                               │
│         ByteTrackWrapper (persistent track IDs ~0.3 ms)     │
│             │                                               │
│             ▼                                               │
│         TargetLock (state machine + Kalman predictor)       │
│          L2 held → lock nearest enemy in aim cone           │
│          lock NEVER jumps to different enemy                │
│          target lost → Kalman predicts for ≤50 frames       │
│             │                                               │
│             ▼                                               │
│         DualAxisPID (separate X / Y, with deadzone + EMA)   │
│             │                                               │
│             ▼                                               │
│   Physical DualSense ──► VirtualGamepad (vgamepad / ViGEm) │
│     (DualSenseReader)       │                               │
│                             ▼                               │
│                       Chiaki → PS5                          │
└─────────────────────────────────────────────────────────────┘
```

**Total pipeline latency target: < 10 ms**

Latency breakdown (run `python tools/benchmark.py` on your hardware to get actual numbers):
| Stage            | Expected (TRT, RTX 3090+) |
|------------------|--------------------------|
| Screen capture   | ~0.8 ms                  |
| TensorRT detect  | ~3–5 ms                  |
| HSV classify     | ~0.4 ms                  |
| Object filter    | ~0.1 ms                  |
| ByteTrack        | ~0.3 ms                  |
| Target lock      | ~0.1 ms                  |
| PID + gamepad    | ~0.3 ms                  |
| **Total**        | **~5–7 ms**              |

> Note: TensorRT engine must be exported first (`python -m training.tensorrt_export`).
> PyTorch-only inference on CPU is ~50–100 ms and NOT suitable for live use.

---

## System Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA RTX 3070 or better (CUDA 11.8+) |
| CPU | Intel i7-10700 / AMD Ryzen 7 5700X |
| RAM | 16 GB |
| OS | Windows 10/11 64-bit |
| Python | 3.10 – 3.12 |
| Driver | NVIDIA 535+ |

---

## Prerequisites (install in this order)

### 1. ViGEm Bus Driver
Download and install from:
`https://github.com/nefarius/ViGEmBus/releases`

This is a kernel-mode driver that creates virtual gamepads.
**The system cannot send controller commands without it.**

### 2. PyTorch with CUDA
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Match your CUDA version (cu118 / cu121 / cu124).

### 3. TensorRT (for sub-10 ms inference)
Download TensorRT 10.x from NVIDIA:
`https://developer.nvidia.com/tensorrt`

Install the Python wheels:
```powershell
pip install tensorrt
```

### 4. Python dependencies
```powershell
cd uc4_aim_assist
pip install -r requirements.txt
```

---

## Setup & Deployment

### Step 1 — Prepare the dataset

```powershell
# Extract frames from DevoManiac YouTube videos
python -m training.frame_extractor `
    --urls "https://youtube.com/..." `
    --output dataset/frames/raw `
    --fps 3

# Annotate enemies using the GUI
python tools/annotate_gui.py `
    --frames dataset/frames/raw `
    --labels dataset/frames/labels

# Organise into train/val/test splits
python -m training.dataset_pipeline `
    --frames dataset/frames/raw `
    --labels dataset/frames/labels `
    --output dataset/yolo
```

### Step 2 — Train YOLOv8

```powershell
python -m training.trainer `
    --data dataset/yolo/data.yaml `
    --model yolov8n.pt `
    --epochs 150 `
    --batch 16 `
    --device 0
```

Target metrics: mAP50 > 0.70, mAP50-95 > 0.45.

### Step 3 — Export to TensorRT

```powershell
python -m training.tensorrt_export `
    --weights models/training/enemy_detector/weights/best.pt `
    --output  models/enemy_detector.engine
```

The engine is device-specific — export on the deployment machine.

### Step 4 — Benchmark

```powershell
python tools/benchmark.py `
    --source dataset/frames/raw `
    --model  models/enemy_detector.engine `
    --iterations 500
```

Confirm total avg < 10 ms before going live.

### Step 5 — Run

```powershell
# 1. Start Chiaki and connect to PS5
# 2. Configure Chiaki → Settings → Gamepad → select "Virtual DS4"
#    (the gamepad our system creates)
# 3. Connect your physical DualSense via USB

# Run with debug overlay
python main.py --debug

# Run silently (production)
python main.py
```

---

## Configuration

All settings are in `config/config.yaml`.  Key sections:

### `detection`
- `model_path` — TensorRT engine path (auto-fallback to `.pt`).
- `confidence_threshold` — Lower → more detections but more false positives.

### `enemy_classification`
- HSV ranges for red / blue marker detection.
- `min_marker_pixels` — How many coloured pixels confirm a marker.

### `target_lock`
- `aim_cone_degrees` — Cone width for initial target selection.
- `kalman_max_predict_frames` — How many frames to predict when target disappears.

### `pid`
- `assist_strength` — Scale the final correction (0 = off, 1 = full).
- `output_smoothing` — Higher = smoother but slower response.
- `deadzone_fraction` — Dead zone to prevent micro-stick jitter.

### `controller`
- `l2_activation_threshold` — L2 must exceed this (0–1) to activate assist.

---

## Target Lock Behaviour

```
L2 released
    → State: UNLOCKED  (no assist)

L2 held, enemies visible within aim cone
    → State: LOCKED on nearest enemy (ID frozen)
    → Aim gently corrects toward chest/neck ROI of locked enemy
    → Other enemies IGNORED completely

L2 held, locked enemy temporarily hidden (smoke, corner)
    → State: PREDICTING (Kalman filter extrapolates position)
    → Stick continues gentle correction toward predicted position
    → If enemy reappears within reacquire_distance_px → LOCKED

L2 held, lock times out (> kalman_max_predict_frames lost)
    → State: UNLOCKED

L2 held, user aims far from locked target AND another enemy is closer
    → State: LOCKED on new target (manual retarget)
```

---

## Enemy Classification Logic

Each detected bounding box is passed to `EnemyClassifier`:

1. Extract a small region **above** the bounding box (where floating marker appears).
2. Convert to HSV and count red pixels (H=0–10 or H=165–180).
3. If red pixels ≥ threshold → **enemy** (aim at it).
4. If blue pixels ≥ threshold → **teammate** (reject silently).
5. If no clear marker → check motion (frame-diff):
   - Changed pixels ≥ threshold → tentative enemy.
   - No motion → static object (statue, painting, scenery) → reject.

---

## ROI (Region of Interest)

The system only aims at the **waist-to-head** region of the enemy body.

Configured by `roi.y_start_ratio` and `roi.y_end_ratio` (fraction of bbox height):
- `0.10` → 10% from top (buffer above head)
- `0.65` → 65% from top (waist)
- Aim point at `0.30` → upper chest / neck area

The debug overlay shows the ROI as a yellow box inside the green lock box.

---

## File Structure

```
uc4_aim_assist/
├── config/
│   └── config.yaml              ← All tunable parameters
├── src/
│   ├── capture/
│   │   └── chiaki_capture.py    ← Threaded Chiaki screen capture
│   ├── detection/
│   │   ├── detector.py          ← YOLOv8 / TensorRT inference
│   │   ├── enemy_classifier.py  ← HSV red/blue marker analysis
│   │   └── object_filter.py     ← HUD zone / aspect / area filter
│   ├── tracking/
│   │   ├── bytetrack_wrapper.py ← Persistent track IDs
│   │   ├── kalman_predictor.py  ← Kalman prediction when target hidden
│   │   └── target_lock.py       ← Lock state machine
│   ├── control/
│   │   ├── pid_controller.py    ← Dual-axis PID with deadzone + EMA
│   │   ├── dualsense_reader.py  ← Physical DualSense HID reader
│   │   └── virtual_gamepad.py   ← Virtual DS4 output via ViGEm
│   ├── pipeline/
│   │   └── inference_pipeline.py ← Main orchestrator
│   └── utils/
│       ├── logger.py             ← Structured logging
│       └── profiler.py           ← Per-section latency profiler
├── training/
│   ├── frame_extractor.py       ← yt-dlp + OpenCV frame extraction
│   ├── dataset_pipeline.py      ← Train/val/test split + data.yaml
│   ├── trainer.py               ← YOLOv8 fine-tuning
│   └── tensorrt_export.py       ← .pt → .engine conversion
├── tools/
│   ├── annotate_gui.py          ← Semi-auto annotation GUI
│   └── benchmark.py             ← End-to-end latency benchmark
├── models/                      ← Place trained models here
├── main.py                      ← Entry point
└── requirements.txt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `hid.HIDException` — DualSense not found | Connect DualSense via USB before launching |
| `vgamepad` error | Install ViGEm Bus Driver first |
| Chiaki not found | Ensure Chiaki window title contains "Chiaki"; check `capture.window_title` in config |
| `FileNotFoundError` — no engine | Run TensorRT export step first, or use `.pt` fallback |
| Latency > 10 ms | Ensure TensorRT engine is used (not PyTorch), lower `input_size` to 480 |
| Aim assist too aggressive | Decrease `pid.assist_strength` and/or `pid.x.max_output` |
| Lock jumps between enemies | Increase `target_lock.reacquire_distance_px` threshold |
| Teammates detected as enemies | Check HSV blue ranges in `enemy_classification`; increase `min_marker_pixels` |

---

## Deployment Checklist

- [ ] ViGEm Bus Driver installed
- [ ] PyTorch CUDA version installed (matches system CUDA)
- [ ] TensorRT installed and `pip install tensorrt` succeeded
- [ ] `pip install -r requirements.txt` succeeded
- [ ] Dataset annotated and YOLOv8 trained (`mAP50 > 0.70`)
- [ ] TensorRT engine exported to `models/enemy_detector.engine`
- [ ] Benchmark passes (avg < 10 ms)
- [ ] Chiaki open and streaming PS5 gameplay
- [ ] DualSense connected via USB
- [ ] Chiaki configured to use virtual gamepad
- [ ] `python main.py --debug` shows correct detections
- [ ] Lock behaviour verified (no auto-switching)

---

*Built for Uncharted 4 Multiplayer on PS5 via Chiaki remote play.*
*Accessibility tool — assists with precise aiming.*
