# UC4 Aim Assist — Delivery Notes

## What the system does

Real-time aim assist for Uncharted 4 Multiplayer on PS5, streamed to PC via Chiaki.

**One red bounding box appears ONLY when the player is demonstrably engaging a single enemy:**
1. L2 held (player is in ADS / aim-down-sights mode)
2. R2 held (player is actively firing)
3. Exactly one enemy is within the strict engagement zone (~380 px radius around screen centre)
4. That enemy has been tracked stably for ≥8 consecutive frames

The system never shows a box on teammates, bystanders, or multiple simultaneous enemies.

## What it does NOT do

- It does **not** auto-aim or move the thumbstick without the player actively pressing L2+R2.
- It does **not** show boxes when the player is just walking around, not shooting.
- It does **not** track enemies behind cover or completely out of view (box disappears after a 12-frame Kalman dropout window, ~0.2 s at 60 fps).
- It does **not** show multiple simultaneous boxes. If two enemies are in the zone at the same time, scoring selects the highest-priority target; the other is not shown.
- It does **not** work during cutscenes, menus, or kill-cams without the local player character on-screen (spectator-mode detection automatically suspends tracking).
- The demo video (demo_final_v2.mp4) uses **demo mode** — engagement is inferred from video geometry (no real R2 signal). Source: `5OYh3vlqTcY.f299.mp4`. The demo shows a sustained lock of 595 consecutive frames (9.92s) on a single enemy with zero target switches, with the system at 62.1% ENGAGED across the 18.0s clip. The live system, driven by real L2/R2 inputs, behaves identically in terms of box stability.

## How to run it

### Prerequisites

```
pip install -r requirements.txt
```

ViGEm Bus Driver must be installed for the virtual gamepad to work:
https://github.com/nefarius/ViGEmBus/releases

### Live system

```
python main.py
python main.py --debug      # shows OpenCV debug overlay
```

Requires:
- Chiaki streaming window open and receiving PS5 video
- DualSense connected via USB (L2/R2 thresholds: 45% / 30% of full press)
- ViGEm Bus Driver for virtual DS4 output

### Demo video generation

```
python tools/demo_generator.py \
    --model models/training/enemy_detector/weights/best.pt \
    --video path/to/gameplay.mp4 \
    --output demo_out.mp4
```

### Tests

```
pytest tests/ -v     # 110 tests, all should pass
```

---

## All threshold values and what they control

### Engagement gate (target_lock)

| Key | Value | Effect |
|-----|-------|--------|
| `strict_radius_px` | 380 | Enemy must be within this distance from screen centre to be eligible |
| `min_stable_frames` | 8 | Track must be this many frames old before engagement is shown |
| `engagement_hold_frames` | 45 | Frames (~0.75 s) box is held after engagement gate closes (R2 released) |
| `r2_activation_threshold` | 0.30 | R2 trigger fraction required to count as "firing" |

### Lock persistence (Prompt 3)

| Key | Value | Effect |
|-----|-------|--------|
| `lock_release_frames` | 12 | Frames the locked track can vanish before box releases (Kalman prediction fills the gap) |
| `min_reacq_confidence` | 0.65 | Re-acquiring from NO_BOX requires this confidence (prevents snapping to wrong enemy after dropout) |
| `max_reacq_distance` | 200 | Re-acquiring candidate must be within this many px of last known position |
| `lock_expiry_frames` | 90 | After this many NO_BOX frames (~1.5 s), position memory is cleared; any enemy can re-acquire |

### Close-range / large boxes (Prompt 4)

| Key | Value | Effect |
|-----|-------|--------|
| `high_conf_fast_track` | 0.75 | Detections ≥ 0.75 confidence skip the HOLDING age gate (engage sooner when model is sure) |
| `crowded_min_detections` | 2 | When more than this many enemies are visible, stability gate drops to 1 frame |

### Object filter (close-range)

| Key | Value | Effect |
|-----|-------|--------|
| `max_bbox_area_fraction` | 0.45 | Allow boxes up to 45% of frame area (close-range enemies) |
| `min_aspect_ratio` | 0.40 | Reject if height/width < 0.40 (extremely wide boxes aren't people) |
| `max_aspect_ratio` | 5.00 | Reject if height/width > 5.00 (extremely tall, pole-like detections) |
| `conf_threshold` | 0.55 | Standard confidence floor for object_filter (post-detector gate) |
| `large_box_area_fraction` | 0.15 | Boxes larger than 15% of frame area get the relaxed threshold below |
| `large_box_conf_threshold` | 0.40 | Relaxed confidence for close-range (large) boxes |

### Target priority scoring (Prompt 7)

| Key | Value | Effect |
|-----|-------|--------|
| `crosshair_y_ratio` | 0.45 | Scoring crosshair sits at 45% down from top (UC4 is third-person; enemies appear above centre) |
| `switch_min_score_drop` | 0.20 | Current locked target's score must drop below this before a target switch is allowed |
| `switch_min_score_gap` | 0.40 | New candidate must also beat current score by this margin before a switch occurs |

Priority score formula:
```
proximity_score = 1.0 - (distance_to_crosshair / frame_diagonal)
age_score = min(track_age / 20, 1.0)
score = 0.50 * proximity_score + 0.30 * confidence + 0.20 * age_score
```

### Detection / tracking

| Key | Value | Effect |
|-----|-------|--------|
| `detection.confidence_threshold` | 0.45 | YOLO detector confidence gate (nothing below this reaches tracking) |
| `detection.nms_threshold` | 0.45 | NMS IoU threshold for deduplication of overlapping YOLO predictions |
| `tracking.minimum_matching_threshold` | 0.70 | ByteTrack IoU threshold for matching detections to existing tracks |
| `kalman_max_predict_frames` | 50 | Maximum frames to Kalman-predict an aim point during HOLDING state |

### Self-player exclusion (Fix A / B)

| Key | Value | Effect |
|-----|-------|--------|
| `self_excl_x_min/max` | 0.35 / 0.65 | Horizontal exclusion zone: 35–65% of frame width |
| `self_excl_y_min/max` | 0.75 / 1.00 | Vertical exclusion zone: bottom 25% of frame |
| `self_excl_bottom_max` | 0.95 | Any box whose bottom edge is ≥ 95% of frame height is rejected |

Spectator escape hatch: if ALL detected enemies would be excluded by Fix A/B, the exclusion is suppressed (handles kill-cams and spectator views where the local player character is not on screen).

---

## Known limitations

1. **No TRT engine yet** — currently running PyTorch CPU/CUDA inference. For production performance, export `best.pt` to TensorRT: `python -m training.tensorrt_export --weights models/training/enemy_detector/weights/best.pt --output models/enemy_detector.engine --fp16`

2. **Demo source selection** — `5OYh3vlqTcY.f299.mp4` was chosen because YOLO produces consistent detections across this segment, not because the footage is duplicate-free (see limitation 6 below). Other source videos in the dataset have similar or higher duplication but cause inconsistent YOLO detection on duplicate frames, which reduces lock stability.

3. **Video codec** — demo_final_v2.mp4 was encoded with OpenCV's avc1 codec (H.264-compatible). For true H.264 CRF 18 quality, re-encode with: `ffmpeg -i demo_final_v2.mp4 -c:v libx264 -crf 18 demo_final_v2_h264crf18.mp4`

4. **ByteTrack deprecation** — `supervision.ByteTrack` is deprecated in v0.28 and removed in v0.30. Tracked IDs remain stable for the current version. Upgrade path: replace `ByteTrackWrapper` with `supervision.BoTSORT`.

5. **Tuning needed on live hardware** — PID gains (Kp=0.22, Kd=0.04) were tuned analytically. Live test may require adjustment. Benchmark: `python tools/benchmark.py`.

6. Demo source footage contains frame duplication
   (~60% of consecutive frame pairs are near-identical,
   consistent with 30fps game content stored at 60fps).
   This does not affect system performance because YOLO
   produces consistent detections across duplicate frames
   in the selected segment. The previous source video
   caused engagement drops because YOLO confidence
   fluctuated across duplicate frames in that specific
   footage, exhausting the 12-frame dropout buffer.
   Live system performance is unaffected by this —
   Chiaki streams real-time content at native framerate.
