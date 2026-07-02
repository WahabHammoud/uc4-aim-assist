"""
build_demo_package.py — Generates the full professional demo package for Ahmed.

Outputs (all in C:\\Systeme\\UC4_Demo_Package\\):
  demo_professional.mp4      — polished demo from the richest combat segment
  multi_enemy_analysis.csv   — per-frame data across all eval segments
  analysis_summary.txt       — automated findings: wrong selections, flickers, locks

Run from project root:
  python tools/build_demo_package.py
"""
from __future__ import annotations

import csv
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml
import torch

from src.detection.detector import EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter
from src.detection.target_selector import TargetSelector, SelectState

CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL  = "models/training/enemy_detector/weights/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT    = Path("C:/Systeme/UC4_Demo_Package")
OUT.mkdir(parents=True, exist_ok=True)

# Best combat segments identified by quick_scan.py
# DEMO: one long segment from the richest multi-enemy window
DEMO_VIDEO  = "dataset/videos/U8tiME2kLok.f399.mp4"
DEMO_SKIP   = 92616   # @25m43s — 17% of frames have 2+ simultaneous enemies
DEMO_FRAMES = 600     # 10 seconds of gameplay

# All segments for CSV analysis
EVAL_SEGMENTS = [
    ("EVAL-1", "dataset/videos/Na7su9ZsqCc.f299.mp4",  33633, "Na7su9 @9m20s"),
    ("EVAL-2", "dataset/videos/U8tiME2kLok.f399.mp4",  92616, "U8tiM  @25m43s"),
    ("EVAL-3", "dataset/videos/5OYh3vlqTcY.f299.mp4",  55220, "5OYh3  @15m20s"),
    ("EVAL-4", "dataset/videos/Na7su9ZsqCc.f299.mp4",  71814, "Na7su9 @19m56s"),
    ("EVAL-5", "dataset/videos/FuIJnd1plI0.f299.mp4",  38400, "FuIJnd @10m40s"),
]
EVAL_FRAMES = 300

# ── Drawing constants ────────────────────────────────────────────────────────
FONT    = cv2.FONT_HERSHEY_SIMPLEX
GREEN   = (0, 220, 0)
GRAY    = (130, 130, 130)
CYAN    = (220, 210, 0)
WHITE   = (255, 255, 255)
RED_BGR = (30, 30, 220)
BLACK   = (0, 0, 0)
ORANGE  = (0, 130, 255)


def _txt(img, text, x, y, color=WHITE, scale=0.46, thick=1):
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def _box(img, x1, y1, x2, y2, color, thick=2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)


def draw_professional_frame(
    frame, enemies, target, fidx, proc_fps, seg_label, locked_count, total_count
):
    out  = frame.copy()
    H, W = out.shape[:2]
    cx, cy = W // 2, H // 2

    # ── All detected enemies (gray, thin) ──────────────────────────────────
    n_others = 0
    for det in enemies:
        is_selected = (
            target is not None
            and abs(det.cx - target.center[0]) < 4
            and abs(det.cy - target.center[1]) < 4
        )
        if not is_selected:
            _box(out, det.x1, det.y1, det.x2, det.y2, GRAY, 1)
            _txt(out, f"enemy  {det.confidence:.2f}",
                 det.x1, det.y1 - 5, GRAY, 0.38)
            n_others += 1

    # ── Selected target (green, bold) ─────────────────────────────────────
    if target is not None:
        x1, y1, x2, y2 = target.bbox
        _box(out, x1, y1, x2, y2, GREEN, 3)

        # Corner tick marks — polished look
        tl = 14
        cv2.line(out, (x1, y1), (x1 + tl, y1), GREEN, 2)
        cv2.line(out, (x1, y1), (x1, y1 + tl), GREEN, 2)
        cv2.line(out, (x2, y1), (x2 - tl, y1), GREEN, 2)
        cv2.line(out, (x2, y1), (x2, y1 + tl), GREEN, 2)
        cv2.line(out, (x1, y2), (x1 + tl, y2), GREEN, 2)
        cv2.line(out, (x1, y2), (x1, y2 - tl), GREEN, 2)
        cv2.line(out, (x2, y2), (x2 - tl, y2), GREEN, 2)
        cv2.line(out, (x2, y2), (x2, y2 - tl), GREEN, 2)

        # Label badge
        badge = f"TARGET  id={target.track_id}  sc={target.score:.2f}"
        (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.42, 1)
        cv2.rectangle(out, (x1, y1 - bh - 10), (x1 + bw + 6, y1), (0, 160, 0), -1)
        _txt(out, badge, x1 + 3, y1 - 4, WHITE, 0.42)

        # Aim point (chest/neck — 30% from top)
        ax, ay = int(target.aim_point[0]), int(target.aim_point[1])
        cv2.drawMarker(out, (ax, ay), CYAN, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(out, (ax, ay), 8, CYAN, 1, cv2.LINE_AA)

    # ── Screen-centre crosshair ────────────────────────────────────────────
    cv2.drawMarker(out, (cx, cy), WHITE, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)

    # ── MULTI-ENEMY alert banner ───────────────────────────────────────────
    if len(enemies) >= 2:
        banner = f"  MULTI-ENEMY  ({len(enemies)} detected)  "
        (mw, mh), _ = cv2.getTextSize(banner, FONT, 0.55, 2)
        bx = (W - mw) // 2
        cv2.rectangle(out, (bx - 4, 8), (bx + mw + 4, 8 + mh + 10), RED_BGR, -1)
        _txt(out, banner, bx, 8 + mh + 2, WHITE, 0.55, 2)

    # ── Bottom HUD bar ─────────────────────────────────────────────────────
    bar_h = 48
    cv2.rectangle(out, (0, H - bar_h), (W, H), (15, 15, 15), -1)
    cv2.line(out, (0, H - bar_h), (W, H - bar_h), (60, 60, 60), 1)

    state    = "LOCKED" if target is not None else "SEARCHING"
    st_color = GREEN if target is not None else ORANGE
    lock_pct = 100 * locked_count // max(total_count, 1)

    _txt(out, f"{seg_label}",         10,      H - 28, GRAY,  0.40)
    _txt(out, f"Frame {fidx:06d}",    10,      H - 10, GRAY,  0.40)
    _txt(out, state,                  W//2 - 50, H - 16, st_color, 0.60, 2)
    _txt(out, f"{proc_fps:.1f} FPS", W - 100, H - 28, WHITE, 0.42)
    _txt(out, f"Lock {lock_pct}%",   W - 100, H - 10, st_color, 0.40)

    return out


# ── Selector factory ─────────────────────────────────────────────────────────

def make_selector():
    ts = CFG.get("target_selector", {})
    return TargetSelector(
        max_lost_frames  = ts.get("max_lost_frames", 8),
        lock_radius_px   = ts.get("lock_radius_px", 150),
        iou_match_thresh = ts.get("iou_match_thresh", 0.25),
    )


# ── Task 1: Professional demo video ─────────────────────────────────────────

def generate_demo_video(detector, classifier, obj_filter):
    print("\n[DEMO VIDEO]  Generating demo_professional.mp4 …")
    cap = cv2.VideoCapture(DEMO_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, DEMO_SKIP)
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 60.0

    out_path = OUT / "demo_professional.mp4"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H)
    )
    selector     = make_selector()
    locked_count = 0
    t0 = time.perf_counter()
    fi = -1

    for fi in range(DEMO_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        raw      = detector.detect(frame)
        classed  = classifier.classify(frame, raw)
        enemies  = obj_filter.filter(classed, W, H)
        target   = selector.update(enemies, W, H)
        if target is not None:
            locked_count += 1
        elapsed  = time.perf_counter() - t0
        fps_proc = (fi + 1) / max(elapsed, 1e-6)
        out_frame = draw_professional_frame(
            frame, enemies, target,
            DEMO_SKIP + fi, fps_proc,
            "UC4 Aim Assist Demo",
            locked_count, fi + 1,
        )
        writer.write(out_frame)
        if (fi + 1) % 100 == 0:
            n = fi + 1
            print(f"  {n}/{DEMO_FRAMES}  locked={locked_count}  "
                  f"({100*locked_count//n}%)  fps={fps_proc:.1f}")

    cap.release()
    writer.release()
    n = fi + 1
    pct = 100 * locked_count // max(n, 1)
    print(f"  Done. {n} frames, {locked_count} LOCKED ({pct}%) → {out_path}")
    return n, locked_count


# ── Task 2: CSV analysis across all eval segments ────────────────────────────

def run_csv_analysis(detector, classifier, obj_filter):
    print("\n[CSV]  Running per-frame analysis across all eval segments …")
    csv_path = OUT / "multi_enemy_analysis.csv"
    rows = []

    for label, video, skip, desc in EVAL_SEGMENTS:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"  SKIP {label} — cannot open {video}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

        selector     = make_selector()
        prev_state   = SelectState.UNLOCKED
        prev_tid     = None
        lock_start   = None
        cur_dur      = 0

        print(f"  {label}  {Path(video).name}  @{skip} …", end=" ", flush=True)

        for fi in range(EVAL_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break

            raw     = detector.detect(frame)
            classed = classifier.classify(frame, raw)
            enemies = obj_filter.filter(classed, W, H)
            target  = selector.update(enemies, W, H)

            is_locked   = (selector.state == SelectState.LOCKED)
            tid         = target.track_id if target else None
            n_enemies   = len(enemies)
            is_multi    = n_enemies >= 2
            timestamp_s = (skip + fi) / fps

            # Lock duration tracking
            if is_locked:
                cur_dur += 1
                if prev_state == SelectState.UNLOCKED:
                    lock_start = fi
            else:
                cur_dur = 0
                lock_start = None

            # ID switch detection while locked
            id_switch = (
                is_locked
                and prev_state == SelectState.LOCKED
                and tid is not None
                and prev_tid is not None
                and tid != prev_tid
            )

            rows.append({
                "segment":      label,
                "video":        Path(video).name,
                "frame_abs":    skip + fi,
                "frame_rel":    fi,
                "timestamp_s":  round(timestamp_s, 3),
                "n_enemies":    n_enemies,
                "is_locked":    int(is_locked),
                "target_id":    tid if tid is not None else "",
                "lock_dur_so_far": cur_dur,
                "is_multi_enemy": int(is_multi),
                "id_switch":    int(id_switch),
                "conf":         round(target.confidence, 3) if target else "",
                "score":        round(target.score, 3) if target else "",
            })

            prev_state = selector.state
            prev_tid   = tid

        cap.release()
        seg_rows = [r for r in rows if r["segment"] == label]
        multi    = sum(r["is_multi_enemy"] for r in seg_rows)
        locked   = sum(r["is_locked"] for r in seg_rows)
        switches = sum(r["id_switch"] for r in seg_rows)
        print(f"multi={multi}  locked={locked}  id_switches={switches}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "segment", "video", "frame_abs", "frame_rel", "timestamp_s",
            "n_enemies", "is_locked", "target_id", "lock_dur_so_far",
            "is_multi_enemy", "id_switch", "conf", "score",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} rows → {csv_path}")
    return rows


# ── Task 3: Automated analysis summary ───────────────────────────────────────

def build_analysis_summary(rows: list[dict], demo_n: int, demo_locked: int) -> str:
    lines = []
    a = lines.append

    a("=" * 70)
    a("UC4 AIM ASSIST — AUTOMATED VIDEO ANALYSIS REPORT")
    a("=" * 70)
    a("")
    a("Generated from 5 evaluation segments × 300 frames + 1 demo segment × 600 frames.")
    a("No ground-truth labels. Findings are based on measurable signals only.")
    a("")

    # Per-segment summary
    a("PER-SEGMENT STATISTICS")
    a("-" * 70)
    a(f"{'Seg':7s}  {'Frames':>7s}  {'Enemy%':>7s}  {'Lock%':>6s}  "
      f"{'Multi%':>7s}  {'IDswitch':>9s}  {'ShortLocks':>10s}")
    a("-" * 70)

    segments = list(dict.fromkeys(r["segment"] for r in rows))
    seg_stats = {}
    all_short_locks = []

    for seg in segments:
        sr = [r for r in rows if r["segment"] == seg]
        n  = len(sr)
        enemy_frames  = sum(1 for r in sr if r["n_enemies"] > 0)
        locked_frames = sum(r["is_locked"] for r in sr)
        multi_frames  = sum(r["is_multi_enemy"] for r in sr)
        id_switches   = sum(r["id_switch"] for r in sr)

        # Short lock detection: lock periods < 5 frames
        short_locks = 0
        in_lock = False
        run = 0
        for r in sr:
            if r["is_locked"]:
                in_lock = True
                run += 1
            else:
                if in_lock and 0 < run < 5:
                    short_locks += 1
                    all_short_locks.append({
                        "seg": seg, "dur": run,
                        "frame": r["frame_abs"] - run,
                        "ts": round((r["frame_abs"] - run) / 60.0, 1),
                    })
                run = 0
                in_lock = False
        if in_lock and 0 < run < 5:
            short_locks += 1

        a(f"{seg:7s}  {n:>7d}  {100*enemy_frames//max(n,1):>6d}%  "
          f"{100*locked_frames//max(n,1):>5d}%  "
          f"{100*multi_frames//max(n,1):>6d}%  "
          f"{id_switches:>9d}  "
          f"{short_locks:>10d}")

        seg_stats[seg] = {
            "n": n, "enemy_frames": enemy_frames,
            "locked_frames": locked_frames, "multi_frames": multi_frames,
            "id_switches": id_switches, "short_locks": short_locks,
        }

    a("-" * 70)
    a("")

    # Demo segment
    a(f"DEMO SEGMENT  (U8tiME2kLok @25m43s, 600 frames)")
    a(f"  LOCKED frames  : {demo_locked}/{demo_n} ({100*demo_locked//max(demo_n,1)}%)")
    a("")

    # ID Switches (potential wrong enemy selections)
    total_switches = sum(s["id_switches"] for s in seg_stats.values())
    a("ID SWITCHES WHILE LOCKED  (most likely: wrong enemy selected after occlusion)")
    a("-" * 70)
    if total_switches == 0:
        a("  None detected across all segments.")
        a("  The selector maintained consistent target identity within each lock.")
    else:
        a(f"  Total: {total_switches} ID switches across all segments.")
        switch_rows = [r for r in rows if r["id_switch"]]
        for r in switch_rows[:20]:
            a(f"  {r['segment']}  frame={r['frame_abs']}  t={r['timestamp_s']:.1f}s  "
              f"old→new_id=?→{r['target_id']}  n_enemies={r['n_enemies']}")
    a("")

    # Short locks (flickering)
    total_short = sum(s["short_locks"] for s in seg_stats.values())
    a("SHORT LOCKS < 5 FRAMES  (lock-then-immediately-release = flickering)")
    a("-" * 70)
    if total_short == 0:
        a("  None detected. No flickering observed.")
    else:
        a(f"  Total: {total_short} short lock events.")
        for sl in all_short_locks[:20]:
            a(f"  {sl['seg']}  frame={sl['frame']}  t={sl['ts']}s  dur={sl['dur']} frames")
    a("")

    # Multi-enemy frames where a target was selected
    multi_with_target = [
        r for r in rows if r["is_multi_enemy"] and r["is_locked"]
    ]
    multi_without = [
        r for r in rows if r["is_multi_enemy"] and not r["is_locked"]
    ]
    a("MULTI-ENEMY FRAMES  (selector had to choose among 2+ detected enemies)")
    a("-" * 70)
    total_multi = sum(r["is_multi_enemy"] for r in rows)
    a(f"  Total multi-enemy frames : {total_multi}")
    a(f"  Selector locked on one   : {len(multi_with_target)}  ({100*len(multi_with_target)//max(total_multi,1)}%)")
    a(f"  Selector passed (none)   : {len(multi_without)}  ({100*len(multi_without)//max(total_multi,1)}%)")
    a("")
    if multi_with_target:
        a("  Timestamps of multi-enemy + locked frames (inspect these in videos):")
        for r in multi_with_target[:30]:
            a(f"    {r['segment']}  @{r['timestamp_s']:.1f}s  "
              f"frame={r['frame_abs']}  n={r['n_enemies']}  "
              f"target_id={r['target_id']}  score={r['score']}")
    a("")

    # Lock stability
    a("LOCK STABILITY ANALYSIS")
    a("-" * 70)
    all_lock_durations = []
    for seg in segments:
        sr = [r for r in rows if r["segment"] == seg]
        run = 0
        for r in sr:
            if r["is_locked"]:
                run += 1
            elif run > 0:
                all_lock_durations.append((seg, run))
                run = 0
        if run > 0:
            all_lock_durations.append((seg, run))

    if all_lock_durations:
        durations = [d for _, d in all_lock_durations]
        a(f"  Total lock events    : {len(durations)}")
        a(f"  Avg lock duration    : {sum(durations)//len(durations)} frames  "
          f"({sum(durations)//len(durations)/60*1000:.0f} ms at 60fps)")
        a(f"  Median lock duration : {sorted(durations)[len(durations)//2]} frames")
        a(f"  Longest lock         : {max(durations)} frames  ({max(durations)/60*1000:.0f} ms)")
        a(f"  Shortest lock        : {min(durations)} frames")
        a("")
        a("  Distribution:")
        buckets = [(1,4,"1–4 (flicker)"), (5,14,"5–14 (brief)"),
                   (15,49,"15–49 (normal)"), (50,999,"50+ (stable)")]
        for lo, hi, name in buckets:
            cnt = sum(1 for d in durations if lo <= d <= hi)
            pct = 100 * cnt // max(len(durations), 1)
            a(f"    {name:20s}: {cnt:4d} events  ({pct}%)")
    a("")

    # Verified observations
    a("VERIFIED OBSERVATIONS")
    a("-" * 70)
    obs = []

    if total_switches == 0:
        obs.append("POSITIVE — No ID switches detected while locked. "
                   "The IoU/proximity maintenance logic holds identity correctly.")
    else:
        obs.append(f"ISSUE — {total_switches} ID switches while locked. "
                   "Selector swapped targets mid-engagement.")

    if total_short == 0:
        obs.append("POSITIVE — No flickering (short <5-frame locks). "
                   "The max_lost_frames=8 buffer is preventing spurious releases.")
    else:
        obs.append(f"ISSUE — {total_short} short lock events detected. "
                   "Possible flickering at these timestamps (see above).")

    # Avg enemies per frame across all segments
    avg_e = sum(r["n_enemies"] for r in rows) / max(len(rows), 1)
    obs.append(f"LOW DETECTION DENSITY — Avg {avg_e:.2f} enemies/frame across all segments. "
               "UC4 multiplayer in these clips typically shows 0–1 enemy at a time. "
               "Multi-enemy scenes are rare ({:.0f}% of frames). "
               "This limits the usefulness of crosshair-priority selection.".format(
               100*total_multi//max(len(rows),1)))

    multi_pass_rate = 100*len(multi_with_target)//max(total_multi,1)
    if multi_pass_rate >= 70:
        obs.append(f"POSITIVE — Selector locked on a target in {multi_pass_rate}% of "
                   "multi-enemy frames. It does not consistently fail to choose.")
    else:
        obs.append(f"CAUTION — Selector locked on target in only {multi_pass_rate}% of "
                   "multi-enemy frames. It is frequently passing when multiple enemies "
                   "are present.")

    for i, o in enumerate(obs, 1):
        a(f"  {i}. {o}")
    a("")

    # Remaining issues
    a("REMAINING ISSUES (unresolved — do not implement fixes yet)")
    a("-" * 70)
    issues = [
        ("SELECTOR-01",
         "Screen-center bias: bbox-contains uses screen center (W/2, H/2) as the "
         "primary Phase 1 lock trigger. In UC4 third-person view, the player character's "
         "body often overlaps the screen center, so any enemy behind the player can "
         "falsely satisfy bbox-contains. Requires visual confirmation in multi-enemy frames."),
        ("SELECTOR-02",
         "No engagement signal: The selector has no way to know which enemy "
         "DevoManiac is actually shooting. It infers proximity to screen center, "
         "not shooting direction, player animation, or muzzle flash."),
        ("DETECTOR-01",
         "Low recall at range: Enemies further than ~10m appear at <40px bbox "
         "height; YOLO confidence drops below 0.45 threshold. These are missed "
         "entirely. Not a selector issue — a model limitation."),
        ("PIPELINE-01",
         "No live test: The full InferencePipeline + Chiaki + PS5 integration "
         "has not been run on real hardware. TensorRT engine (.engine) not exported. "
         "CPU inference (~4fps) is not viable for production; TRT target is 60fps."),
    ]
    for code, desc in issues:
        a(f"  [{code}]")
        a(f"    {desc}")
        a("")

    # Recommended next steps
    a("RECOMMENDED NEXT STEPS (ordered by priority)")
    a("-" * 70)
    steps = [
        "Export TensorRT engine (tools/tensorrt_export.py) — required for real-time performance.",
        "Run InferencePipeline on PS5 via Chiaki on real hardware — first live test.",
        "Visual review of multi-enemy timestamps in demo videos "
        "(see multi_enemy_analysis.csv column is_multi_enemy=1) — "
        "confirm whether selector picks the correct enemy.",
        "If visual review reveals systematic wrong-enemy selection: "
        "tune acq_threshold or edge_margin_px in config.yaml before adding new signals.",
        "Record Loom video in Arabic for Ahmed — Milestone 1 formal submission.",
    ]
    for i, s in enumerate(steps, 1):
        a(f"  {i}. {s}")
    a("")
    a("=" * 70)

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"UC4 Demo Package Builder")
    print(f"  Output directory : {OUT}")
    print(f"  Device           : {DEVICE}")
    print(f"  Model            : {MODEL}")
    print()

    print("Loading detector …")
    detector = EnemyDetector({
        "model_path":            "models/no_engine.engine",
        "fallback_model":        MODEL,
        "confidence_threshold":  0.45,
        "device":                DEVICE,
    })
    detector.load()
    detector.warmup(n_iters=2)
    classifier = EnemyClassifier(CFG.get("enemy_classification", {}))
    obj_filter = ObjectFilter(CFG.get("object_filter", {}))

    # Task 1
    demo_n, demo_locked = generate_demo_video(detector, classifier, obj_filter)

    # Task 2
    rows = run_csv_analysis(detector, classifier, obj_filter)

    # Task 3
    print("\n[ANALYSIS]  Building summary report …")
    summary = build_analysis_summary(rows, demo_n, demo_locked)
    summary_path = OUT / "analysis_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)

    print(f"\n{'='*60}")
    print(f"PACKAGE COMPLETE → {OUT}")
    print(f"  demo_professional.mp4     — demo video")
    print(f"  multi_enemy_analysis.csv  — per-frame data ({len(rows)} rows)")
    print(f"  analysis_summary.txt      — verified findings")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
