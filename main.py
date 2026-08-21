"""
UC4 Aim Assist — Entry Point.

Usage:
    python main.py                         # Run with default config
    python main.py --config config/config.yaml --debug
    python main.py --config config/config.yaml --no-gamepad   # dry-run (detection only)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Instance lock — prevents multiple simultaneous runs from stacking windows.
# The lock file stores the PID of the running process so a second launch can
# kill the first one cleanly before starting.
# ---------------------------------------------------------------------------

_LOCK_FILE = r"C:\temp\uc4_lock.txt"


def _acquire_instance_lock() -> None:
    """Kill any previous instance and write our PID to the lock file."""
    try:
        os.makedirs(r"C:\temp", exist_ok=True)
    except OSError:
        return  # can't create directory — skip locking rather than crashing

    if os.path.exists(_LOCK_FILE):
        try:
            with open(_LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            result = subprocess.run(
                ["taskkill", "/f", "/pid", str(old_pid)],
                capture_output=True,
            )
            if result.returncode == 0:
                print(f"[INFO] Killed previous instance (PID {old_pid}).")
            # Give the process a moment to release cv2 windows
            import time as _time; _time.sleep(0.3)
        except (ValueError, OSError, subprocess.SubprocessError):
            pass
        try:
            os.remove(_LOCK_FILE)
        except OSError:
            pass

    try:
        with open(_LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass


def _release_instance_lock() -> None:
    """Remove the lock file if it belongs to this process."""
    try:
        if os.path.exists(_LOCK_FILE):
            with open(_LOCK_FILE) as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                os.remove(_LOCK_FILE)
    except (ValueError, OSError):
        pass


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge `overrides` into `base` in-place.

    Dict values are merged recursively; all other types are replaced.
    This lets config_local.yaml override individual keys within a section
    without having to repeat the entire section.
    """
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def main() -> None:
    _acquire_instance_lock()

    parser = argparse.ArgumentParser(
        description="UC4 Aim Assist — PS5 Enemy Detection & Target Lock"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show real-time OpenCV debug overlay with detections and lock state"
    )
    parser.add_argument(
        "--no-gamepad", action="store_true",
        help="Skip virtual gamepad output (useful for testing detection without a DualSense)"
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Show transparent always-on-top red box overlay on the Chiaki window"
    )
    parser.add_argument(
        "--capture-card", action="store_true",
        help="Use UVC capture card (KASTWAVE AvedioLink / any UVC) instead of Chiaki screen capture"
    )
    parser.add_argument(
        "--device-index", type=int, default=0,
        help="Capture card device index (default: 0). Run tools/find_capture_device.py to list devices."
    )
    parser.add_argument(
        "--auto-detect", action="store_true",
        help="Auto-detect capture card device index by scanning 0–9 for 1920×1080 (use with --capture-card)"
    )
    parser.add_argument(
        "--show-feed", action="store_true",
        help="Show capture card feed in a window with the red box drawn on frame (use with --capture-card)"
    )
    parser.add_argument(
        "--windowed", action="store_true",
        help="Open the feed window at 960x540 windowed mode instead of fullscreen. Use on 4K monitors."
    )
    parser.add_argument(
        "--test-video", action="store_true",
        help="Use a local video file instead of Chiaki/capture card. Opens a file picker if --video-path is not given."
    )
    parser.add_argument(
        "--video-path", type=str, default=None,
        help="Path to a video file for --test-video mode (skips file picker dialog)."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Demo mode: L2 gating disabled — box shows whenever an enemy is detected."
    )
    args = parser.parse_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        print("        Run from the uc4_aim_assist/ directory.")
        sys.exit(1)

    # Import here so errors surface cleanly
    from src.pipeline.inference_pipeline import InferencePipeline
    from src.utils.logger import get_logger
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load user-local overrides (gitignored, never committed).
    # config_local.yaml lives next to config.yaml and only needs to contain
    # the keys you want to change — everything else stays at the committed default.
    config_local_path = config_path.parent / "config_local.yaml"
    if config_local_path.exists():
        with open(config_local_path) as f:
            local_overrides = yaml.safe_load(f) or {}
        _deep_merge(cfg, local_overrides)

    log = get_logger("main", cfg.get("logging", {}))
    if config_local_path.exists():
        log.info("Local overrides applied from %s", config_local_path)
    log.info("=" * 60)
    log.info("UC4 Aim Assist — Uncharted 4 PS5 Enemy Tracking System")
    log.info("=" * 60)

    if args.no_gamepad:
        cfg.setdefault("controller", {})["virtual_gamepad_type"] = "none"
        log.info(
            "--no-gamepad active: PID corrections computed but NOT sent to ViGEm. "
            "Right stick will not move. Detection and lock logic run normally."
        )

    if args.capture_card:
        cfg.setdefault("capture", {})["mode"] = "capture_card"
        if args.auto_detect:
            cfg["capture"]["capture_card_index"] = -1
            log.info("--capture-card --auto-detect active: will scan devices 0–9 for 1920×1080.")
        else:
            cfg["capture"]["capture_card_index"] = args.device_index
            log.info(
                "--capture-card active: using UVC capture card on device index %d.",
                args.device_index,
            )

    if args.test_video or args.video_path:
        video_path = args.video_path
        if not video_path:
            import tkinter as tk
            from tkinter import filedialog
            _root = tk.Tk()
            _root.withdraw()
            video_path = filedialog.askopenfilename(
                title="Select a video file for testing",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.ts *.m4v"),
                    ("All files", "*.*"),
                ],
            )
            _root.destroy()
            if not video_path:
                print("[ERROR] No video file selected. Exiting.")
                sys.exit(1)
        cfg.setdefault("capture", {})["mode"] = "video_file"
        cfg["capture"]["video_path"] = video_path
        log.info("--test-video active: using video file: %s", video_path)

    if args.demo:
        log.info("--demo active: L2 gating disabled — box shows on every detected enemy.")

    # Pass the (possibly patched) config dict directly so in-memory patches
    # are not lost when InferencePipeline re-reads the YAML from disk.
    pipeline = InferencePipeline(config_path=str(config_path), config=cfg)

    overlay = None
    if args.overlay:
        from src.overlay.overlay_window import OverlayWindow
        window_title = cfg.get("capture", {}).get("window_title", "Chiaki")
        overlay = OverlayWindow(window_title=window_title)
        overlay.start()
        log.info("Overlay started — red box will appear on the Chiaki window.")

    try:
        pipeline.start()
        pipeline.run(show_debug=args.debug, overlay=overlay, show_feed=args.show_feed, windowed=args.windowed, demo_mode=args.demo)
    except Exception as exc:
        log.exception("Fatal error in pipeline: %s", exc)
        sys.exit(1)
    finally:
        if overlay:
            overlay.stop()
        _release_instance_lock()


if __name__ == "__main__":
    main()
