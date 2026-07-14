"""
UC4 Aim Assist — Entry Point.

Usage:
    python main.py                         # Run with default config
    python main.py --config config/config.yaml --debug
    python main.py --config config/config.yaml --no-gamepad   # dry-run (detection only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
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
        help="Show capture card feed in a fullscreen window with the red box drawn on frame (use with --capture-card)"
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

    log = get_logger("main", cfg.get("logging", {}))
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
        pipeline.run(show_debug=args.debug, overlay=overlay, show_feed=args.show_feed)
    except Exception as exc:
        log.exception("Fatal error in pipeline: %s", exc)
        sys.exit(1)
    finally:
        if overlay:
            overlay.stop()


if __name__ == "__main__":
    main()
