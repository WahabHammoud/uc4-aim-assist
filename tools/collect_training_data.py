"""
Capture card training data collector.

Reads frames from the KASTWAVE AvedioLink (or any UVC capture card) and
saves one JPEG every 0.5 seconds.  Produces raw gameplay frames ready for
labelling with a tool like LabelImg or Roboflow.

Usage:
    python tools/collect_training_data.py
    python tools/collect_training_data.py --device 1 --interval 0.3 --output dataset/ahmed_training
"""

import argparse
import os
import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Save capture card frames for training data collection"
    )
    parser.add_argument("--device",   type=int,   default=0,
                        help="Capture card device index (default: 0)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between saved frames (default: 0.5)")
    parser.add_argument("--output",   type=str,
                        default="dataset/ahmed_training",
                        help="Output directory for saved frames")
    args = parser.parse_args()

    import cv2

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: could not open device {args.device}")
        print("Run tools/find_capture_device.py to list available devices.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture card: {actual_w}x{actual_h} on device {args.device}")
    print(f"Saving to:    {os.path.abspath(output_dir)}")
    print(f"Interval:     {args.interval}s  (~{1/args.interval:.0f} frames/min)")
    print("Press Q in the preview window to stop.\n")

    # Flush stale buffer frames before starting
    for _ in range(10):
        cap.read()

    count = 0
    last_save = 0.0
    session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: dropped frame")
            continue

        now = time.time()
        if now - last_save >= args.interval:
            filename = os.path.join(output_dir, f"{session_start}_{count:05d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
            last_save = now
            print(f"Saved frame {count:4d}  →  {os.path.basename(filename)}")

        cv2.imshow("Recording — press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone — {count} frames saved to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
