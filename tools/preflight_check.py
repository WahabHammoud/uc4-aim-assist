"""
UC4 Aim Assist — Pre-flight check.
Run this BEFORE main.py to verify the capture card, controller, and overlay.

Usage:
    python tools/preflight_check.py [device_index]
    python tools/preflight_check.py 0
"""

import sys
import time

import cv2


def check_capture_card(device_index: int) -> bool:
    print(f"\n[1] Testing capture card on device {device_index}...")

    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"FAIL: Cannot open device {device_index}")
        print("Fix: Check USB cable is connected to a USB 3.0 port")
        return False
    print("OK: Device opened")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if w != 1920:
        print(f"WARN: Got {w}x{h} instead of 1920x1080")
        print("Fix: Use a USB 3.0 port (not USB 2.0)")
        print("Trying MJPG fallback...")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w != 1920:
            print(f"FAIL: Still {w}x{h} after MJPG fallback")
            cap.release()
            return False
    print(f"OK: Resolution {w}x{h} @ {fps:.0f}fps")

    print("Waiting for frame from PS5 (make sure PS5 is on)...")
    t0 = time.time()
    frame = None
    while time.time() - t0 < 8.0:
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break

    if frame is None:
        print("FAIL: No frame received in 8 seconds")
        print("Fix: Check HDMI cable from PS5 to capture card INPUT port")
        print("Fix: Make sure PS5 is turned on")
        cap.release()
        return False

    elapsed = time.time() - t0
    print(f"OK: Frame received in {elapsed:.1f}s — size {frame.shape}")

    cv2.imwrite("preflight_frame.jpg", frame)
    print("OK: Test frame saved as preflight_frame.jpg")
    print("    Open this file to verify you see the PS5 game image")

    cap.release()
    return True


def check_dualsense() -> bool:
    print("\n[2] Testing DualSense controller...")
    try:
        import hid
        dualsense_pids = {3302, 3570, 3308}
        found = None
        for d in hid.enumerate():
            if d["vendor_id"] == 1356 and d["product_id"] in dualsense_pids:
                found = d
                break
        if found:
            name = "DualSense Edge" if found["product_id"] == 3570 else "DualSense"
            print(f"OK: {name} found (pid={found['product_id']})")
            return True
        else:
            print("WARN: No DualSense found via USB")
            print("Fix: Connect DualSense to PC via USB cable")
            print("Note: System will run in AUTO mode without controller")
            return True
    except Exception as e:
        print(f"WARN: Cannot check controller: {e}")
        return True


def check_pygetwindow() -> bool:
    print("\n[3] Testing overlay (pygetwindow)...")
    try:
        import pygetwindow as gw
        wins = gw.getAllTitles()
        chiaki_found = any("chiaki" in w.lower() for w in wins if w.strip())
        if chiaki_found:
            print("OK: Chiaki window found — overlay will attach")
        else:
            print("WARN: Chiaki window not found")
            print("Fix: Open Chiaki before running the main system")
            print("Note: Use --show-feed instead of --overlay for capture card mode")
        return True
    except Exception as e:
        print(f"FAIL: pygetwindow error: {e}")
        return False


if __name__ == "__main__":
    device_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("=" * 50)
    print("UC4 Aim Assist — Pre-flight Check")
    print("=" * 50)

    results = []
    results.append(check_capture_card(device_index))
    results.append(check_dualsense())
    results.append(check_pygetwindow())

    print("\n" + "=" * 50)
    if all(results):
        print("ALL CHECKS PASSED — Ready to run main system")
        print(f"Run: python main.py --capture-card --device-index {device_index} --show-feed")
    else:
        print("SOME CHECKS FAILED — Fix issues above before running")
    print("=" * 50)
