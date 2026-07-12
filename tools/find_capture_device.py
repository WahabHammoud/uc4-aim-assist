import cv2

print("Scanning for video capture devices (DirectShow)...")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        ret, frame = cap.read()
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        status = "OK" if ret else "no frame"
        print(f"  Device {i}: {w}x{h} @ {fps:.0f}fps — {status}")
        if w == 1920:
            print(f"  *** USE THIS ONE: --device-index {i} ***")
        cap.release()
    else:
        print(f"  Device {i}: not available")

try:
    from cv2_enumerate_cameras import enumerate_cameras
    print("\nDevices by name:")
    for cam in enumerate_cameras(cv2.CAP_DSHOW):
        print(f"  Index {cam.index}: {cam.name}")
        if "usb" in cam.name.lower() or "video" in cam.name.lower():
            print(f"  *** Possible capture card: {cam.name} ***")
except ImportError:
    print("(cv2-enumerate-cameras not installed, skipping name scan)")

print("Done. Use the device number with --device-index")
