import cv2

print("Scanning for video capture devices...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        status = "OK - frame received" if ret else "opened but no frame"
        print(f"  Device {i}: {w}x{h} — {status}")
        cap.release()
    else:
        print(f"  Device {i}: not available")
print("Done. Use the device number with --device-index")
