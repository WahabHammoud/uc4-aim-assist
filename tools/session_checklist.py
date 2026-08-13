"""
Pre-session readiness checklist for UC4 Aim Assist.

Run this before every session with Ahmed to confirm the system is fully ready:
    python tools/session_checklist.py

Prints GREEN checkmark for each passing check, RED X for each failure.
At the end: "READY FOR SESSION" or a list of things to fix.
"""

import sys
import os

# Force working directory to repo root so relative paths resolve
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Colours (Windows-safe via ANSI escape, enabled via os.system trick) ──────
os.system("")   # enable ANSI on Windows 10/11 terminals

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

OK   = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[ FAIL ]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {OK}  {label}{suffix}")


def _fail(label: str, fix: str = "") -> None:
    suffix = f"\n         Fix: {fix}" if fix else ""
    print(f"  {FAIL}  {label}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {WARN}  {label}{suffix}")


# ── Individual checks ─────────────────────────────────────────────────────────

def check_python() -> bool:
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        _ok(label)
        return True
    _fail(label, "Requires Python ≥ 3.10")
    return False


def check_packages() -> bool:
    packages = [
        ("cv2",         "opencv-python"),
        ("numpy",       "numpy"),
        ("yaml",        "PyYAML"),
        ("torch",       "torch"),
        ("ultralytics", "ultralytics"),
        ("supervision", "supervision"),
        ("mss",         "mss"),
        ("hid",         "hidapi"),
    ]
    all_ok = True
    for module, pip_name in packages:
        try:
            __import__(module)
            _ok(f"Package: {pip_name}")
        except ImportError:
            _fail(f"Package: {pip_name}", f"pip install {pip_name}")
            all_ok = False
    return all_ok


def check_onnxruntime() -> bool:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            _ok("onnxruntime-gpu", "CUDAExecutionProvider available")
        else:
            _warn("onnxruntime (CPU only)",
                  "Run setup_gpu.bat to enable CUDA — will still work but slower")
        return True
    except ImportError:
        _fail("onnxruntime", "pip install onnxruntime-gpu")
        return False


def check_cuda() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            _ok(f"CUDA / GPU: {name} ({vram} GB)")
            return True
        else:
            _warn("CUDA not available", "Run setup_gpu.bat — will fall back to ONNX CPU")
            return True   # warn only, not fatal
    except Exception as e:
        _warn(f"CUDA check failed: {e}")
        return True


def check_vigem() -> bool:
    try:
        import vgamepad as vg
    except ImportError:
        _fail("ViGEm / vgamepad", "pip install vgamepad  (then install ViGEm Bus Driver)")
        return False

    try:
        g = vg.VDS4Gamepad()
        del g
        _ok("ViGEm / vgamepad", "virtual DS4 created successfully")
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if any(kw in msg for kw in ("vigem", "file not found", "winerror 2", "cannot find")):
            _fail("ViGEm Bus Driver not installed",
                  "Download from github.com/nefarius/ViGEmBus/releases — see tools/install_vigem.md")
        else:
            _fail(f"ViGEm error: {exc}", "See tools/install_vigem.md")
        return False


def check_capture_card() -> bool:
    try:
        import cv2
        found = False
        for idx in range(5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 60)
            ret, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            if ret and w == 1920:
                _ok(f"Capture card", f"device index {idx}, 1920x1080")
                found = True
                break
        if not found:
            _fail("Capture card not found",
                  "Check USB 3.0 cable and HDMI cable from PS5. "
                  "Run tools/find_capture_device.py for diagnostics.")
        return found
    except Exception as e:
        _fail(f"Capture card check failed: {e}")
        return False


def check_dualsense() -> bool:
    SUPPORTED = [
        (0x054C, 0x0CE6, "DualSense standard"),
        (0x054C, 0x0DF2, "DualSense Edge"),
        (0x054C, 0x0CEC, "DualSense USB alt"),
    ]
    try:
        import hid
    except ImportError:
        _fail("DualSense HID", "pip install hidapi")
        return False

    for vid, pid, name in SUPPORTED:
        try:
            dev = hid.device()
            dev.open(vid, pid)
            dev.close()
            _ok(f"DualSense connected", f"{name} (VID={vid:#06x} PID={pid:#06x})")
            return True
        except Exception:
            pass

    _warn("DualSense not detected",
          "Connect via USB. System runs in auto-mode without it (L2 gate always active).")
    return True   # warn only — system degrades gracefully


def check_model() -> bool:
    model = "models/uc4_enemy_v2_final.onnx"
    engine = "models/uc4_enemy_v2_final.engine"
    if os.path.exists(engine):
        size_mb = os.path.getsize(engine) / (1024 ** 2)
        _ok(f"TensorRT engine", f"{engine}  ({size_mb:.0f} MB)")
        return True
    if os.path.exists(model):
        size_mb = os.path.getsize(model) / (1024 ** 2)
        _ok(f"ONNX model", f"{model}  ({size_mb:.0f} MB)")
        _warn("No TensorRT engine yet",
              "Run: python tools/export_tensorrt.py  for maximum GPU performance")
        return True
    _fail(f"Model file missing", f"Expected: {model}")
    return False


def check_config() -> bool:
    try:
        import yaml
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        _fail(f"config.yaml load failed: {e}")
        return False

    issues = []
    det = cfg.get("detection", {})
    cap = cfg.get("capture", {})

    if det.get("detector_mode") != "finetuned":
        issues.append(f"detector_mode={det.get('detector_mode')} — expected 'finetuned'")
    if det.get("confidence_threshold", 0) < 0.40:
        issues.append(f"confidence_threshold={det.get('confidence_threshold')} — low for custom model (suggest 0.55)")
    if cap.get("mode") != "capture_card":
        issues.append(f"capture.mode={cap.get('mode')} — expected 'capture_card'")
    if "uc4_enemy_v2_final" not in det.get("model_path", ""):
        issues.append(f"model_path={det.get('model_path')} — expected uc4_enemy_v2_final.onnx")

    if issues:
        for issue in issues:
            _fail(f"Config: {issue}", "Edit config/config.yaml")
        return False

    _ok("config.yaml", f"detector_mode=finetuned, model={os.path.basename(det['model_path'])}, "
                        f"capture_mode={cap['mode']}, conf={det['confidence_threshold']}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  UC4 Aim Assist — Pre-Session Readiness Checklist{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()

    results = {}

    print(f"{BOLD}[1] Python{RESET}")
    results["python"] = check_python()

    print(f"\n{BOLD}[2] Python Packages{RESET}")
    results["packages"] = check_packages()

    print(f"\n{BOLD}[3] ONNX Runtime{RESET}")
    results["onnx"] = check_onnxruntime()

    print(f"\n{BOLD}[4] CUDA / GPU{RESET}")
    results["cuda"] = check_cuda()

    print(f"\n{BOLD}[5] ViGEm / vgamepad  (REQUIRED for stick assist){RESET}")
    results["vigem"] = check_vigem()

    print(f"\n{BOLD}[6] Capture Card{RESET}")
    results["capture"] = check_capture_card()

    print(f"\n{BOLD}[7] DualSense Controller{RESET}")
    results["dualsense"] = check_dualsense()

    print(f"\n{BOLD}[8] Model File{RESET}")
    results["model"] = check_model()

    print(f"\n{BOLD}[9] Config (config.yaml){RESET}")
    results["config"] = check_config()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    failures = [k for k, v in results.items() if not v]
    if not failures:
        print(f"{GREEN}{BOLD}  READY FOR SESSION — all checks passed!{RESET}")
        print(f"  Start the system with:  START.bat")
    else:
        print(f"{RED}{BOLD}  NOT READY — fix these before the session:{RESET}")
        labels = {
            "python":    "Python version",
            "packages":  "Missing Python packages",
            "onnx":      "onnxruntime not installed",
            "cuda":      "CUDA / GPU issue",
            "vigem":     "ViGEm Bus Driver (stick assist will not work!)",
            "capture":   "Capture card not detected",
            "dualsense": "DualSense not connected",
            "model":     "Model file missing",
            "config":    "config.yaml misconfigured",
        }
        for k in failures:
            print(f"  {RED}x{RESET}  {labels.get(k, k)}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()


if __name__ == "__main__":
    main()
