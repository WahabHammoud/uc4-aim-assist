"""
Semi-automatic annotation GUI.

Provides a keyboard-driven OpenCV interface to quickly label frames with
enemy bounding boxes in YOLO format.

Controls
--------
  Mouse drag      : draw bounding box
  ENTER / SPACE   : confirm box and save label
  Z               : undo last box in current frame
  D / Right arrow : next frame
  A / Left arrow  : previous frame
  S               : save current frame labels and move to next
  ESC / Q         : quit

The tool also runs simple auto-detection (using the current model if
available) and shows suggested boxes in yellow.  The user confirms, adjusts,
or deletes them.

Labels are saved to --labels-dir in YOLO format:
  <class_id> <cx> <cy> <w> <h>   (normalised 0-1)

Usage:
    python tools/annotate_gui.py \
        --frames dataset/frames/raw \
        --labels dataset/frames/labels \
        --model  models/enemy_detector.pt   # optional: for pre-suggestions
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels
CLASS_ID = 0                        # 0 = enemy


class AnnotationGUI:
    WINDOW = "UC4 Annotator"
    BOX_COLOR      = (0, 255, 0)      # Confirmed boxes
    SUGGEST_COLOR  = (0, 200, 255)    # Model-suggested boxes
    DRAWING_COLOR  = (255, 100, 100)  # While dragging

    def __init__(self, frames_dir: Path, labels_dir: Path, model_path: Optional[Path]):
        self._frames_dir = frames_dir
        self._labels_dir = labels_dir
        self._model_path = model_path
        self._model      = None

        # Frame list
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self._frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
        if not self._frames:
            raise ValueError(f"No images found in {frames_dir}")

        self._idx         = 0
        self._boxes: List[BBox] = []
        self._suggestions: List[BBox] = []

        # Drawing state
        self._drawing = False
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_end:   Optional[Tuple[int, int]] = None

        self._current_frame: Optional[np.ndarray] = None
        self._frame_w = 1
        self._frame_h = 1

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1280, 720)
        cv2.setMouseCallback(self.WINDOW, self._mouse_callback)

        if self._model_path and self._model_path.exists():
            self._load_model()

        self._load_frame(0)
        print(f"Annotating {len(self._frames)} frames.  Press H for help.")

        while True:
            self._draw()
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):                         # Q / ESC → quit
                break
            elif key in (ord("d"), 83):                       # D / Right → next
                self._save_labels()
                self._load_frame(min(self._idx + 1, len(self._frames) - 1))
            elif key in (ord("a"), 81):                       # A / Left → prev
                self._save_labels()
                self._load_frame(max(self._idx - 1, 0))
            elif key in (ord("s"),):                          # S → save + next
                self._save_labels()
                if self._idx < len(self._frames) - 1:
                    self._load_frame(self._idx + 1)
            elif key == ord("z"):                             # Z → undo
                if self._boxes:
                    self._boxes.pop()
            elif key == ord("c"):                             # C → clear all
                self._boxes.clear()
            elif key == ord("h"):
                self._print_help()
            elif key in (13, 32):                             # Enter / Space → accept suggestions
                self._accept_suggestions()

        self._save_labels()
        cv2.destroyAllWindows()
        print(f"\nAnnotation session ended.  Labels saved to {self._labels_dir}")

    # ------------------------------------------------------------------
    # Frame management
    # ------------------------------------------------------------------

    def _load_frame(self, idx: int) -> None:
        self._idx = idx
        path = self._frames[idx]
        self._current_frame = cv2.imread(str(path))
        if self._current_frame is None:
            print(f"[WARNING] Cannot load {path}")
            self._current_frame = np.zeros((720, 1280, 3), np.uint8)
        self._frame_h, self._frame_w = self._current_frame.shape[:2]
        self._boxes = self._load_existing_labels(path)
        self._suggestions = self._run_suggestions()
        title = (f"[{idx+1}/{len(self._frames)}] {path.name}  "
                 f"| Boxes: {len(self._boxes)}  | Suggestions: {len(self._suggestions)}")
        cv2.setWindowTitle(self.WINDOW, title)

    def _load_existing_labels(self, img_path: Path) -> List[BBox]:
        lbl_path = self._labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            return []
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = [float(x) for x in parts]
                x1 = int((cx - w / 2) * self._frame_w)
                y1 = int((cy - h / 2) * self._frame_h)
                x2 = int((cx + w / 2) * self._frame_w)
                y2 = int((cy + h / 2) * self._frame_h)
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _save_labels(self) -> None:
        img_path = self._frames[self._idx]
        self._labels_dir.mkdir(parents=True, exist_ok=True)
        lbl_path = self._labels_dir / (img_path.stem + ".txt")
        with open(lbl_path, "w") as f:
            for (x1, y1, x2, y2) in self._boxes:
                cx = ((x1 + x2) / 2) / self._frame_w
                cy = ((y1 + y2) / 2) / self._frame_h
                bw = abs(x2 - x1) / self._frame_w
                bh = abs(y2 - y1) / self._frame_h
                f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        if self._current_frame is None:
            return
        canvas = self._current_frame.copy()

        # Suggestions (yellow)
        for (x1, y1, x2, y2) in self._suggestions:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.SUGGEST_COLOR, 1)
            cv2.putText(canvas, "suggest", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.SUGGEST_COLOR, 1)

        # Confirmed boxes (green)
        for i, (x1, y1, x2, y2) in enumerate(self._boxes):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
            cv2.putText(canvas, f"#{i}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.BOX_COLOR, 1)

        # Live drag
        if self._drawing and self._drag_start and self._drag_end:
            cv2.rectangle(canvas, self._drag_start, self._drag_end, self.DRAWING_COLOR, 1)

        # HUD
        info = (f"Frame {self._idx+1}/{len(self._frames)}  "
                f"Boxes:{len(self._boxes)}  [D]=next [A]=prev [Z]=undo [S]=save+next")
        cv2.putText(canvas, info, (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        cv2.imshow(self.WINDOW, canvas)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._drag_start = (x, y)
            self._drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1, y1 = self._drag_start
            x2, y2 = x, y
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self._boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            self._drag_start = self._drag_end = None

    # ------------------------------------------------------------------
    # Model suggestions
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            print(f"Suggestion model loaded: {self._model_path}")
        except Exception as e:
            print(f"[WARNING] Could not load suggestion model: {e}")

    def _run_suggestions(self) -> List[BBox]:
        if self._model is None or self._current_frame is None:
            return []
        try:
            results = self._model.predict(
                source=self._current_frame, conf=0.4, verbose=False
            )
            boxes = []
            if results and results[0].boxes is not None:
                for xyxy in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    boxes.append((x1, y1, x2, y2))
            return boxes
        except Exception:
            return []

    def _accept_suggestions(self) -> None:
        self._boxes.extend(self._suggestions)
        self._suggestions = []

    @staticmethod
    def _print_help() -> None:
        print("""
=== UC4 Annotation GUI ===
  Drag        : draw bounding box
  Enter/Space : accept all suggestions
  Z           : undo last box
  C           : clear all boxes for current frame
  S           : save + next frame
  D / →       : next frame (auto-save)
  A / ←       : previous frame (auto-save)
  H           : show this help
  Q / ESC     : quit
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UC4 Annotation GUI")
    parser.add_argument("--frames",  default="dataset/frames/raw")
    parser.add_argument("--labels",  default="dataset/frames/labels")
    parser.add_argument("--model",   default=None,  help="Optional .pt for suggestions")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    gui = AnnotationGUI(
        frames_dir = Path(args.frames),
        labels_dir = Path(args.labels),
        model_path = model_path,
    )
    gui.run()


if __name__ == "__main__":
    main()
