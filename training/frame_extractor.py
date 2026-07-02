"""
Frame extraction pipeline for dataset creation.

Supports:
  - Single YouTube video URLs
  - Full YouTube PLAYLIST URLs  ← paste the DevoManiac UMPL playlist URL here

Downloads with yt-dlp, then extracts frames at a configurable FPS.
Deduplication skips near-identical frames so the annotation set stays lean.

Usage (single video):
    python -m training.frame_extractor \
        --urls "https://youtube.com/watch?v=..." \
        --fps 3

Usage (full playlist — recommended for DevoManiac UMPL):
    python -m training.frame_extractor \
        --playlist "https://www.youtube.com/playlist?list=XXXXXXXX" \
        --fps 3 \
        --max-per-video 1500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paste the DevoManiac UMPL playlist URL here (copy it from YouTube)
# ---------------------------------------------------------------------------
DEVO_PLAYLIST_URL: str = "https://www.youtube.com/playlist?list=PLPY8De9y4puIi0Csx4oNqa5FH27k-hgct"   # e.g. "https://www.youtube.com/playlist?list=PLxxxxxxxx"

# Individual video fallback (used only if no --playlist and no --urls given)
DEVO_VIDEOS: List[str] = []


# ---------------------------------------------------------------------------
# Playlist helpers
# ---------------------------------------------------------------------------

def get_playlist_video_urls(playlist_url: str) -> List[str]:
    """
    Use yt-dlp --flat-playlist to extract all video URLs from a playlist
    without downloading anything.
    """
    print(f"Fetching playlist info: {playlist_url}")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--print", "%(url)s",
        "--no-warnings",
        playlist_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Could not read playlist:\n{result.stderr}")
        return []

    urls = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("http")
    ]

    # yt-dlp sometimes prints bare video IDs instead of full URLs
    full_urls = []
    for u in urls:
        if u.startswith("https://"):
            full_urls.append(u)
        else:
            full_urls.append(f"https://www.youtube.com/watch?v={u}")

    print(f"  Found {len(full_urls)} videos in playlist.")
    return full_urls


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: Path) -> Optional[Path]:
    """
    Download a single YouTube video (no playlist expansion).
    Returns the path of the downloaded .mp4, or None on failure.
    Skips download if the file already exists (resume-safe).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "%(id)s.%(ext)s")

    # First, get the video ID to check if already downloaded
    id_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--print", "%(id)s",
        "--no-warnings",
        "--no-playlist",
        url,
    ]
    id_result = subprocess.run(id_cmd, capture_output=True, text=True)
    video_id = id_result.stdout.strip()
    if video_id:
        existing = output_dir / f"{video_id}.mp4"
        if existing.exists():
            print(f"  [SKIP] Already downloaded: {existing.name}")
            return existing

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", template,
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "3",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] yt-dlp failed for {url}")
        print(f"          {result.stderr[:300]}")
        return None

    # Return the file that was just created
    if video_id:
        candidate = output_dir / f"{video_id}.mp4"
        if candidate.exists():
            return candidate

    # Fallback: find the newest mp4
    mp4s = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0] if mp4s else None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 3.0,
    max_frames: int = 1500,
    deduplicate: bool = True,
    sim_threshold: float = 0.94,
) -> int:
    """
    Extract frames from a video file and save as JPEG.

    Parameters
    ----------
    video_path    : source .mp4 file.
    output_dir    : directory to save frames.
    target_fps    : frames per second of video to extract (3 is plenty).
    max_frames    : hard cap per video.
    deduplicate   : skip visually similar frames (menus, kill-cams, etc.).
    sim_threshold : histogram correlation above this → duplicate.

    Returns the number of frames saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return 0

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step        = max(1, int(video_fps / target_fps))
    video_stem  = video_path.stem

    print(f"  Extracting: {video_path.name}")
    print(f"    video_fps={video_fps:.1f}  total_frames={total_frames}  step={step}")

    saved     = 0
    frame_idx = 0
    prev_hist = None

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            if deduplicate:
                hist = _compute_hist(frame)
                if prev_hist is not None:
                    sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if sim >= sim_threshold:
                        frame_idx += 1
                        continue
                prev_hist = hist

            fname = output_dir / f"{video_stem}_f{frame_idx:07d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1

            if saved % 100 == 0:
                print(f"    … {saved} frames saved")

        frame_idx += 1

    cap.release()
    print(f"  Done: {saved} frames saved → {output_dir}")
    return saved


def _compute_hist(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UC4 Frame Extractor — supports single videos and full playlists"
    )
    parser.add_argument(
        "--playlist", default=None,
        help="YouTube PLAYLIST URL — downloads ALL videos in the playlist. "
             "Example: https://www.youtube.com/playlist?list=PLxxxxxxxx"
    )
    parser.add_argument(
        "--urls", nargs="*", default=None,
        help="One or more individual YouTube video URLs (alternative to --playlist)"
    )
    parser.add_argument("--video-dir",      default="dataset/videos")
    parser.add_argument("--output",         default="dataset/frames/raw")
    parser.add_argument("--fps",            type=float, default=3.0)
    parser.add_argument("--max-per-video",  type=int,   default=1500)
    parser.add_argument("--no-deduplicate", action="store_true")
    args = parser.parse_args()

    # Determine source URLs
    if args.playlist:
        video_urls = get_playlist_video_urls(args.playlist)
    elif args.urls:
        video_urls = args.urls
    elif DEVO_PLAYLIST_URL:
        video_urls = get_playlist_video_urls(DEVO_PLAYLIST_URL)
    elif DEVO_VIDEOS:
        video_urls = DEVO_VIDEOS
    else:
        print(
            "Nothing to download.\n"
            "  Option 1 (recommended): --playlist <YouTube playlist URL>\n"
            "  Option 2: --urls <url1> <url2> ...\n"
            "  Option 3: set DEVO_PLAYLIST_URL at top of this file"
        )
        return

    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output)
    total      = 0

    print(f"\n{'='*60}")
    print(f"UC4 Frame Extractor")
    print(f"  Videos to process : {len(video_urls)}")
    print(f"  Target FPS         : {args.fps}")
    print(f"  Max frames/video   : {args.max_per_video}")
    print(f"  Output dir         : {output_dir.resolve()}")
    print(f"{'='*60}\n")

    for i, url in enumerate(video_urls, 1):
        print(f"\n[{i}/{len(video_urls)}] {url}")
        video_path = download_video(url, video_dir)
        if video_path is None:
            print(f"  Skipping (download failed).")
            continue

        n = extract_frames(
            video_path  = video_path,
            output_dir  = output_dir,
            target_fps  = args.fps,
            max_frames  = args.max_per_video,
            deduplicate = not args.no_deduplicate,
        )
        total += n
        print(f"  Running total: {total} frames")

    print(f"\n{'='*60}")
    print(f"COMPLETE — Total frames extracted: {total}")
    print(f"Output: {output_dir.resolve()}")
    print(f"\nNext step:")
    print(f"  python tools/annotate_gui.py --frames {output_dir}")


if __name__ == "__main__":
    main()
