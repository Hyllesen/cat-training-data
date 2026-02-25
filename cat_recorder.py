"""
cat_recorder.py — 24/7 RTSP Cat Detection & Recorder
======================================================
Monitors an RTSP stream, runs YOLO object detection, and records segmented
MP4 clips strictly when a cat (COCO class 16) is detected.

Dependencies:
    pip install ultralytics opencv-python python-dotenv

Usage:
    1. Copy .env.example to .env and set RTSP_URL
    2. python cat_recorder.py
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.parsers.expat import model

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION — adjust these values for your environment
# ---------------------------------------------------------------------------
RTSP_URL        = os.getenv("RTSP_URL", "")
OUTPUT_DIR      = Path("recordings")           # directory where clips are saved
MODEL_PATH      = "yolo26n.pt"                 # YOLO26 nano — NMS-free, fastest variant
DEVICE          = "mps"                        # Apple Silicon GPU/NPU
# --- Updated CONFIGURATION ---
CONF_THRESHOLD  = 0.50                         # Bumped slightly to reduce weak chicken-cat guesses
CAT_CLASS_ID    = 16                           
BIRD_CLASS_ID   = 14                           # COCO index for birds/chickens
ABSENCE_TIMEOUT = 4.0                          # seconds of no-cat before stopping recording
RECONNECT_DELAY = 5                            # seconds to wait before RTSP reconnect attempt
CODEC           = "avc1"                       # H.264; falls back to 'mp4v' if unavailable
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def connect_stream(url: str) -> tuple[cv2.VideoCapture, int, int, float]:
    """
    Open the RTSP stream and return (cap, width, height, fps).
    Forces TCP transport to reduce packet loss on RTSP feeds.
    Raises RuntimeError if the stream cannot be opened.
    """
    # CAP_PROP_OPEN_TIMEOUT_MSEC and the RTSP TCP flag improve stability
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    # Request TCP transport via FFMPEG options (avoids UDP packet loss)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {url}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0   # default 15 fps if not reported
    log.info("Stream opened — %dx%d @ %.1f fps", width, height, fps)
    return cap, width, height, fps


def open_writer(width: int, height: int, fps: float) -> tuple[cv2.VideoWriter, Path]:
    """
    Create a new VideoWriter with a timestamped filename.
    Tries the H.264 (avc1) codec first; falls back to mp4v for compatibility.
    Returns (writer, file_path).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = OUTPUT_DIR / f"cat_{timestamp}.mp4"

    for codec in (CODEC, "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
        if writer.isOpened():
            log.info("Recording started → %s  [codec: %s]", filepath.name, codec)
            return writer, filepath
        writer.release()

    raise RuntimeError("No compatible video codec found (tried avc1, mp4v)")


def detect_cat(results) -> bool:
    """
    Return True only if a cat is detected AND no birds (chickens) 
    are the dominant detection in the frame.
    """
    bird_detected = False
    cat_detected = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)

            if conf >= CONF_THRESHOLD:
                if class_id == BIRD_CLASS_ID:
                    bird_detected = True
                elif class_id == CAT_CLASS_ID:
                    cat_detected = True

    # If a chicken is in the frame, we ignore everything to prevent false positives
    if bird_detected:
        return False
        
    return cat_detected


def release_writer(writer: Optional[cv2.VideoWriter], filepath: Optional[Path]) -> None:
    """Safely release a VideoWriter and log the saved file."""
    if writer is not None and writer.isOpened():
        writer.release()
        log.info("Recording saved → %s", filepath.name if filepath else "unknown")


def main() -> None:
    if not RTSP_URL:
        log.error("RTSP_URL not set in .env file. Please configure it and try again.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the model ONCE outside the reconnect loop — avoid repeated disk I/O
    log.info("Loading model: %s on device=%s", MODEL_PATH, DEVICE)
    model = YOLO(MODEL_PATH)
    log.info("Model loaded. Starting stream monitor...")

    # -----------------------------------------------------------------------
    # OUTER RECONNECT LOOP — runs forever; re-enters after any stream failure
    # -----------------------------------------------------------------------
    while True:
        cap    = None
        writer = None
        clip_path: Optional[Path] = None

        # Hysteresis state — tracks whether we are currently recording
        is_recording  = False
        last_seen_ts  = 0.0   # timestamp of the most recent cat detection

        try:
            cap, width, height, fps = connect_stream(RTSP_URL)

            # -------------------------------------------------------------------
            # INNER READ LOOP — runs per-frame while the stream is healthy
            # -------------------------------------------------------------------
            while True:
                ret, frame = cap.read()

                # --- RTSP disconnection detection ---
                # cap.read() returns False or an empty frame when the feed drops.
                # Break out to the reconnect loop rather than spinning on errors.
                if not ret or frame is None:
                    log.warning("Stream read failed — triggering reconnect")
                    break

                # Run YOLO inference on the current frame
                # Specifically monitor for both birds and cats
                results = model(frame, device=DEVICE, classes=[14, 16], verbose=False)
                cat_present = detect_cat(results)

                now = time.monotonic()

                # --- HYSTERESIS / DEBOUNCE LOGIC ---
                # The goal is to avoid flickering recordings: we START immediately
                # when a cat appears, but STOP only after the cat has been absent
                # for ABSENCE_TIMEOUT seconds (4 s by default).

                if cat_present:
                    # Update the "last time we saw a cat" timestamp every positive frame
                    last_seen_ts = now

                    if not is_recording:
                        # Cat just appeared — open a new video clip
                        writer, clip_path = open_writer(width, height, fps)
                        is_recording = True

                if is_recording:
                    # Write the current frame into the active clip
                    writer.write(frame)

                    # Check whether the absence window has elapsed.
                    # (now - last_seen_ts) grows while no cat is detected;
                    # once it exceeds ABSENCE_TIMEOUT we close the clip.
                    if not cat_present and (now - last_seen_ts) > ABSENCE_TIMEOUT:
                        release_writer(writer, clip_path)
                        writer     = None
                        clip_path  = None
                        is_recording = False
                        log.info("Cat absent for %.1fs — recording stopped", ABSENCE_TIMEOUT)

        except KeyboardInterrupt:
            # --- CLEAN SHUTDOWN via Ctrl+C ---
            log.info("Interrupted by user — shutting down cleanly")
            release_writer(writer, clip_path)
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            return

        except Exception as exc:
            log.error("Unexpected error: %s", exc, exc_info=True)

        finally:
            # Always release stream resources before attempting reconnect
            release_writer(writer, clip_path)
            if cap is not None:
                cap.release()

        # --- RECONNECT BACK-OFF ---
        # Pause before retrying so we don't hammer a temporarily-down camera
        log.info("Reconnecting in %ds...", RECONNECT_DELAY)
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    main()
