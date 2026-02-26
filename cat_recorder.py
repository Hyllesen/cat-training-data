"""
cat_recorder_v2.py — Optimized for Custom YOLO11n Model
======================================================
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RTSP_URL        = os.getenv("RTSP_URL", "")
OUTPUT_DIR      = Path("recordings")
# Pointing to your new best.pt from the M4 training run
MODEL_PATH      = "models/detector/best.pt" 
DEVICE          = "mps" 

CONF_THRESHOLD  = 0.50 
# In your custom model, 'cat' is index 0
CAT_CLASS_ID    = 0                            
ABSENCE_TIMEOUT = 4.0                          
RECONNECT_DELAY = 5                            
CODEC           = "avc1"                       
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def connect_stream(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {url}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    log.info("Stream opened — %dx%d @ %.1f fps", width, height, fps)
    return cap, width, height, fps

def open_writer(width: int, height: int, fps: float, conf: tuple[float, float]):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    conf_tag   = f"c{int(conf[0]*100)}-{int(conf[1]*100)}"
    filepath   = OUTPUT_DIR / f"cat_{timestamp}_{conf_tag}.mp4"

    for codec in (CODEC, "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
        if writer.isOpened():
            log.info("Recording started → %s", filepath.name)
            return writer, filepath
        writer.release()
    raise RuntimeError("No compatible video codec found")

def detect_cat(results) -> Optional[tuple[float, float]]:
    """
    Simplified: Your custom model already ignores chickens, 
    so we just look for Class 0 (Cat).
    """
    cat_confs: list[float] = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)

            if conf >= CONF_THRESHOLD and class_id == CAT_CLASS_ID:
                cat_confs.append(conf)

    if not cat_confs:
        return None

    return (min(cat_confs), max(cat_confs))

def main() -> None:
    if not RTSP_URL:
        log.error("RTSP_URL not set in .env")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading Custom Model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    while True:
        cap, writer, clip_path = None, None, None
        is_recording, last_seen_ts = False, 0.0

        try:
            cap, width, height, fps = connect_stream(RTSP_URL)

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # We only ask the model for Class 0
                results = model(frame, device=DEVICE, classes=[0], verbose=False)
                cat_conf = detect_cat(results)
                cat_present = cat_conf is not None
                now = time.monotonic()

                if cat_present:
                    last_seen_ts = now
                    if not is_recording:
                        log.info("Cat detected!")
                        writer, clip_path = open_writer(width, height, fps, cat_conf)
                        is_recording = True

                if is_recording:
                    writer.write(frame)
                    if not cat_present and (now - last_seen_ts) > ABSENCE_TIMEOUT:
                        writer.release()
                        log.info("Recording saved → %s", clip_path.name)
                        is_recording = False

        except KeyboardInterrupt:
            break
        except Exception as exc:
            log.error("Error: %s", exc)
        finally:
            if writer: writer.release()
            if cap: cap.release()
        
        time.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    main()