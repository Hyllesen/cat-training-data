import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
RTSP_URL = os.getenv("RTSP_URL")
DETECTOR_MODEL = os.getenv("DETECTOR_PATH", "models/detector/best.pt")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_PATH", "runs/classify/cat_identity_v1/weights/best.pt")
DETECTIONS_DIR = "detections"

CONF_THRESHOLD = 0.7
ALERT_COOLDOWN = 30 
VIDEO_BUFFER_SECONDS = 5 # How long to record after the last detection

# --- PREPARE DIRECTORY ---
os.makedirs(DETECTIONS_DIR, exist_ok=True)

detector = YOLO(DETECTOR_MODEL)
classifier = YOLO(CLASSIFIER_MODEL)

def run_monitor():
    cap = cv2.VideoCapture(RTSP_URL)
    
    # Get camera properties for the VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20 # Fallback to 20 if camera doesn't report FPS
    
    # Video Recording State
    video_writer = None
    recording_until = 0
    
    print("--- Garden Monitoring Active (Video Mode) ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = detector(frame, verbose=False, device='mps')
        cat_detected_this_frame = False
        primary_identity = "unknown"

        for r in results:
            for box in r.boxes:
                if box.conf > CONF_THRESHOLD:
                    cat_detected_this_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Classification
                    cat_crop = frame[y1:y2, x1:x2]
                    if cat_crop.size == 0: continue
                    id_results = classifier(cat_crop, verbose=False, device='mps')
                    primary_identity = id_results[0].names[id_results[0].probs.top1]
                    
                    # Visual Markers
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- VIDEO SAVING LOGIC ---
        if cat_detected_this_frame:
            recording_until = time.time() + VIDEO_BUFFER_SECONDS
            
            # Start a new file if we aren't already recording
            if video_writer is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{primary_identity}_{timestamp}.mp4"
                save_path = os.path.join(DETECTIONS_DIR, filename)
                
                # Define Codec (avc1 is great for Mac)
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
                print(f"📹 Recording started: {filename}")

        # If we are in "recording mode", write the frame
        if video_writer is not None:
            video_writer.write(frame)
            
            # Stop recording if time is up
            if time.time() > recording_until:
                video_writer.release()
                video_writer = None
                print("🏁 Recording finished and saved.")

        cv2.imshow("Garden Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer: video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_monitor()