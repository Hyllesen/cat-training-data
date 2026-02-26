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
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_PATH", "runs/classify/cat_identity_v3/weights/best.pt")
DETECTIONS_DIR = "detections"

CONF_THRESHOLD = 0.7
ALERT_COOLDOWN = 30 
VIDEO_BUFFER_SECONDS = 5 

os.makedirs(DETECTIONS_DIR, exist_ok=True)

detector = YOLO(DETECTOR_MODEL)
classifier = YOLO(CLASSIFIER_MODEL)

def run_monitor():
    cap = cv2.VideoCapture(RTSP_URL)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20 
    
    video_writer = None
    recording_until = 0
    
    print("--- Garden Monitoring Active ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = detector(frame, verbose=False, device='mps')
        
        current_frame_identity = None
        current_frame_conf = 0.0

        for r in results:
            for box in r.boxes:
                if box.conf > CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Identify the cat
                    cat_crop = frame[y1:y2, x1:x2]
                    if cat_crop.size == 0: continue
                    
                    id_results = classifier(cat_crop, verbose=False, device='mps')
                    conf = id_results[0].probs.top1conf.item()
                    label = id_results[0].names[id_results[0].probs.top1]

                    # Update frame-level tracking for naming
                    if conf > current_frame_conf:
                        current_frame_conf = conf
                        current_frame_identity = label
                    
                    # --- DRAW LABELS ON FRAME ---
                    # Red for stray, Green for residents
                    color = (0, 0, 255) if label == "horny_meow" else (0, 255, 0)
                    
                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw Label Background (makes text easier to read)
                    label_str = f"{label.upper()} {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    
                    # Draw Text
                    cv2.putText(frame, label_str, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- VIDEO SAVING LOGIC ---
        if current_frame_identity and current_frame_identity != "background":
            recording_until = time.time() + VIDEO_BUFFER_SECONDS
            
            if video_writer is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prob_int = int(current_frame_conf * 100)
                filename = f"{current_frame_identity}_p{prob_int}_{timestamp}.mp4"
                save_path = os.path.join(DETECTIONS_DIR, filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
                print(f"📹 Recording: {filename}")

        if video_writer is not None:
            # We write the 'frame' AFTER drawing on it
            video_writer.write(frame)
            if time.time() > recording_until:
                video_writer.release()
                video_writer = None
                print("🏁 Saved.")

        cv2.imshow("Garden Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer: video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_monitor()