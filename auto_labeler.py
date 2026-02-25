import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
BASE_DATASET_DIR = Path("dataset")
INPUT_FOLDERS = {
    "positives": BASE_DATASET_DIR / "positives", # Cats/Both
    "negatives": BASE_DATASET_DIR / "negatives"  # Just Chickens
}

# Where the training data will live
OUT_IMAGE_DIR = BASE_DATASET_DIR / "train" / "images"
OUT_LABEL_DIR = BASE_DATASET_DIR / "train" / "labels"

MODEL_NAME = "yolo26n.pt"
CONF_THRESHOLD = 0.50
FRAMES_TO_SKIP = 15 # Extract 1 frame per second for 15fps video

# Create directories
OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Load model on M4 hardware
model = YOLO(MODEL_NAME)

def process_folder(folder_path, is_negative=False):
    if not folder_path.exists():
        print(f"Skipping {folder_path}, folder not found.")
        return

    print(f"\n--- Processing {'NEGATIVES (Chickens)' if is_negative else 'POSITIVES (Cats)'} ---")
    
    for video_file in folder_path.glob("*.mp4"):
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        saved_count = 0
        base_name = video_file.stem

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % FRAMES_TO_SKIP == 0:
                img_name = f"{base_name}_f{frame_count}.jpg"
                txt_name = f"{base_name}_f{frame_count}.txt"
                
                img_path = OUT_IMAGE_DIR / img_name
                txt_path = OUT_LABEL_DIR / txt_name

                if is_negative:
                    # FOR CHICKENS: Save image and a completely empty label file
                    cv2.imwrite(str(img_path), frame)
                    open(txt_path, 'w').close() 
                    saved_count += 1
                else:
                    # FOR CATS: Run detection to assist labeling
                    results = model.predict(source=frame, device='mps', classes=[16], conf=CONF_THRESHOLD, verbose=False)
                    
                    if len(results[0].boxes) > 0:
                        cv2.imwrite(str(img_path), frame)
                        with open(txt_path, 'w') as f:
                            for box in results[0].boxes:
                                x_c, y_c, w, h = box.xywhn[0].tolist()
                                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                        saved_count += 1
            
            frame_count += 1
        cap.release()
        print(f"Processed {video_file.name}: Saved {saved_count} frames")

# Run the process
process_folder(INPUT_FOLDERS["positives"], is_negative=False)
process_folder(INPUT_FOLDERS["negatives"], is_negative=True)

print("\nProcessing Complete!")
print(f"Images: {len(list(OUT_IMAGE_DIR.glob('*.jpg')))}")
print(f"Labels: {len(list(OUT_LABEL_DIR.glob('*.txt')))}")