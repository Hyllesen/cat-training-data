import os
import shutil
import cv2

# --- CONFIGURATION ---
INPUT_DIR = "recordings"      # Where your chicken/cat spam is currently saved
CAT_DIR   = "dataset/positives" # Folder for Cat or Cat+Chicken
CHICK_DIR = "dataset/negatives" # Folder for Just Chickens

os.makedirs(CAT_DIR, exist_ok=True)
os.makedirs(CHICK_DIR, exist_ok=True)

print("--- Dataset Sorter ---")
print("Controls: [C] = Cat/Both, [N] = Negative (Chicken), [S] = Skip, [Q] = Quit")

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith((".mp4", ".mov")): continue
    
    video_path = os.path.join(INPUT_DIR, filename)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if ret:
        # Show a preview of the video to decide
        cv2.imshow("Sort this clip", cv2.resize(frame, (800, 450)))
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('c'):
            shutil.move(video_path, os.path.join(CAT_DIR, filename))
            print(f"Moved {filename} to POSITIVES (Cat/Both)")
        elif key == ord('n'):
            shutil.move(video_path, os.path.join(CHICK_DIR, filename))
            print(f"Moved {filename} to NEGATIVES (Chicken)")
        elif key == ord('q'):
            break
        else:
            print(f"Skipped {filename}")
            
    cap.release()

cv2.destroyAllWindows()