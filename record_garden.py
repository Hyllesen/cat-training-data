import cv2
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
RTSP_URL = os.getenv("RTSP_URL")
OUTPUT_FILE = "empty_garden_background.mp4"
RECORD_SECONDS = 60  # Duration of the clip

def record_background():
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("❌ Error: Could not connect to the RTSP stream. Check your .env file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    
    # Define the codec (MP4 format for Mac compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))

    print(f"🎬 Recording {RECORD_SECONDS} seconds of background...")
    print("⚠️ Make sure there are NO cats in the frame!")

    start_time = time.time()
    
    while int(time.time() - start_time) < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        
        # Optional: Show the feed so you can see if a cat walks in
        cv2.imshow("Recording Background - Press 'q' to stop early", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Finished! Saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    record_background()