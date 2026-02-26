import cv2
import os
import shutil

# --- CONFIGURATION ---
INPUT_DIR = "dataset_raw"
OUTPUT_DIR = "training_data"
WINDOW_NAME = "Identity Sorter"
DISPLAY_SIZE = 800  # Forces the longest edge of the image to be 800 pixels

CLASS_MAP = {
    ord('s'): "squaky",
    ord('o'): "orange",
    ord('h'): "horny_meow",
    ord('b'): "background",
}

UNDO_KEYS = [63234, 2, 81, ord('u'), ord('U'), 2424832, 65361] 

for folder in CLASS_MAP.values():
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

def resize_for_display(img, max_size):
    """Scales the image proportionally so its longest side matches max_size."""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    # INTER_CUBIC makes upscaled small images look a bit smoother/less pixelated
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

def sort_images():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} folder not found.")
        return

    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(('.jpg', '.png'))]
    total = len(files)
    
    if total == 0:
        print("No images found in dataset_raw to sort.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    print(f"Loaded {total} images.")
    history = []
    i = 0
    
    while i < total:
        filename = files[i]
        img_path = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(img_path):
            i += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            i += 1
            continue

        # Resize the image copy specifically for viewing
        display_img = resize_for_display(img, DISPLAY_SIZE)
        
        cv2.setWindowTitle(WINDOW_NAME, f"[{i+1}/{total}] {filename} - S/O/H/B or Left Arrow to Undo")
        cv2.imshow(WINDOW_NAME, display_img)
        
        key = cv2.waitKeyEx(0)
        char_key = key & 0xFF
        
        if char_key == ord('q'):
            break
        elif key in UNDO_KEYS:
            if history:
                last_orig, last_new, action = history.pop()
                if action == "move":
                    shutil.move(last_new, last_orig)
                i -= 1
            continue
        elif char_key == 32: # Space to skip
            history.append((None, None, "skip"))
            i += 1
        elif char_key in CLASS_MAP:
            target_folder = CLASS_MAP[char_key]
            new_path = os.path.join(OUTPUT_DIR, target_folder, filename)
            shutil.move(img_path, new_path)
            history.append((img_path, new_path, "move"))
            i += 1
            
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)
    print("\nSorting session complete.")

if __name__ == "__main__":
    sort_images()