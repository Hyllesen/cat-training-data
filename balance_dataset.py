import os
import random

# Target count based on your smallest cat class
TARGET_COUNT = 1000 
DATA_DIR = "training_data"

for folder in ["orange", "squaky", "background"]:
    path = os.path.join(DATA_DIR, folder)
    if not os.path.exists(path): continue
    
    files = os.listdir(path)
    if len(files) > TARGET_COUNT:
        print(f"⚖️ Balancing {folder}: {len(files)} -> {TARGET_COUNT}")
        
        # Pick random files to delete
        to_delete = random.sample(files, len(files) - TARGET_COUNT)
        
        for f in to_delete:
            os.remove(os.path.join(path, f))

print("✅ Dataset balanced!")