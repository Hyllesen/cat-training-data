from ultralytics import YOLO

def train_cat_identity_model():
    # 1. Load the base classification model
    # 'yolov8n-cls' is the "nano" version—perfect for speed and real-time use
    #model = YOLO('yolov8n-cls.pt')
    # Start from the previous model to continue improving it
    model=YOLO('runs/classify/cat_identity_v3/weights/best.pt') 

    outputname = 'cat_identity_v4'  # Name for the new model version

    # 2. Start training
    results = model.train(
        data='training_data',    # Path to your sorted folders
        epochs=50,               # 50 passes through the data
        imgsz=320,               # Standard size for classification
        device='mps',            # USE THE M4 GPU!
        batch=16,                # Number of images processed at once
        name=outputname,   # Name of the output folder
        fliplr=0.5,              # Randomly flip images horizontally for augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4  # Randomly adjust hue, saturation, and brightness for augmentation       
    )

    print("Training Complete!")
    print(f"Your new model is saved at: runs/classify/{outputname}/weights/best.pt")

if __name__ == "__main__":
    train_cat_identity_model()