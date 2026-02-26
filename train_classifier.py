from ultralytics import YOLO

def train_cat_identity_model():
    # 1. Load the base classification model
    # 'yolov8n-cls' is the "nano" version—perfect for speed and real-time use
    model = YOLO('yolov8n-cls.pt')

    # 2. Start training
    results = model.train(
        data='training_data',    # Path to your sorted folders
        epochs=50,               # 50 passes through the data
        imgsz=224,               # Standard size for classification
        device='mps',            # USE THE M4 GPU!
        batch=32,                # Number of images processed at once
        name='cat_identity_v1'   # Name of the output folder
    )

    print("Training Complete!")
    print(f"Your new model is saved at: runs/classify/cat_identity_v1/weights/best.pt")

if __name__ == "__main__":
    train_cat_identity_model()