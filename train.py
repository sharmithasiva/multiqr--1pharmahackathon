from ultralytics import YOLO

# Loading the yolo model
model = YOLO("last.pt")

# Train
model.train(
    data="src/data/data.yaml",
    epochs=50,
    imgsz=[640, 1024],   
    batch=8,
    workers=4,
    name="multiqr_yolov8_train_continue",
    augment=True,        # data augmentation
    mosaic=True,
    mixup=True,
    lrf=0.01,
    box=0.05,
    patience=20,
    optimizer="SGD",
    resume = True
)
