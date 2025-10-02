from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("src/models/best.pt")

# Path to a test image
img_path = "src/data/images/test/img235.jpg"

# Run inference
results = model.predict(img_path, conf=0.25)  # conf=confidence threshold

# Show the image with bounding boxes
results[0].show()  # Opens a window with the image + boxes

# Optional: save the image with drawn boxes
results[0].save(save_dir="outputs/visuals")
