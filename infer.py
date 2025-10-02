import os
import json
import argparse
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

def main(input_dir, output_file, model_path='src/models/best.pt', conf_thres=0.5):
    input_dir = Path(input_dir)
    assert input_dir.exists(), f"Input directory {input_dir} does not exist."

    # Load YOLOv8 model
    model = YOLO(model_path)

    results_json = []

    # Iterate through all images
    for img_path in sorted(input_dir.glob("*.*")):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        # Run inference
        results = model.predict(source=str(img_path), conf=conf_thres, verbose=False)[0]

        qrs = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Round coordinates to integers
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                qrs.append({"bbox": bbox})

        # Append result for this image
        results_json.append({
            "image_id": img_path.stem,
            "qrs": qrs
        })

    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Inference complete. JSON saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 QR Code Detection Inference")
    parser.add_argument("--input", type=str, required=True, help="Folder of test images")
    parser.add_argument("--output", type=str, default="outputs/submission_detection_1.json", help="Output JSON file")
    parser.add_argument("--model", type=str, default="src/models/best.pt", help="Path to trained YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)
    main(args.input, args.output, args.model, args.conf)
