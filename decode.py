import os
import json
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
# from pyzbar.pyzbar import decode as pyzbar_decode

def classify_qr(value: str) -> str:
    """Classify QR code based on its value."""
    val = value.upper()
    if val.startswith("B") or "BATCH" in val:
        return "batch"
    elif "MFR" in val or "MANUFACTURER" in val:
        return "manufacturer"
    elif "DST" in val or "DISTRIBUTOR" in val:
        return "distributor"
    else:
        return "other"

def decode_qr_opencv(crop):
    """Try decoding QR with OpenCV QRCodeDetector."""
    detector = cv2.QRCodeDetector()
    value, pts, _ = detector.detectAndDecode(crop)
    if value:
        return value
    # fallback to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    value, pts, _ = detector.detectAndDecode(gray)
    return value

# def decode_qr_pyzbar(crop):
#     """Fallback decoder using pyzbar."""
#     decoded_objects = pyzbar_decode(crop)
#     if decoded_objects:
#         return decoded_objects[0].data.decode("utf-8")
#     return ""

def run_inference(weights, input_dir, output_file, conf=0.5, pad=10, resize=None):
    """Run YOLO detection and QR decoding on images."""
    model = YOLO(weights)
    results_json = []

    image_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
    print(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        image_id = Path(img_path).stem

        detections = model.predict(img, conf=conf, verbose=False)[0]
        qr_entries = []

        for box in detections.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            # optional padding
            x1_pad = max(0, x1 - pad)
            y1_pad = max(0, y1 - pad)
            x2_pad = min(w, x2 + pad)
            y2_pad = min(h, y2 + pad)
            crop = img[y1_pad:y2_pad, x1_pad:x2_pad]

            # optional resizing
            if resize:
                crop = cv2.resize(crop, resize, interpolation=cv2.INTER_LINEAR)

            # Try OpenCV decoding
            value = decode_qr_opencv(crop)

            # Fallback to pyzbar if OpenCV failed
            # if not value:
            #     value = decode_qr_pyzbar(crop)

            qr_type = classify_qr(value) if value else "other"

            qr_entries.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "value": value if value else "",
                "type": qr_type
            })

        results_json.append({
            "image_id": image_id,
            "qrs": qr_entries
        })

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--input", type=str, default="demo_images/", help="Folder with images")
    parser.add_argument("--output", type=str, default="outputs/submission_decoding.json", help="Output JSON file")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--pad", type=int, default=10, help="Padding around detected boxes")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="Resize cropped QR (width height)")
    args = parser.parse_args()

    run_inference(args.weights, args.input, args.output, args.conf, args.pad, tuple(args.resize) if args.resize else None)
