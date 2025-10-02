import json
import os
import glob
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

# -------------------------------
# IoU function
# -------------------------------
def iou(box1, box2):
    """box format = [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)

    union = area1 + area2 - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

# -------------------------------
# Load ground truth (YOLO txt → pixel coords)
# -------------------------------
def load_ground_truth(gt_dir, img_dir):
    gt_data = {}
    for txt_file in glob.glob(os.path.join(gt_dir, "*.txt")):
        image_id = Path(txt_file).stem
        img_path_jpg = os.path.join(img_dir, image_id + ".jpg")
        img_path_png = os.path.join(img_dir, image_id + ".png")

        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            print(f"Warning: Image for {image_id} not found, skipping.")
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        boxes = []
        with open(txt_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x, y, w, h = map(float, parts)

                # Convert normalized YOLO → absolute pixel coords
                x_c, y_c = x * img_w, y * img_h
                bw, bh = w * img_w, h * img_h
                x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
                x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)
                boxes.append([x1, y1, x2, y2])
        gt_data[image_id] = boxes
    return gt_data

# -------------------------------
# Load predictions
# -------------------------------
def load_predictions(pred_json):
    with open(pred_json, "r") as f:
        preds = json.load(f)

    pred_data = {}
    for item in preds:
        image_id = item["image_id"]
        pred_data[image_id] = [qr["bbox"] for qr in item["qrs"]]
    return pred_data

# -------------------------------
# mAP computation
# -------------------------------
def compute_map(gt_data, pred_data, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5]  # default

    aps = []
    for thr in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for img_id, gt_boxes in gt_data.items():
            preds = pred_data.get(img_id, [])
            matched = [False] * len(gt_boxes)

            for pred_box in preds:
                ious = [iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(ious) if ious else 0
                max_idx = np.argmax(ious) if ious else -1

                if max_iou >= thr and not matched[max_idx]:
                    tp += 1
                    matched[max_idx] = True
                else:
                    fp += 1

            fn += matched.count(False)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        aps.append(precision)  # simplification: AP≈precision at given IoU

    return np.mean(aps)

# -------------------------------
# Main Evaluation
# -------------------------------
def evaluate(pred_json, gt_dir, img_dir):
    gt_data = load_ground_truth(gt_dir, img_dir)
    pred_data = load_predictions(pred_json)

    # Compute metrics
    tp, fp, fn = 0, 0, 0
    for img_id, gt_boxes in gt_data.items():
        preds = pred_data.get(img_id, [])
        matched = [False] * len(gt_boxes)

        for pred_box in preds:
            ious = [iou(pred_box, gt_box) for gt_box in gt_boxes]
            max_iou = max(ious) if ious else 0
            max_idx = np.argmax(ious) if ious else -1

            if max_iou >= 0.5 and not matched[max_idx]:
                tp += 1
                matched[max_idx] = True
            else:
                fp += 1

        fn += matched.count(False)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    # mAP@0.5
    map50 = compute_map(gt_data, pred_data, [0.5])
    # mAP@[.5:.95] step 0.05
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map5095 = compute_map(gt_data, pred_data, iou_thresholds)

    print("Results:")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  mAP@0.5:   {map50:.4f}")
    print(f"  mAP@.5:.95: {map5095:.4f}")

if __name__ == "__main__":
    pred_json = "outputs/submission_detection_1.json" 
    gt_dir = "src/data/labels/test/"                           
    img_dir = "src/data/images/test/"                          
    evaluate(pred_json, gt_dir, img_dir)
