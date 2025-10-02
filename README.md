# Multi-QR Code Detection and Recognition for Medicine Packs

This repository contains a solution for detecting multiple QR codes on medicine packs using a lightweight YOLOv8 model. It also optionally decodes and classifies QR codes into categories such as batch, manufacturer, distributor, or other.

This project is built to comply with the hackathon submission requirements.

---
    ```
    multiqr-hackathon/
    │
    ├── README.md                        # Setup & usage instructions
    ├── requirements.txt                 # Python dependencies
    ├── train.py                         # YOLOv8 training script
    ├── infer.py                         # Inference script (Stage 1 → outputs JSON)
    ├── evaluate.py                      # Evaluate model predictions (optional)
    ├── visualise.py                     # Visualise predicted bounding boxes
    ├── decode.py                        # Decoding & classification (Stage 2)
    │
    ├── outputs/                         
    │   ├── submission_detection_1.json  # Output file (Stage 1)
    │   └── submission_decoding_2.json   # Output file (Stage 2, bonus)
    │
    └── src/
        ├── data/
        │   ├── images/
        │   │   ├── train/               # Training images
        │   │   ├── val/                 # Validation images
        │   │   └── test/                # Test images
        │   ├── labels/
        │   │   ├── train/               # YOLO .txt annotations for train
        │   │   ├── val/                 # YOLO .txt annotations for val
        │   │   └── test/                # YOLO .txt annotations for test
        │   └── data.yaml                # Dataset config (paths, classes)
        │
        ├── models/                      
        │   └── best.pt                  # Trained YOLO model weights
    
        

data.yaml : config file that contains datat information like path, classes etc.

path: ./src/data   

train: images/train
val: images/val
test: src/data/images/test

nc: 1   
names: ["qr"]


## Annotation Workflow

- Annotation Tool: Used LabelImg installed via 'pip install labelImg' to annotate QR codes.

- Bounding Boxes: For each QR code in an image, a rectangle box was drawn around it.

- Labeling: Each bounding box was labeled as 'qr'.

- Saving Annotations: Annotations were saved in YOLO .txt format. Each line contains:

  <class_id> <x_center> <y_center> <width> <height>
   All values are normalized between 0 and 1 relative to the image width and height.

Note :  The annotations can be found in src/data/labels/

## Setup Instructions

1. Clone the repository
   ```bash
   git clone https://github.com/sharmithasiva/multiqr--1pharmahackathon.git
   cd multiqr-hackathon

2. Install dependencies
   ```bash
   pip install -r requirements.txt

Stage 1 : QR Code Detection using Yolov8

1. Training the model
   ```bash
   python train.py
- Outputs (weights) will be saved in src/models/best.pt.
- Change training parameters in train.py if needed (epochs, batch size, etc).

2. Inference
   ```bash
   python infer.py --input src/data/images/test/ --output outputs/submission_detection_1.json --model src/models/best.pt --conf 0.5
- Generates submission_detection_1.json in the specified format
  [
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]

3. Evaluation and Visualising Model Predictions (Optional)
   ```bash
   python evaluate.py
- Outputs:
  True Positives (TP), False Positives (FP), False Negatives (FN)
  Precision & Recall
  mAP@0.5
  mAP@0.5:0.95

  ```bash
   python visualise.py
Visualise the predicted bounding boxes for further verification and model fine-tuning.

Stage 2 :  Decoding the detected QR codes

     ```bash
    python decode_qr.py --input src/data/images/test --weights src/models/best.pt --output outputs/submission_decoding_2.json --conf 0.5

Produces submission_decoding_2.json in format:
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max], "value": "B12345", "type": "batch"},
      {"bbox": [x_min, y_min, x_max, y_max], "value": "MFR56789", "type": "manufacturer"}
    ]
  }
]

## Environment
- Python 3.10+
- PyTorch >= 2.0
- Ultralytics >= 8.0
- OpenCV >= 4.7
- Numpy, Matplotlib
- Works on Windows/Linux

## Reproducing Submission

To reproduce submission files run the following commands:

      # Stage 1 Detection Submission
       python infer.py --input src/data/images/test --output outputs/submission_detection_1.json --model src/models/best.pt

       # Stage 2 Decoding Submission (bonus)
       python decode_qr.py --input src/data/images/test --weights src/models/best.pt --output outputs/submission_decoding_2.json


## Notes
- The model is trained on 200 provided images with YOLOv8.
- Handles tilt, blur, and partial occlusion (based on training set).
- No external APIs were used for detection or decoding.
- Decoding relies only on OpenCV only.






   

