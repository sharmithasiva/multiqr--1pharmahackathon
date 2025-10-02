import json
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------
# Paths
# ------------------------
json_file = "outputs/submission_decoding_2.json"  # Your prediction JSON
img_folder = "src/data/images/test"               # Test images folder

# ------------------------
# Load JSON predictions
# ------------------------
with open(json_file, "r") as f:
    data = json.load(f)

# ------------------------
# Visualization loop
# ------------------------
plt.figure(figsize=(10,10))

for item in data:
    image_id = item["image_id"]
    img_path_jpg = Path(img_folder) / f"{image_id}.jpg"
    img_path_png = Path(img_folder) / f"{image_id}.png"

    if img_path_jpg.exists():
        img_path = str(img_path_jpg)
    elif img_path_png.exists():
        img_path = str(img_path_png)
    else:
        print(f"Image {image_id} not found, skipping")
        continue

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Draw boxes and labels
    for qr in item["qrs"]:
        x1, y1, x2, y2 = qr["bbox"]
        value = qr.get("value", "")
        qr_type = qr.get("type", "")
        text = f"{value} ({qr_type})" if value else qr_type if qr_type else "QR"

        # Color: green if decoded, red if empty
        color = (0, 255, 0) if value else (255, 0, 0)  # RGB

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_rgb, text, (x1, max(y1-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display with Matplotlib
    plt.clf()
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(image_id)
    plt.pause(3)  # Show each image for 3 seconds

    # Optional: manual review
    input("Press Enter to move to next image...")

plt.close()
print("All images processed.")
