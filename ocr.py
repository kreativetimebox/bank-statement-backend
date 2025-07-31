import os
import json
import cv2
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Set single input file path

import cv2

# Resize the image manually before OCR
input_image_path = "images/5098.png"
resized_image_path = "resized_input.jpg"

image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {input_image_path}")

resized_image = cv2.resize(image, (1280, 960))
cv2.imwrite(resized_image_path, resized_image)

INPUT_IMAGE = resized_image_path
OUTPUT_FILE = "annotations/ocr_task_single.json"

# Ensure output directory exists
os.makedirs("annotations", exist_ok=True)

# Run OCR
result = ocr.predict(INPUT_IMAGE)

# Prepare task
filename = os.path.basename(INPUT_IMAGE)
image = cv2.imread(INPUT_IMAGE)
h, w = image.shape[:2]

task = {
    "data": {
        "image": f"/data/local-files/?d=images/{filename}"
    },
    "predictions": [{
        "result": []
    }]
}

# Parse results
for line in result:
    for detection in line:
        box = detection[0]  # 4-point polygon
        (text, conf) = detection[1]

        # Optional confidence threshold
        if conf < 0.5:
            continue

        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        task["predictions"][0]["result"].append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x_min * 100 / w,
                "y": y_min * 100 / h,
                "width": (x_max - x_min) * 100 / w,
                "height": (y_max - y_min) * 100 / h,
                "rectanglelabels": [text]
            }
        })

# Save to file
with open(OUTPUT_FILE, "w") as f:
    json.dump([task], f, indent=2)

print(f"✅ OCR task for single image saved to {OUTPUT_FILE}")