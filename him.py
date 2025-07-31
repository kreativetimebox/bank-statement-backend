import os
import json
import cv2
from flask import Flask, send_from_directory
from paddleocr import PaddleOCR

# ==============================
# CONFIG
# ==============================
input_folder = r"./images"
output_file = r"output"
server_port = 8000
base_url = f"http://127.0.0.1:{server_port}"
# ==============================

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Debug: Confirm folder & files
print("📂 Serving from folder:", input_folder)
if not os.path.exists(input_folder):
    print("❌ ERROR: Folder does not exist!")
else:
    print("📄 Files in folder:")
    for f in os.listdir(input_folder):
        print("   -", f)

# Normalize PaddleOCR box format
def normalize_box_format(box):
    if all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box):
        return box
    if isinstance(box, (list, tuple)) and all(isinstance(v, (int, float)) for v in box):
        return [[box[i], box[i + 1]] for i in range(0, len(box), 2)]
    if len(box) == 1 and isinstance(box[0], (list, tuple)):
        inner = box[0]
        if all(isinstance(v, (int, float)) for v in inner):
            return [[inner[i], inner[i + 1]] for i in range(0, len(inner), 2)]
    return []

# Run OCR on all images
def run_ocr_on_folder(folder_path):
    extracted_data = []
    all_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for file in all_files:
        image_path = os.path.join(folder_path, file)
        print(f"🔍 Processing: {file}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠ Skipping unreadable image: {file}")
            continue

        result = ocr.predict(image_path)
        h, w = img.shape[:2]
        annotations = []

        for detection in result[0]:
            raw_box = detection[0]
            box = normalize_box_format(raw_box)

            if isinstance(detection[1], (list, tuple)) and len(detection[1]) >= 2:
                text = detection[1][0]
                confidence = float(detection[1][1])
            else:
                text = str(detection[1])
                confidence = None

            points = [[round((x / w) * 100, 2), round((y / h) * 100, 2)] for x, y in box]

            annotations.append({
                "original_width": w,
                "original_height": h,
                "image_rotation": 0,
                "value": {
                    "points": points,
                    "polygonlabels": ["text"],
                    "text": [text]
                },
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "score": confidence,
                "origin": "auto"
            })

        image_url = f"{base_url}/{file}"
        extracted_data.append({
            "data": {"image": image_url},
            "annotations": [{"result": annotations}]
        })

    return extracted_data

# Flask server
app = Flask(__name__)

@app.route('/<path:filename>')
def serve_image(filename):
    # Debug: Print every image request
    print(f"📥 Request for file: {filename}")
    return send_from_directory(input_folder, filename)

if __name__ == "__main__":
    # Run OCR
    results = run_ocr_on_folder(input_folder)

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved OCR results to: {output_file}")
    print("📌 Import this JSON into Label Studio.")

    # Start Flask server
    print(f"🚀 Starting Flask server at {base_url}")
    app.run(host="127.0.0.1", port=server_port)