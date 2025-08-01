import os
import json
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import easyocr

# Initialize EasyOCR reader (language: English)
reader = easyocr.Reader(['en'])

def process_image(image_path, image_url):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_np = np.array(image)

    # Perform OCR using EasyOCR
    ocr_results = reader.readtext(image_np)

    results = []
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for (bbox, text, conf) in ocr_results:
        try:
            if not text.strip() or conf < 0.3:
                continue

            # EasyOCR bbox is [top-left, top-right, bottom-right, bottom-left]
            # We need to find min x, y and max x, y for the rectangle
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]

            x0, y0 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            region_id = str(uuid.uuid4())

            results.append({
                "id": region_id,
                "type": "rectangle",
                "from_name": "bbox",
                "to_name": "image",
                "value": {
                    "x": (x0 / width) * 100,
                    "y": (y0 / height) * 100,
                    "width": ((x2 - x0) / width) * 100,
                    "height": ((y2 - y0) / height) * 100
                }
            })

            results.append({
                "id": region_id,
                "type": "labels",
                "from_name": "label",
                "to_name": "image",
                "value": {
                    "labels": ["Text"]
                }
            })

            results.append({
                "id": region_id,
                "type": "textarea",
                "from_name": "transcription",
                "to_name": "image",
                "value": {
                    "text": [text]
                }
            })

            # Draw rectangle and text on the image
            draw.rectangle([x0, y0, x2, y2], outline="red", width=2)
            draw.text((x0, y0 - 10), text, fill="blue", font=font)

        except Exception as e:
            print(f"Skipping region due to: {e}")
            continue

    return results, image


def process_folder(input_folder, output_folder, json_output_path, web_server_base_url):
    os.makedirs(output_folder, exist_ok=True)
    all_tasks = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)
        image_url = f"{web_server_base_url}{filename}"

        results, annotated_image = process_image(image_path, image_url)
        annotated_image.save(output_image_path)

        all_tasks.append({
            "data": {
                "image": image_url
            },
            "annotations": [{
                "result": results
            }]
        })

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"✅ All annotations saved to {json_output_path}")


# === Configuration ===
input_folder = "input"  # <- Folder with your input JPG/PNG images
output_folder = "output"       # <- Folder for output image preview + JSON
json_output_path = os.path.join(output_folder, "annotations.json")
web_server_base_url = "http://localhost:8000/"

process_folder(input_folder, output_folder, json_output_path, web_server_base_url)