import os
import pytesseract
from PIL import Image
import json
import cv2 
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys


IMAGE_DIR = "surya_i" # Directory containing your receipt images
OUTPUT_JSON_FILE = 'receipt_ocr_tasks.json' # Output JSON file for Label Studio
LABEL_STUDIO_STATIC_SERVER_URL = 'http://localhost:8081/' # URL where your images will be served



def get_ocr_predictions(image_path):

    try:
        img = Image.open(image_path).convert("RGB")  # Ensure compatible format
    except Exception as e:
        print(f"Failed to open image {image_path}: {e}")
        return []

    img_width, img_height = img.size

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    predictions = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i]
        conf = int(data['conf'][i]) # Confidence score

        # Only consider words with reasonable confidence and non-empty text
        # Tesseract returns -1 for blocks/lines, so filter for actual words
        if conf > 0 and text.strip():
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            # Label Studio expects coordinates as percentage of original width/height
            # and bounding box coordinates for a rectangle label
            # Note: Tesseract's origin is top-left, which matches Label Studio.
            value = {
                "x": (x / img_width) * 100,
                "y": (y / img_height) * 100,
                "width": (w / img_width) * 100,
                "height": (h / img_height) * 100,
                "rotation": 0, # Assuming no rotation for now
                "text": [text] # OCR text goes here
            }

            predictions.append({
                "id": f"bbox_{i}{os.path.basename(image_path).replace('.', '')}", # Unique ID for the region
                "from_name": "label", # Refers to the name of your RectangleLabels tag in LS config
                "to_name": "image",   # Refers to the name of your Image tag in LS config
                "type": "rectanglelabels", # Type of annotation
                "value": value,
                "score": conf / 100.0 # Score should be between 0 and 1
            })
    return predictions

def main():
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory '{IMAGE_DIR}' not found. Please create it and add images.")
        return

    tasks = []
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in '{IMAGE_DIR}'. Please add images to this directory.")
        return

    for i, image_filename in enumerate(image_files):
        image_path = os.path.join(IMAGE_DIR, image_filename)
        print(f"Processing {image_path}...")

        image_url = f"{LABEL_STUDIO_STATIC_SERVER_URL}{image_filename}"

        ocr_predictions = get_ocr_predictions(image_path)

        task = {
            "data": {
                "image": image_url # This is the variable referenced by <Image value="$image"/>
            },
            "predictions": [
                {
                    "model_version": "tesseract_ocr_v1", # A version name for your OCR predictions
                    "score": 0.8, # An overall score for the prediction (can be an average of confidences)
                    "result": ocr_predictions
                }
            ]
        }
        tasks.append(task)

    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully generated {len(tasks)} tasks with OCR pre-annotations.")
    print(f"Output saved to {OUTPUT_JSON_FILE}")
    print("\nNext steps:")
    print(f"1. Ensure your static server serves images from '{os.path.abspath(IMAGE_DIR)}' at '{LABEL_STUDIO_STATIC_SERVER_URL}'.")
    print("2. Start Label Studio: label-studio start")
    print("3. Create a new project in Label Studio and set up the labeling interface.")
    print("4. Import the generated JSON file ('receipt_ocr_tasks.json') into your Label Studio project.")

if __name__ == "__main__":
    main()

    from http.server import HTTPServer, test

    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()

    os.chdir(IMAGE_DIR)
    print(f"Serving images from http://localhost:8081/")
    test(CORSRequestHandler, HTTPServer, port=8081)
    

# class CORSRequestHandler (SimpleHTTPRequestHandler):
#     def end_headers (self):
#         self.send_header('Access-Control-Allow-Origin', '*')
#         SimpleHTTPRequestHandler.end_headers(self)

# if __name__ == '__main__':
#     main()  # Run the OCR + task generation

#     # Start static server with CORS
#     os.chdir(IMAGE_DIR)  # Serve files from 'Receipt' directory
#     test(CORSRequestHandler, HTTPServer, port=8081)