import pytesseract
import cv2
import json
import os
import uuid
import threading
from PIL import Image, ImageDraw, ImageFont
import http.server
import socketserver

# Configuration
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output"
JSON_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "annotations.json")
WEB_SERVER_BASE_URL = "http://localhost:8000/"
THRESHOLD = 60
PORT = 8000

# --- OCR + Annotation Logic ---
def process_image(image_path, image_url, threshold=60):
    results = []

    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = image_cv.shape
    pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config='--psm 6')

    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        try:
            conf = int(ocr_data['conf'][i])
        except ValueError:
            conf = 0

        if not text or conf < threshold:
            continue

        x, y, bw, bh = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
        x_pct = x / w * 100
        y_pct = y / h * 100
        bw_pct = bw / w * 100
        bh_pct = bh / h * 100

        region_id = str(uuid.uuid4())

        results.append({
            "id": region_id,
            "type": "rectangle",
            "from_name": "bbox",
            "to_name": "image",
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": bw_pct,
                "height": bh_pct
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

        # Draw box
        draw.rectangle([(x, y), (x + bw, y + bh)], outline="red", width=2)
        draw.text((x, y - 12 if y - 12 > 0 else y + bh + 2), f"{text}", fill="blue", font=font)

    return results, pil_image


def process_folder(input_folder, output_folder, json_output_path, web_server_base_url, threshold=60):
    os.makedirs(output_folder, exist_ok=True)
    all_tasks = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image_url = f"{web_server_base_url}{filename}"

        results, annotated_image = process_image(input_path, image_url, threshold)
        annotated_image.save(output_path)

        task = {
            "data": {
                "image": image_url
            },
            "annotations": [{
                "result": results
            }]
        }
        all_tasks.append(task)

    with open(json_output_path, "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"✅ All annotations saved to: {json_output_path}")

# --- CORS HTTP Server Logic ---
class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUTPUT_FOLDER, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'x-api-key,Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

def start_server():
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🚀 Serving from {os.path.abspath(OUTPUT_FOLDER)} at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

# --- Main Execution ---
if __name__ == "__main__":
    print("🔍 Starting OCR and annotation generation...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, JSON_OUTPUT_PATH, WEB_SERVER_BASE_URL, THRESHOLD)

    print("🌐 Starting HTTP server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    input("🔁 Press ENTER to stop the server.\n")