import os
import cv2
import json
import uuid
import threading
from PIL import Image, ImageDraw, ImageFont
import http.server
import socketserver
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Configuration
INPUT_FOLDER = "receipt_1"
OUTPUT_FOLDER = "output"
JSON_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "annotations.json")
WEB_SERVER_BASE_URL = "http://localhost:8000/"
PORT = 8000

# Load doctr OCR model
doctr_model = ocr_predictor(pretrained=True)

def process_image(image_path, image_url):
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

    doc = DocumentFile.from_images(image_path)
    result = doctr_model(doc)

    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                word_text = word.value.strip()
                if not word_text:
                    continue

                region_id = str(uuid.uuid4())
                (x_min, y_min), (x_max, y_max) = word.geometry
                x = int(x_min * w)
                y = int(y_min * h)
                bw = int((x_max - x_min) * w)
                bh = int((y_max - y_min) * h)

                x_pct = x / w * 100
                y_pct = y / h * 100
                bw_pct = bw / w * 100
                bh_pct = bh / h * 100

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
                    "type": "textarea",
                    "from_name": "transcription",
                    "to_name": "image",
                    "value": {
                        "text": [word_text]
                    }
                })

                draw.rectangle([(x, y), (x + bw, y + bh)], outline="red", width=2)
                draw.text((x, y - 12 if y - 12 > 0 else y + bh + 2), word_text, fill="blue", font=font)

    return results, pil_image

def process_folder(input_folder, output_folder, json_output_path, web_server_base_url):
    os.makedirs(output_folder, exist_ok=True)
    all_tasks = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image_url = f"{web_server_base_url}{filename}"

        results, annotated_image = process_image(input_path, image_url)
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

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def init(self, *args, **kwargs):
        super().init(*args, directory=OUTPUT_FOLDER, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin", "*')
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key,Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

def start_server():
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🌐 Serving images at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("🛑 Server stopped.")

if __name__ == "__main__":
    print("🔍 Running OCR and annotation generation...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, JSON_OUTPUT_PATH, WEB_SERVER_BASE_URL)

    print("🚀 Starting local HTTP server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    input("📦 Press ENTER to stop the server.\n")