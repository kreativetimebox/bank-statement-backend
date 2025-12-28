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

# ========================
# Configuration
# ========================
INPUT_FOLDER = "receipt_remain"
OUTPUT_FOLDER = "output_remain"
JSON_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "annotations1.json")
WEB_SERVER_BASE_URL = "http://localhost:8001/"
PORT = 8001

# Load Doctr OCR model
doctr_model = ocr_predictor(pretrained=True)


def process_image(image_path, image_url):
    """Run OCR on the image, annotate bounding boxes, and return label-studio style results."""
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

    try:
        doc = DocumentFile.from_images(image_path)
        result = doctr_model(doc)
        exported = result.export()
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return [], pil_image

    for block in exported['pages'][0]['blocks']:
        for line in block['lines']:
            for word in line['words']:
                word_text = word['value'].strip()
                if not word_text:
                    continue

                region_id = str(uuid.uuid4())
                (x_min, y_min), (x_max, y_max) = word['geometry']

                # Convert relative coordinates to pixel values
                x = int(x_min * w)
                y = int(y_min * h)
                bw = int((x_max - x_min) * w)
                bh = int((y_max - y_min) * h)

                # Convert to percentage (Label Studio format)
                x_pct = (x / w) * 100
                y_pct = (y / h) * 100
                bw_pct = (bw / w) * 100
                bh_pct = (bh / h) * 100

                # Add bounding box annotation
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

                # Add transcription annotation
                results.append({
                    "id": region_id,
                    "type": "textarea",
                    "from_name": "transcription",
                    "to_name": "image",
                    "value": {
                        "text": [word_text]
                    }
                })

                # Draw annotation on image
                draw.rectangle([(x, y), (x + bw, y + bh)], outline="red", width=2)
                draw.text(
                    (x, y - 12 if y - 12 > 0 else y + bh + 2),
                    word_text, fill="blue", font=font
                )

    return results, pil_image


def process_folder(input_folder, output_folder, json_output_path, web_server_base_url):
    """Process all images in the folder and generate annotation JSON."""
    os.makedirs(output_folder, exist_ok=True)
    all_tasks = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image_url = f"{web_server_base_url}{filename}"

        print(f"  Processing {filename}...")
        results, annotated_image = process_image(input_path, image_url)
        annotated_image.save(output_path)

        task = {
            "data": {"image": image_url},
            "annotations": [{"result": results}]
        }
        all_tasks.append(task)

    with open(json_output_path, "w") as f:
        json.dump(all_tasks, f, indent=2)

    print(f"  All annotations saved to: {json_output_path}")


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP server with CORS headers for Label Studio."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUTPUT_FOLDER, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-api-key,ContentType")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()


def start_server():
    """Start local HTTP server to serve annotated images."""
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"  Serving from {os.path.abspath(OUTPUT_FOLDER)} at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


if __name__ == "__main__":
    print("  Running OCR and annotation generation...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, JSON_OUTPUT_PATH, WEB_SERVER_BASE_URL)
    print("  Starting local HTTP server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    input("  Press ENTER to stop the server.\n")