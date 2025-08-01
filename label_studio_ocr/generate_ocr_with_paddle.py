import pytesseract
import cv2
import json
import os

# Set Tesseract path (adjust if needed)
# IMPORTANT: Replace this with the actual path to your tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Please change this to your Tesseract path

def extract_text_and_generate_labelstudio_json(image_path, preview_output_path, json_output_path, web_server_base_url="http://localhost:8000/"):
    """
    Extracts text bounding boxes from an image using Tesseract,
    generates a preview image with boxes, and creates a Label Studio JSON file.

    Args:
        image_path (str): Path to the input image.
        preview_output_path (str): Path to save the preview image with bounding boxes.
        json_output_path (str): Path to save the generated Label Studio JSON file.
        web_server_base_url (str): The base URL of your local web server (e.g., "http://localhost:8000/").
                                   Ensure it ends with a slash if it's a directory.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    h, w, _ = img.shape
    boxes = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    results = []
    for i in range(len(boxes['text'])):
        if boxes['text'][i].strip() == "":
            continue

        x, y, bw, bh = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
        x_pct = x / w * 100
        y_pct = y / h * 100
        bw_pct = bw / w * 100
        bh_pct = bh / h * 100

        results.append({
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": bw_pct,
                "height": bh_pct,
                "rectanglelabels": ["Text"]
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels"
        })

        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

    os.makedirs(os.path.dirname(preview_output_path), exist_ok=True)
    cv2.imwrite(preview_output_path, img)
    print(f"Preview image saved to: {preview_output_path}")

    # Construct the URL for the image
    # Assuming the web server is started in the parent directory of 'output'
    # E.g., if preview_output_path is 'output/16_preview.jpg'
    # and web_server_base_url is 'http://localhost:8000/'
    # The final URL will be 'http://localhost:8000/output/16_preview.jpg'
    image_url = f"{web_server_base_url}{os.path.basename(os.path.dirname(preview_output_path))}/{os.path.basename(preview_output_path)}"


    label_studio_json = [{
        "data": {
            "image": image_url
        },
        "annotations": [{
            "result": results
        }]
    }]

    with open(json_output_path, 'w') as f:
        json.dump(label_studio_json, f, indent=2)
    print(f"Label Studio JSON saved to: {json_output_path}")

# --- Example Usage ---
# Make sure '16.jpg' exists in the same directory as your script
# Or provide a full path: image_path="D:/label_studio_ocr/16.jpg"

# Define output paths
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

input_image_name = "5095.png"
preview_image_name = f"{os.path.splitext(input_image_name)[0]}_preview.jpg"
annotations_file_name = "annotations.json"

# Important: This base URL should match where your web server is serving files from.
# If you run 'python -m http.server 8000' from D:\label_studio_ocr,
# and your image is in D:\label_studio_ocr\output, then the base URL needs to include 'output/'
WEB_SERVER_BASE_URL = "http://localhost:8000/" # Ensure this matches your server's address and port

extract_text_and_generate_labelstudio_json(
    image_path=input_image_name,
    preview_output_path=os.path.join(output_dir, preview_image_name),
    json_output_path=os.path.join(output_dir, annotations_file_name),
    web_server_base_url=WEB_SERVER_BASE_URL
)

print("\n--- Next Steps ---")
print(f"1. Start a simple HTTP server in the directory containing your 'output' folder (e.g., `D:\label_studio_ocr`):")
print(f"   `python -m http.server 8000`")
print("2. Start Label Studio: `label-studio start`")
print(f"3. In Label Studio UI (http://localhost:8080):")
print(f"   - Go to Data Import and upload your generated JSON: {os.path.join(output_dir, annotations_file_name)}")
print(f"   - The image should now appear with pre-annotated bounding boxes, loaded from {WEB_SERVER_BASE_URL}output/{preview_image_name}")