import os
import json
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PPStructureV3
from PIL import Image

# === Utility to convert NumPy & float types to JSON serializable ===
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

# === Clean PPStructure Result ===
def clean_ppstructure_result(result):
    data = result.get("res", result)

    output = {
        "input_path": data.get("input_path"),
        "page_index": data.get("page_index"),
        "model_settings": data.get("model_settings", {}),
        "layout_det_res": {
            "boxes": data.get("layout_det_res", {}).get("boxes", [])
        },
        "overall_ocr_res": {
            "rec_texts": data.get("overall_ocr_res", {}).get("rec_texts", [])
        },
        "table_res_list": data.get("table_res_list", [])
    }

    return make_serializable(output)

# === Resize image if width or height > 4000 ===
def resize_image_if_needed(image, max_dim=4000):
    width, height = image.size
    if width > max_dim or height > max_dim:
        ratio = min(max_dim / width, max_dim / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

# === Convert PDF to images and process ===
def process_pdf(pdf_path, output_json_path, image_output_dir="/content/pdf_pages"):
    os.makedirs(image_output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=300)

    ocr_engine = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)
    final_results = []

    for i, page in enumerate(pages):
        image_path = os.path.join(image_output_dir, f"page_{i+1}.png")

        resized_image = resize_image_if_needed(page)
        resized_image.save(image_path)

        # Run PPStructureV3
        results = ocr_engine.predict(image_path)
        for result in results:
            cleaned = clean_ppstructure_result(result)
            final_results.append(cleaned)

    # Save final structured data
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"✅ All pages processed. Results saved to: {output_json_path}")

# === Example usage ===
pdf_file = "/content/sample.pdf"  # Replace with your PDF path
output_json = "/content/ppstructurev3_from_pdf.json"
process_pdf(pdf_file, output_json)
