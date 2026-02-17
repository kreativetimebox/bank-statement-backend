from paddleocr import PPStructureV3
import json
import numpy as np

def make_serializable(obj):
    """Recursively convert NumPy and float32 types to serializable Python types."""
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
        # 🔥 Preserve full table details
        "table_res_list": data.get("table_res_list", [])
    }

    return make_serializable(output)

# === Load your image ===
image_path = "/kaggle/input/pdfimage/page_1.png"  # Replace with your image
ocr_engine = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)
results = ocr_engine.predict(image_path)

# === Clean and preserve important data ===
final_data = [clean_ppstructure_result(r) for r in results]

# === Save to JSON ===
output_path = "/kaggle/working/ppstructurev3.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"✅ Output with full `table_res_list` saved to: {output_path}")
