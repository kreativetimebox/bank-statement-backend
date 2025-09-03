import json
import os
from urllib.parse import urlparse
from collections import defaultdict

INPUT_FILE = "grouped_key_value.json"   # new input
OUTPUT_FILE = "donut_format.jsonl"

FIELD_ORDER = [
    "Invoice Number",
    "Receipt Date",
    "Supplier Name",
    "Item Name",
    "Item Quantity",
    "Item Unit Price",
    "Item Amount",
    "Item VAT Code",
    "VAT Code",
    "VAT Percent",
    "VAT Amount",
    "Net Amount",
    "Total Amount",
    "Payment Method",
    "Total Item Count",
    "Currency",
    "Coupon Name",
    "Total Discount"
]

def image_filename(path: str) -> str:
    """Extract only filename from path or URL."""
    if not path:
        return ""
    parsed = urlparse(path)
    candidate = parsed.path if parsed.path else path
    candidate = candidate.split("?")[0].split("#")[0]
    return os.path.basename(candidate)

def extract_fields(item: dict):
    """Extract label→text pairs (key,value) directly from 'fields'."""
    pairs = []
    for f in item.get("fields", []):
        key = f.get("key", "").strip()
        val = f.get("value", "").strip()
        if key:
            pairs.append((key, val))
    return pairs

def build_ground_truth_str(pairs):
    """
    Build the ground_truth string manually:
      - Keep duplicates
      - Missing fields → ""
      - Order fixed by FIELD_ORDER
    """
    field_values = defaultdict(list)
    for lbl, txt in pairs:
        field_values[lbl].append(txt)

    lines = []
    for field in FIELD_ORDER:
        if field in field_values:
            for v in field_values[field]:
                lines.append(f' "{field}": "{v}"')
        else:
            lines.append(f' "{field}": ""')

    body = ",".join(lines)
    return "{ " + body + " }"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for item in data:
            img_name = image_filename(item.get("image_path", ""))
            pairs = extract_fields(item)
            ground_truth = build_ground_truth_str(pairs)

            entry = {
                "image": img_name,
                "ground_truth": ground_truth
            }
            out.write(
                '{"image": "' + img_name + '", "ground_truth": ' + json.dumps(ground_truth, ensure_ascii=False) + "} \n"
            )

    print(f"✅ Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
