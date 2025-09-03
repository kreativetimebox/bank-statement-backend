import json

input_file = "1_10_Final_JSON.json"
output_file = "grouped_key_value.json"

def extract_grouped_by_image(data):
    extracted = []

    for item in data:
        image_path = item.get("data", {}).get("image", None)
        fields = []

        for ann in item.get("annotations", []):
            results = ann.get("result", [])
            temp_map = {}

            for r in results:
                rid = r.get("id")
                value = r.get("value", {})

                if r["type"] == "rectangle":
                    temp_map.setdefault(rid, {})["bbox"] = {
                        "x": value.get("x"),
                        "y": value.get("y"),
                        "width": value.get("width"),
                        "height": value.get("height")
                    }

                elif r["type"] == "textarea":
                    text = value.get("text", [])
                    if text:
                        temp_map.setdefault(rid, {})["text"] = " ".join(text)

                elif r["type"] == "labels":
                    labels = value.get("labels", [])
                    if labels:
                        temp_map.setdefault(rid, {})["label"] = labels[0]

            # Collect only complete key-value pairs
            for rid, info in temp_map.items():
                if "label" in info and "text" in info:
                    fields.append({
                        "key": info["label"],
                        "value": info["text"],
                        "bbox": info.get("bbox", {})
                    })

        if fields:
            extracted.append({
                "image_path": image_path,
                "fields": fields
            })

    return extracted


if __name__ == "__main__":
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = extract_grouped_by_image(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Saved structured output to {output_file}")
