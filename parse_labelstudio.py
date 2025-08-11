# parse_labelstudio.py
import json
import os
from typing import List, Dict, Any

def load_labelstudio_export(path: str) -> List[Dict[str, Any]]:
    """
    Expect a list of task dicts (Label Studio export). Returns the tasks list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def percent_to_pixels(box_percent: Dict[str, Any], W: int, H: int):
    # Label Studio stores x,y,width,height as percents (0-100)
    x = box_percent["x"]
    y = box_percent["y"]
    w = box_percent["width"]
    h = box_percent["height"]
    x0 = (x / 100.0) * W
    y0 = (y / 100.0) * H
    x1 = ((x + w) / 100.0) * W
    y1 = ((y + h) / 100.0) * H
    return [x0, y0, x1, y1]

def parse_task(task: Dict[str, Any], images_root: str = None):
    """
    Convert one Label Studio task into an example with:
      - image_path
      - words: list[str]
      - word_boxes_px: list[[x0,y0,x1,y1]]
      - region_labels: list[{'box':..., 'labels':[...]}]  (region-level)
    """
    image_ref = task.get("image")
    # image_ref may be a local path or a url like http://localhost:8000/IMG0288.jpg
    if images_root and image_ref and not os.path.isabs(image_ref):
        # if your export uses local filenames at the end of url, try to map
        filename = os.path.basename(image_ref)
        image_path = os.path.join(images_root, filename)
    else:
        image_path = image_ref

    W = None
    H = None
    # Some exports include original_width/height inside bbox items; fallback to label entries
    if "bbox" in task and len(task["bbox"]) > 0:
        W = task["bbox"][0].get("original_width")
        H = task["bbox"][0].get("original_height")
    elif "label" in task and len(task["label"]) > 0:
        W = task["label"][0].get("original_width")
        H = task["label"][0].get("original_height")

    if W is None or H is None:
        raise ValueError("Could not determine original image width/height. Check export fields.")

    # word-level boxes (percent coords) -> px boxes
    word_boxes_px = []
    if "bbox" in task:
        for b in task["bbox"]:
            word_boxes_px.append(percent_to_pixels(b, W, H))
    else:
        raise ValueError("No word-level bbox found in 'bbox' key.")

    words = task.get("transcription", [])
    if len(words) != len(word_boxes_px):
        # keep going but warn
        print(f"[WARN] transcription length {len(words)} != bbox length {len(word_boxes_px)} for id {task.get('id')}")

    # region labels: each element has x,y,width,height and labels[] (region-level)
    region_labels = []
    for reg in task.get("label", []):
        box_px = percent_to_pixels(reg, W, H)
        region_labels.append({
            "box_px": box_px,
            "labels": reg.get("labels", [])
        })

    # assign a region label to each word (center-in-rect)
    assigned_labels = []
    for i, wb in enumerate(word_boxes_px):
        x0, y0, x1, y1 = wb
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        assigned = "O"  # default outside any region
        for reg in region_labels:
            rx0, ry0, rx1, ry1 = reg["box_px"]
            # check containment (allow words on boundary)
            if cx >= rx0 and cx <= rx1 and cy >= ry0 and cy <= ry1:
                # take first label name (if multiple, adjust as needed)
                assigned = reg["labels"][0]
                break
        assigned_labels.append(assigned)

    example = {
        "image_path": image_path,
        "words": words,
        "word_boxes_px": word_boxes_px,
        "assigned_labels": assigned_labels,
        "image_width": W,
        "image_height": H,
        "id": task.get("id")
    }
    return example

def parse_dataset(export_json: str, images_root: str = None):
    tasks = load_labelstudio_export(export_json)
    examples = []
    for t in tasks:
        try:
            ex = parse_task(t, images_root)
            examples.append(ex)
        except Exception as e:
            print(f"[ERROR] skipping task {t.get('id')}: {e}")
    return examples

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="label studio export json")
    ap.add_argument("--images_root", default=None, help="folder where images are stored (optional)")
    ap.add_argument("--out", default="examples.json", help="path to save parsed examples")
    args = ap.parse_args()

    exs = parse_dataset(args.export, args.images_root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(exs, f, indent=2)
    print(f"Saved {len(exs)} examples to {args.out}")