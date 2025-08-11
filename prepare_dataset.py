# prepare_dataset.py
import json
from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
from transformers import AutoProcessor
import os

def px_to_1000(box_px, W, H):
    x0, y0, x1, y1 = box_px
    x0n = int(round((x0 / W) * 1000))
    y0n = int(round((y0 / H) * 1000))
    x1n = int(round((x1 / W) * 1000))
    y1n = int(round((y1 / H) * 1000))
    # clip
    return [max(0, min(1000, x0n)), max(0, min(1000, y0n)),
            max(0, min(1000, x1n)), max(0, min(1000, y1n))]

def labels_to_bio(word_labels):
    # word_labels: list of labels where 'O' indicates no entity
    bio = []
    prev = None
    for lab in word_labels:
        if not lab or lab == "O":
            bio.append("O")
            prev = None
        else:
            # if previous label same, I-, else B-
            if prev == lab:
                bio.append("I-" + lab)
            else:
                bio.append("B-" + lab)
            prev = lab
    return bio

def build_label_set(examples):
    # collect all entity base names (excluding 'O')
    s = set()
    for ex in examples:
        for lab in ex["assigned_labels"]:
            if lab and lab != "O":
                s.add(lab)
    return sorted(list(s))

def convert_examples_to_hf(examples_json, out_dir="hf_dataset", processor_name="microsoft/layoutlmv3-base"):
    examples = json.load(open(examples_json, "r", encoding="utf-8"))

    # create label list
    entity_names = build_label_set(examples)  # e.g. ["Supplier Name","Item","Item Amount",...]
    # form BIO label_list: O plus B- and I- for each entity
    label_list = ["O"]
    for name in entity_names:
        label_list.append("B-" + name)
        label_list.append("I-" + name)

    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}

    dataset_items = []
    for ex in examples:
        W = ex["image_width"]; H = ex["image_height"]
        boxes_1000 = [px_to_1000(b, W, H) for b in ex["word_boxes_px"]]
        bio_list = labels_to_bio(ex["assigned_labels"])
        labels_ids = [label2id.get(t, label2id["O"]) for t in bio_list]
        dataset_items.append({
            "image_path": ex["image_path"],
            "words": ex["words"],
            "bboxes": boxes_1000,
            "labels": labels_ids
        })

    # split: simple random split (adjust or use explicit split)
    np.random.seed(42)
    np.random.shuffle(dataset_items)
    n = len(dataset_items)
    train = dataset_items[: int(0.8*n)]
    val = dataset_items[int(0.8*n): int(0.9*n)]
    test = dataset_items[int(0.9*n):]

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test)
    })

    # save metadata (label mappings)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "label_list.json"), "w", encoding="utf-8") as f:
        json.dump({"label_list": label_list, "label2id": label2id, "id2label": id2label}, f, indent=2)

    ds_dict.save_to_disk(out_dir)
    print(f"Saved HF dataset to {out_dir} with sizes: ",
          {k: len(ds_dict[k]) for k in ds_dict.keys()})
    return out_dir

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--examples", required=True, help="examples.json produced earlier")
    p.add_argument("--out", default="hf_dataset", help="output dataset dir")
    args = p.parse_args()
    convert_examples_to_hf(args.examples, args.out)