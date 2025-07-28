'''import sys
sys.path.extend(['src/docformer'])

import modeling
import dataset
from transformers import BertTokenizerFast

# Configuration
config = {
    "coordinate_size": 96,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "vocab_size": 30522,
    "max_2d_position_embeddings": 1000,
    "max_position_embeddings": 512,
    "shape_size":1024,
    'layer_norm_eps': 1e-12,
    'hidden_dropout_prob': 0.1,
    'max_relative_positions': 64,
    'intermediate_ff_size_factor': 4,
}

# Load tokenizer and image path
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
image_path = "ABC 2.png"

# Extract features
encoding = dataset.create_features(image_path, tokenizer, add_batch_dim=True)

# Model setup
feature_extractor = modeling.ExtractFeatures(config)
docformer = modeling.DocFormerEncoder(config)

# Forward pass
v_bar, t_bar, v_bar_s, t_bar_s = feature_extractor(encoding)
output = docformer(v_bar, t_bar, v_bar_s, t_bar_s)

print("Doc Encoding Shape:", output.shape)'''

import os, json
from tqdm import tqdm
from PIL import Image
import pytesseract

# Use raw string for Windows paths or forward slashes
DATA_DIR = r"Data\training_data"
OUT_FILE = r"Data\train.txt"

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

with open(OUT_FILE, 'w', encoding='utf-8') as fw:
    for file in tqdm(os.listdir(DATA_DIR)):
        if file.endswith(".png"):
            image_path = os.path.join(DATA_DIR, file)
            image = Image.open(image_path)
            width, height = image.size

            ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            line = []

            for i in range(len(ocr_result["text"])):
                text = ocr_result["text"][i]
                if text.strip() == "":
                    continue
                x, y, w, h = ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i]
                bbox = normalize_bbox([x, y, x + w, y + h], width, height)
                label = "O"  # Set all labels as "O" for now (no entity)
                line.append(f"{text}\t{' '.join(map(str, bbox))}\t{label}")

            line.append("")  # Newline after each image
            fw.write("\n".join(line) + "\n")
