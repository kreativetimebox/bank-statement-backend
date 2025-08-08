import torch
import json
import os
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import pipeline

# --- 1. Load the NER model from Hugging Face ---
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# --- 2. Load the image and perform OCR with doctr ---
image_path = r'C:\Users\Lenovo\Desktop\DOCTR\data\IMG0264_638882805886603164.jpg'  # change to your image path
doc = DocumentFile.from_images(image_path)
model = ocr_predictor(pretrained=True)
result = model(doc)
output = result.export()

# --- 3. Extract text, process with NER, and structure for JSON output ---
final_output = {"pages": []}

for page_idx, page in enumerate(output["pages"]):
    page_data = {
        "page_number": page_idx + 1,
        "lines": []
    }
    
    for block in page["blocks"]:
        for line in block["lines"]:
            line_text = ""
            line_confidences = []

            for word in line["words"]:
                line_text += word["value"] + " "
                line_confidences.append(word["confidence"])

            clean_text = line_text.strip()
            
            if line_confidences:
                avg_confidence = sum(line_confidences) / len(line_confidences)
            else:
                avg_confidence = 0

            entities = ner_pipeline(clean_text)
            
            found_entities = []
            if entities:
                for entity in entities:
                    found_entities.append({
                        "text": entity['word'],
                        "type": entity['entity_group'],
                        "ner_confidence": float(entity['score'])
                    })

            page_data["lines"].append({
                "text": clean_text,
                "average_ocr_confidence": float(avg_confidence),
                "named_entities": found_entities
            })

    final_output["pages"].append(page_data)

# --- 4. Generate a unique filename and write the JSON file ---

# Get the base name of the input image without the extension
input_filename = os.path.splitext(os.path.basename(image_path))[0]
# Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Combine them to create a unique output filename
output_filepath = f"{input_filename}_{timestamp}.json"

with open(output_filepath, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print(f"✅ Results successfully saved to: {output_filepath}")