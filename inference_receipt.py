"""
Receipt Entity Extraction using Fine-tuned LayoutLMv3 + DocTR OCR
"""

import json
import torch
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification
from PIL import Image
import numpy as np

def setup_model():
    """Load the fine-tuned LayoutLMv3 model"""
    model_path = "layoutlmv3_receipt_model"
    
    # Load tokenizer and model
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_path)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    
    # Load label mapping
    with open('layoutlmv3_data/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    return tokenizer, model, id_to_label

def perform_ocr_with_doctr(image_path):
    """Perform OCR using DocTR"""
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        # Initialize DocTR model
        model = ocr_predictor(pretrained=True)
        
        # Load and process image
        doc = DocumentFile.from_images(image_path)
        
        # Perform OCR
        result = model(doc)
        
        # Extract text and bounding boxes
        tokens = []
        bboxes = []
        
        # Process DocTR results
        for page in result.pages:
            page_width = page.dimensions[1]  # width
            page_height = page.dimensions[0]  # height
            
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Get word text
                        text = word.value
                        
                        # Get bounding box (normalized coordinates)
                        bbox = word.geometry
                        
                        # Convert to absolute coordinates then to 0-1000 scale
                        x_min = int(bbox[0][0] * page_width)
                        y_min = int(bbox[0][1] * page_height)
                        x_max = int(bbox[1][0] * page_width)
                        y_max = int(bbox[1][1] * page_height)
                        
                        # Normalize to 0-1000 scale for LayoutLMv3
                        x_min_norm = int((x_min / page_width) * 1000)
                        y_min_norm = int((y_min / page_height) * 1000)
                        x_max_norm = int((x_max / page_width) * 1000)
                        y_max_norm = int((y_max / page_height) * 1000)
                        
                        tokens.append(text)
                        bboxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
        
        return tokens, bboxes, (page_width, page_height)
        
    except ImportError:
        print("DocTR not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "python-doctr[torch]"], check=True)
        print("DocTR installed. Please run the script again.")
        return None, None, None

def extract_entities(image_path):
    """Main function to extract entities from receipt image"""
    print(f"🔍 Processing receipt: {image_path}")
    
    # Setup model
    print("📦 Loading fine-tuned LayoutLMv3 model...")
    tokenizer, model, id_to_label = setup_model()
    
    # Perform OCR
    print("📄 Performing OCR with DocTR...")
    tokens, bboxes, image_size = perform_ocr_with_doctr(image_path)
    
    if tokens is None:
        return
    
    print(f"✅ OCR completed: {len(tokens)} tokens extracted")
    print(f"📏 Image size: {image_size}")
    
    # Prepare input for LayoutLMv3
    print("🤖 Running entity extraction...")
    
    # Truncate if too long
    max_length = 512
    if len(tokens) > max_length - 2:
        tokens = tokens[:max_length - 2]
        bboxes = bboxes[:max_length - 2]
    
    # Tokenize
    encoding = tokenizer(
        tokens,
        boxes=bboxes,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Extract predictions for actual tokens (not padding)
    input_ids = encoding['input_ids'][0]
    attention_mask = encoding['attention_mask'][0]
    predicted_labels = predictions[0]
    
    # Map predictions back to tokens
    results = []
    token_idx = 0
    
    for i, (input_id, mask, pred_label) in enumerate(zip(input_ids, attention_mask, predicted_labels)):
        if mask == 0:  # Padding token
            continue
            
        # Skip special tokens
        if input_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
            
        if token_idx < len(tokens):
            token = tokens[token_idx]
            bbox = bboxes[token_idx]
            label = id_to_label.get(pred_label.item(), 'O')
            
            results.append({
                'token': token,
                'bbox': bbox,
                'label': label,
                'confidence': torch.softmax(outputs.logits[0][i], dim=0)[pred_label].item()
            })
            
            token_idx += 1
    
    return results

def display_results(results):
    """Display extraction results in a nice format"""
    print("\n🎯 ENTITY EXTRACTION RESULTS:")
    print("=" * 80)
    
    # Group by entity type
    entities = {}
    for result in results:
        if result['label'] != 'O':
            entity_type = result['label'].replace('B-', '').replace('I-', '')
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(result)
    
    # Display grouped entities
    for entity_type, entity_results in entities.items():
        print(f"\n📋 {entity_type.replace('_', ' ').title()}:")
        print("-" * 40)
        
        # Combine B- and I- tags for multi-token entities
        current_entity = ""
        current_confidence = []
        
        for result in entity_results:
            if result['label'].startswith('B-'):
                # Start new entity
                if current_entity:
                    avg_conf = sum(current_confidence) / len(current_confidence)
                    print(f"  • {current_entity} (confidence: {avg_conf:.2f})")
                current_entity = result['token']
                current_confidence = [result['confidence']]
            elif result['label'].startswith('I-') and current_entity:
                # Continue entity
                current_entity += " " + result['token']
                current_confidence.append(result['confidence'])
            else:
                # Single token entity
                if current_entity:
                    avg_conf = sum(current_confidence) / len(current_confidence)
                    print(f"  • {current_entity} (confidence: {avg_conf:.2f})")
                print(f"  • {result['token']} (confidence: {result['confidence']:.2f})")
                current_entity = ""
                current_confidence = []
        
        # Don't forget the last entity
        if current_entity:
            avg_conf = sum(current_confidence) / len(current_confidence)
            print(f"  • {current_entity} (confidence: {avg_conf:.2f})")
    
    # Show summary
    total_entities = len([r for r in results if r['label'] != 'O'])
    total_tokens = len(results)
    
    print(f"\n📊 SUMMARY:")
    print(f"  • Total tokens: {total_tokens}")
    print(f"  • Entity tokens: {total_entities}")
    print(f"  • Entity coverage: {total_entities/total_tokens*100:.1f}%")
    print(f"  • Unique entity types: {len(entities)}")

if __name__ == "__main__":
    # Process the specified receipt
    image_path = "/Users/utkarshupadhyay/Computer Science/Kreativetimebox/Layoutlmv3_1/images/IMG0133_638882809282735163.jpg"
    
    # Check if image exists
    import os
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        exit(1)
    
    try:
        results = extract_entities(image_path)
        if results:
            display_results(results)
        else:
            print("❌ Entity extraction failed")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
