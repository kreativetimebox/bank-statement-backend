#!/usr/bin/env python3
"""
Improved Receipt Entity Extraction with Better Post-processing
"""

import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import json
import re

def load_model():
    """Load the fine-tuned LayoutLMv3 model"""
    model_path = "/Users/utkarshupadhyay/Computer Science/Kreativetimebox/Layoutlmv3_1/layoutlmv3_receipt_model"
    
    # Load processor from the original model with apply_ocr=False since we're providing our own OCR
    from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast
    
    image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
    
    # Load the fine-tuned model
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    
    # Load label mapping
    with open("/Users/utkarshupadhyay/Computer Science/Kreativetimebox/Layoutlmv3_1/layoutlmv3_data/label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    # Create reverse mapping from the label_to_id section
    label_to_id = label_mapping["label_to_id"]
    id2label = {v: k for k, v in label_to_id.items()}
    
    return processor, model, id2label

def perform_ocr(image_path):
    """Perform OCR using DocTR"""
    # Load the OCR model
    model = ocr_predictor(pretrained=True)
    
    # Load and process the document
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    
    # Extract words with their bounding boxes
    words = []
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # Get geometry coordinates
                    geometry = word.geometry
                    
                    # Handle different geometry formats
                    if len(geometry) == 4:
                        x1, y1, x2, y2 = geometry
                    elif len(geometry) == 2:
                        # Format might be [(x1, y1), (x2, y2)]
                        (x1, y1), (x2, y2) = geometry
                    else:
                        print(f"Warning: Unexpected geometry format: {geometry}")
                        continue
                    
                    # Convert to absolute coordinates
                    bbox = [
                        int(x1 * img_width),
                        int(y1 * img_height), 
                        int(x2 * img_width),
                        int(y2 * img_height)
                    ]
                    
                    words.append({
                        'text': word.value,
                        'bbox': bbox,
                        'confidence': word.confidence
                    })
    
    return words, (img_width, img_height)

def is_price(text):
    """Check if text looks like a price"""
    price_pattern = r'^\d+\.\d{2}$|^-\d+\.\d{2}$|^\d+,\d{2}$'
    return bool(re.match(price_pattern, text))

def is_quantity(text):
    """Check if text looks like a quantity"""
    quantity_patterns = [
        r'^\d+$',  # Just numbers
        r'^\d+x$', r'^x\d+$',  # With x
        r'^\d+kg$', r'^\d+g$', r'^\d+ml$', r'^\d+l$'  # With units
    ]
    return any(re.match(pattern, text.lower()) for pattern in quantity_patterns)

def is_item_code(text):
    """Check if text looks like an item code"""
    code_patterns = [
        r'^\d{5,}$',  # Long numbers (5+ digits)
        r'^[A-Z]{2,}\d+$',  # Letters followed by numbers
        r'^\d+[A-Z]+$'  # Numbers followed by letters
    ]
    return any(re.match(pattern, text) for pattern in code_patterns)

def post_process_entities(entities):
    """Apply business rules to improve entity classification"""
    processed = []
    
    for entity in entities:
        text = entity['text']
        original_label = entity['label']
        confidence = entity['confidence']
        
        # Skip very low confidence predictions (lower threshold for supplier names)
        min_confidence = 0.4 if 'SUPPLIER' in original_label else 0.6
        if confidence < min_confidence:
            continue
            
        # Apply business rules
        new_label = original_label
        
        # Price detection
        if is_price(text):
            new_label = 'Item_Amount'
            confidence = min(confidence + 0.2, 0.95)  # Boost confidence
            
        # Quantity detection
        elif is_quantity(text) and 'quantity' not in original_label.lower():
            new_label = 'Item_Quantity'
            confidence = min(confidence + 0.15, 0.9)
            
        # Item code detection
        elif is_item_code(text):
            new_label = 'Item_Code'
            confidence = min(confidence + 0.1, 0.85)
            
        # Supplier name rules
        elif 'SUPPLIER' in original_label:
            # Common store names
            store_names = ['ALDI', 'TESCO', 'ASDA', 'SAINSBURY', 'MORRISONS', 'LIDL', 'STORES']
            if text.upper() in store_names:
                confidence = min(confidence + 0.2, 0.95)
            elif len(text) < 2 or text.isdigit():
                continue  # Skip very short or numeric "supplier names"
                
        # Filter out obvious misclassifications
        if (original_label == 'Item_Quantity' and 
            text.lower() in ['a', 'the', 'card', 'there', 'and', 'or']):
            continue
            
        if (original_label == 'Supplier_Name' and 
            (is_price(text) or text.lower() in ['gbp', 'eur', 'usd'])):
            continue
            
        processed.append({
            'text': text,
            'label': new_label,
            'confidence': confidence,
            'bbox': entity['bbox'],
            'original_label': original_label
        })
    
    return processed

def combine_multi_token_entities(entities):
    """Combine B- and I- tagged tokens into complete entities"""
    combined = []
    current_entity = None
    
    for entity in entities:
        label = entity['label']
        
        if label.startswith('B-'):
            # Start new entity
            if current_entity:
                combined.append(current_entity)
            
            current_entity = {
                'text': entity['text'],
                'label': label[2:],  # Remove B- prefix
                'confidence': [entity['confidence']],
                'bbox': entity['bbox'],
                'tokens': [entity]
            }
            
        elif label.startswith('I-') and current_entity and label[2:] == current_entity['label']:
            # Continue current entity
            current_entity['text'] += ' ' + entity['text']
            current_entity['confidence'].append(entity['confidence'])
            current_entity['tokens'].append(entity)
            
        else:
            # End current entity and start new single-token entity
            if current_entity:
                combined.append(current_entity)
                
            current_entity = {
                'text': entity['text'],
                'label': label.replace('B-', '').replace('I-', ''),
                'confidence': [entity['confidence']],
                'bbox': entity['bbox'],
                'tokens': [entity]
            }
    
    # Don't forget last entity
    if current_entity:
        combined.append(current_entity)
    
    # Calculate average confidence for each entity
    for entity in combined:
        entity['avg_confidence'] = sum(entity['confidence']) / len(entity['confidence'])
    
    return combined

def extract_entities(image_path):
    """Extract entities from receipt image with improved processing"""
    print(f"🔍 Processing receipt: {image_path}")
    
    # Load model
    print("📦 Loading fine-tuned LayoutLMv3 model...")
    processor, model, id2label = load_model()
    
    # Perform OCR
    print("📄 Performing OCR with DocTR...")
    words, image_size = perform_ocr(image_path)
    print(f"✅ OCR completed: {len(words)} words extracted")
    
    if not words:
        print("❌ No text found in image")
        return []
    
    # Prepare data for LayoutLMv3
    tokens = [word['text'] for word in words]
    bboxes = []
    
    img_width, img_height = image_size
    print(f"📏 Image size: {image_size}")
    
    # Normalize bboxes to 0-1000 scale (LayoutLMv3 format)
    for word in words:
        x1, y1, x2, y2 = word['bbox']
        normalized_bbox = [
            int((x1 / img_width) * 1000),
            int((y1 / img_height) * 1000),
            int((x2 / img_width) * 1000),
            int((y2 / img_height) * 1000)
        ]
        bboxes.append(normalized_bbox)
    
    # Process with LayoutLMv3
    print("🤖 Running entity extraction...")
    image = Image.open(image_path)
    
    # For LayoutLMv3, we provide tokens and boxes since we disabled OCR
    encoding = processor(
        image,
        text=tokens,
        boxes=bboxes,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Extract predictions
    predictions = torch.argmax(outputs.logits, dim=2)
    probabilities = torch.softmax(outputs.logits, dim=2)
    
    # Convert to entities
    results = []
    token_idx = 0
    
    for i, (input_id, pred_label) in enumerate(zip(encoding.input_ids[0], predictions[0])):
        if input_id in [processor.tokenizer.cls_token_id, processor.tokenizer.sep_token_id, processor.tokenizer.pad_token_id]:
            continue
            
        if token_idx < len(tokens):
            label = id2label.get(pred_label.item(), 'O')
            
            if label != 'O':
                confidence = probabilities[0][i][pred_label].item()
                
                results.append({
                    'text': tokens[token_idx],
                    'bbox': bboxes[token_idx],
                    'label': label,
                    'confidence': confidence
                })
            
            token_idx += 1
    
    # Apply post-processing
    print("🔧 Applying post-processing rules...")
    processed_entities = post_process_entities(results)
    
    # Combine multi-token entities
    combined_entities = combine_multi_token_entities(processed_entities)
    
    return combined_entities

def display_improved_results(entities):
    """Display results with better formatting"""
    print("\n🎯 IMPROVED ENTITY EXTRACTION RESULTS:")
    print("=" * 80)
    
    # Group by entity type
    grouped = {}
    for entity in entities:
        entity_type = entity['label']
        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(entity)
    
    # Sort by confidence within each group
    for entity_type in grouped:
        grouped[entity_type].sort(key=lambda x: x['avg_confidence'], reverse=True)
    
    # Display results
    for entity_type, entity_list in grouped.items():
        print(f"\n📋 {entity_type.replace('_', ' ').title()}:")
        print("-" * 50)
        
        for entity in entity_list:
            confidence_bar = "█" * int(entity['avg_confidence'] * 10)
            print(f"  • {entity['text']} (confidence: {entity['avg_confidence']:.2f}) {confidence_bar}")
    
    # Enhanced summary
    total_entities = len(entities)
    high_conf_entities = len([e for e in entities if e['avg_confidence'] >= 0.7])
    
    print(f"\n📊 ENHANCED SUMMARY:")
    print(f"  • Total entities found: {total_entities}")
    print(f"  • High confidence entities (≥0.7): {high_conf_entities}")
    print(f"  • Entity types detected: {len(grouped)}")
    print(f"  • Average confidence: {sum(e['avg_confidence'] for e in entities)/len(entities):.2f}")

if __name__ == "__main__":
    # Process the receipt
    image_path = "/Users/utkarshupadhyay/Computer Science/Kreativetimebox/Layoutlmv3_1/images/IMG0133_638882809282735163.jpg"
    
    import os
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        exit(1)
    
    try:
        entities = extract_entities(image_path)
        if entities:
            display_improved_results(entities)
        else:
            print("❌ No entities extracted")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
