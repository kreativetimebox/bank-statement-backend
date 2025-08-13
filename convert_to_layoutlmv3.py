import json
import os
from collections import defaultdict

def convert_to_layoutlmv3_format(input_file, output_dir):
    """
    Convert receipt dataset to LayoutLMv3 training format
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create label mapping
    label_to_id = {
        'O': 0,  # Outside/Other
        'B-SUPPLIER_NAME': 1,
        'I-SUPPLIER_NAME': 2,
        'B-RECEIPT_DATE': 3,
        'I-RECEIPT_DATE': 4,
        'B-RECEIPT_ID': 5,
        'I-RECEIPT_ID': 6,
        'B-ITEM': 7,
        'I-ITEM': 8,
        'B-ITEM_CODE': 9,
        'I-ITEM_CODE': 10,
        'B-ITEM_QUANTITY': 11,
        'I-ITEM_QUANTITY': 12,
        'B-UNIT_PRICE': 13,
        'I-UNIT_PRICE': 14,
        'B-ITEM_AMOUNT': 15,
        'I-ITEM_AMOUNT': 16,
        'B-TAX_AMOUNT': 17,
        'I-TAX_AMOUNT': 18,
        'B-TOTAL_AMOUNT': 19,
        'I-TOTAL_AMOUNT': 20,
        'B-VAT_TAX_CODE': 21,
        'I-VAT_TAX_CODE': 22,
        'B-CURRENCY': 23,
        'I-CURRENCY': 24
    }
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    # Convert each sample
    converted_samples = []
    
    for sample in data:
        if not all(key in sample for key in ['bbox', 'transcription', 'label']):
            continue
            
        # Get basic info
        image_path = sample['image']
        sample_id = sample['id']
        
        # Get text tokens and bboxes
        tokens = sample['transcription']
        bboxes = sample['bbox']
        
        if len(tokens) != len(bboxes):
            print(f"Warning: Mismatch in sample {sample_id}: {len(tokens)} tokens vs {len(bboxes)} bboxes")
            continue
        
        # Initialize all tokens as 'O' (Outside)
        ner_tags = [0] * len(tokens)  # 0 = 'O'
        
        # Create a mapping from bbox coordinates to token indices
        bbox_to_token = {}
        for i, bbox in enumerate(bboxes):
            bbox_key = (bbox['x'], bbox['y'], bbox['width'], bbox['height'])
            bbox_to_token[bbox_key] = i
        
        # Process labels and assign NER tags
        for label_info in sample['label']:
            if 'labels' not in label_info:
                continue
                
            entity_type = label_info['labels'][0]  # Take first label
            
            # Map entity types to our format
            entity_mapping = {
                'Supplier Name': 'SUPPLIER_NAME',
                'Receipt Date': 'RECEIPT_DATE', 
                'Receipt ID': 'RECEIPT_ID',
                'Item': 'ITEM',
                'Item Code': 'ITEM_CODE',
                'Item Quantity': 'ITEM_QUANTITY',
                'Unit Price': 'UNIT_PRICE',
                'Item Amount': 'ITEM_AMOUNT',
                'Tax Amount': 'TAX_AMOUNT',
                'Total Amount': 'TOTAL_AMOUNT',
                'VAT/Tax Code': 'VAT_TAX_CODE',
                'Currency': 'CURRENCY'
            }
            
            if entity_type not in entity_mapping:
                continue
                
            mapped_entity = entity_mapping[entity_type]
            
            # Find overlapping tokens for this label region
            label_bbox = {
                'x': label_info['x'],
                'y': label_info['y'], 
                'width': label_info['width'],
                'height': label_info['height']
            }
            
            # Find tokens that overlap with this label region
            overlapping_tokens = []
            for i, token_bbox in enumerate(bboxes):
                if boxes_overlap(token_bbox, label_bbox):
                    overlapping_tokens.append(i)
            
            # Assign BIO tags
            if overlapping_tokens:
                # First token gets B- tag
                b_tag = f"B-{mapped_entity}"
                if b_tag in label_to_id:
                    ner_tags[overlapping_tokens[0]] = label_to_id[b_tag]
                
                # Remaining tokens get I- tags
                i_tag = f"I-{mapped_entity}"
                if i_tag in label_to_id:
                    for token_idx in overlapping_tokens[1:]:
                        ner_tags[token_idx] = label_to_id[i_tag]
        
        # Convert bboxes to normalized format [x0, y0, x1, y1] in 0-1000 scale
        normalized_bboxes = []
        for bbox in bboxes:
            original_width = bbox['original_width']
            original_height = bbox['original_height']
            
            # Convert percentage to absolute coordinates
            x0 = int((bbox['x'] / 100) * original_width)
            y0 = int((bbox['y'] / 100) * original_height) 
            x1 = int(((bbox['x'] + bbox['width']) / 100) * original_width)
            y1 = int(((bbox['y'] + bbox['height']) / 100) * original_height)
            
            # Normalize to 0-1000 scale
            x0_norm = int((x0 / original_width) * 1000)
            y0_norm = int((y0 / original_height) * 1000)
            x1_norm = int((x1 / original_width) * 1000)
            y1_norm = int((y1 / original_height) * 1000)
            
            normalized_bboxes.append([x0_norm, y0_norm, x1_norm, y1_norm])
        
        # Create the converted sample
        converted_sample = {
            "id": str(sample_id),
            "tokens": tokens,
            "bboxes": normalized_bboxes,
            "ner_tags": ner_tags,
            "image": {
                "path": image_path,
                "width": bboxes[0]['original_width'] if bboxes else 1512,
                "height": bboxes[0]['original_height'] if bboxes else 2016
            }
        }
        
        converted_samples.append(converted_sample)
    
    # Save the converted dataset
    output_file = os.path.join(output_dir, 'layoutlmv3_training_data.json')
    with open(output_file, 'w') as f:
        json.dump(converted_samples, f, indent=2)
    
    # Save label mapping
    label_file = os.path.join(output_dir, 'label_mapping.json')
    with open(label_file, 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label
        }, f, indent=2)
    
    print(f"Conversion completed!")
    print(f"- Converted samples: {len(converted_samples)}")
    print(f"- Training data saved to: {output_file}")
    print(f"- Label mapping saved to: {label_file}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    total_tokens = sum(len(sample['tokens']) for sample in converted_samples)
    total_entities = sum(sum(1 for tag in sample['ner_tags'] if tag != 0) for sample in converted_samples)
    
    print(f"- Total tokens: {total_tokens}")
    print(f"- Total entity tokens: {total_entities}")
    print(f"- Entity coverage: {total_entities/total_tokens*100:.1f}%")
    
    # Entity distribution
    entity_counts = defaultdict(int)
    for sample in converted_samples:
        for tag in sample['ner_tags']:
            if tag != 0:
                entity_counts[id_to_label[tag]] += 1
    
    print(f"\nEntity Distribution:")
    for entity, count in sorted(entity_counts.items()):
        print(f"- {entity}: {count}")
    
    return converted_samples

def boxes_overlap(box1, box2, threshold=0.1):
    """
    Check if two bounding boxes overlap significantly
    """
    # Convert to absolute coordinates for comparison
    box1_x1 = box1['x']
    box1_y1 = box1['y']
    box1_x2 = box1['x'] + box1['width']
    box1_y2 = box1['y'] + box1['height']
    
    box2_x1 = box2['x']
    box2_y1 = box2['y']
    box2_x2 = box2['x'] + box2['width']
    box2_y2 = box2['y'] + box2['height']
    
    # Check for overlap
    x_overlap = max(0, min(box1_x2, box2_x2) - max(box1_x1, box2_x1))
    y_overlap = max(0, min(box1_y2, box2_y2) - max(box1_y1, box2_y1))
    
    overlap_area = x_overlap * y_overlap
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    
    if box1_area == 0:
        return False
        
    overlap_ratio = overlap_area / box1_area
    return overlap_ratio >= threshold

if __name__ == "__main__":
    # Convert the dataset
    input_file = "combined_receipt_dataset.json"
    output_dir = "layoutlmv3_data"
    
    convert_to_layoutlmv3_format(input_file, output_dir)
