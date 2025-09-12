import json
from pathlib import Path

def convert_labelstudio_to_donut_format():
    """
    Convert Label Studio annotations to Donut training format
    """
    
    print("🔄 Converting Label Studio annotations to Donut format")
    print("=" * 55)
    
    # Process both train and validation sets
    for split in ['train', 'validation']:
        print(f"\n📁 Processing {split} set...")
        
        split_dir = Path(f"dataset/{split}")
        annotations_file = split_dir / "annotations.json"
        output_file = split_dir / "metadata.jsonl"
        
        if not annotations_file.exists():
            print(f"❌ {annotations_file} not found!")
            continue
            
        # Load Label Studio annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            labelstudio_data = json.load(f)
        
        print(f"📊 Loaded {len(labelstudio_data)} annotation records")
        
        # Group annotations by image
        image_annotations = {}
        for record in labelstudio_data:
            if 'data' in record and 'image' in record['data']:
                image_path = record['data']['image']
                image_name = image_path.split('/')[-1]
                
                if image_name not in image_annotations:
                    image_annotations[image_name] = {
                        'image_path': image_path,
                        'annotations': []
                    }
                
                # Extract annotations from the record
                if 'annotations' in record and len(record['annotations']) > 0:
                    annotation = record['annotations'][0]  # Take first annotation
                    if 'result' in annotation:
                        image_annotations[image_name]['annotations'].extend(annotation['result'])
        
        print(f"📷 Processing {len(image_annotations)} unique images")
        
        # Convert to Donut format and save as JSONL
        donut_records = []
        for image_name, data in image_annotations.items():
            # Extract text fields and their labels from annotations
            ground_truth = {}
            
            for annotation in data['annotations']:
                if annotation.get('type') == 'textarea' and 'value' in annotation:
                    # Get the text content
                    text_content = annotation['value'].get('text', [])
                    if text_content and len(text_content) > 0:
                        text = text_content[0]
                        
                        # Find corresponding label
                        annotation_id = annotation.get('id')
                        label = None
                        
                        # Look for matching label annotation
                        for label_ann in data['annotations']:
                            if (label_ann.get('id') == annotation_id and 
                                label_ann.get('type') == 'labels' and 
                                'value' in label_ann and 
                                'labels' in label_ann['value']):
                                labels = label_ann['value']['labels']
                                if labels:
                                    label = labels[0]
                                    break
                        
                        if label and text:
                            # Normalize label names for Donut
                            label_normalized = label.lower().replace(' ', '_').replace('/', '_')
                            if label_normalized not in ground_truth:
                                ground_truth[label_normalized] = []
                            ground_truth[label_normalized].append(text)
            
            # Create Donut record
            donut_record = {
                "file_name": image_name,
                "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
            }
            
            donut_records.append(donut_record)
        
        # Save as JSONL (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in donut_records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"💾 Saved {len(donut_records)} records to {output_file}")
        
        # Show sample of converted data
        if donut_records:
            print(f"📋 Sample record:")
            sample = donut_records[0]
            print(f"   File: {sample['file_name']}")
            print(f"   Ground truth: {sample['ground_truth'][:100]}...")
    
    print(f"\n✅ Conversion completed!")
    return True

if __name__ == "__main__":
    convert_labelstudio_to_donut_format()
