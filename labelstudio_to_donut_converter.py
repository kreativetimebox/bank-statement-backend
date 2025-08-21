#!/usr/bin/env python3
"""
Label Studio to Donut Format Converter

This script converts Label Studio annotations to Donut model format and splits
the dataset into train/validation/test sets with 0.8/0.1/0.1 ratio.

Usage:
    python labelstudio_to_donut_converter.py --annotation_file path/to/annotations.json --image_dir path/to/images --output_dir path/to/output

Author: AI Assistant
"""

import json
import os
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
from collections import defaultdict


class LabelStudioToDonutConverter:
    """Converts Label Studio annotations to Donut format"""
    
    def __init__(self, annotation_file: str, image_dir: str, output_dir: str):
        self.annotation_file = annotation_file
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        
        # Define the mapping from Label Studio labels to Donut format keys
        self.label_mapping = {
            "Supplier Name": "supplier_name",
            "Receipt No.": "receipt_no",
            "Receipt Date": "receipt_date", 
            "Currency": "currency",
            "Transaction ID": "transaction_id",
            "Payment Method": "payment_method",
            "Special Instructions": "special_instructions",
            "Item Code": "item_code",
            "Item Name": "item_name", 
            "Item Quantity": "item_quantity",
            "Item Unit Price": "item_unit_price",
            "Item Discount": "item_discount",
            "Item Amount": "item_amount",
            "Item VAT Code": "item_vat_code",
            "VAT Code": "vat_code",
            "VAT Percent": "vat_percent",
            "vat_percent": "vat_percent",  # Handle both formats
            "VAT Amount": "vat_amount",
            "Sub Total": "sub_total",
            "Net Amount": "net_amount", 
            "Total Discount": "total_discount",
            "Coupon Name": "coupon_name",
            "Coupon Amount": "coupon_amount",
            "Total Amount": "total_amount",
            "Change Amount": "change_amount",
            "Total Item Count": "total_item_count",
            "Sale Amount": "sale_amount",
            "Payment Date": "payment_date",
            "Payment Mode": "payment_mode",
            "Card Details": "card_details"
        }
        
        # Item-related fields that should be grouped together
        self.item_fields = {
            "item_code", "item_name", "item_quantity", 
            "item_unit_price", "item_discount", "item_amount", "item_vat_code"
        }
        
    def load_annotations(self) -> List[Dict]:
        """Load Label Studio annotations from JSON file"""
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_text_and_labels(self, result_list: List[Dict]) -> Dict[str, str]:
        """Extract text content and associated labels from annotation result"""
        # Group by ID first (for same-ID annotations)
        grouped_annotations = defaultdict(dict)
        
        # Also collect all transcriptions and labels separately for coordinate matching
        transcriptions = []
        labels = []
        
        for item in result_list:
            item_id = item.get('id')
            from_name = item.get('from_name')
            item_type = item.get('type')
            
            if item_id not in grouped_annotations:
                grouped_annotations[item_id] = {}
                
            if from_name == 'transcription' and item_type == 'textarea':
                text_content = item.get('value', {}).get('text', [])
                if text_content:
                    grouped_annotations[item_id]['text'] = ' '.join(text_content)
                    # Also store for coordinate matching
                    transcriptions.append({
                        'text': ' '.join(text_content),
                        'id': item_id,
                        'value': item.get('value', {})
                    })
                    
            elif from_name == 'label' and item_type == 'labels':
                labels_list = item.get('value', {}).get('labels', [])
                if labels_list:
                    grouped_annotations[item_id]['label'] = labels_list[0]  # Take first label
                    # Also store for coordinate matching
                    labels.append({
                        'label': labels_list[0],
                        'id': item_id,
                        'value': item.get('value', {})
                    })
        
        # Extract text-label pairs using ID matching first
        extracted_data = {}
        for item_id, data in grouped_annotations.items():
            if 'text' in data and 'label' in data:
                label = data['label']
                text = data['text'].strip()
                
                if label in self.label_mapping and text:
                    donut_key = self.label_mapping[label]
                    extracted_data[donut_key] = text
        
        # If we didn't get many matches, try coordinate-based matching
        if len(extracted_data) < 3:  # Threshold for "not many matches"
            print("  Trying coordinate-based matching...")
            
            def get_bbox_center(value_dict):
                """Get center coordinates of bounding box"""
                x = value_dict.get('x', 0)
                y = value_dict.get('y', 0) 
                width = value_dict.get('width', 0)
                height = value_dict.get('height', 0)
                return (x + width/2, y + height/2)
            
            # Match transcriptions and labels by proximity
            for transcription in transcriptions:
                trans_center = get_bbox_center(transcription['value'])
                
                # Find closest label
                closest_label = None
                min_distance = float('inf')
                
                for label_item in labels:
                    label_center = get_bbox_center(label_item['value'])
                    distance = ((trans_center[0] - label_center[0])**2 + 
                               (trans_center[1] - label_center[1])**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_label = label_item
                
                # If we found a close label (within reasonable distance)
                if closest_label and min_distance < 10:  # Adjust threshold as needed
                    label = closest_label['label']
                    text = transcription['text'].strip()
                    
                    if label in self.label_mapping and text:
                        donut_key = self.label_mapping[label]
                        if donut_key not in extracted_data:  # Don't overwrite existing matches
                            extracted_data[donut_key] = text
                    
        return extracted_data
    
    def group_items(self, extracted_data: Dict[str, str]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """Separate item fields from other fields and group items"""
        non_item_data = {}
        item_data = defaultdict(dict)
        
        for key, value in extracted_data.items():
            if key in self.item_fields:
                # For items, we need to handle multiple items
                # This is a simplified approach - in reality, you might need more sophisticated grouping
                if key == 'item_name':
                    item_index = len([k for k in item_data.keys() if 'item_name' in item_data[k]])
                    item_data[item_index]['item_name'] = value
                elif key == 'item_quantity':
                    # Find the most recent item without quantity or create new item
                    item_index = 0
                    for i in range(len(item_data)):
                        if 'item_quantity' not in item_data[i]:
                            item_index = i
                            break
                    else:
                        item_index = len(item_data)
                    item_data[item_index]['item_quantity'] = value
                elif key == 'item_unit_price':
                    # Similar logic for other item fields
                    item_index = 0
                    for i in range(len(item_data)):
                        if 'item_unit_price' not in item_data[i]:
                            item_index = i
                            break
                    else:
                        item_index = len(item_data)
                    item_data[item_index]['item_unit_price'] = value
                elif key == 'item_amount':
                    item_index = 0
                    for i in range(len(item_data)):
                        if 'item_amount' not in item_data[i]:
                            item_index = i
                            break
                    else:
                        item_index = len(item_data)
                    item_data[item_index]['item_amount'] = value
                # Add other item fields as needed
                else:
                    # For other item fields, just add to the first available item
                    if len(item_data) == 0:
                        item_data[0] = {}
                    item_data[0][key] = value
            else:
                non_item_data[key] = value
        
        # Convert defaultdict to list of dicts
        items_list = [item_data[i] for i in sorted(item_data.keys())]
        
        return non_item_data, items_list
    
    def create_donut_format(self, extracted_data: Dict[str, str]) -> Dict:
        """Convert extracted data to Donut format"""
        non_item_data, items_list = self.group_items(extracted_data)
        
        # Create the base donut structure
        donut_data = {
            "gt_parse": {
                "supplier_name": non_item_data.get("supplier_name", ""),
                "receipt_no": non_item_data.get("receipt_no", ""), 
                "receipt_date": non_item_data.get("receipt_date", ""),
                "currency": non_item_data.get("currency", ""),
                "transaction_id": non_item_data.get("transaction_id", ""),
                "payment_method": non_item_data.get("payment_method", ""),
                "special_instructions": non_item_data.get("special_instructions", ""),
                "vat_code": non_item_data.get("vat_code", ""),
                "vat_percent": non_item_data.get("vat_percent", ""),
                "vat_amount": non_item_data.get("vat_amount", ""),
                "sub_total": non_item_data.get("sub_total", ""),
                "net_amount": non_item_data.get("net_amount", ""),
                "total_discount": non_item_data.get("total_discount", ""),
                "coupon_name": non_item_data.get("coupon_name", ""),
                "coupon_amount": non_item_data.get("coupon_amount", ""),
                "total_amount": non_item_data.get("total_amount", ""),
                "change_amount": non_item_data.get("change_amount", ""),
                "total_item_count": non_item_data.get("total_item_count", ""),
                "sale_amount": non_item_data.get("sale_amount", ""),
                "payment_date": non_item_data.get("payment_date", ""),
                "payment_mode": non_item_data.get("payment_mode", ""),
                "card_details": non_item_data.get("card_details", ""),
                "items": items_list
            }
        }
        
        return donut_data
    
    def get_image_filename(self, image_path: str) -> str:
        """Extract filename from image path"""
        if image_path.startswith("http://"):
            # Handle URL format
            return os.path.basename(image_path)
        else:
            # Handle relative path format
            return os.path.basename(image_path)
    
    def convert_annotations(self) -> List[Tuple[str, Dict]]:
        """Convert all annotations to Donut format"""
        annotations = self.load_annotations()
        converted_data = []
        
        print(f"Processing {len(annotations)} annotations...")
        
        for i, annotation in enumerate(annotations):
            try:
                # Get image path
                image_path = annotation.get('data', {}).get('image', '')
                if not image_path:
                    print(f"Warning: No image path found for annotation {i}")
                    continue
                
                image_filename = self.get_image_filename(image_path)
                
                # Check if image file exists
                image_file_path = self.image_dir / image_filename
                if not image_file_path.exists():
                    print(f"Warning: Image file not found: {image_file_path}")
                    continue
                
                # Get annotation results
                annotation_list = annotation.get('annotations', [])
                if not annotation_list:
                    print(f"Warning: No annotations found for image {image_filename}")
                    continue
                
                # Process the first (and usually only) annotation
                result_list = annotation_list[0].get('result', [])
                if not result_list:
                    print(f"Warning: No results found for image {image_filename}")
                    continue
                
                # Extract text and labels
                extracted_data = self.extract_text_and_labels(result_list)
                
                # Convert to Donut format
                donut_data = self.create_donut_format(extracted_data)
                
                converted_data.append((image_filename, donut_data))
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} annotations...")
                    
            except Exception as e:
                print(f"Error processing annotation {i}: {str(e)}")
                continue
        
        print(f"Successfully converted {len(converted_data)} annotations")
        return converted_data
    
    def split_dataset(self, data: List[Tuple[str, Dict]], 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, 
                     test_ratio: float = 0.1) -> Tuple[List, List, List]:
        """Split dataset into train/validation/test sets"""
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle data
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        total_size = len(data_copy)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_data = data_copy[:train_size]
        val_data = data_copy[train_size:train_size + val_size]
        test_data = data_copy[train_size + val_size:]
        
        print(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_dataset(self, train_data: List, val_data: List, test_data: List):
        """Save the split datasets to output directory"""
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        datasets = [
            ('train', train_data),
            ('val', val_data), 
            ('test', test_data)
        ]
        
        for split_name, dataset in datasets:
            print(f"Saving {split_name} dataset...")
            
            for image_filename, donut_data in dataset:
                # Copy image file
                src_image_path = self.image_dir / image_filename
                dst_image_path = self.output_dir / split_name / 'images' / image_filename
                
                if src_image_path.exists():
                    shutil.copy2(src_image_path, dst_image_path)
                else:
                    print(f"Warning: Source image not found: {src_image_path}")
                    continue
                
                # Save JSON label file
                json_filename = os.path.splitext(image_filename)[0] + '.json'
                label_path = self.output_dir / split_name / 'labels' / json_filename
                
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(donut_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(dataset)} files to {split_name} set")
    
    def run(self):
        """Run the complete conversion process"""
        print("Starting Label Studio to Donut conversion...")
        print(f"Input annotation file: {self.annotation_file}")
        print(f"Input image directory: {self.image_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Convert annotations
        converted_data = self.convert_annotations()
        
        if not converted_data:
            print("No data to process. Exiting.")
            return
        
        # Split dataset
        train_data, val_data, test_data = self.split_dataset(converted_data)
        
        # Save dataset
        self.save_dataset(train_data, val_data, test_data)
        
        print("Conversion completed successfully!")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total processed: {len(converted_data)}")
        print(f"Train set: {len(train_data)} ({len(train_data)/len(converted_data)*100:.1f}%)")
        print(f"Validation set: {len(val_data)} ({len(val_data)/len(converted_data)*100:.1f}%)")
        print(f"Test set: {len(test_data)} ({len(test_data)/len(converted_data)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Convert Label Studio annotations to Donut format')
    parser.add_argument('--annotation_file', required=True, 
                       help='Path to Label Studio annotation JSON file')
    parser.add_argument('--image_dir', required=True,
                       help='Path to directory containing images')
    parser.add_argument('--output_dir', required=True,
                       help='Path to output directory for Donut dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for dataset splitting (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    random.seed(args.seed)
    
    # Validate input paths
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    # Create converter and run
    converter = LabelStudioToDonutConverter(
        annotation_file=args.annotation_file,
        image_dir=args.image_dir, 
        output_dir=args.output_dir
    )
    
    converter.run()


if __name__ == '__main__':
    main()
