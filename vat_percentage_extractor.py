#!/usr/bin/env python3
"""
VAT Percentage Extractor for Label Studio Export

This script processes a Label Studio export JSON file to:
1. Identify VAT percentage values embedded in VAT/Tax fields
2. Create separate 'vat_percent' label fields with proper bounding boxes
3. Preserve all original annotation metadata
4. Generate an updated JSON file
"""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Tuple
import copy

class VATPercentageExtractor:
    def __init__(self, input_json_path: str, output_json_path: str = None):
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path or input_json_path.replace('.json', '_updated.json')
        self.data = None
        self.vat_percentage_patterns = [
            r'(\d+\.?\d*%)',  # Matches percentages like 55.7%, 48.8%, 77.0%
            r'(\d+\.?\d*)\s*%',  # Matches percentages like 55.7 %, 48.8 %
            r'(\d+\.?\d*)\s*percent',  # Matches percentages like 55.7 percent
            r'(\d+\.?\d*)\s*per\s*cent',  # Matches percentages like 55.7 per cent
        ]
        
    def load_data(self) -> bool:
        """Load the JSON data from file"""
        try:
            with open(self.input_json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Successfully loaded {len(self.data)} annotation records")
            return True
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return False
    
    def is_percentage_value(self, text: str) -> bool:
        """Check if text contains a percentage value"""
        if not text:
            return False
        
        text = text.strip()
        for pattern in self.vat_percentage_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def extract_percentage_value(self, text: str) -> str:
        """Extract percentage value from text"""
        if not text:
            return ""
        
        text = text.strip()
        for pattern in self.vat_percentage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                percentage = match.group(1)
                # Ensure it ends with %
                if not percentage.endswith('%'):
                    percentage += '%'
                return percentage
        
        return ""
    
    def create_vat_percent_annotation(self, original_item: Dict[str, Any], percentage_value: str, 
                                    transcription_item: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new vat_percent annotation based on the original item"""
        # Deep copy the original item to preserve all metadata
        new_item = copy.deepcopy(original_item)
        
        # Generate new unique ID
        new_item['id'] = str(uuid.uuid4())
        
        # Update the labels to 'vat_percent'
        if 'value' in new_item and 'labels' in new_item['value']:
            new_item['value']['labels'] = ['vat_percent']
        
        # If we have a transcription item, use its bounding box and text
        if transcription_item and 'value' in transcription_item:
            new_item['value'].update(transcription_item['value'])
            # Update text value to just the percentage
            if 'text' in new_item['value']:
                new_item['value']['text'] = [percentage_value]
        
        return new_item
    
    def find_vat_percentage_transcriptions(self, result_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find all transcription items that contain VAT percentage values"""
        vat_percent_transcriptions = []
        
        for item in result_items:
            if (item.get('type') == 'textarea' and 
                item.get('from_name') == 'transcription'):
                
                text_values = item.get('value', {}).get('text', [])
                if text_values and self.is_percentage_value(text_values[0]):
                    vat_percent_transcriptions.append(item)
        
        return vat_percent_transcriptions
    
    def process_annotations(self) -> Tuple[int, int]:
        """Process all annotations to extract VAT percentages"""
        total_vat_fields = 0
        total_percentages_extracted = 0
        
        print("Processing annotations to extract VAT percentages...")
        
        for record_idx, record in enumerate(self.data):
            annotations = record.get('annotations', [])
            
            for annotation in annotations:
                result = annotation.get('result', [])
                new_result_items = []
                
                # First, find all VAT percentage transcriptions
                vat_percent_transcriptions = self.find_vat_percentage_transcriptions(result)
                
                # Process each item
                for item in result:
                    # Add the original item
                    new_result_items.append(item)
                    
                    # Check if this is a VAT/Tax related label
                    if (item.get('type') == 'labels' and 
                        'value' in item and 
                        'labels' in item['value']):
                        
                        labels = item['value']['labels']
                        is_vat_tax = any(label.lower() in ['vat/tax code', 'tax amount', 'vat'] 
                                       for label in labels)
                        
                        if is_vat_tax:
                            total_vat_fields += 1
                            
                            # Look for corresponding transcription/textarea with the actual value
                            text_value = ""
                            corresponding_transcription = None
                            
                            for other_item in result:
                                if (other_item.get('type') == 'textarea' and 
                                    other_item.get('from_name') == 'transcription'):
                                    # Check if this transcription corresponds to the VAT label
                                    # by comparing bounding box coordinates
                                    if self.bounding_boxes_match(item, other_item):
                                        text_values = other_item.get('value', {}).get('text', [])
                                        if text_values:
                                            text_value = text_values[0]
                                            corresponding_transcription = other_item
                                            break
                            
                            # Check if the text contains a percentage
                            if text_value and self.is_percentage_value(text_value):
                                percentage_value = self.extract_percentage_value(text_value)
                                if percentage_value:
                                    # Create new vat_percent annotation
                                    new_vat_percent = self.create_vat_percent_annotation(
                                        item, percentage_value, corresponding_transcription
                                    )
                                    new_result_items.append(new_vat_percent)
                                    total_percentages_extracted += 1
                                    
                                    print(f"  Extracted VAT percentage: {percentage_value} from '{text_value}'")
                
                # Also add standalone VAT percentage transcriptions that might not have labels
                for transcription_item in vat_percent_transcriptions:
                    # Check if this transcription already has a corresponding label
                    has_corresponding_label = False
                    for item in result:
                        if (item.get('type') == 'labels' and 
                            'value' in item and 
                            'labels' in item['value'] and
                            self.bounding_boxes_match(item, transcription_item)):
                            has_corresponding_label = True
                            break
                    
                    # If no corresponding label, create a new vat_percent label
                    if not has_corresponding_label:
                        text_values = transcription_item.get('value', {}).get('text', [])
                        if text_values:
                            percentage_value = self.extract_percentage_value(text_values[0])
                            if percentage_value:
                                # Create a new label item for this percentage
                                new_label_item = {
                                    "original_width": transcription_item.get('original_width', 1512),
                                    "original_height": transcription_item.get('original_height', 2016),
                                    "image_rotation": transcription_item.get('image_rotation', 0),
                                    "value": {
                                        "x": transcription_item['value']['x'],
                                        "y": transcription_item['value']['y'],
                                        "width": transcription_item['value']['width'],
                                        "height": transcription_item['value']['height'],
                                        "rotation": transcription_item['value'].get('rotation', 0),
                                        "labels": ["vat_percent"]
                                    },
                                    "id": str(uuid.uuid4()),
                                    "from_name": "label",
                                    "to_name": "image",
                                    "type": "labels",
                                    "origin": "manual"
                                }
                                
                                # Create a new transcription item for the percentage
                                new_transcription_item = copy.deepcopy(transcription_item)
                                new_transcription_item['id'] = str(uuid.uuid4())
                                new_transcription_item['value']['text'] = [percentage_value]
                                
                                new_result_items.append(new_label_item)
                                new_result_items.append(new_transcription_item)
                                total_percentages_extracted += 1
                                
                                print(f"  Created new VAT percentage label: {percentage_value} from '{text_values[0]}'")
                
                # Update the annotation result
                annotation['result'] = new_result_items
        
        return total_vat_fields, total_percentages_extracted
    
    def bounding_boxes_match(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Check if two items have matching bounding boxes"""
        if 'value' not in item1 or 'value' not in item2:
            return False
        
        val1 = item1['value']
        val2 = item2['value']
        
        # Check if coordinates are within a small tolerance
        tolerance = 2.0  # 2% tolerance for coordinate matching
        
        for coord in ['x', 'y', 'width', 'height']:
            if coord in val1 and coord in val2:
                diff = abs(val1[coord] - val2[coord])
                if diff > tolerance:
                    return False
        
        return True
    
    def save_updated_data(self) -> bool:
        """Save the updated data to the output file"""
        try:
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            print(f"Updated data saved to: {self.output_json_path}")
            return True
        except Exception as e:
            print(f"Error saving updated data: {e}")
            return False
    
    def run_extraction(self):
        """Run the complete VAT percentage extraction process"""
        if not self.load_data():
            return
        
        total_vat_fields, total_percentages = self.process_annotations()
        
        print(f"\nExtraction Summary:")
        print(f"  Total VAT/Tax fields processed: {total_vat_fields}")
        print(f"  VAT percentages extracted: {total_percentages}")
        
        if self.save_updated_data():
            print(f"\nSuccessfully created updated JSON file with VAT percentage fields!")
            print(f"  Original file: {self.input_json_path}")
            print(f"  Updated file: {self.output_json_path}")
        else:
            print("Failed to save updated data.")

def main():
    """Main function"""
    input_file = "exported_json_58_img.json"
    output_file = "exported_json_58_img_updated.json"
    
    # Initialize extractor
    extractor = VATPercentageExtractor(input_file, output_file)
    
    # Run the extraction
    extractor.run_extraction()

if __name__ == "__main__":
    main()
