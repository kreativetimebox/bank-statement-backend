#!/usr/bin/env python3
"""
VAT Percentage Summary Report

This script analyzes the updated JSON file to provide a detailed summary
of all extracted VAT percentages and their distribution.
"""

import json
from collections import Counter, defaultdict

def analyze_vat_percentages(json_file_path: str):
    """Analyze VAT percentages in the updated JSON file"""
    
    # Load the updated JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("VAT PERCENTAGE EXTRACTION SUMMARY REPORT")
    print("=" * 60)
    
    # Collect all VAT percentage data
    vat_percentages = []
    file_vat_data = defaultdict(list)
    
    for record in data:
        # Extract image filename
        image_url = record.get('data', {}).get('image', '')
        filename = image_url.split('/')[-1] if image_url else "unknown"
        
        annotations = record.get('annotations', [])
        for annotation in annotations:
            result = annotation.get('result', [])
            
            for item in result:
                # Look for vat_percent labels
                if (item.get('type') == 'labels' and 
                    'value' in item and 
                    'labels' in item['value'] and
                    'vat_percent' in item['value']['labels']):
                    
                    # Find corresponding transcription
                    percentage_value = "N/A"
                    for other_item in result:
                        if (other_item.get('type') == 'textarea' and 
                            other_item.get('from_name') == 'transcription' and
                            other_item.get('id') != item.get('id')):
                            
                            # Check if bounding boxes match
                            if bounding_boxes_match(item, other_item):
                                text_values = other_item.get('value', {}).get('text', [])
                                if text_values:
                                    percentage_value = text_values[0]
                                    break
                    
                    vat_percentages.append(percentage_value)
                    file_vat_data[filename].append(percentage_value)
    
    # Generate summary statistics
    print(f"\nTotal VAT Percentage Fields Found: {len(vat_percentages)}")
    print(f"Files with VAT Percentages: {len([f for f, p in file_vat_data.items() if p])}")
    
    # Count unique percentage values
    percentage_counts = Counter(vat_percentages)
    print(f"\nUnique VAT Percentage Values: {len(percentage_counts)}")
    
    print("\nVAT Percentage Distribution:")
    print("-" * 40)
    for percentage, count in sorted(percentage_counts.items(), 
                                  key=lambda x: (x[0] == 'N/A', 
                                               float(x[0].replace('%', '')) if x[0] != 'N/A' and x[0].replace('%', '').replace('.', '').isdigit() else 0)):
        print(f"  {percentage}: {count} occurrences")
    
    # Show files with their VAT percentages
    print(f"\nVAT Percentages by File:")
    print("-" * 40)
    for filename, percentages in sorted(file_vat_data.items()):
        if percentages:
            print(f"  {filename}: {', '.join(percentages)}")
    
    # Calculate statistics for numeric percentages
    numeric_percentages = []
    for p in vat_percentages:
        if p != 'N/A':
            try:
                numeric_percentages.append(float(p.replace('%', '')))
            except ValueError:
                continue
    
    if numeric_percentages:
        print(f"\nNumeric VAT Percentage Statistics:")
        print("-" * 40)
        print(f"  Minimum: {min(numeric_percentages):.1f}%")
        print(f"  Maximum: {max(numeric_percentages):.1f}%")
        print(f"  Average: {sum(numeric_percentages)/len(numeric_percentages):.1f}%")
        
        # Group by ranges
        ranges = {
            "0-25%": 0,
            "25-50%": 0,
            "50-75%": 0,
            "75-100%": 0
        }
        
        for p in numeric_percentages:
            if p <= 25:
                ranges["0-25%"] += 1
            elif p <= 50:
                ranges["25-50%"] += 1
            elif p <= 75:
                ranges["50-75%"] += 1
            else:
                ranges["75-100%"] += 1
        
        print(f"\nVAT Percentage Ranges:")
        print("-" * 40)
        for range_name, count in ranges.items():
            print(f"  {range_name}: {count} occurrences")

def bounding_boxes_match(item1, item2):
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

if __name__ == "__main__":
    # Analyze the updated JSON file
    analyze_vat_percentages("exported_json_58_img_updated.json")
