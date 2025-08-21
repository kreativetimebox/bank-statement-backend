#!/usr/bin/env python3
"""
Analyze Converted Dataset Statistics

This script analyzes the converted Donut dataset and provides statistics
about the extracted data fields and their distribution.

Usage:
    python3 analyze_converted_dataset.py --dataset_dir path/to/converted/dataset
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any


def analyze_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Analyze the converted dataset and return statistics"""
    
    stats = {
        'total_files': 0,
        'splits': {},
        'field_coverage': defaultdict(int),
        'field_examples': defaultdict(list),
        'item_statistics': {
            'files_with_items': 0,
            'total_items': 0,
            'avg_items_per_receipt': 0
        },
        'value_lengths': defaultdict(list),
        'common_values': defaultdict(Counter)
    }
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split / 'labels'
        if not split_dir.exists():
            continue
            
        split_stats = {
            'file_count': 0,
            'non_empty_files': 0,
            'field_coverage': defaultdict(int)
        }
        
        json_files = list(split_dir.glob('*.json'))
        split_stats['file_count'] = len(json_files)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                gt_parse = data.get('gt_parse', {})
                has_non_empty_field = False
                
                # Analyze receipt-level fields
                for field, value in gt_parse.items():
                    if field == 'items':
                        continue
                        
                    if value and value.strip():
                        has_non_empty_field = True
                        stats['field_coverage'][field] += 1
                        split_stats['field_coverage'][field] += 1
                        
                        # Store examples (first 3 unique values)
                        if len(stats['field_examples'][field]) < 3:
                            if value not in [ex['value'] for ex in stats['field_examples'][field]]:
                                stats['field_examples'][field].append({
                                    'value': value,
                                    'file': json_file.name
                                })
                        
                        # Track value lengths
                        stats['value_lengths'][field].append(len(value))
                        
                        # Track common values for certain fields
                        if field in ['currency', 'payment_method', 'payment_mode']:
                            stats['common_values'][field][value] += 1
                
                # Analyze items
                items = gt_parse.get('items', [])
                if items:
                    stats['item_statistics']['files_with_items'] += 1
                    stats['item_statistics']['total_items'] += len(items)
                    has_non_empty_field = True
                    
                    for item in items:
                        for item_field, item_value in item.items():
                            if item_value and str(item_value).strip():
                                field_key = f"items.{item_field}"
                                stats['field_coverage'][field_key] += 1
                                split_stats['field_coverage'][field_key] += 1
                
                if has_non_empty_field:
                    split_stats['non_empty_files'] += 1
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        stats['splits'][split] = split_stats
        stats['total_files'] += split_stats['file_count']
    
    # Calculate averages
    if stats['item_statistics']['files_with_items'] > 0:
        stats['item_statistics']['avg_items_per_receipt'] = (
            stats['item_statistics']['total_items'] / 
            stats['item_statistics']['files_with_items']
        )
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print formatted statistics"""
    
    print("=" * 60)
    print("CONVERTED DATASET ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Total files: {stats['total_files']}")
    
    # Split statistics
    print(f"\nSPLIT BREAKDOWN:")
    total_non_empty = 0
    for split, split_stats in stats['splits'].items():
        non_empty = split_stats['non_empty_files']
        total_files = split_stats['file_count']
        percentage = (non_empty / total_files * 100) if total_files > 0 else 0
        print(f"  {split.upper()}: {total_files} files, {non_empty} with data ({percentage:.1f}%)")
        total_non_empty += non_empty
    
    print(f"  TOTAL: {stats['total_files']} files, {total_non_empty} with data")
    
    # Field coverage
    print(f"\nFIELD COVERAGE (files with non-empty values):")
    receipt_fields = [field for field in stats['field_coverage'].keys() if not field.startswith('items.')]
    item_fields = [field for field in stats['field_coverage'].keys() if field.startswith('items.')]
    
    print(f"  Receipt-level fields:")
    for field in sorted(receipt_fields):
        count = stats['field_coverage'][field]
        percentage = (count / total_non_empty * 100) if total_non_empty > 0 else 0
        print(f"    {field}: {count} files ({percentage:.1f}%)")
    
    if item_fields:
        print(f"  Item-level fields:")
        for field in sorted(item_fields):
            count = stats['field_coverage'][field]
            # For item fields, percentage is out of total items, not files
            print(f"    {field}: {count} items")
    
    # Item statistics
    print(f"\nITEM STATISTICS:")
    item_stats = stats['item_statistics']
    print(f"  Files with items: {item_stats['files_with_items']}")
    print(f"  Total items across all files: {item_stats['total_items']}")
    if item_stats['avg_items_per_receipt'] > 0:
        print(f"  Average items per receipt: {item_stats['avg_items_per_receipt']:.1f}")
    
    # Field examples
    print(f"\nFIELD EXAMPLES:")
    for field in sorted(stats['field_examples'].keys()):
        if not field.startswith('items.'):  # Only show receipt-level examples
            examples = stats['field_examples'][field]
            print(f"  {field}:")
            for example in examples[:3]:
                print(f"    '{example['value']}' (from {example['file']})")
    
    # Common values for specific fields
    print(f"\nCOMMON VALUES:")
    for field in ['currency', 'payment_method', 'payment_mode']:
        if field in stats['common_values'] and stats['common_values'][field]:
            print(f"  {field}:")
            for value, count in stats['common_values'][field].most_common(5):
                print(f"    '{value}': {count} files")
    
    # Value length statistics
    print(f"\nVALUE LENGTH STATISTICS:")
    for field in ['supplier_name', 'total_amount', 'receipt_date']:
        if field in stats['value_lengths']:
            lengths = stats['value_lengths'][field]
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
            print(f"  {field}: avg={avg_len:.1f}, min={min_len}, max={max_len} characters")


def main():
    parser = argparse.ArgumentParser(description='Analyze converted Donut dataset')
    parser.add_argument('--dataset_dir', required=True,
                       help='Path to converted dataset directory')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Analyzing dataset: {dataset_dir}")
    stats = analyze_dataset(dataset_dir)
    print_statistics(stats)


if __name__ == '__main__':
    main()
