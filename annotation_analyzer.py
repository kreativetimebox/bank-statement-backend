#!/usr/bin/env python3
"""
Label Studio Export Annotation Analyzer

This script analyzes a Label Studio export JSON file to:
1. Count annotated image files
2. List common labels across all files
3. List file-specific labels and values
4. Identify VAT and tax related labels
5. Export results to CSV
"""

import json
import csv
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
import urllib.parse

class LabelStudioAnalyzer:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.data = None
        self.image_files = set()
        self.all_labels = set()
        self.file_labels = defaultdict(dict)
        self.vat_tax_labels = set()
        self.vat_tax_data = defaultdict(dict)
        
        # Common VAT and tax related keywords
        self.vat_tax_keywords = {
            'vat', 'tax', 'gst', 'hst', 'pst', 'qst', 'sales tax', 'excise',
            'duty', 'levy', 'tariff', 'customs', 'import tax', 'export tax',
            'withholding tax', 'income tax', 'corporate tax', 'property tax',
            'vat_percent'
        }
        
    def load_data(self):
        """Load the JSON data from file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Successfully loaded {len(self.data)} annotation records")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return False
        return True
    
    def extract_image_filename(self, image_url: str) -> str:
        """Extract filename from image URL"""
        if not image_url:
            return "unknown"
        
        # Handle both localhost URLs and direct filenames
        if image_url.startswith('http'):
            # Extract filename from URL
            parsed = urllib.parse.urlparse(image_url)
            filename = Path(parsed.path).name
        else:
            filename = Path(image_url).name
            
        return filename
    
    def is_image_file(self, filename: str) -> bool:
        """Check if file is an image based on extension"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return Path(filename).suffix.lower() in image_extensions
    
    def is_vat_tax_related(self, label: str) -> bool:
        """Check if a label is VAT or tax related"""
        label_lower = label.lower()
        return any(keyword in label_lower for keyword in self.vat_tax_keywords)
    
    def analyze_annotations(self):
        """Analyze all annotations in the data"""
        if not self.data:
            print("No data loaded. Please load the JSON file first.")
            return
        
        print("Analyzing annotations...")
        
        for record in self.data:
            # Extract image filename
            image_url = record.get('data', {}).get('image', '')
            filename = self.extract_image_filename(image_url)
            
            if not self.is_image_file(filename):
                continue
                
            self.image_files.add(filename)
            
            # Process annotations
            annotations = record.get('annotations', [])
            for annotation in annotations:
                result = annotation.get('result', [])
                
                for item in result:
                    # Check for label type annotations
                    if item.get('type') == 'labels' and 'labels' in item.get('value', {}):
                        labels = item['value']['labels']
                        text_values = item['value'].get('text', [])
                        
                        for label in labels:
                            self.all_labels.add(label)
                            
                            # Store label and value for this file
                            if filename not in self.file_labels:
                                self.file_labels[filename] = {}
                            
                            # Get the text value if available
                            text_value = text_values[0] if text_values else "N/A"
                            self.file_labels[filename][label] = text_value
                            
                            # Check if it's VAT/tax related
                            if self.is_vat_tax_related(label):
                                self.vat_tax_labels.add(label)
                                if filename not in self.vat_tax_data:
                                    self.vat_tax_data[filename] = {}
                                self.vat_tax_data[filename][label] = text_value
    
    def get_common_labels(self) -> Set[str]:
        """Get labels that appear in all files"""
        if not self.file_labels:
            return set()
        
        # Get labels that appear in all files
        all_file_labels = [set(labels.keys()) for labels in self.file_labels.values()]
        if not all_file_labels:
            return set()
        
        common_labels = set.intersection(*all_file_labels)
        return common_labels
    
    def get_file_specific_labels(self) -> Dict[str, Set[str]]:
        """Get labels that are specific to individual files"""
        if not self.file_labels:
            return {}
        
        # Count label occurrences across all files
        label_counts = Counter()
        for labels in self.file_labels.values():
            label_counts.update(labels.keys())
        
        # Labels that appear only once are file-specific
        file_specific = defaultdict(set)
        for filename, labels in self.file_labels.items():
            for label in labels:
                if label_counts[label] == 1:
                    file_specific[filename].add(label)
        
        return file_specific
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("LABEL STUDIO ANNOTATION ANALYSIS REPORT")
        print("="*60)
        
        # 1. Image files count
        print(f"\n1. ANNOTATED IMAGE FILES:")
        print(f"   Total count: {len(self.image_files)}")
        print(f"   Files: {', '.join(sorted(self.image_files))}")
        
        # 2. All labels
        print(f"\n2. ALL ANNOTATED LABELS:")
        print(f"   Total count: {len(self.all_labels)}")
        for label in sorted(self.all_labels):
            print(f"   - {label}")
        
        # 3. Common labels across all files
        common_labels = self.get_common_labels()
        print(f"\n3. COMMON LABELS ACROSS ALL FILES:")
        print(f"   Count: {len(common_labels)}")
        for label in sorted(common_labels):
            print(f"   - {label}")
        
        # 4. File-specific labels
        file_specific = self.get_file_specific_labels()
        print(f"\n4. FILE-SPECIFIC LABELS:")
        for filename, labels in sorted(file_specific.items()):
            if labels:
                print(f"   {filename}: {', '.join(sorted(labels))}")
        
        # 5. VAT and tax related labels
        print(f"\n5. VAT AND TAX RELATED LABELS:")
        print(f"   Count: {len(self.vat_tax_labels)}")
        for label in sorted(self.vat_tax_labels):
            print(f"   - {label}")
        
        # 6. File labels and values summary
        print(f"\n6. FILE LABELS AND VALUES SUMMARY:")
        for filename, labels in sorted(self.file_labels.items()):
            print(f"   {filename}:")
            for label, value in sorted(labels.items()):
                print(f"     {label}: {value}")
    
    def export_to_csv(self, output_file: str = "annotation_analysis.csv"):
        """Export analysis results to CSV file"""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Analysis Type', 'File Name', 'Label', 'Value', 'Is VAT/Tax Related'])
                
                # Write all file labels and values
                for filename, labels in sorted(self.file_labels.items()):
                    for label, value in sorted(labels.items()):
                        is_vat_tax = "Yes" if self.is_vat_tax_related(label) else "No"
                        writer.writerow(['All Labels', filename, label, value, is_vat_tax])
                
                # Write VAT/tax specific data
                for filename, labels in sorted(self.vat_tax_data.items()):
                    for label, value in sorted(labels.items()):
                        writer.writerow(['VAT/Tax Only', filename, label, value, "Yes"])
                
                # Write summary statistics
                writer.writerow([])
                writer.writerow(['SUMMARY STATISTICS'])
                writer.writerow(['Total Image Files', len(self.image_files), '', '', ''])
                writer.writerow(['Total Labels', len(self.all_labels), '', '', ''])
                writer.writerow(['Common Labels', len(self.get_common_labels()), '', '', ''])
                writer.writerow(['VAT/Tax Labels', len(self.vat_tax_labels), '', '', ''])
                
            print(f"\nResults exported to: {output_file}")
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    def run_analysis(self):
        """Run the complete analysis"""
        if not self.load_data():
            return
        
        self.analyze_annotations()
        self.generate_report()
        self.export_to_csv()

def main():
    """Main function"""
    # Initialize analyzer with your JSON file
    analyzer = LabelStudioAnalyzer("exported_json_58_img.json")
    
    # Run the complete analysis
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
