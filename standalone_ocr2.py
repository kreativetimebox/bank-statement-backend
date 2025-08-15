#!/usr/bin/env python3
"""
Enhanced Receipt Parser using docTR.
Extracts and formats receipt data including vendor, items, prices, taxes, and totals.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

@dataclass
class Entity:
    type: str
    text: str
    value: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}

class ReceiptParser:
    def __init__(self):
        self.patterns = {
            'vendor': [
                r'^([A-Z][A-Za-z0-9&\s.,-]+(?:\s+[A-Z][A-Za-z0-9&\s.,-]+)*)\s*$',
                r'^[A-Z0-9\s&.,-]+(?:\s+[A-Z0-9\s&.,-]+)*$',
            ],
            'date': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            ],
            'tax_code': [
                r'VAT\s*[Nn]o[.:]?\s*([A-Z0-9]+)',
                r'GST\s*[Nn]o[.:]?\s*([A-Z0-9]+)',
                r'TAX\s*[Nn]o[.:]?\s*([A-Z0-9]+)',
            ],
            'total': [
                r'(?:total|balance\s+to\s+pay|amount\s+due)[^\d]*([£$€]?\s*\d+[.,]\d{2})',
                r'([£$€]?\s*\d+[.,]\d{2})\s*(?:total|balance|amount)',
            ],
            'tax_amount': [
                r'(?:tax|vat|gst)[^\d]*([£$€]?\s*\d+[.,]\d{2})',
                r'([£$€]?\s*\d+[.,]\d{2})\s*(?:tax|vat|gst)',
            ],
            'item': [
                # Format: ITEM PRICE
                r'^\s*([A-Z][A-Za-z0-9\s&.-]+?)\s+([£$€]?\s*\d+[.,]\d{2})\s*$',
                # Format: QTY ITEM PRICE
                r'^\s*(\d+)\s+([A-Za-z0-9\s&.-]+?)\s+([£$€]?\s*\d+[.,]\d{2})\s*$',
            ]
        }
        self.currency_symbols = ['£', '$', '€']
        self.seen_values = set()

    def clean_amount(self, amount: str) -> str:
        """Clean and format currency amount."""
        if not amount:
            return ""
        # Remove any non-digit except decimal point/comma
        cleaned = re.sub(r'[^\d,.]', '', amount)
        # Replace comma with dot if it's used as decimal separator
        if ',' in cleaned and '.' in cleaned:
            if cleaned.find(',') < cleaned.find('.'):
                cleaned = cleaned.replace(',', '')
            else:
                cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            cleaned = cleaned.replace(',', '.')
        return cleaned

    def extract_entity(self, field: str, text: str, confidence: float = 0.95) -> List[Entity]:
        """Extract entities using predefined patterns."""
        entities = []
        for pattern in self.patterns.get(field, []):
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                groups = match.groups()
                if not groups:
                    continue
                    
                value = groups[-1]  # Last group is typically the value we want
                if not value:
                    continue

                # Clean up the value
                if field in ['total', 'tax_amount', 'item']:
                    value = self.clean_amount(value)
                
                # Skip duplicates
                if value in self.seen_values:
                    continue
                    
                self.seen_values.add(value)
                
                entities.append(Entity(
                    type=field.upper(),
                    text=match.group(0).strip(),
                    value=value,
                    confidence=confidence,
                    metadata={'pattern': pattern}
                ))
                
        return entities

    def extract_items(self, text: str) -> List[Entity]:
        """Extract line items from receipt."""
        items = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # First try to find items with prices
        for i in range(len(lines)):
            line = lines[i]
            
            # Skip lines that are too short or look like headers/totals
            if len(line) < 3 or any(term in line.lower() for term in ['total', 'subtotal', 'tax', 'balance', 'amount']):
                continue
                
            # Try to match item patterns
            for pattern in self.patterns['item']:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:  # ITEM PRICE
                        item_name, price = groups
                        quantity = '1'
                    else:  # QTY ITEM PRICE
                        quantity, item_name, price = groups
                    
                    # Clean up values
                    item_name = item_name.strip()
                    quantity = quantity.strip()
                    price = self.clean_amount(price)
                    
                    # Calculate total if not provided
                    try:
                        total = str(float(price) * int(quantity))
                    except:
                        total = price
                    
                    # Add item and its components
                    items.extend([
                        Entity('ITEM', item_name, item_name, 0.90),
                        Entity('QUANTITY', quantity, quantity, 0.95),
                        Entity('PRICE', price, price, 0.95),
                        Entity('ITEM_TOTAL', total, total, 0.95)
                    ])
                    break
        
        return items

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse text and extract all entities into a structured format."""
        self.seen_values = set()  # Reset seen values for each parse
        
        # Initialize result structure
        result = {
            'vendor': '',
            'date': '',
            'tax_code': '',
            'items': [],
            'subtotal': '',
            'tax_amount': '',
            'total': '',
            'currency': '£',  # Default to GBP
            'raw_text': text
        }
        
        # Extract basic entities
        vendor_matches = self.extract_entity('vendor', text, 0.90)
        if vendor_matches:
            # Prefer longer vendor names (more likely to be correct)
            result['vendor'] = max(vendor_matches, key=lambda x: len(x.text)).text
            result['vendor_confidence'] = max(vendor_matches, key=lambda x: len(x.text)).confidence
        
        date_matches = self.extract_entity('date', text, 0.95)
        if date_matches:
            result['date'] = date_matches[0].value
            result['date_confidence'] = date_matches[0].confidence
        
        tax_code_matches = self.extract_entity('tax_code', text, 0.90)
        if tax_code_matches:
            result['tax_code'] = tax_code_matches[0].value
            result['tax_code_confidence'] = tax_code_matches[0].confidence
        
        # Extract items
        items = self.extract_items(text)
        item_entities = {}
        
        # Group items by their position
        for item in items:
            if item.type not in item_entities:
                item_entities[item.type] = []
            item_entities[item.type].append(item)
        
        # Pair up items with their quantities and prices
        item_count = len(item_entities.get('ITEM', []))
        for i in range(item_count):
            item = {
                'name': item_entities.get('ITEM', [{}] * (i+1))[i].value,
                'quantity': item_entities.get('QUANTITY', [{}] * (i+1))[i].value if 'QUANTITY' in item_entities else '1',
                'price': item_entities.get('PRICE', [{}] * (i+1))[i].value,
                'total': item_entities.get('ITEM_TOTAL', [{}] * (i+1))[i].value,
                'confidence': item_entities.get('ITEM', [{}] * (i+1))[i].confidence
            }
            result['items'].append(item)
        
        # Extract totals
        total_matches = self.extract_entity('total', text, 0.95)
        if total_matches:
            result['total'] = total_matches[0].value
            result['total_confidence'] = total_matches[0].confidence
            # Try to determine currency from total
            for symbol in self.currency_symbols:
                if symbol in total_matches[0].text:
                    result['currency'] = symbol
                    break
        
        # Extract tax amount
        tax_matches = self.extract_entity('tax_amount', text, 0.90)
        if tax_matches:
            result['tax_amount'] = tax_matches[0].value
            result['tax_amount_confidence'] = tax_matches[0].confidence
        
        return result

class DocTR_OCR:
    def __init__(self, det_arch: str = 'db_resnet50', reco_arch: str = 'crnn_vgg16_bn'):
        """Initialize the docTR OCR model."""
        print(f"Initializing docTR OCR with {det_arch} detector and {reco_arch} recognizer...")
        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
        print("OCR model loaded successfully")

    def process_image(self, image_path: str) -> Tuple[str, List[str]]:
        """Process an image and return extracted text and lines."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            doc = DocumentFile.from_images(image_path)
            result = self.model(doc)
            
            full_text = ""
            lines = []
            
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        lines.append(line_text)
                        if full_text:
                            full_text += "\n"
                        full_text += line_text
            
            print(f"Processed {len(lines)} lines of text")
            return full_text, lines
            
        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            raise

def print_receipt(receipt: Dict[str, Any]):
    """Print receipt data in the requested table format."""
    if not receipt:
        print("No receipt data to display.")
        return
    
    def print_section(title, items, is_items=False):
        print(f"\n=== {title} ===")
        if is_items:
            print(f"{'Text':<40} | {'Quantity':<8} | {'Price':<10} | {'Confidence':<10}")
            print("-" * 80)
            for item in items:
                name = item.get('name', 'N/A')[:38]
                qty = item.get('quantity', '1')
                price = item.get('price', '0.00')
                conf = f"{item.get('confidence', 0) * 100:.1f}%" if item.get('confidence') else 'N/A'
                print(f"{name:<40} | {str(qty):<8} | {receipt.get('currency', '£')}{price:<9} | {conf}")
        else:
            print(f"{'Text':<40} | {'Value':<18} | {'Confidence'}")
            print("-" * 80)
            for item in items:
                text = str(item.get('text', ''))[:38]
                value = str(item.get('value', ''))
                conf = f"{item.get('confidence', 0) * 100:.1f}%" if item.get('confidence') else 'N/A'
                print(f"{text:<40} | {value:<18} | {conf}")
    
    # Vendor section
    vendor_data = [
        {'text': receipt.get('vendor', ''), 'value': '', 'confidence': receipt.get('vendor_confidence', 0.95)}
    ]
    print_section("STORE", vendor_data)
    
    # Date section
    if 'date' in receipt:
        date_data = [
            {'text': receipt['date'], 'value': '', 'confidence': receipt.get('date_confidence', 0.95)}
        ]
        print_section("DATE", date_data)
    
    # Items section
    if receipt.get('items'):
        print_section("ITEM", receipt['items'], is_items=True)
    
    # Subtotal section
    if 'subtotal' in receipt:
        subtotal_data = [
            {'text': 'Subtotal', 'value': f"{receipt.get('currency', '£')}{receipt['subtotal']}", 
             'confidence': receipt.get('subtotal_confidence', 0.95)}
        ]
        print_section("SUBTOTAL", subtotal_data)
    
    # Tax section
    tax_data = []
    if 'tax_code' in receipt:
        tax_data.append({
            'text': 'VAT', 
            'value': receipt['tax_code'], 
            'confidence': receipt.get('tax_code_confidence', 0.9)
        })
    if 'tax_amount' in receipt:
        tax_data.append({
            'text': 'VAT Amount', 
            'value': f"{receipt.get('currency', '£')}{receipt['tax_amount']}", 
            'confidence': receipt.get('tax_amount_confidence', 0.9)
        })
    if tax_data:
        print_section("TAX", tax_data)
    
    # Total section
    if 'total' in receipt:
        total_data = [
            {'text': 'Total', 'value': f"{receipt.get('currency', '£')}{receipt['total']}", 
             'confidence': receipt.get('total_confidence', 0.95)}
        ]
        print_section("TOTAL", total_data)

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Extract structured data from receipts')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--det', default='db_resnet50', 
                       help='Detection architecture (default: db_resnet50)')
    parser.add_argument('--reco', default='crnn_vgg16_bn',
                       help='Recognition architecture (default: crnn_vgg16_bn)')
    parser.add_argument('--output-json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    try:
        # Initialize OCR and Parser
        ocr = DocTR_OCR(det_arch=args.det, reco_arch=args.reco)
        parser = ReceiptParser()
        
        # Process image
        print(f"\nProcessing image: {os.path.basename(args.image_path)}")
        full_text, lines = ocr.process_image(args.image_path)
        
        # Parse text and extract receipt data
        print("Extracting receipt information...")
        receipt_data = parser.parse(full_text)
        
        # Add raw text to output
        receipt_data['raw_text'] = full_text
        
        # Print receipt in table format
        print_receipt(receipt_data)
        
        # Optionally output as JSON
        if args.output_json:
            print("\n" + "=" * 50)
            print("JSON OUTPUT:")
            print("=" * 50)
            print(json.dumps(receipt_data, indent=2))
        
        return receipt_data
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
