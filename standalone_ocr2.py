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
            'store': [
                (r'^([A-Z][A-Za-z0-9\s&.-]+?)\s*(?:STORE|MARKET|SHOP|SUPERMARKET|TESCO|SAINSBURY|ASDA|MORRISONS|WAITROSE|M&S|MARKS[\s&]*SPENCER)', 0.95),
                (r'^([A-Z][A-Za-z0-9\s&.-]+?)(?:\s*\n\s*[A-Z0-9\s]+)?\s*\n\s*(?:\d+[\s\w,.-]+\n)?\s*[A-Z0-9\s-]+\s*\n\s*[A-Z0-9\s-]+\s*$', 0.90),
            ],
            'date': [
                (r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', 0.95),
                (r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})', 0.90),
            ],
            'item': [
                (r'^\s*([A-Z][A-Za-z0-9\s&.-]+?)\s+([£$€]?\s*\d+[.,]\d{2})\s*$', 0.90),  # Item with price
                (r'^\s*([A-Z][A-Za-z0-9\s&.-]+?)\s*[xX]?\s*(\d+)\s*[xX]?\s*([£$€]?\s*\d+[.,]\d{2})\s*$', 0.90),  # Item with quantity and price
            ],
            'total': [
                (r'(?:TOTAL|AMOUNT DUE|BALANCE|GRAND TOTAL)[^\d£$€]*(?:[£$€]?\s*\d+[.,]\d{2})', 0.95),
                (r'(?:[£$€]?\s*\d+[.,]\d{2})\s*$', 0.90),  # Last number in receipt
            ],
            'vat': [
                (r'VAT\s*(?:NO\.?|NUMBER|#)?\s*[:]?\s*([A-Z0-9\s-]+)', 0.95),
                (r'(GB\d{9})|(\d{11})', 0.90),  # UK VAT number format
            ],
            'tax_amount': [
                (r'VAT\s*(?:AMOUNT)?[^\d£$€]*([£$€]?\s*\d+[.,]\d{2})', 0.90),
                (r'TAX\s*(?:AMOUNT)?[^\d£$€]*([£$€]?\s*\d+[.,]\d{2})', 0.90),
            ]
        }
        self.currency_symbols = ['£', '$', '€']

    def extract_entity(self, entity_type: str, text: str, min_confidence: float = 0.8) -> List[Entity]:
        """Extract entities using predefined patterns."""
        entities = []
        patterns = self.patterns.get(entity_type, [])
        
        for pattern, confidence in patterns:
            if confidence < min_confidence:
                continue
                
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                if not match:
                    continue
                    
                try:
                    # Handle both group(1) and group(0) cases safely
                    if match.groups():
                        value = match.group(1) if match.group(1) is not None else match.group(0)
                    else:
                        value = match.group(0)
                        
                    if value is None:
                        continue
                        
                    value = str(value).strip()
                    if not value or len(value) < 2:  # Filter out very short matches
                        continue
                        
                    entities.append(Entity(
                        type=entity_type.upper(),
                        text=match.group(0).strip(),
                        value=value,
                        confidence=confidence
                    ))
                    
                except (AttributeError, IndexError) as e:
                    # Skip any problematic matches
                    continue
                    
        return entities

    def extract_items(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract items with quantity and price from receipt lines."""
        items = []
        current_item = None
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
                
            # Check for price patterns
            price_match = re.search(r'([£$€]?\s*\d+[.,]\d{2})\s*$', line)
            
            if price_match and current_item:
                # If we have a current item and found a price, add it
                current_item['price'] = price_match.group(1).replace(' ', '').replace(',', '.')
                current_item['total'] = current_item['price']  # Default total to price if no quantity
                items.append(current_item)
                current_item = None
                continue
                
            # Check for item patterns
            item_match = re.match(r'^\s*([A-Z][A-Za-z0-9\s&.-]+?)\s+([£$€]?\s*\d+[.,]\d{2})\s*$', line, re.IGNORECASE)
            if item_match:
                current_item = {
                    'name': item_match.group(1).strip(),
                    'quantity': '1',
                    'price': item_match.group(2).replace(' ', '').replace(',', '.'),
                    'total': item_match.group(2).replace(' ', '').replace(',', '.'),
                    'confidence': 0.90
                }
                items.append(current_item)
                current_item = None
                continue
                
            # If no price pattern but looks like an item, start a new item
            if re.match(r'^[A-Z][A-Za-z0-9\s&.-]+$', line) and not any(term in line.lower() for term in ['total', 'subtotal', 'tax', 'vat', 'change', 'card', 'cash', 'balance']):
                current_item = {
                    'name': line.strip(),
                    'quantity': '1',
                    'price': None,
                    'total': None,
                    'confidence': 0.85
                }
                
        return items

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse receipt text and extract structured data."""
        result = {
            'vendor': '',
            'date': '',
            'items': [],
            'total': '',
            'tax_code': '',
            'tax_amount': '',
            'currency': '£',  # Default to GBP
            'vendor_confidence': 0,
            'date_confidence': 0,
            'total_confidence': 0,
            'tax_code_confidence': 0,
            'tax_amount_confidence': 0
        }
        
        # Split text into lines for line-by-line processing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract vendor (store name) - usually at the top of the receipt
        for i, line in enumerate(lines[:5]):  # Check first 5 lines for store name
            if len(line) > 5 and line.isupper() and not any(c.isdigit() for c in line):
                result['vendor'] = line
                result['vendor_confidence'] = 0.95
                break
        
        # Extract date
        date_matches = self.extract_entity('date', text)
        if date_matches:
            result['date'] = date_matches[0].value
            result['date_confidence'] = date_matches[0].confidence
        
        # Extract items
        result['items'] = self.extract_items(lines)
        
        # Extract total amount
        total_matches = self.extract_entity('total', text)
        if total_matches:
            # Get the highest confidence total match
            best_total = max(total_matches, key=lambda x: x.confidence)
            result['total'] = best_total.value
            result['total_confidence'] = best_total.confidence
            
            # Try to determine currency from total
            for symbol in self.currency_symbols:
                if symbol in best_total.text:
                    result['currency'] = symbol
                    break
        
        # Extract tax information
        vat_matches = self.extract_entity('vat', text)
        if vat_matches:
            result['tax_code'] = vat_matches[0].value
            result['tax_code_confidence'] = vat_matches[0].confidence
        
        tax_matches = self.extract_entity('tax_amount', text)
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
