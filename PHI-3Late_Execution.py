import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from datetime import datetime
import pandas as pd
import re
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device logic: OCR on CPU, Phi-3 on GPU if available
ocr_device = "cpu"
lm_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_id = "microsoft/phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set a distinct pad_token to avoid eos_token conflict
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

class EnhancedPDFExtractor:
    def __init__(self):
        self.pytesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?@#$%^&*()_+-=[]{}|\\/<> '
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing for better OCR accuracy"""
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        processed_image = Image.fromarray(cleaned)
        
        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(processed_image)
        processed_image = enhancer.enhance(1.2)
        
        return processed_image
    
    def extract_text_with_layout(self, file_path: str) -> Dict[str, Any]:
        """Extract text while preserving layout and structure"""
        ext = os.path.splitext(file_path)[1].lower()
        
        extracted_data = {
            'raw_text': '',
            'structured_text': '',
            'tables': [],
            'text_blocks': [],
            'metadata': {}
        }
        
        if ext in ['.png', '.jpg', '.jpeg']:
            logger.info("🖼️ Processing image file...")
            return self._extract_from_image(file_path, extracted_data)
        
        elif ext == '.pdf':
            return self._extract_from_pdf(file_path, extracted_data)
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _extract_from_image(self, file_path: str, extracted_data: Dict) -> Dict:
        """Extract text from image files with enhanced preprocessing"""
        image = Image.open(file_path).convert("RGB")
        
        # Original OCR
        original_text = pytesseract.image_to_string(image, config=self.pytesseract_config)
        
        # Enhanced OCR with preprocessing
        processed_image = self.preprocess_image(image)
        enhanced_text = pytesseract.image_to_string(processed_image, config=self.pytesseract_config)
        
        # Get detailed OCR data for layout
        ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config=self.pytesseract_config)
        
        # Combine results
        extracted_data['raw_text'] = original_text if len(original_text) > len(enhanced_text) else enhanced_text
        extracted_data['structured_text'] = self._structure_text_from_ocr_data(ocr_data)
        extracted_data['text_blocks'] = self._extract_text_blocks_from_ocr_data(ocr_data)
        
        return extracted_data
    
    def _extract_from_pdf(self, file_path: str, extracted_data: Dict) -> Dict:
        """Comprehensive PDF extraction with multiple methods"""
        # Method 1: Try digital text extraction with layout
        try:
            logger.info("📄 Attempting digital text extraction...")
            digital_text, tables, text_blocks = self._extract_digital_pdf_content(file_path)
            
            if digital_text.strip() and len(digital_text.split()) > 10:  # Meaningful content threshold
                logger.info("✅ Digital extraction successful")
                extracted_data['raw_text'] = digital_text
                extracted_data['tables'] = tables
                extracted_data['text_blocks'] = text_blocks
                extracted_data['structured_text'] = self._structure_digital_text(digital_text)
                return extracted_data
        except Exception as e:
            logger.warning(f"❌ Digital extraction failed: {str(e)}")
        
        # Method 2: OCR with high-resolution conversion
        logger.info("🔁 Using enhanced OCR extraction...")
        return self._extract_pdf_via_ocr(file_path, extracted_data)
    
    def _extract_digital_pdf_content(self, file_path: str) -> Tuple[str, List, List]:
        """Extract content from PDF while preserving structure"""
        doc = fitz.open(file_path)
        full_text = ""
        all_tables = []
        all_text_blocks = []
        
        for page_num, page in enumerate(doc):
            # Extract text with layout
            text_dict = page.get_text("dict")
            page_text = page.get_text()
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + page_text
            
            # Extract tables
            tables = page.find_tables()
            for table in tables:
                try:
                    table_data = table.extract()
                    if table_data:
                        all_tables.append({
                            'page': page_num + 1,
                            'data': table_data,
                            'bbox': table.bbox
                        })
                except:
                    pass
            
            # Extract text blocks with positioning
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5:  # Text block
                    all_text_blocks.append({
                        'page': page_num + 1,
                        'text': block[4],
                        'bbox': block[:4]
                    })
        
        doc.close()
        return full_text, all_tables, all_text_blocks
    
    def _extract_pdf_via_ocr(self, file_path: str, extracted_data: Dict) -> Dict:
        """OCR extraction with high DPI and preprocessing"""
        # Convert PDF to high-resolution images
        images = convert_from_path(file_path, dpi=300, first_page=None, last_page=None)
        
        all_text = []
        all_structured_text = []
        all_text_blocks = []
        
        for i, image in enumerate(images):
            logger.info(f"🔍 Processing page {i+1}/{len(images)}")
            
            # Original OCR
            page_text = pytesseract.image_to_string(image.convert("RGB"), config=self.pytesseract_config)
            
            # Enhanced OCR
            processed_image = self.preprocess_image(image.convert("RGB"))
            enhanced_page_text = pytesseract.image_to_string(processed_image, config=self.pytesseract_config)
            
            # Choose better result
            final_page_text = enhanced_page_text if len(enhanced_page_text) > len(page_text) else page_text
            
            # Get structured data
            ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config=self.pytesseract_config)
            
            all_text.append(f"\n--- PAGE {i+1} ---\n" + final_page_text)
            all_structured_text.append(self._structure_text_from_ocr_data(ocr_data))
            all_text_blocks.extend(self._extract_text_blocks_from_ocr_data(ocr_data, page_num=i+1))
        
        extracted_data['raw_text'] = '\n'.join(all_text)
        extracted_data['structured_text'] = '\n'.join(all_structured_text)
        extracted_data['text_blocks'] = all_text_blocks
        
        return extracted_data
    
    def _structure_text_from_ocr_data(self, ocr_data: Dict) -> str:
        """Structure OCR data into readable format preserving layout"""
        structured_lines = []
        current_line = []
        current_top = None
        tolerance = 10  # Pixel tolerance for same line
        
        # Group words by lines based on vertical position
        for i, word in enumerate(ocr_data['text']):
            if int(ocr_data['conf'][i]) > 30 and word.strip():  # Filter low confidence
                top = ocr_data['top'][i]
                left = ocr_data['left'][i]
                
                if current_top is None or abs(top - current_top) <= tolerance:
                    current_line.append((left, word))
                    current_top = top
                else:
                    # Start new line
                    if current_line:
                        current_line.sort(key=lambda x: x[0])  # Sort by horizontal position
                        structured_lines.append(' '.join([word for _, word in current_line]))
                    current_line = [(left, word)]
                    current_top = top
        
        # Add last line
        if current_line:
            current_line.sort(key=lambda x: x[0])
            structured_lines.append(' '.join([word for _, word in current_line]))
        
        return '\n'.join(structured_lines)
    
    def _extract_text_blocks_from_ocr_data(self, ocr_data: Dict, page_num: int = 1) -> List[Dict]:
        """Extract text blocks with position information"""
        blocks = []
        for i, word in enumerate(ocr_data['text']):
            if int(ocr_data['conf'][i]) > 30 and word.strip():
                blocks.append({
                    'page': page_num,
                    'text': word,
                    'confidence': int(ocr_data['conf'][i]),
                    'bbox': [
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    ]
                })
        return blocks
    
    def _structure_digital_text(self, text: str) -> str:
        """Clean and structure digitally extracted text"""
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = ' '.join(line.split())  # Normalize whitespace
            if cleaned_line:  # Skip empty lines
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)

def extract_structured_data_with_enhanced_phi3(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced Phi-3 extraction with comprehensive prompting"""
    
    # Combine all available text sources for better extraction
    combined_text = extracted_data.get('raw_text', '')
    structured_text = extracted_data.get('structured_text', '')
    tables = extracted_data.get('tables', [])
    
    # Add table information to context
    table_context = ""
    if tables:
        table_context = "\n\nTABLE DATA FOUND:\n"
        for i, table in enumerate(tables):
            table_context += f"Table {i+1}:\n"
            if isinstance(table.get('data'), list):
                for row in table['data'][:5]:  # Limit to first 5 rows
                    table_context += f"  {row}\n"
    
    # Use structured text if available and longer
    analysis_text = structured_text if len(structured_text) > len(combined_text) else combined_text
    analysis_text += table_context
    
    prompt = f"""
You are an expert invoice and receipt extraction system. Analyze the following document text and extract ALL available information in a structured JSON format. Pay special attention to:

1. INVOICE DETAILS: number, date, due date, reference numbers
2. VENDOR/SUPPLIER: complete name, address, contact details, tax IDs
3. CUSTOMER/BUYER: complete name, address, contact details  
4. ITEMIZED BREAKDOWN: description, quantity, unit price, line totals
5. FINANCIAL TOTALS: subtotal, taxes (VAT/GST/sales tax), discounts, final total
6. PAYMENT INFO: terms, methods, bank details, payment status
7. ADDITIONAL DATA: PO numbers, delivery info, notes, terms & conditions

For each extracted field, include a confidence score (0.0-1.0). If information is unclear or missing, still include the field with confidence 0.0 and value as null or "".

Extract EVERYTHING visible in the document - don't leave out any details, numbers, addresses, or text that might be relevant.

IMPORTANT: Return ONLY valid JSON with no additional text or explanations.

JSON Structure:
{{
  "document_type": {{"value": "invoice/receipt/bill", "confidence": 0.0}},
  "invoice_number": {{"value": "", "confidence": 0.0}},
  "invoice_date": {{"value": "", "confidence": 0.0}},
  "due_date": {{"value": "", "confidence": 0.0}},
  "reference_numbers": {{"value": [], "confidence": 0.0}},
  
  "vendor": {{
    "name": {{"value": "", "confidence": 0.0}},
    "address": {{"value": "", "confidence": 0.0}},
    "phone": {{"value": "", "confidence": 0.0}},
    "email": {{"value": "", "confidence": 0.0}},
    "website": {{"value": "", "confidence": 0.0}},
    "tax_id": {{"value": "", "confidence": 0.0}},
    "registration_number": {{"value": "", "confidence": 0.0}}
  }},
  
  "customer": {{
    "name": {{"value": "", "confidence": 0.0}},
    "address": {{"value": "", "confidence": 0.0}},
    "phone": {{"value": "", "confidence": 0.0}},
    "email": {{"value": "", "confidence": 0.0}},
    "customer_id": {{"value": "", "confidence": 0.0}}
  }},
  
  "items": [
    {{
      "description": {{"value": "", "confidence": 0.0}},
      "quantity": {{"value": 0, "confidence": 0.0}},
      "unit_price": {{"value": 0.0, "confidence": 0.0}},
      "line_total": {{"value": 0.0, "confidence": 0.0}},
      "tax_rate": {{"value": 0.0, "confidence": 0.0}},
      "category": {{"value": "", "confidence": 0.0}}
    }}
  ],
  
  "totals": {{
    "subtotal": {{"value": 0.0, "confidence": 0.0}},
    "tax_amount": {{"value": 0.0, "confidence": 0.0}},
    "tax_rate": {{"value": 0.0, "confidence": 0.0}},
    "discount_amount": {{"value": 0.0, "confidence": 0.0}},
    "discount_percentage": {{"value": 0.0, "confidence": 0.0}},
    "total_amount": {{"value": 0.0, "confidence": 0.0}},
    "currency": {{"value": "", "confidence": 0.0}}
  }},
  
  "payment_info": {{
    "payment_terms": {{"value": "", "confidence": 0.0}},
    "payment_method": {{"value": "", "confidence": 0.0}},
    "bank_details": {{"value": "", "confidence": 0.0}},
    "payment_status": {{"value": "", "confidence": 0.0}}
  }},
  
  "additional_info": {{
    "purchase_order": {{"value": "", "confidence": 0.0}},
    "delivery_date": {{"value": "", "confidence": 0.0}},
    "delivery_address": {{"value": "", "confidence": 0.0}},
    "notes": {{"value": "", "confidence": 0.0}},
    "terms_conditions": {{"value": "", "confidence": 0.0}},
    "other_references": {{"value": [], "confidence": 0.0}}
  }}
}}

--- DOCUMENT TEXT TO ANALYZE ---
{analysis_text}
--- END OF DOCUMENT TEXT ---
"""

    # Tokenize with attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=32000, return_attention_mask=True).to(lm_device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and clean JSON output
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return _parse_json_from_response(decoded, outputs)

def _parse_json_from_response(response: str, outputs) -> Dict[str, Any]:
    """Parse JSON from model response with error handling"""
    try:
        # Find JSON boundaries
        json_start = response.find("{")
        if json_start == -1:
            raise ValueError("No JSON found in response")
        
        # Extract JSON string
        json_str = response[json_start:]
        
        # Find matching closing brace
        brace_count = 0
        json_end = -1
        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > 0:
            json_str = json_str[:json_end]
        
        # Clean up common formatting issues
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
        json_str = json_str.strip()
        
        # Parse JSON
        parsed_json = json.loads(json_str)
        logger.info("✅ Successfully parsed JSON from model output")
        
        return parsed_json
        
    except json.JSONDecodeError as e:
        logger.error(f"⚠️ JSON parsing failed: {str(e)}")
        
        # Fallback: try to extract key information with regex
        fallback_data = _extract_fallback_data(response)
        
        return {
            "error": "JSON parsing failed",
            "raw_response": response[:1000],
            "parsing_error": str(e),
            "fallback_extraction": fallback_data
        }

def _extract_fallback_data(text: str) -> Dict[str, Any]:
    """Fallback extraction using regex patterns"""
    patterns = {
        'invoice_number': r'(?:invoice|inv)(?:\s*#?|\s+no\.?|\s+number)[\s:]*([A-Z0-9-]+)',
        'total_amount': r'(?:total|amount due|final total)[\s:$]*([0-9,]+\.?[0-9]*)',
        'date': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        'vendor_name': r'^([A-Z][A-Za-z\s&.,]+?)(?:\n|$)',
    }
    
    extracted = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        extracted[key] = matches[0] if matches else None
    
    return extracted

# Enhanced main pipeline
def run_enhanced_invoice_pipeline(file_path: str, output_dir: str = "output") -> Tuple[str, Dict[str, Any]]:
    """Enhanced main pipeline with comprehensive extraction"""
    logger.info(f"🔍 Starting enhanced processing for: {file_path}")
    
    try:
        # Initialize extractor
        extractor = EnhancedPDFExtractor()
        
        # Extract text with layout preservation
        logger.info("📄 Extracting text and structure...")
        extracted_data = extractor.extract_text_with_layout(file_path)
        
        # Structured data extraction with Phi-3
        logger.info("🤖 Analyzing with enhanced Phi-3...")
        structured_data = extract_structured_data_with_enhanced_phi3(extracted_data)
        
        # Prepare final output
        output_data = {
            "metadata": {
                "input_file": os.path.basename(file_path),
                "file_size_bytes": os.path.getsize(file_path),
                "extraction_timestamp": datetime.now().isoformat(),
                "processing_device": lm_device,
                "model_used": model_id,
                "extraction_method": "enhanced_digital_ocr_hybrid"
            },
            "raw_extraction": {
                "raw_text_length": len(extracted_data.get('raw_text', '')),
                "structured_text_length": len(extracted_data.get('structured_text', '')),
                "tables_found": len(extracted_data.get('tables', [])),
                "text_blocks_found": len(extracted_data.get('text_blocks', [])),
                "raw_text_preview": extracted_data.get('raw_text', '')[:1000] + "..." if len(extracted_data.get('raw_text', '')) > 1000 else extracted_data.get('raw_text', '')
            },
            "structured_extraction": structured_data,
            "full_raw_data": extracted_data  # Complete extraction data
        }
        
        # Save output
        output_path = generate_output_filename(file_path, output_dir)
        
        if save_json_output(output_data, output_path):
            logger.info(f"✅ Processing completed successfully!")
            logger.info(f"📁 Output saved to: {output_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("EXTRACTION SUMMARY")
            print("="*60)
            print(f"📄 File: {os.path.basename(file_path)}")
            print(f"📊 Text extracted: {len(extracted_data.get('raw_text', ''))} characters")
            print(f"📋 Tables found: {len(extracted_data.get('tables', []))}")
            print(f"🔍 Text blocks: {len(extracted_data.get('text_blocks', []))}")
            
            if not isinstance(structured_data, dict) or "error" not in structured_data:
                print(f"✅ Structured extraction: SUCCESS")
            else:
                print(f"⚠️ Structured extraction: PARTIAL (check output file)")
            
            return output_path, structured_data
        else:
            logger.error("❌ Failed to save output file")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        return None, None

# Utility functions (keeping original ones)
def generate_output_filename(input_file_path: str, output_dir: str = "output") -> str:
    """Generate unique output filename"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_enhanced_extracted_{timestamp}.json"
    return os.path.join(output_dir, output_filename)

def save_json_output(data: Dict[str, Any], output_path: str) -> bool:
    """Save JSON output with error handling"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"✅ Output saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving file: {str(e)}")
        return False

def process_multiple_files_enhanced(file_paths: List[str], output_dir: str = "output") -> List[Dict[str, Any]]:
    """Enhanced batch processing"""
    results = []
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE {i}/{len(file_paths)}: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        if os.path.exists(file_path):
            output_path, extracted_data = run_enhanced_invoice_pipeline(file_path, output_dir)
            results.append({
                "input_file": file_path,
                "output_file": output_path,
                "success": output_path is not None,
                "extraction_summary": {
                    "has_structured_data": extracted_data is not None and "error" not in str(extracted_data),
                    "file_size": os.path.getsize(file_path)
                }
            })
        else:
            logger.error(f"❌ File not found: {file_path}")
            results.append({
                "input_file": file_path,
                "output_file": None,
                "success": False,
                "error": "File not found"
            })
    
    # Print final summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r["success"])
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {len(results) - successful}/{len(results)}")
    
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{status}: {os.path.basename(result['input_file'])}")
        if result.get("output_file"):
            print(f"   📁 Output: {os.path.basename(result['output_file'])}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Single file processing with enhanced extraction
    file_path = "C:\\Users\\Lenovo\\Desktop\\PHI-3\\776548925_638693417193744395 - Copy.pdf"
    
    # Process single file
    output_path, extracted_data = run_enhanced_invoice_pipeline(file_path, output_dir="enhanced_extracted_invoices")
    
    # Example: Process multiple files
    # file_list = [
    #     "path/to/invoice1.pdf",
    #     "path/to/invoice2.pdf", 
    #     "path/to/invoice3.jpg"
    # ]
    # results = process_multiple_files_enhanced(file_list, output_dir="batch_enhanced_extractions")