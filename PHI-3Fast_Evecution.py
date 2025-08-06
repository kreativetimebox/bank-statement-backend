"""
OPTIMIZED FAST PDF EXTRACTOR
This version is designed for speed while maintaining accuracy
Typical processing time: 30-60 seconds per document
"""

import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import json
from datetime import datetime
import logging
import re
from typing import Dict, Any, Tuple, Optional
import concurrent.futures
import threading

# Optional: Lightweight OCR preprocessing
try:
    from PIL import ImageFilter
    PIL_FILTERS_AVAILABLE = True
except ImportError:
    PIL_FILTERS_AVAILABLE = False

# Use lighter AI models or local alternatives
USE_HEAVY_AI = False  # Set to True if you want full Phi-3 processing

if USE_HEAVY_AI:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastPDFExtractor:
    """Optimized extractor focusing on speed"""
    
    def __init__(self):
        # Optimized OCR config for speed vs accuracy balance
        self.fast_ocr_config = r'--oem 3 --psm 6'  # Faster than character whitelist
        self.detailed_ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?@#$%^&*()_+-=[]{}|\\/<> '
        
        # AI model initialization (lazy loading)
        self.ai_model = None
        self.ai_tokenizer = None
        self._model_loaded = False
        
        print(f"🚀 FastPDFExtractor initialized")
        print(f"💡 Heavy AI processing: {'ENABLED' if USE_HEAVY_AI else 'DISABLED (faster)'}")
    
    def quick_preprocess_image(self, image: Image.Image) -> Image.Image:
        """Fast image preprocessing - basic but effective"""
        # Convert to grayscale for faster processing
        if image.mode != 'L':
            image = image.convert('L')
        
        # Quick contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # Optional: apply unsharp mask if available
        if PIL_FILTERS_AVAILABLE:
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        
        return image
    
    def extract_text_fast(self, file_path: str) -> Dict[str, Any]:
        """Fast text extraction with smart method selection"""
        start_time = datetime.now()
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            result = self._extract_pdf_fast(file_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            result = self._extract_image_fast(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Add timing info
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time_seconds'] = processing_time
        
        logger.info(f"⏱️ Extraction completed in {processing_time:.2f} seconds")
        return result
    
    def _extract_pdf_fast(self, file_path: str) -> Dict[str, Any]:
        """Fast PDF processing with smart fallback"""
        
        # Step 1: Try digital extraction first (fastest method)
        logger.info("📄 Attempting fast digital extraction...")
        try:
            digital_result = self._extract_digital_text_fast(file_path)
            
            # Quality check: if we got substantial text, use it
            if len(digital_result['raw_text'].strip()) > 100:
                logger.info(f"✅ Digital extraction successful ({len(digital_result['raw_text'])} chars)")
                digital_result['extraction_method'] = 'digital_fast'
                return digital_result
        except Exception as e:
            logger.warning(f"Digital extraction failed: {e}")
        
        # Step 2: Smart OCR - process only first few pages for speed test
        logger.info("🔍 Using smart OCR extraction...")
        return self._extract_ocr_smart(file_path)
    
    def _extract_digital_text_fast(self, file_path: str) -> Dict[str, Any]:
        """Fast digital text extraction"""
        doc = fitz.open(file_path)
        
        all_text = []
        page_count = len(doc)
        
        # Limit pages for very large documents
        max_pages = min(page_count, 50)  # Process max 50 pages for speed
        
        for page_num in range(max_pages):
            page = doc[page_num]
            page_text = page.get_text()
            
            if page_text.strip():
                all_text.append(f"--- PAGE {page_num + 1} ---\n{page_text}")
        
        doc.close()
        
        combined_text = '\n'.join(all_text)
        
        return {
            'raw_text': combined_text,
            'page_count': page_count,
            'processed_pages': max_pages,
            'tables': [],  # Skip table extraction for speed
            'metadata': {
                'total_pages': page_count,
                'processed_pages': max_pages
            }
        }
    
    def _extract_ocr_smart(self, file_path: str) -> Dict[str, Any]:
        """Smart OCR with speed optimizations"""
        
        # Convert with moderate DPI for speed/quality balance
        logger.info("🖼️ Converting PDF to images (DPI: 200)...")
        try:
            # Try to get page count first
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()
            
            # Limit pages for very large documents
            max_pages = min(total_pages, 20)  # Process max 20 pages
            
            images = convert_from_path(
                file_path, 
                dpi=200,  # Lower DPI for speed
                first_page=1,
                last_page=max_pages
            )
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return {'raw_text': '', 'error': str(e)}
        
        # Process pages (potentially in parallel for speed)
        if len(images) > 3:
            # Use parallel processing for multiple pages
            return self._process_images_parallel(images)
        else:
            # Process sequentially for small documents
            return self._process_images_sequential(images)
    
    def _process_images_sequential(self, images: list) -> Dict[str, Any]:
        """Process images one by one"""
        all_text = []
        
        for i, image in enumerate(images):
            logger.info(f"🔍 OCR processing page {i+1}/{len(images)}")
            
            # Quick preprocessing
            processed_image = self.quick_preprocess_image(image)
            
            # Fast OCR
            page_text = pytesseract.image_to_string(
                processed_image, 
                config=self.fast_ocr_config
            )
            
            if page_text.strip():
                all_text.append(f"--- PAGE {i+1} ---\n{page_text}")
        
        return {
            'raw_text': '\n'.join(all_text),
            'page_count': len(images),
            'extraction_method': 'ocr_sequential'
        }
    
    def _process_images_parallel(self, images: list) -> Dict[str, Any]:
        """Process multiple images in parallel for speed"""
        logger.info(f"⚡ Processing {len(images)} pages in parallel...")
        
        def process_single_page(args):
            i, image = args
            try:
                processed_image = self.quick_preprocess_image(image)
                page_text = pytesseract.image_to_string(
                    processed_image, 
                    config=self.fast_ocr_config
                )
                return i, page_text
            except Exception as e:
                logger.error(f"Error processing page {i+1}: {e}")
                return i, ""
        
        # Use ThreadPoolExecutor for I/O bound OCR tasks
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all pages
            future_to_page = {
                executor.submit(process_single_page, (i, img)): i 
                for i, img in enumerate(images)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_page):
                try:
                    page_num, page_text = future.result()
                    results.append((page_num, page_text))
                except Exception as e:
                    logger.error(f"Page processing failed: {e}")
        
        # Sort results by page number and combine
        results.sort(key=lambda x: x[0])
        all_text = []
        
        for page_num, page_text in results:
            if page_text.strip():
                all_text.append(f"--- PAGE {page_num+1} ---\n{page_text}")
        
        return {
            'raw_text': '\n'.join(all_text),
            'page_count': len(images),
            'extraction_method': 'ocr_parallel'
        }
    
    def _extract_image_fast(self, file_path: str) -> Dict[str, Any]:
        """Fast image processing"""
        logger.info("🖼️ Processing image file...")
        
        image = Image.open(file_path)
        
        # Quick preprocessing
        processed_image = self.quick_preprocess_image(image)
        
        # OCR
        text = pytesseract.image_to_string(processed_image, config=self.fast_ocr_config)
        
        return {
            'raw_text': text,
            'extraction_method': 'image_fast'
        }
    
    def load_ai_model_lazy(self):
        """Load AI model only when needed"""
        if not USE_HEAVY_AI or self._model_loaded:
            return
        
        logger.info("📥 Loading AI model (one-time setup)...")
        try:
            model_id = "microsoft/phi-3-mini-128k-instruct"
            self.ai_tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            if self.ai_tokenizer.pad_token is None:
                self.ai_tokenizer.pad_token = self.ai_tokenizer.eos_token
            
            # Use lower precision and optimization for speed
            self.ai_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            self._model_loaded = True
            logger.info("✅ AI model loaded")
            
        except Exception as e:
            logger.error(f"AI model loading failed: {e}")
            self._model_loaded = False
    
    def extract_structured_data_fast(self, raw_text: str) -> Dict[str, Any]:
        """Fast structured data extraction"""
        
        if USE_HEAVY_AI:
            return self._extract_with_ai(raw_text)
        else:
            return self._extract_with_regex(raw_text)
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Fast regex-based extraction"""
        logger.info("🔍 Using fast regex extraction...")
        
        # Comprehensive regex patterns
        patterns = {
            'invoice_number': [
                r'(?:invoice|inv)(?:\s*#?|\s+no\.?|\s+number)[\s:]*([A-Z0-9-]{3,})',
                r'(?:bill|receipt)\s*#?[\s:]*([A-Z0-9-]{3,})',
                r'#?\s*([A-Z0-9-]{6,})'  # Generic number pattern
            ],
            'date': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})'
            ],
            'total_amount': [
                r'(?:total|amount due|balance|final)[\s:$]*([0-9,]+\.?[0-9]*)',
                r'\$\s*([0-9,]+\.?[0-9]*)',
                r'([0-9,]+\.[0-9]{2})\s*(?:total|due|balance)'
            ],
            'vendor_name': [
                r'^([A-Z][A-Za-z\s&.,]{3,50}?)(?:\n|$)',  # First line often vendor
                r'from[\s:]+([A-Z][A-Za-z\s&.,]{3,50})',
                r'bill from[\s:]+([A-Z][A-Za-z\s&.,]{3,50})'
            ],
            'email': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            'phone': [
                r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
                r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
            ]
        }
        
        extracted = {}
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Take first non-empty match
                    value = next((m for m in matches if m.strip()), None)
                    if value:
                        extracted[field] = {
                            'value': value.strip(),
                            'confidence': 0.8  # Moderate confidence for regex
                        }
                        break
            
            # Default empty value if not found
            if field not in extracted:
                extracted[field] = {'value': '', 'confidence': 0.0}
        
        # Try to extract line items (basic)
        line_items = self._extract_line_items_regex(text)
        extracted['line_items'] = line_items
        
        return extracted
    
    def _extract_line_items_regex(self, text: str) -> list:
        """Extract line items using patterns"""
        
        # Look for table-like structures
        lines = text.split('\n')
        items = []
        
        # Patterns that might indicate line items
        item_patterns = [
            r'(.+?)\s+(\d+)\s+\$?([0-9,]+\.?[0-9]*)\s+\$?([0-9,]+\.?[0-9]*)',  # desc qty price total
            r'(.+?)\s+\$?([0-9,]+\.?[0-9]*)\s+(\d+)\s+\$?([0-9,]+\.?[0-9]*)',  # desc price qty total
            r'(.+?)\s+\$?([0-9,]+\.?[0-9]*)',  # desc total (simple)
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:  # Skip short lines
                continue
            
            for pattern in item_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        items.append({
                            'description': {'value': groups[0].strip(), 'confidence': 0.7},
                            'amount': {'value': groups[-1], 'confidence': 0.7}  # Last group is usually total
                        })
                    break
        
        return items[:10]  # Limit to first 10 items for speed
    
    def _extract_with_ai(self, text: str) -> Dict[str, Any]:
        """AI-based extraction (if enabled)"""
        logger.info("🤖 Using AI extraction...")
        
        self.load_ai_model_lazy()
        
        if not self._model_loaded:
            logger.warning("AI model not available, falling back to regex")
            return self._extract_with_regex(text)
        
        # Truncate text for faster processing
        max_text_length = 8000  # Much shorter for speed
        if len(text) > max_text_length:
            text = text[:max_text_length] + "\n... [TRUNCATED FOR SPEED]"
        
        # Simplified prompt for faster processing
        prompt = f"""Extract key information from this invoice/receipt as JSON:

{{
  "invoice_number": "",
  "date": "",
  "vendor_name": "",
  "total_amount": 0.0,
  "currency": "",
  "customer_name": ""
}}

Document text:
{text}

JSON:"""
        
        try:
            # Faster tokenization and generation
            inputs = self.ai_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=10000,  # Shorter context
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.ai_model.generate(
                    **inputs,
                    max_new_tokens=500,  # Shorter output
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.ai_tokenizer.pad_token_id
                )
            
            response = self.ai_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON
            json_start = response.find('{')
            if json_start != -1:
                json_str = response[json_start:]
                json_end = json_str.rfind('}') + 1
                if json_end > 0:
                    json_str = json_str[:json_end]
                    return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"AI extraction failed: {e}")
        
        # Fallback to regex
        return self._extract_with_regex(text)

def run_fast_extraction(file_path: str, output_dir: str = "fast_output") -> Tuple[Optional[str], Dict[str, Any]]:
    """Main fast extraction pipeline"""
    
    start_total = datetime.now()
    logger.info(f"🚀 Starting FAST extraction: {os.path.basename(file_path)}")
    
    try:
        # Initialize fast extractor
        extractor = FastPDFExtractor()
        
        # Step 1: Fast text extraction
        logger.info("📄 Fast text extraction...")
        extraction_result = extractor.extract_text_fast(file_path)
        
        raw_text = extraction_result.get('raw_text', '')
        
        if not raw_text.strip():
            logger.error("❌ No text extracted from document")
            return None, {'error': 'No text extracted'}
        
        # Step 2: Fast structured extraction
        logger.info("🔍 Fast structured extraction...")
        structured_data = extractor.extract_structured_data_fast(raw_text)
        
        # Prepare output
        total_time = (datetime.now() - start_total).total_seconds()
        
        output_data = {
            'metadata': {
                'file_name': os.path.basename(file_path),
                'file_size_mb': round(os.path.getsize(file_path) / (1024*1024), 2),
                'processing_time_seconds': total_time,
                'extraction_timestamp': datetime.now().isoformat(),
                'method': extraction_result.get('extraction_method', 'unknown'),
                'ai_used': USE_HEAVY_AI
            },
            'extraction_summary': {
                'text_length': len(raw_text),
                'pages_processed': extraction_result.get('page_count', 1),
                'structured_fields_found': len(structured_data)
            },
            'structured_data': structured_data,
            'raw_text_sample': raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        }
        
        # Save output
        output_path = save_fast_output(output_data, file_path, output_dir)
        
        # Success summary
        print(f"\n{'='*60}")
        print("⚡ FAST EXTRACTION COMPLETED")
        print(f"{'='*60}")
        print(f"📄 File: {os.path.basename(file_path)}")
        print(f"⏱️ Time: {total_time:.1f} seconds")
        print(f"📊 Text: {len(raw_text):,} characters")
        print(f"🔧 Method: {extraction_result.get('extraction_method', 'unknown')}")
        print(f"💾 Output: {os.path.basename(output_path) if output_path else 'None'}")
        
        # Show key extracted data
        if isinstance(structured_data, dict) and 'invoice_number' in structured_data:
            inv_num = structured_data.get('invoice_number', {}).get('value', 'Not found')
            total_amt = structured_data.get('total_amount', {}).get('value', 'Not found')
            vendor = structured_data.get('vendor_name', {}).get('value', 'Not found')
            
            print(f"🏷️ Invoice #: {inv_num}")
            print(f"💰 Total: {total_amt}")
            print(f"🏢 Vendor: {vendor}")
        
        print("="*60)
        
        return output_path, structured_data
        
    except Exception as e:
        logger.error(f"❌ Fast extraction failed: {str(e)}")
        return None, {'error': str(e)}

def save_fast_output(data: Dict[str, Any], input_path: str, output_dir: str) -> str:
    """Save output quickly"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_fast_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return output_path
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        return None

def batch_process_fast(file_paths: list, output_dir: str = "fast_batch_output") -> list:
    """Fast batch processing"""
    results = []
    total_start = datetime.now()
    
    print(f"⚡ Starting FAST batch processing: {len(file_paths)} files")
    print("="*60)
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n📄 Processing {i}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            results.append({'file': file_path, 'success': False, 'error': 'File not found'})
            continue
        
        output_path, structured_data = run_fast_extraction(file_path, output_dir)
        
        results.append({
            'file': file_path,
            'output': output_path,
            'success': output_path is not None,
            'data_preview': {
                'invoice_number': structured_data.get('invoice_number', {}).get('value', 'N/A') if isinstance(structured_data, dict) else 'N/A',
                'total_amount': structured_data.get('total_amount', {}).get('value', 'N/A') if isinstance(structured_data, dict) else 'N/A'
            }
        })
    
    # Final summary
    total_time = (datetime.now() - total_start).total_seconds()
    successful = sum(1 for r in results if r['success'])
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(file_paths)}")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    print(f"📊 Average: {total_time/len(file_paths):.1f} seconds per file")
    print(f"🚀 Speed: {'FAST' if total_time/len(file_paths) < 120 else 'SLOW'}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Single file - FAST processing
    file_path = "C:\\Users\\Lenovo\\Desktop\\PHI-3\\Arul Paul Invoice-1.pdf"
    
    print("🚀 OPTIMIZED FAST PDF EXTRACTOR")
    print("="*60)
    print(f"⚡ Speed mode: {'AI + Regex' if USE_HEAVY_AI else 'Regex only (fastest)'}")
    print(f"🎯 Target time: 30-60 seconds per document")
    print("="*60)
    
    # Process single file
    output_path, data = run_fast_extraction(file_path, output_dir="fast_extractions")
    
    # Example batch processing
    # file_list = ["file1.pdf", "file2.pdf", "file3.pdf"]
    # batch_results = batch_process_fast(file_list, "fast_batch_results")