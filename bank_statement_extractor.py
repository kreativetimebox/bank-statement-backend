import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from datetime import datetime
import time

# Use CPU for everything
lm_device = "cpu"
ocr_device = "cpu"

# Smaller model that works well with text prompts
model_id = "tiiuae/falcon-rw-1b"  # better than flan-t5-base for JSON-style tasks
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(lm_device)

print(f"📌 Model loaded on device: {lm_device}")

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.png', '.jpg', '.jpeg']:
        image = Image.open(file_path).convert("RGB")
        return pytesseract.image_to_string(image)

    elif ext == '.pdf':
        try:
            with fitz.open(file_path) as doc:
                text = "".join([page.get_text() for page in doc])
            if text.strip():
                return text
        except:
            pass
        # fallback to OCR, only first page
        images = convert_from_path(file_path, first_page=1, last_page=1)
        return "\n".join([pytesseract.image_to_string(img.convert("RGB")) for img in images])

    raise ValueError("Unsupported file type.")

def preprocess_bank_statement_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\d+)/(\d+)/(\d+)', r'\1/\2/\3', text)
    text = re.sub(r'(\d+\.\d+)\s*([A-Z]+)', r'\1 \2', text)
    return text.strip()

def extract_bank_statement_data(text):
    text = text[:3000]  # Trim to prevent token overload
    prompt = f"""
Extract the following JSON from this bank statement text:

{{
  "account_holder": {{
    "name": "...",
    "address": "..."
  }},
  "account_details": {{
    "account_number": "...",
    "sort_code": "...",
    "bank_name": "..."
  }},
  "statement_period": {{
    "start_date": "...",
    "end_date": "..."
  }},
  "transactions": [
    {{
      "transaction_date": "...",
      "description": "...",
      "withdrawal": "...",
      "deposit": "...",
      "balance": "..."
    }}
  ]
}}

Only respond with the JSON. Text:
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(lm_device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_start = output_text.find("{")

    if json_start == -1:
        print("⚠️ Model did not return any JSON-like content.")
        print("Raw model output:\n", output_text)
        return None

    json_str = output_text[json_start:]
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}")
        print("⚠️ Raw output (showing partial):")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
        return json_str  # Fallback raw string

def validate_bank_statement_data(json_data):
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        result = {"valid": True, "issues": []}
        for field in ['account_holder', 'account_details', 'statement_period', 'transactions']:
            if field not in data:
                result["valid"] = False
                result["issues"].append(f"Missing: {field}")
        return result
    except Exception as e:
        return {"valid": False, "issues": [str(e)]}

def run_bank_statement_pipeline(file_path):
    print("🏦 Starting bank statement analysis")
    print(f"📁 File: {file_path}")

    try:
        raw_text = extract_text_from_file(file_path)
        print(f"✅ Extracted text (length={len(raw_text)})")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return

    processed = preprocess_bank_statement_text(raw_text)
    print("✅ Preprocessing complete")

    start = time.time()
    structured_json = extract_bank_statement_data(processed)
    print(f"✅ LLM inference done in {time.time() - start:.2f} seconds")

    if not structured_json:
        print("❌ No structured output produced")
        return

    validation = validate_bank_statement_data(structured_json)
    if validation["valid"]:
        print("✅ Validation passed")
    else:
        print("⚠️ Validation issues:")
        for issue in validation["issues"]:
            print("   -", issue)

    print("\n📊 FINAL OUTPUT:\n" + "="*60)
    print(structured_json)
    print("="*60)
    return structured_json

def save_extracted_data(json_data, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_data if isinstance(json_data, str) else json.dumps(json_data, indent=2))
        print(f"💾 Output saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save output: {e}")

# Run the full pipeline
if __name__ == "__main__":
    file_path = "C:\\Users\\Lenovo\\Desktop\\PHI-3_BankStatement\\Barclays_uk_bank_statement.pdf"
    result = run_bank_statement_pipeline(file_path)
    if result:
        output_file = f"bank_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_extracted_data(result, output_file)
        print("🎉 Done.")
    else:
        print("❌ Pipeline did not produce a result.")
