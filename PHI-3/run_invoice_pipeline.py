import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set device logic: OCR on CPU, Phi-3 on GPU if available
ocr_device = "cpu"
lm_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_id = "microsoft/phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set a distinct pad_token to avoid eos_token conflict
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token (common workaround)
    # Alternatively, add a new pad_token: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# Text extractor (OCR + digital PDF)
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.png', '.jpg', '.jpeg']:
        print("🖼️ Extracting from image using pytesseract...")
        image = Image.open(file_path).convert("RGB")
        return pytesseract.image_to_string(image)

    elif ext == '.pdf':
        try:
            print("📄 Trying digital extraction from PDF...")
            with fitz.open(file_path) as doc:
                text = "".join([page.get_text() for page in doc])
            if text.strip():
                print("✅ Digital text extraction succeeded.")
                return text
        except:
            print("❌ Digital extraction failed.")

        print("🔁 Falling back to OCR for scanned PDF...")
        images = convert_from_path(file_path)
        return "\n".join([pytesseract.image_to_string(img.convert("RGB")) for img in images])

    raise ValueError("Unsupported file type.")

# Phi-3 structured extractor
def extract_structured_data_with_phi3(text):
    prompt = f"""
You are a receipt and invoice extraction assistant. From the following invoice text, extract the following structured JSON fields only:

- invoice_number
- invoice_date
- vendor_name
- items: list of items where each item includes:
    - description (like date range or label)
    - quantity (days/items)
    - unit_price (rate per day/item)
    - total_price (for that line)
- totals_section: include all explicitly written totals such as Net Total, VAT, Final Total, etc.
- payment_terms: include the payment due date and any specific payment instructions
- customer_details: include customer name, address, and contact information
- additional_notes: any special instructions or comments

Return only JSON. Don't add explanations.

--- START OF INVOICE TEXT ---
{text}
--- END OF TEXT ---
"""

    # Tokenize with attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(lm_device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly pass attention_mask
            max_new_tokens=1024,
            do_sample=False
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_start = decoded.find("{")
    return decoded[json_start:]

# Main pipeline function
def run_invoice_pipeline(file_path):
    print("🔍 Extracting text from file...")
    raw_text = extract_text_from_file(file_path)

    print("🤖 Sending to Phi-3 for structured extraction...")
    structured_json = extract_structured_data_with_phi3(raw_text)

    print("\n✅ Final Output from Phi-3:\n")
    print(structured_json)
    return structured_json

# Run the pipeline (replace with your local file path)
file_path = "C:\\Users\\Lenovo\\Desktop\\PHI-3\\Multiple Invoices Meged invoices Ref 3.pdf"  # Adjust to your local PDF file path
run_invoice_pipeline(file_path)