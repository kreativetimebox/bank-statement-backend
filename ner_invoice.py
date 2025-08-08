from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import pipeline
import re
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_first_match(patterns, text):
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            groups = [g for g in match.groups() if g]
            if groups:
                return groups[-1]
            return match.group(1)
    return None

def extract_supplier(ocr_lines, ner_results):
    # Try NER first with confidence threshold
    org_entities = [ent for ent in ner_results if ent['entity_group'] == 'ORG' and ent.get('score', 0) > 0.4]
    org_entities = sorted(org_entities, key=lambda x: x.get('score', 0), reverse=True)

    # Try to find best fuzzy match in top lines
    top_lines = ocr_lines[:5]
    for ent in org_entities:
        for line in top_lines:
            if similar(ent['word'], line) > 0.6:
                return ent['word'].strip()

    # Keyword proximity heuristic
    supplier_keywords = ["supplier", "from", "vendor", "seller", "billed by", "bill from"]
    for i, line in enumerate(top_lines):
        if any(k in line.lower() for k in supplier_keywords):
            if i + 1 < len(top_lines):
                candidate = top_lines[i + 1].strip()
                if len(candidate) > 3:
                    return candidate

    # Fallback to uppercase lines at top
    for line in top_lines:
        if line.isupper() and len(line) > 3:
            return line.strip()

    return None

def extract_invoice_fields(ocr_text, ocr_lines):
    # Patterns for invoice number, date, total amount
    invoice_number_patterns = [
        r"Invoice\s*(Number|No\.?|#)[:\s]*([^\s]+)",
        r"Inv\s*No\.?[:\s]*([^\s]+)",
        r"Invoice\s*ID[:\s]*([^\s]+)",
        r"Bill\s*No\.?[:\s]*([^\s]+)"
    ]

    invoice_date_patterns = [
        r"(Invoice\s*)?Date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"Date\s*of\s*Issue[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"Bill\s*Date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    ]

    total_amount_patterns = [
        r"(Total|Amount Due|Grand Total|Balance Due)[:\s]*([£$€₹]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        r"Total\s*Amount[:\s]*([£$€₹]?\d+(?:,\d{3})*(?:\.\d{2})?)"
    ]

    invoice_number = find_first_match(invoice_number_patterns, ocr_text)
    invoice_date = find_first_match(invoice_date_patterns, ocr_text)
    total_amount = find_first_match(total_amount_patterns, ocr_text)

    currency = None
    if total_amount:
        for symbol, code in [("£", "GBP"), ("$", "USD"), ("€", "EUR"), ("₹", "INR")]:
            if symbol in total_amount:
                currency = code
                total_amount = total_amount.replace(symbol, "").replace(",", "")
                break
        else:
            total_amount = total_amount.replace(",", "")

    return invoice_number, invoice_date, total_amount, currency

def main(image_path):
    doc = DocumentFile.from_images(image_path)
    ocr_model = ocr_predictor(pretrained=True)
    result = ocr_model(doc)
    output = result.export()

    # Extract lines and build text
    ocr_lines = []
    for page in output["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                line_text = " ".join(word["value"] for word in line["words"])
                ocr_lines.append(line_text)

    ocr_text = "\n".join(ocr_lines)

    # NER on full OCR text
    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    ner_results = ner_model(ocr_text)

    supplier = extract_supplier(ocr_lines, ner_results)
    invoice_number, invoice_date, total_amount, currency = extract_invoice_fields(ocr_text, ocr_lines)

    structured_data = {
        "supplier": supplier,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "total_amount": total_amount,
        "currency": currency
    }

    print("\n===== Structured Data Extracted =====")
    print(structured_data)
    return structured_data

# Example usage
image_path = r"E:\bank-statement-ui\Bank statements\Categorize_Data\Invoice_Image\IMG0265_638882805887603264.jpg"
main(image_path)
