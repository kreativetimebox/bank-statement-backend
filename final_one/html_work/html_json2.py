import json
from bs4 import BeautifulSoup

json_path = "../json/Amex.json"
html_path = "../html file/Amex.html"
out_path = "final_with_boxes_amex.json"

def norm(s: str) -> str:
    return " ".join(s.replace("\xa0", " ").split())

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load HTML
with open(html_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

html_pages = {}
page_num = 1
html_pages[page_num] = []

for tag in soup.find_all(["a", "p"]):
    if tag.name == "a" and tag.get("name"):
        try:
            page_num = int(tag["name"])
            html_pages[page_num] = []
        except ValueError:
            continue
    elif tag.name == "p":
        text = tag.get_text(" ", strip=True).replace("\xa0", " ")
        if text.strip():
            html_pages[page_num].append(text)

# Extract tokens + boxes per page
def extract_table_tokens_with_boxes(doc):
    pages_tokens = {}
    for i, page in enumerate(doc, start=1):
        tokens = []
        boxes = []
        for table in page.get("table_res_list", []):
            table_pred = table.get("table_ocr_pred", {})
            rec_texts = table_pred.get("rec_texts", [])
            rec_boxes = table_pred.get("rec_boxes", [])
            tokens.extend(rec_texts)
            boxes.extend(rec_boxes)
        pages_tokens[i] = list(zip(tokens, boxes))
    return pages_tokens

def merge_tokens_by_html(tokens_with_boxes, html_lines):
    norm_lines = [norm(l) for l in html_lines]
    merged = []
    buf_text, buf_boxes = None, []

    for tok, box in tokens_with_boxes:
        t = norm(tok)
        if buf_text is None:
            buf_text, buf_boxes = t, [box]
            continue
        candidate = (buf_text + " " + t).strip()
        if any(candidate in line for line in norm_lines):
            buf_text = candidate
            buf_boxes.append(box)
        else:
            if buf_boxes:
                lefts   = [b[0] for b in buf_boxes]
                tops    = [b[1] for b in buf_boxes]
                rights  = [b[2] for b in buf_boxes]
                bottoms = [b[3] for b in buf_boxes]
                merged.append({
                    "text": buf_text,
                    "rec_box": [min(lefts), min(tops), max(rights), max(bottoms)]
                })
            buf_text, buf_boxes = t, [box]

    if buf_text and buf_boxes:
        lefts   = [b[0] for b in buf_boxes]
        tops    = [b[1] for b in buf_boxes]
        rights  = [b[2] for b in buf_boxes]
        bottoms = [b[3] for b in buf_boxes]
        merged.append({
            "text": buf_text,
            "rec_box": [min(lefts), min(tops), max(rights), max(bottoms)]
        })

    return merged

# Process all pages
pages_tokens = extract_table_tokens_with_boxes(data)
final = {}
for p_idx, tokens_with_boxes in pages_tokens.items():
    final[f"Page_{p_idx}"] = merge_tokens_by_html(tokens_with_boxes, html_pages.get(p_idx, []))

# Save
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

print("✅ Saved:", out_path)
