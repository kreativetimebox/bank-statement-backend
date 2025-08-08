# invoice_extractor_simple.py
# pip install "python-doctr[torch]" transformers numpy python-dateutil rapidfuzz

import re, json
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dateutil import parser as dateparser

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import pipeline as hf_pipeline

DEBUG = False  # set True to print why/where values were found

def jprint(obj): print(json.dumps(obj, indent=2, ensure_ascii=False))
def norm_space(s: str) -> str: return re.sub(r"\s+", " ", s or "").strip()
def similar(a, b): return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
def safe_float(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s = s.replace(",", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None
def parse_date_any(s: str) -> Optional[str]:
    if not s: return None
    try:
        d = dateparser.parse(s, dayfirst=False, yearfirst=False, fuzzy=True)
        if d and 1990 <= d.year <= 2100: return d.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None
def clean_ner_word(w: str) -> str:
    return re.sub(r"^#+", "", (w or "")).replace("##", "").strip()
def looks_like_company(s: str) -> bool:
    if not s or len(s) < 3: return False
    bad = ["invoice","date","total","amount","vat","due","customer"]
    return not any(b in s.lower() for b in bad)

# -------- OCR lines with word-level positions --------
def build_lines(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines = []
    for p in output.get("pages", []):
        for b in p.get("blocks", []):
            for ln in b.get("lines", []):
                words = ln.get("words", [])
                if not words:
                    continue
                txt = " ".join(w.get("value","") for w in words)
                xs, confs, wlist = [], [], []
                for w in words:
                    g = w.get("geometry")
                    val = w.get("value", "")
                    if isinstance(g, (list, tuple)) and len(g) == 2:
                        try:
                            x0, y0 = float(g[0]), float(g[1])
                            x1, y1 = float(g[1]), float(g[1][1])
                            xc = 0.5 * (x0 + x1)
                            xs.append(xc)
                            wlist.append({"text": val, "xc": xc})
                        except Exception:
                            wlist.append({"text": val, "xc": None})
                    else:
                        wlist.append({"text": val, "xc": None})
                    c = w.get("confidence")
                    if c is not None:
                        try: confs.append(float(c))
                        except Exception: pass
                ocr_conf = (sum(confs) / len(confs)) if confs else 0.0
                cy = None
                lg = ln.get("geometry")
                if isinstance(lg, (list, tuple)) and len(lg) == 2:
                    try: cy = 0.5 * (float(lg[0][1]) + float(lg[1][1]))
                    except Exception: cy = None
                lines.append({
                    "text": norm_space(txt),
                    "ocr_conf": float(ocr_conf),
                    "xs": xs,
                    "cy": cy,
                    "words": wlist
                })
    return lines

# -------- NER once on full text --------
def ner_full_text(text: str):
    if not text.strip(): return []
    ner = hf_pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner(text)

# -------- Parties --------
def pick_supplier(lines: List[Dict[str,Any]], ents: List[Dict[str,Any]]) -> Optional[str]:
    top_lines = [l["text"] for l in lines[:8]]
    orgs = [e for e in ents if e.get("entity_group") == "ORG"]
    orgs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    # Prefer company-like names that match header lines
    for e in orgs:
        w = looks_like_company(clean_ner_word(e.get("word","")))
        if not w: continue
    for e in orgs:
        w = clean_ner_word(e.get("word",""))
        if len(w) < 3: continue
        if any(similar(w, tl) >= 0.5 or w in tl for tl in top_lines):
            if looks_like_company(w): return w
    # Supplier hints then next line
    hints = ["supplier","from","vendor","seller","billed by","bill from","invoiced by"]
    for i, tl in enumerate(top_lines):
        if any(h in tl.lower() for h in hints):
            if i+1 < len(top_lines) and len(top_lines[i+1]) > 3:
                return top_lines[i+1]
    # Fallback uppercase-like header line
    for tl in top_lines:
        if len(tl) > 3 and (tl.isupper() or re.search(r"[A-Z]{3,}", tl)):
            return tl
    return None

def pick_buyer(lines: List[Dict[str,Any]], ents: List[Dict[str,Any]]) -> Tuple[Optional[str], Optional[str]]:
    anchors = ["bill to","billed to","sold to","ship to","deliver to","invoice to","recipient","customer","buyer","to:"]
    texts = [l["text"] for l in lines]
    for i, t in enumerate(texts):
        if any(a in t.lower() for a in anchors):
            name = t
            addr=[]
            for j in range(i+1, min(i+4, len(texts))):
                s=texts[j].lower()
                if len(texts[j])>=4 and not any(k in s for k in ["invoice","date","total","amount","grand total","terms"]):
                    addr.append(texts[j])
            return name, (norm_space(" ".join(addr)) if addr else None)
    # Fallback to second-best ORG
    orgs = [clean_ner_word(e.get("word","")) for e in ents if e.get("entity_group")=="ORG"]
    orgs = [o for o in orgs if looks_like_company(o)]
    if len(orgs) >= 2: return orgs[1], None
    return (orgs, None) if orgs else (None, None)

# -------- Header fields --------
def extract_header_fields(text: str, lines: List[Dict[str,Any]]) -> Dict[str,Any]:
    t = norm_space(text)
    money = r"(?:[£$€₹¥]\s*)?\d{1,3}(?:,\d{3})*(?:\.\d{2})"
    patt_num = [
        r"invoice\s*(number|no|#|num|ref(?:erence)?)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/\.]+)",
        r"\binv\s*(no|#)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/\.]+)",
        r"invoice\s*id\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/\.]+)",
        r"bill\s*no\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/\.]+)",
    ]
    patt_date = [
        r"(?:invoice\s*)?date\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"date\s*of\s*issue\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"bill\s*date\s*[:\-]?\s*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        r"(?:invoice\s*)?date\s*[:\-]?\s*([0-3]?\d\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})",
        r"(?:invoice\s*)?date\s*[:\-]?\s*((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+[0-3]?\d,\s*\d{2,4})",
        r"(?:invoice\s*)?date\s*[:\-]?\s*(\d{4}[\/\-.][01]?\d[\/\-.][0-3]?\d)",
    ]
    patt_due = [
        r"due\s*date\s*[:\-]?\s*([A-Za-z0-9,\/\-. ]+)",
        r"payment\s*due\s*[:\-]?\s*([A-Za-z0-9,\/\-. ]+)",
        r"pay\s*by\s*[:\-]?\s*([A-Za-z0-9,\/\-. ]+)",
    ]
    # Require >=3 chars for PO id to avoid short noise like "RA"
    patt_po = [r"(?:\bpo\b|purchase\s*order|order\s*number)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/\.]{2,})"]
    patt_terms = [r"(?:payment\s*terms|terms)\s*[:\-]?\s*([A-Za-z0-9 \-]+)", r"\bnet\s*(\d{1,3})\b"]
    patt_sub = [rf"sub[\- ]?total\s*[:\-]?\s*({money})"]
    patt_tax = [rf"(?:tax|vat|gst|igst|cgst|sgst)\s*[:\-]?\s*({money})", rf"vat\s*\d+%?\s*[:\-]?\s*({money})"]
    patt_tot = [
        rf"(grand\s*total|invoice\s*total|amount\s*due|total\s*due|total\s*gbp|net|total)\s*[:\-]?\s*({money})",
        rf"total\s*amount\s*[:\-]?\s*({money})",
    ]
    def grab(pats):
        for p in pats:
            m = re.search(p, t, re.IGNORECASE)
            if m:
                g = [x for x in m.groups() if x]
                return g[-1] if g else m.group(0)
        return None
    inv_num = grab(patt_num)
    inv_date_raw = grab(patt_date)
    due_raw = grab(patt_due)
    po = grab(patt_po)
    terms = grab(patt_terms)
    subtotal_raw = grab(patt_sub)
    tax_raw = grab(patt_tax)
    total_raw = grab(patt_tot)
    # Bottom-region fallback for total
    if not total_raw:
        bottom = [l for l in lines if l.get("cy") is not None and l["cy"] >= 0.7]
        cues = ["grand total","invoice total","amount due","total due","total"]
        for l in bottom[::-1]:
            low = l["text"].lower()
            if any(c in low for c in cues):
                m = re.search(rf"{money}", l["text"])
                if m: total_raw = m.group(); break
    # Currency
    currency=None
    for sym, code in [("£","GBP"),("$","USD"),("€","EUR"),("₹","INR"),("¥","JPY")]:
        if any(v and sym in v for v in [subtotal_raw, tax_raw, total_raw]): currency=code; break
    if not currency:
        m=re.search(r"\b(USD|GBP|EUR|INR|JPY|AUD|CAD|CHF|CNY|HKD|SGD|NZD|ZAR|SEK|NOK|DKK|AED|SAR|QAR)\b", t)
        if m: currency=m.group(1)
    def norm_money(s):
        if not s: return None
        s = s.replace(",", "")
        s = re.sub(r"[£$€₹¥]\s*", "", s)
        return s
    out = {
        "number": inv_num,
        "date": parse_date_any(inv_date_raw) if inv_date_raw else None,
        "due_date": parse_date_any(due_raw) if due_raw else None,
        "po_number": po,
        "payment_terms": f"Net {terms}" if terms and terms.isdigit() else terms,
        "currency": currency,
        "subtotal": float(norm_money(subtotal_raw)) if subtotal_raw else None,
        "tax": float(norm_money(tax_raw)) if tax_raw else None,
        "total": float(norm_money(total_raw)) if total_raw else None,
    }
    if DEBUG:
        print("[header]", dict(inv_num=inv_num, inv_date=inv_date_raw, due=due_raw, po=po, terms=terms, subtotal=subtotal_raw, tax=tax_raw, total=total_raw, currency=currency))
    return out

# -------- Items: header detect, column estimate, word-level assignment --------
def detect_items_header(lines: List[Dict[str,Any]]) -> Optional[int]:
    cues = ["description","item","code","sku","qty","quantity","unit","unit price","rate","vat","amount","total","value","price"]
    for i,l in enumerate(lines):
        low=l["text"].lower()
        hits=sum(1 for c in cues if c in low)
        digits=len(re.findall(r"\d", l["text"]))
        if hits>=2 or (hits>=1 and digits>=2):
            return i
    return None

def estimate_columns(line: Dict[str,Any]) -> List[Tuple[float,float]]:
    xs = sorted(line.get("xs", []))
    if not xs: return [(0.05,0.55),(0.55,0.72),(0.72,0.86),(0.86,0.98)]
    bins = np.linspace(0.02, 0.98, 12)
    hist, edges = np.histogram(xs, bins=bins)
    bands=[]
    for b in range(len(hist)):
        if hist[b] > 0:
            bands.append([float(edges[b]), float(edges[b+1])])
    merged=[]
    for c in bands:
        if not merged or c[0] - merged[-1][1] > 0.03: merged.append(c)
        else: merged[-1][1] = c[1]
    if len(merged) < 2: merged = [(0.05,0.55),(0.55,0.72),(0.72,0.86),(0.86,0.98)]
    # Slightly widen bands to be tolerant
    widened = []
    for a,b in merged:
        widened.append((max(0.0, a-0.01), min(1.0, b+0.01)))
    return widened

def assign_cells_by_words(line: Dict[str, Any], cols: List[Tuple[float, float]]) -> List[str]:
    n = max(2, len(cols))
    buckets = [[] for _ in range(n)]
    for w in line.get("words", []):
        xc = w.get("xc")
        t = w.get("text","")
        if not t: continue
        if xc is None:
            # if no position, push to last (amount)
            buckets[-1].append(t)
            continue
        placed = False
        for k,(a,b) in enumerate(cols):
            if a <= xc <= b:
                buckets[k].append(t); placed = True; break
        if not placed:
            buckets[-1].append(t)
    cells = [norm_space(" ".join(v)) if v else None for v in buckets]
    # Rightmost-amount safeguard
    if cells[-1] is None or not re.search(r"\d", cells[-1] or ""):
        m2 = re.findall(r"[£$€₹¥]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})", line.get("text",""))
        if m2:
            cells[-1] = m2[-1]
            if not cells[0]:
                cells = (line.get("text","").rsplit(m2[-1], 1)).strip()
    return cells

def extract_items(lines: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    idx = detect_items_header(lines)
    if idx is None: return []
    cols = estimate_columns(lines[idx])
    items=[]
    stop_cues = ["invoice total","grand total","total amount","amount due","subtotal","total vat","vat total","outstanding","waiting to pay","totals:"]
    for i in range(idx+1, len(lines)):
        txt = lines[i]["text"] or ""
        low = txt.lower()
        if any(c in low for c in stop_cues): break
        if len(txt.split()) < 2 and not re.search(r"\d", txt): continue
        try:
            cells = assign_cells_by_words(lines[i], cols)
        except Exception:
            cells = [txt, None]
        # Final amount fallback
        if (not cells or cells[-1] is None):
            m2 = re.findall(r"[£$€₹¥]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})", txt)
            if m2:
                amt_txt = m2[-1]
                if not cells: cells = [txt, amt_txt]
                else: cells[-1] = amt_txt
                if not cells[0]:
                    cells = txt.rsplit(amt_txt, 1).strip()
        n = len(cells)
        row: Dict[str, Any] = {}
        row["description"] = cells if n >= 1 else None
        if n >= 4:
            row["qty"] = safe_float(cells[1]) if cells[1] else None
            row["unit_price"] = safe_float(cells[-2]) if cells[-2] else None
            row["amount"] = safe_float(cells[-1]) if cells[-1] else None
        elif n == 3:
            row["qty"] = safe_float(cells[1]) if cells[1] else None
            row["amount"] = safe_float(cells[2]) if cells[2] else None
        else:
            row["amount"] = safe_float(cells[-1]) if cells[-1] else None
        m = re.search(r"\b(\d{1,2}(?:\.\d+)?)\s*%(\s*vat)?\b", low)
        if m: row["vat_percent"] = safe_float(m.group(1))
        row["conf"] = min(0.98, 0.5 + 0.4 * float(lines[i].get("ocr_conf", 0.0)))
        if any(v is not None for k,v in row.items() if k != "conf"):
            items.append(row)
    for r, it in enumerate(items, 1): it["row"] = r
    if DEBUG: print(f"[items] header_idx={idx}, cols={cols}, rows={len(items)}")
    return items

# -------- Assemble --------
def assemble_result(vendor, buyer, header, items):
    if header.get("subtotal") is not None and header.get("tax") is not None and header.get("total") is None:
        maybe = header["subtotal"] + header["tax"]
        header["total"] = round(maybe, 2)
    buyer_name, buyer_addr = (buyer if isinstance(buyer, tuple) else (buyer, None))
    return {
        "vendor": {"name": vendor, "address": None, "tax_id": None},
        "buyer": {"name": buyer_name, "address": buyer_addr, "tax_id": None},
        "invoice": {
            "number": header["number"],
            "date": header["date"],
            "due_date": header["due_date"],
            "po_number": header["po_number"],
            "payment_terms": header["payment_terms"],
            "currency": header["currency"],
            "subtotal": header["subtotal"],
            "tax": header["tax"],
            "total": header["total"],
        },
        "line_items": items,
        "metadata": {"ocr_engine": "doctr:default", "ner_model": "dslim/bert-base-NER"}
    }

# -------- Orchestrator --------
def extract(image_path: str) -> Dict[str,Any]:
    doc = DocumentFile.from_images(image_path)
    ocr = ocr_predictor(pretrained=True, detect_orientation=True)
    out = ocr(doc).export()
    lines = build_lines(out)
    text = "\n".join(l["text"] for l in lines)
    ents = ner_full_text(text)
    vendor = pick_supplier(lines, ents)
    buyer = pick_buyer(lines, ents)
    header = extract_header_fields(text, lines)
    items = extract_items(lines)
    return assemble_result(vendor, buyer, header, items)

if __name__ == "__main__":
    image_path = r"E:\bank-statement-ui\Bank statements\Categorize_Data\Invoice_Image\IMG0264_638882805886603164.jpg"
    res = extract(image_path)
    jprint(res)
