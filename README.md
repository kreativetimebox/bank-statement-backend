This document explains why the current Named Entity Recognition (NER)–assisted invoice extraction is failing on some files and returning null values, what we’ve verified, and the immediate remediations and next steps.

The pipeline sometimes returns nulls for vendor/buyer and header fields because the upstream OCR-to-text structuring is inconsistent across invoices, leaving the NER and regex stages with little or no usable context.

Even when OCR returns text, generic NER is not layout-aware and can miss or mis-rank organization names on noisy/scanned receipts, leading to empty or incorrect parties.

As a result, key fields (vendor/buyer, invoice number/date, totals) can be null when anchors aren’t found and NER fails to supply reliable candidates.

What’s failing
Null/empty vendor and buyer names on receipts and mixed-layout invoices.

Header fields remain null where labels differ (e.g., “Ref No.”, “Amount Payable”) or where the pipeline can’t find text lines due to OCR structuring differences.

On some images, items/lines parse poorly; however, this README focuses on the NER-related party/header failures.

What is working
On clean, templated invoices with standard labels, regex recovers invoice_number/date/total.

On clear layouts, NER contributes helpful ORG tags to pick the supplier, and anchors retrieve buyer.

This images contains the ouput of this code using Doctr , ner , regex the most values we are getting in structured json format are null 
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2936fe2d-9b95-4691-9d60-ec2bf6b1e789" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/05833eb4-62af-4a7d-8640-7aaceec72863" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e8816481-3c26-4861-afb4-fa6bb5b39ead" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9c728ceb-5699-43bc-b07a-9ec6c5b2a5f2" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b97ec8d0-80f7-42a5-bc2b-6c42a322f9d9" />



This ner model can only give us small output which is also not correct around 30-40 percent time when used with regex 
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c35ab500-1475-4f3e-9bcc-5b17c4ecd306" />



I am adding both the files one with regex(ner_invoice1.py) and one without regex(ner_invoice.py)
