import ocrmypdf

input_pdf = "bankpdf/revolt_scan.pdf"
output_pdf = "revolt_ocr.pdf"

# Run OCR
ocrmypdf.ocr(input_pdf, output_pdf, deskew=True, rotate_pages=True)

print("OCR completed:", output_pdf)
