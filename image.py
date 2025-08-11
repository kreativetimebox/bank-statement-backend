from pdf2image import convert_from_path
import os

# Path to the PDF
pdf_path = 'C:\Kreativetime_box\invoices_to_image\Invoice_Scanned_PDF-20250807T065945Z-1-001\Invoice_Scanned_PDF\V067926-LEE010-INV 123573-APPROVED BY VELOU.pdf'

# Output folder to save images
output_folder = 'scaned_output_image'
os.makedirs(output_folder, exist_ok=True)

# Convert all pages
images = convert_from_path(pdf_path, dpi=300)

# Save each image
for i, image in enumerate(images):
    image_path = os.path.join(output_folder, f'page_{i+1}.png')
    image.save(image_path, 'PNG')

print(f"Saved {len(images)} pages as images in '{output_folder}'")