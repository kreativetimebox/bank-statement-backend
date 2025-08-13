from pdf2image import convert_from_path
import os

# Folder containing PDF files
pdf_folder = 'receipts_PDF'
output_folder = 'receipts images'
os.makedirs(output_folder, exist_ok=True)

# Loop through each file in the folder
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Processing: {filename}")
        
        # Convert all pages in the PDF
        images = convert_from_path(pdf_path, dpi=300)
        
        # Save each page as an image
        for i, image in enumerate(images):
            image_name = f"{os.path.splitext(filename)[0]}_page_{i+1}.png"
            image_path = os.path.join(output_folder, image_name)
            image.save(image_path, 'PNG')

print(f"✅ Finished converting PDFs in '{pdf_folder}' to images in '{output_folder}'")
