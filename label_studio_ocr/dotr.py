#Requirements
#pip install python-doctr[torch]


import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image, ImageDraw, ImageOps

# === 1. Load image and model ===
image_path = "your_image.jpg"  # Replace with your image
model = ocr_predictor(pretrained=True)

# === 2. Load and analyze the image ===
doc = DocumentFile.from_images(image_path)
result = model(doc)
page = result.pages[0]

# === 3. Open and correct orientation ===
image = Image.open(image_path)
image = ImageOps.exif_transpose(image)
draw = ImageDraw.Draw(image)
width, height = image.size

# === 4. Draw boxes + Print words with confidence ===
print("Detected Text with Confidence Scores:\n")

for block in page.blocks:
    for line in block.lines:
        for word in line.words:
            (x_min, y_min), (x_max, y_max) = word.geometry
            box = (
                x_min * width,
                y_min * height,
                x_max * width,
                y_max * height
            )

            # Draw bounding box and label
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 10), word.value, fill="blue")

            # Print text and confidence
            print(f"Text: '{word.value}'  |  Confidence: {word.confidence:.2f}")

# === 5. Show the image with boxes ===
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.show()
