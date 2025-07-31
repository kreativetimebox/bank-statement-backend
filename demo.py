# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
import faulthandler
faulthandler.enable()

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

import cv2

# Resize the image manually before OCR
input_image_path = "Data/receipts images/5098.png"
resized_image_path = "resized_input.jpg"

image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {input_image_path}")

resized_image = cv2.resize(image, (1280, 960))
cv2.imwrite(resized_image_path, resized_image)

# Run OCR inference on the resized image
result = ocr.predict(input=resized_image_path)

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")