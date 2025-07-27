import pdfplumber
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Ask user for expected number of lines
expected_h = int(input("Enter expected number of horizontal lines: "))  # e.g., 14
expected_v = int(input("Enter expected number of vertical lines: "))    # e.g., 5

# Helper: Group close positions
def group_positions(pos_list, threshold=5):
    grouped = []
    for p in sorted(pos_list):
        if not grouped or abs(p - grouped[-1]) > threshold:
            grouped.append(p)
    return grouped

# PDF file
pdf_path = "pdf/SBI-March.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages):
        print(f"\n--- Page {page_num + 1} ---")

        # Convert to grayscale image
        pil_img = page.to_image(resolution=200).original.convert("L")
        img_np_full = np.array(pil_img)

        # Crop to lower part (table zone)
        h, w = img_np_full.shape
        crop_y_start = int(h * 0.4)
        cropped_img = img_np_full[crop_y_start:, :]
        y_offset = crop_y_start

        # Threshold to binary
        _, binary = cv2.threshold(cropped_img, 180, 255, cv2.THRESH_BINARY_INV)

        # Morphology to detect lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # Get contours
        contours_h, _ = cv2.findContours(h_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_v, _ = cv2.findContours(v_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get positions
        h_positions = sorted([cv2.boundingRect(c)[1] for c in contours_h])
        v_positions = sorted([cv2.boundingRect(c)[0] for c in contours_v])

        # Group closely spaced lines
        grouped_h = group_positions(h_positions)
        grouped_v = group_positions(v_positions)

        # Select closest to expected count
        final_h = grouped_h[:expected_h]
        final_v = grouped_v[:expected_v]

        print(f"  Using {len(final_h)} horizontal lines")
        print(f"  Using {len(final_v)} vertical lines")

        # Draw on full image
        vis_img = cv2.cvtColor(img_np_full, cv2.COLOR_GRAY2BGR)

        for y in final_h:
            y_adj = y + y_offset
            cv2.line(vis_img, (0, y_adj), (w, y_adj), (0, 255, 0), 1)

        for x in final_v:
            cv2.line(vis_img, (x, y_offset), (x, h), (255, 0, 0), 1)

        # Show result
        plt.figure(figsize=(10, 12))
        plt.imshow(vis_img)
        plt.title(f"User-Guided Lines on Page {page_num + 1}")
        plt.axis("off")
        plt.show()
