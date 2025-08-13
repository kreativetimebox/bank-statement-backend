import json
import math  # for abs
import csv

# --- Load JSON ---
with open("Amex.json", "r", encoding="utf-8") as f:
    data = json.load(f)

table_data = data[0]["table_res_list"][0]

cell_box_list = table_data["cell_box_list"]
rec_texts = table_data["table_ocr_pred"]["rec_texts"]
rec_polys = table_data["table_ocr_pred"]["rec_polys"]

def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]

rec_bboxes = [poly_to_bbox(p) for p in rec_polys]

# Step 1: Get all headers first from user
headers = []
print("Enter all headers (enter 'q' when done):")
while True:
    header_name = input("Header name: ").strip()
    if header_name.lower() == "q":
        break
    if header_name:
        headers.append(header_name)

        header_text_bbox = None
        header_cell_box = None

        for idx, (text, bbox) in enumerate(zip(rec_texts, rec_bboxes)):
            if header_name.lower() in text.lower():
                header_text_bbox = bbox
                break

        if header_text_bbox:
            header_cell_box = min(
                cell_box_list,
                key=lambda cb: abs(cb[0] - header_text_bbox[0]) + abs(cb[1] - header_text_bbox[1])
            )
            print(f"Header '{header_name}' found:")
            print(f"  Text bbox coordinates: {header_text_bbox}")
            print(f"  Closest cell_box: {header_cell_box}")
        else:
            print(f"Header '{header_name}' not found in OCR texts.")

# Step 2: Find header text bbox and cell_box for each header
header_info_list = []
for header_name in headers:
    header_text_bbox = None
    header_cell_box = None

    for idx, (text, bbox) in enumerate(zip(rec_texts, rec_bboxes)):
        if header_name.lower() in text.lower():
            header_text_bbox = bbox
            break

    if not header_text_bbox:
        print(f"❌ Header text '{header_name}' not found.")
        continue

    header_cell_box = min(
        cell_box_list,
        key=lambda cb: abs(cb[0] - header_text_bbox[0]) + abs(cb[1] - header_text_bbox[1])
    )

    header_info_list.append({
        "name": header_name,
        "text_bbox": header_text_bbox,
        "cell_box": header_cell_box,
    })

if not header_info_list:
    print("No valid headers found, exiting.")
    exit()

header_info_list.sort(key=lambda h: h["text_bbox"][0])

matched_indexes = set()

# Step 3: Extract texts for each header column applying your full logic
for i, header_info in enumerate(header_info_list):
    name = header_info["name"]
    cb_left, cb_top, cb_right, cb_bottom = header_info["cell_box"]

    header_text_left_x = header_info["text_bbox"][0]
    header_text_right_x = header_info["text_bbox"][2]

    prev_header_text_right_x = None
    if i > 0:
        prev_header_text_right_x = header_info_list[i - 1]["text_bbox"][2]

    next_header_text_left_x = None
    if i + 1 < len(header_info_list):
        next_header_text_left_x = header_info_list[i + 1]["text_bbox"][0]

    col_texts = []

    for idx, (text, bbox) in enumerate(zip(rec_texts, rec_bboxes)):
        text_left_x, text_top_y, text_right_x, text_bottom_y = bbox

        crosses_cell_box = (text_left_x < cb_right) and (text_right_x > cb_left)

        if not crosses_cell_box:
            continue

        if i == 0:
            if next_header_text_left_x is None or text_right_x < next_header_text_left_x:
                col_texts.append((bbox, text))
                matched_indexes.add(idx)
        else:
            cond_prev = (prev_header_text_right_x is None or text_left_x > prev_header_text_right_x)
            cond_next = (next_header_text_left_x is None or text_right_x < next_header_text_left_x)

            if cond_prev and cond_next:
                col_texts.append((bbox, text))
                matched_indexes.add(idx)

    col_texts.sort(key=lambda b: b[0][1])

    print(f"\n📌 Column '{name}' (cell_box X-range: {cb_left} → {cb_right})")
    for bbox, text in col_texts:
        print(f"Coords: {bbox}, Text: {text}")

# Step 4: Print unassigned texts
unassigned = [
    (bbox, text)
    for idx, (text, bbox) in enumerate(zip(rec_texts, rec_bboxes))
    if idx not in matched_indexes
]

if unassigned:
    print("\n⚠️ Unassigned values:")
    for bbox, text in unassigned:
        print(f"Coords: {bbox}, Text: {text}")

# --- Step 5: Assign unassigned texts based on overlapping header TEXT bbox ranges AND vertical chaining ---

print("\n🔄 Assigning unassigned texts based on overlapping header TEXT bbox ranges and vertical chaining...")

header_text_xranges = {
    h["name"]: (h["text_bbox"][0], h["text_bbox"][2])
    for h in header_info_list
}

# Prepare new assignments: header_name -> list of (bbox, text)
new_assignments = {h["name"]: [] for h in header_info_list}

still_unassigned = []

# First pass: horizontal overlap with header text bbox ranges
for bbox, text in unassigned:
    text_left_x = bbox[0]
    text_right_x = bbox[2]
    assigned_headers = []
    for header_name, (hx_left, hx_right) in header_text_xranges.items():
        if text_left_x <= hx_right and text_right_x >= hx_left:
            new_assignments[header_name].append((bbox, text))
            assigned_headers.append(header_name)
    if not assigned_headers:
        still_unassigned.append((bbox, text))

# Vertical chaining for remaining unassigned with threshold=50
threshold = 50

# Build a map of all assigned bbox to their headers from Step 3 + Step 5
assigned_map = {}

# Add Step 3 assigned bboxes with headers (approximate using header_text_xranges)
for idx in matched_indexes:
    bbox = rec_bboxes[idx]
    text_left_x = bbox[0]
    text_right_x = bbox[2]
    assigned_headers = []
    for header_name, (hx_left, hx_right) in header_text_xranges.items():
        if text_left_x <= hx_right and text_right_x >= hx_left:
            assigned_headers.append(header_name)
    assigned_map[tuple(bbox)] = assigned_headers

# Add Step 5 new assignments
for header_name, items in new_assignments.items():
    for bbox, text in items:
        key = tuple(bbox)
        if key in assigned_map:
            assigned_map[key].append(header_name)
            # remove duplicates
            assigned_map[key] = list(set(assigned_map[key]))
        else:
            assigned_map[key] = [header_name]

# Repeat vertical chaining until no new assignments
still_unassigned_current = still_unassigned
iteration = 0

while still_unassigned_current:
    iteration += 1
    print(f"\n🔄 Vertical chaining iteration {iteration} - unassigned count: {len(still_unassigned_current)}")
    newly_assigned = []
    still_unassigned_next = []

    for u_bbox, u_text in still_unassigned_current:
        u_top = u_bbox[1]
        u_left = u_bbox[0]
        u_right = u_bbox[2]

        matched_headers = set()

        for a_bbox, a_headers in assigned_map.items():
            a_bottom = a_bbox[3]
            # Check vertical proximity within threshold
            if abs(u_top - a_bottom) <= threshold:
                a_left = a_bbox[0]
                a_right = a_bbox[2]
                # Check horizontal overlap or meet
                if not (u_right < a_left or u_left > a_right):
                    matched_headers.update(a_headers)

        if matched_headers:
            # Assign to matched headers
            for h in matched_headers:
                new_assignments[h].append((u_bbox, u_text))
            assigned_map[tuple(u_bbox)] = list(matched_headers)
            newly_assigned.append((u_bbox, u_text))
        else:
            still_unassigned_next.append((u_bbox, u_text))

    print(f"🟢 Newly assigned in this iteration: {len(newly_assigned)}")

    if not newly_assigned:
        break

    still_unassigned_current = still_unassigned_next

# Final reporting
for header_name, items in new_assignments.items():
    if items:
        print(f"\n🟢 Assigned under '{header_name}' header after all assignments:")
        for bbox, text in items:
            print(f"Coords: {bbox}, Text: {text}")

if still_unassigned_current:
    print("\n⚠️ Still unassigned texts after vertical chaining:")
    for bbox, text in still_unassigned_current:
        print(f"Coords: {bbox}, Text: {text}")
else:
    print("\n✅ All texts assigned after vertical chaining.")

# --- Step 6: Save all columns in CSV ---

# Combine all assigned texts for each header (Step 3 + Step 5)
all_columns = {}

for header_info in header_info_list:
    header = header_info["name"]
    combined_bbox_texts = []

    # From Step 3 (matched_indexes)
    for idx in matched_indexes:
        bbox = rec_bboxes[idx]
        text = rec_texts[idx]
        if header in assigned_map.get(tuple(bbox), []):
            combined_bbox_texts.append((bbox, text))

    # From Step 5 (new_assignments)
    combined_bbox_texts.extend(new_assignments.get(header, []))

    # Sort by bbox top coordinate (y) and keep all texts including duplicates
    combined_bbox_texts_sorted = sorted(combined_bbox_texts, key=lambda x: x[0][1])

    combined = [text for bbox, text in combined_bbox_texts_sorted]

    all_columns[header] = combined

# Add unassigned texts as an extra column "Unassigned"
all_columns["Unassigned"] = [text for bbox, text in still_unassigned_current] if still_unassigned_current else []

# Align rows by max length column
max_rows = max(len(texts) for texts in all_columns.values())

# Prepare rows for CSV
headers = list(all_columns.keys())
csv_rows = [headers]

for i in range(max_rows):
    row = []
    for header in headers:
        col = all_columns[header]
        row.append(col[i] if i < len(col) else "")
    csv_rows.append(row)

# Save to CSV
csv_filename = "extracted_table.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_rows)

print(f"\n✅ Saved extracted table data to '{csv_filename}'.")
