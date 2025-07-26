import re
import xml.etree.ElementTree as ET
import csv
import os
import json

xml_file_path = r"xml/output_sbi_march.xml"
csv_output_path = r"csv/output_sbi_march.csv"
CONFIG_FILE = "bank_configs.json" 
date_patterns = [
    r'\b\d{1,2}[-/|]\d{1,2}[-/|]\d{2,4}\b',
    r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
    r'\b\d{1,2}[-/](?:[A-Za-z]{3,9})[-/]\d{2,4}\b',
    r'\b\d{1,2}\s+(?:[A-Za-z]{3,9})\s+\d{2,4}\b',
    r'\b(?:[A-Za-z]{3,9})\s+\d{2,4}\b',
    r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+TO\s+\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
    r'\b\d{1,2}\s+(?:[A-Za-z]{3})\b' 
]

def is_date(text):

    for pattern in date_patterns:
        if re.fullmatch(pattern, text.strip()):
            return pattern
    return None

def match_row(rows, top, bottom, threshold=5):

    for row in rows:
        if abs(row['top'] - top) <= threshold and abs(row['bottom'] - bottom) <= threshold:
            return row
    return None

vertical_threshold = 10 
tight_chain_tolerance = 6 

TOLERANCE_X = 10 # General horizontal tolerance for column range matching
TOLERANCE_Y = 15 # General vertical tolerance

ALIGNMENT_TOLERANCE_X = 8 # New: Tolerance for checking alignment with original header coordinates

def find_chained_header_segments(segments_on_page, target_header_name):

    target_header_upper = target_header_name.strip().upper()
    print(f"\n--- Debug: find_chained_header_segments: Searching for header: '{target_header_name}' (Upper: '{target_header_upper}') ---")
    
    sorted_segments = sorted(segments_on_page, key=lambda x: (x['top'], x['x0']))
    # 1. Try to find the full phrase as a single segment first (more robust check)
    for segment in sorted_segments:
        seg_text_raw = segment['text']
        print(f"--- Debug:   Examining segment: Text='{seg_text_raw}', ID={segment.get('id', 'N/A')}, Top={segment['top']:.2f}") # Added segment debug
        if seg_text_raw and seg_text_raw.strip().upper() == target_header_upper:
            print(f"--- Debug:   Found EXACT match for '{target_header_name}' as single segment: Text='{seg_text_raw}', ID={segment.get('id', 'N/A')}, x0={segment['x0']:.2f}, x1={segment['x1']:.2f}, top={segment['top']:.2f}, bottom={segment['bottom']:.2f}")
            return [segment]

    # 2. If full phrase not found, attempt to chain words for multi-line headers
    words = target_header_upper.split()
    if not words:
        print("--- Debug: find_chained_header_segments: No words to chain. ---")
        return None

    print(f"--- Debug: find_chained_header_segments: Attempting multi-word chaining for '{target_header_name}' ---")
    for i, current_segment in enumerate(sorted_segments):
        current_text_lower = current_segment['text'].lower()
        
        # Check if the first word (or part of it) is in the current segment
        if words[0].lower() in current_text_lower:
            potential_chain = [current_segment]
            current_word_index = 0

            # Check how many words from the target header are already in the current segment
            temp_seg_words = current_text_lower.split()
            for target_word in words:
                if target_word.lower() in temp_seg_words:
                    current_word_index += 1
                else:
                    break # Stop if a word is missing

            if current_word_index == len(words):
                 print(f"--- Debug: find_chained_header_segments: Found all words in initial segment: '{current_segment['text']}' ---")
                 return [current_segment]

            # Try to chain subsequent segments
            for next_segment_index in range(i + 1, len(sorted_segments)):
                if current_word_index >= len(words):
                    break # All words found

                next_segment = sorted_segments[next_segment_index]
                next_text_lower = next_segment['text'].lower()
                next_expected_word = words[current_word_index].lower()

                if next_expected_word in next_text_lower:
                    vertical_distance = next_segment['top'] - potential_chain[-1]['bottom']

                    # Check for horizontal alignment: either x0, x1 are close, or segments overlap horizontally
                    is_horizontally_aligned = (
                        abs(potential_chain[-1]['x0'] - next_segment['x0']) < TOLERANCE_X or
                        abs(potential_chain[-1]['x1'] - next_segment['x1']) < TOLERANCE_X or
                        (max(potential_chain[-1]['x0'], next_segment['x0']) < min(potential_chain[-1]['x1'], next_segment['x1']))
                    )

                    # Chain if vertically close and horizontally aligned
                    if (vertical_distance >= -TOLERANCE_Y and vertical_distance < TOLERANCE_Y and
                        is_horizontally_aligned):
                        potential_chain.append(next_segment)
                        current_word_index += 1
                        print(f"--- Debug: find_chained_header_segments:   Chained '{next_segment['text']}' (ID: {next_segment.get('id', 'N/A')}) for '{next_expected_word}'. Current chain: {[s['text'] for s in potential_chain]} ---")
                    elif next_segment['top'] > potential_chain[-1]['bottom'] + TOLERANCE_Y * 2:
                        # If the next segment is too far below, break the chain attempt
                        print(f"--- Debug: find_chained_header_segments:   Segment '{next_segment['text']}' too far vertically from '{potential_chain[-1]['text']}'. Breaking chain attempt. ---")
                        break
                elif next_segment['top'] > potential_chain[-1]['bottom'] + TOLERANCE_Y * 2:
                    # If the next segment is too far below, break the chain attempt
                    print(f"--- Debug: find_chained_header_segments:   Segment '{next_segment['text']}' too far vertically from '{potential_chain[-1]['text']}'. Breaking chain attempt. ---")
                    break

            if current_word_index == len(words):
                print(f"--- Debug: find_chained_header_segments: Successfully chained all words for '{target_header_name}': {[s['text'] for s in potential_chain]} ---")
                return potential_chain

    print(f"Debug: find_chained_header_segments: Header '{target_header_name}' NOT found after all attempts. ---")
    return None

# --- New: Bank name and config loading/saving ---
bank_name = input("Enter the bank name: ").strip()
bank_name_key = bank_name.replace(" ", "_").lower() # Create a clean key for JSON

bank_configs = {}
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            bank_configs = json.load(f)
        print(f"Loaded existing configurations from {CONFIG_FILE}")
    except json.JSONDecodeError:
        print(f"Error reading {CONFIG_FILE}. Starting with empty configurations.")
        bank_configs = {}

user_defined_headers_raw = []
output_date_column_name = ""

if bank_name_key in bank_configs:
    print(f"\nConfiguration found for '{bank_name}'. Using stored headers.")
    config = bank_configs[bank_name_key]
    output_date_column_name = config.get("date_column_name", "")
    user_defined_headers_raw = config.get("other_headers", [])
    
    # Ensure the date column name is always in user_defined_headers_raw for header scanning
    if output_date_column_name and output_date_column_name not in user_defined_headers_raw:
        user_defined_headers_raw.insert(0, output_date_column_name) # Add it at the beginning for consistency
    
    # Give option to re-enter
    re_enter = input(f"Do you want to re-enter headers for '{bank_name}'? (yes/no): ").strip().lower()
    if re_enter == 'yes':
        print("Please re-enter headers.")
        user_defined_headers_raw = [] # Clear existing to prompt for new
        output_date_column_name = ""
else:
    print(f"\nNo configuration found for '{bank_name}'. Please enter headers.")

# Prompt for headers if not loaded or re-entering
if not user_defined_headers_raw or not output_date_column_name:
    output_date_column_name = input("Enter the desired column name for DATE in the output CSV (e.g., DATE or Transaction Date): ").strip()
    print(f"--- Debug: User entered for Date Column Name: '{output_date_column_name}' ---") # NEW DEBUG
    
    # Ensure the date column name is always in user_defined_headers_raw for header scanning
    if output_date_column_name and output_date_column_name not in user_defined_headers_raw:
        user_defined_headers_raw.insert(0, output_date_column_name) # Add it at the beginning for consistency

    print("\nEnter additional column headers one by one. Type 'done' when finished.")
    while True:
        header_name = input("Enter column header name (or 'done'): ").strip()
        if header_name.lower() == 'done':
            break
        if header_name:
            if header_name not in user_defined_headers_raw:
                user_defined_headers_raw.append(header_name)
        else:
            print("Header name cannot be empty. Please try again.")

    bank_configs[bank_name_key] = {
        "date_column_name": output_date_column_name,
        "other_headers": [h for h in user_defined_headers_raw if h != output_date_column_name] # Store others
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(bank_configs, f, indent=4)
    print(f"Configuration for '{bank_name}' saved to {CONFIG_FILE}")


user_defined_headers_upper = [h.strip().upper() for h in user_defined_headers_raw if h.strip()]
user_defined_headers_upper = list(dict.fromkeys(user_defined_headers_upper))

output_date_column_name_upper = output_date_column_name.strip().upper()

print(f"\nUsing the following user-defined column headers for CSV output: {', '.join(user_defined_headers_raw)}")
print(f"Output Date Column (CSV): {output_date_column_name}")
print("-" * 50)

try:
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
except FileNotFoundError:
    print(f"Error: XML file not found at {xml_file_path}")
    exit()
except ET.ParseError:
    print(f"Error: Could not parse XML file at {xml_file_path}. Check if it's a valid XML.")
    exit()

column_ranges = [] # Stores the horizontal ranges (start, end) and names of columns
header_bottom = None # Stores the bottom coordinate of the header row on Page 1
all_rows = [] # Accumulates all extracted data rows
page1_header_names_upper = [] # Stores uppercase names of headers from Page 1 for subsequent page checks
unnamed_header_added = False # New: Flag to track if "Unnamed_Header" has been added

# Iterate through each page in the XML document
for page_idx, page in enumerate(root.findall('Page'), start=1):
    segments = page.findall('Segment')
    rows = [] # Rows for the current page

    # Variables to track date and row boundaries on the current page
    first_date_top_on_page = None
    last_date_bottom_on_page_raw = None 
    last_date_value_on_page = None 

    current_page_header_found_bottom = None # Stores the bottom of the header if found on the current page

    current_page_segments_data = []
    for s_elem in segments:
        text_elem = s_elem.find('Text')
        if text_elem is None or not text_elem.text:
            continue
        try:
            current_page_segments_data.append({
                'text': text_elem.text.strip(),
                'x0': float(s_elem.find('x0').text),
                'x1': float(s_elem.find('x1').text),
                'top': float(s_elem.find('top').text),
                'bottom': float(s_elem.find('bottom').text),
                'id': s_elem.get('id') # Capture segment ID for debugging
            })
        except (TypeError, ValueError):
            print(f"--- Debug: Skipping malformed segment in Page {page_idx}: {ET.tostring(s_elem).decode().strip()}")
            continue

    # --- Page 1: Header and Column Range Definition (DYNAMIC) ---
    # This block runs only for the first page to establish column boundaries dynamically.
    if page_idx == 1:
        print(f"\n--- Debug: Page 1 Header Scan ---")
        found_header_segments_page1 = []
        for user_header_name_raw in user_defined_headers_raw:
            chained_segments = find_chained_header_segments(current_page_segments_data, user_header_name_raw)
            if chained_segments:
                min_x0 = min(s['x0'] for s in chained_segments)
                max_x1 = max(s['x1'] for s in chained_segments)
                min_top = min(s['top'] for s in chained_segments)
                max_bottom = max(s['bottom'] for s in chained_segments)
                found_header_segments_page1.append({
                    'name': user_header_name_raw, # Use user-defined name
                    'x0': min_x0,
                    'x1': max_x1,
                    'top': min_top,
                    'bottom': max_bottom
                })
                print(f"--- Debug: Successfully found header '{user_header_name_raw}'. Coordinates: x0={min_x0:.2f}, x1={max_x1:.2f}, top={min_top:.2f}, bottom={max_bottom:.2f}")
            else:
                print(f"--- Debug: Header '{user_header_name_raw}' NOT found by find_chained_header_segments on Page 1.")


        # Sort found headers by their x0 to process them left-to-right
        found_header_segments_page1.sort(key=lambda s: s['x0'])

        column_ranges = []
        date_column_found_coords = None
        
        # Implement the new dynamic logic for column ranges based on user's guidance
        prev_header_x1 = None
        for i, h_segment in enumerate(found_header_segments_page1):
            col_name = h_segment['name']
            col_start = 0.0
            col_end = h_segment['x1']

            if i == 0: # First column: start slightly before its x0
                col_start = h_segment['x0'] - 5
            else: # Subsequent columns: start slightly before the previous header's x1
                if prev_header_x1 is not None:
                    col_start = prev_header_x1 - 5
                else: # Fallback if prev_header_x1 somehow not set (shouldn't happen with sorted list)
                    col_start = h_segment['x0'] - 5 
            
            column_ranges.append({
                'name': col_name,
                'start': col_start, # Buffered start for data extraction
                'end': col_end,     # Buffered end for data extraction
                'top': h_segment['top'], 
                'bottom': h_segment['bottom'],
                'original_x0': h_segment['x0'],   # New: Store original header x0
                'original_x1': h_segment['x1']    # New: Store original header x1
            })
            prev_header_x1 = h_segment['x1'] # Store x1 of current header for next iteration

            if col_name.strip().upper() == output_date_column_name_upper:
                date_column_found_coords = {'top': h_segment['top'], 'bottom': h_segment['bottom']}
                print(f"✅ Found Date Column Header '{col_name}' and set its coordinates: Top: {h_segment['top']:.2f}, Bottom: {h_segment['bottom']:.2f}")


        # Extend the last column's end significantly to capture any overflowing text
        if column_ranges:
            column_ranges[-1]['end'] += 100

        if date_column_found_coords is None:
            print(f"❌ Date column header '{output_date_column_name}' was NOT found/identified correctly during Page 1 header scan.")
            header_bottom_candidates = [h['bottom'] for h in column_ranges if h.get('bottom', -1) != -1]
            if header_bottom_candidates:
                header_bottom = max(header_bottom_candidates)
                print(f"--- Debug: Falling back to max header bottom: {header_bottom:.2f}")
            else:
                header_bottom = 0 # Fallback
                print(f"--- Debug: No header bottoms found, header_bottom defaulted to {header_bottom:.2f}")
        else:
            header_bottom = date_column_found_coords['bottom']
            print(f"--- Debug: header_bottom set from found date column header: {header_bottom:.2f}")


        page1_header_names_upper = [h['name'].strip().upper() for h in column_ranges]

        print("\n🧱 Column ranges from Page 1 (dynamically calculated):")
        for col in column_ranges:
            print(f"'{col['name']}': {col['start']:.1f} to {col['end']:.1f} (Original Header: {col.get('original_x0', 'N/A'):.1f} to {col.get('original_x1', 'N/A'):.1f})")
        print()

    current_column_ranges = column_ranges # All pages use the column ranges defined on Page 1

    # --- Check for header on subsequent pages and set data start point ---
    if page_idx > 1 and page1_header_names_upper: # Only check if page 1 headers were successfully found
        found_header_on_current_page = False
        potential_header_segments_on_current_page = []

        # Try to find all original headers on the current page to confirm a header row
        for user_header_name_raw in user_defined_headers_raw:
            found_chained_segments = find_chained_header_segments(current_page_segments_data, user_header_name_raw)
            if found_chained_segments:
                # Consolidate found segments to get their overall vertical span
                min_top = min(s['top'] for s in found_chained_segments)
                max_bottom = max(s['bottom'] for s in found_chained_segments)
                potential_header_segments_on_current_page.append({
                    'text': user_header_name_raw, # For debugging
                    'top': min_top,
                    'bottom': max_bottom
                })
        
        # Heuristic: Consider it a header row if at least 50% of the original headers are found
        # and they are roughly vertically aligned.
        if len(potential_header_segments_on_current_page) >= len(user_defined_headers_raw) * 0.5:
            tops = [s['top'] for s in potential_header_segments_on_current_page]
            if tops:
                min_top_headers_on_page = min(tops)
                max_top_headers_on_page = max(tops)
                # Check for vertical alignment within a small threshold (2 times vertical_threshold)
                # Corrected variable names here
                if (max_top_headers_on_page - min_top_headers_on_page) <= vertical_threshold * 2:
                    current_page_header_found_bottom = max(s['bottom'] for s in potential_header_segments_on_current_page)
                    found_header_on_current_page = True
                    print(f"✅ Page {page_idx} - Found matching header row. Data extraction will start below Y: {current_page_header_found_bottom:.1f}")
                else:
                    print(f"⚠️ Page {page_idx} - Matching header segments found but not vertically aligned enough ({max_top_headers_on_page - min_top_headers_on_page:.1f} > {vertical_threshold * 2}). Falling back to date pattern detection for data start.")
            else:
                print(f"⚠️ Page {page_idx} - No header segments found on this page to confirm header row. Falling back to date pattern detection for data start.")
        else:
            print(f"⚠️ Page {page_idx} - Not enough matching header segments found to confirm header row ({len(potential_header_segments_on_current_page)} < {len(user_defined_headers_raw) * 0.5}). Falling back to date pattern detection for data start.")

    # Determine the effective start of data extraction for the current page
    # Initialize data_start_y_coord to be just below the header row.
    data_start_y_coord = header_bottom 
    if page_idx > 1 and current_page_header_found_bottom is not None:
        data_start_y_coord = current_page_header_found_bottom 

    date_column_range = None
    for col in column_ranges:
        if col['name'].strip().upper() == output_date_column_name_upper:
            date_column_range = col
            break

    if date_column_range is None:
        print(f"Error: Date column range for '{output_date_column_name}' not established dynamically. Skipping data extraction for this page.")
        continue

    # Find first and last date on the page for data boundary
    # This loop now only considers segments that are below the determined data_start_y_coord
    for segment in current_page_segments_data:
        text = segment['text']
        seg_x0 = segment['x0']
        seg_top = segment['top']
        seg_bottom = segment['bottom']

        # ONLY consider segments that are below the header/data_start_y_coord
        if seg_top < data_start_y_coord:
            continue

        # Use the dynamically determined date column range for identifying dates
        is_in_date_column_area = (date_column_range['start'] - TOLERANCE_X) <= seg_x0 <= (date_column_range['end'] + TOLERANCE_X)

        if is_in_date_column_area:
            matched_pattern = is_date(text)

            if matched_pattern: # If the text is indeed a date
                if first_date_top_on_page is None:
                    first_date_top_on_page = seg_top
                    print(f"📄 Page {page_idx} - First Data Date Found: '{text}' (Pattern: '{matched_pattern}') (Top: {seg_top:.2f}, Bottom: {seg_bottom:.2f})")
                
                # Update last date information
                last_date_bottom_on_page_raw = seg_bottom
                last_date_value_on_page = text
    
    # AFTER finding the first date on the page, adjust data_start_y_coord
    if first_date_top_on_page is not None:
        # Ensure data extraction starts at or below the first date found.
        # This implicitly means segments with seg_top < first_date_top_on_page will be skipped.
        # We use max to ensure we are always below the header OR the first date, whichever is lower on the page.
        data_start_y_coord = max(data_start_y_coord, first_date_top_on_page)
        print(f"📄 Page {page_idx} - Final data_start_y_coord: {data_start_y_coord:.2f} (based on first date found or header bottom).")
    else:
        print(f"⚠️ Page {page_idx} - No date found on page below header. Data extraction will proceed using header bottom (Y: {data_start_y_coord:.2f}).")


    # --- Vertical Chaining Logic to determine true_last_row_bottom_on_page ---
    true_last_row_bottom_on_page = -1.0 # Initialize with a very small value.

    if last_date_value_on_page:
        print(f"📄 Page {page_idx} - Last Date Found (raw for chaining): {last_date_value_on_page} (bottom: {last_date_bottom_on_page_raw:.2f})")
        true_last_row_bottom_on_page = last_date_bottom_on_page_raw

        MAX_CHAIN_DISTANCE = 200  # Optional: prevent chaining too far down
        chained_bottoms = []

        for col in column_ranges:
            current_col_max_bottom = last_date_bottom_on_page_raw
            chained = False

            # Collect potential segments for chaining in this column
            # Ensure chaining also starts from or below the data_start_y_coord
            col_segments = [
                s for s in current_page_segments_data
                if (col['start'] - TOLERANCE_X) <= s['x0'] <= (col['end'] + TOLERANCE_X)
                and s['top'] >= data_start_y_coord 
            ]
            col_segments.sort(key=lambda s: s['top'])

            for seg in col_segments:
                seg_top = seg['top']
                seg_bottom = seg['bottom']

                if seg_top <= current_col_max_bottom + vertical_threshold:
                    print(f"🔗 Chaining '{seg['text']}' in column '{col['name']}' (top: {seg_top:.2f}, bottom: {seg_bottom:.2f})")
                    current_col_max_bottom = max(current_col_max_bottom, seg_bottom)
                    chained = True
                elif seg_top > current_col_max_bottom + vertical_threshold:
                    # Stop if segment is too far vertically
                    break

            if chained:
                chained_bottoms.append(current_col_max_bottom)

        if chained_bottoms:
            # Final chaining limit — also ensure it doesn't exceed MAX_CHAIN_DISTANCE
            # Added +5 buffer for more 'bottom' value
            true_last_row_bottom_on_page = min(max(chained_bottoms), last_date_bottom_on_page_raw + MAX_CHAIN_DISTANCE) + 5
        else:
            true_last_row_bottom_on_page = last_date_bottom_on_page_raw + 5 # Add buffer even if no chaining

        print(f"📄 Page {page_idx} - True bottom after vertical chaining (with bottom buffer): {true_last_row_bottom_on_page:.2f}")

    else:
        # If no date was found on the page, revert to a fallback
        true_last_row_bottom_on_page = float('inf') 
        print(f"⚠️ Page {page_idx} - No date found on page. Data will be extracted until the end of the page below the header.")


    # --- Main Data Extraction Loop ---
    # This loop processes all segments on the page to extract data into rows.
    for segment in current_page_segments_data:
        text = segment['text']
        seg_x0 = segment['x0']
        seg_x1 = segment['x1']
        seg_top = segment['top']
        seg_bottom = segment['bottom']
        seg_center = (seg_x0 + seg_x1) / 2

        # Apply top boundary filtering based on the calculated data_start_y_coord
        # Segments with `seg_top` less than `data_start_y_coord` are considered headers or irrelevant text.
        if seg_top < data_start_y_coord:
            continue

        # Filter out anything below the true last row bottom
        # This check is now based on the `true_last_row_bottom_on_page` which considers chaining from the last date.
        if true_last_row_bottom_on_page != float('inf') and seg_top > true_last_row_bottom_on_page:
             continue

        col_name = None
        assigned_col_info = None # New: To store the column definition for alignment check

        # Determine which column the segment belongs to based on its horizontal position
        for col_def in current_column_ranges:
            # Prioritize x0 matching
            if (col_def['start'] - TOLERANCE_X) <= seg_x0 <= (col_def['end'] + TOLERANCE_X):
                if col_def['name'].strip().upper() in user_defined_headers_upper:
                    col_name = col_def['name']
                    assigned_col_info = col_def # Store the full column definition
                    break
            # Fallback to center matching only if x0 didn't match directly
            elif (col_def['start'] - TOLERANCE_X) <= seg_center <= (col_def['end'] + TOLERANCE_X):
                if col_def['name'].strip().upper() in user_defined_headers_upper:
                    col_name = col_def['name']
                    assigned_col_info = col_def # Store the full column definition
                    break

        if not col_name:
            continue # Skip segment if it doesn't belong to any defined column

        # New Logic: Check if the segment's coordinates align with the ORIGINAL header coordinates
        if assigned_col_info and 'original_x0' in assigned_col_info and 'original_x1' in assigned_col_info:
            original_header_x0 = assigned_col_info['original_x0']
            original_header_x1 = assigned_col_info['original_x1']

            # Define "alignment" as the segment's x0 being close to or after header's original x0
            # AND segment's x1 being close to or before header's original x1.
            # This allows for segments that are slightly shifted but still conceptually within the header's bounds.
            is_aligned_with_original_header = (
                (seg_x0 >= original_header_x0 - ALIGNMENT_TOLERANCE_X) and
                (seg_x1 <= original_header_x1 + ALIGNMENT_TOLERANCE_X)
            )
            
            # Additional check: Ensure there's at least some overlap with the original header's width
            # (e.g., segment's x0 is not far past header's x1, and segment's x1 is not far before header's x0)
            has_overlap_with_original_header = (
                max(seg_x0, original_header_x0) < min(seg_x1, original_header_x1 + ALIGNMENT_TOLERANCE_X)
            )

            # If the segment is within the column's broad range but NOT aligned with its original header,
            # and also doesn't have significant overlap, assign it to "Unnamed_Header".
            if not is_aligned_with_original_header and not has_overlap_with_original_header:
                print(f"--- Debug: Segment '{text}' (x0:{seg_x0:.2f}, x1:{seg_x1:.2f}) assigned to '{col_name}' (buffered: {assigned_col_info['start']:.2f}-{assigned_col_info['end']:.2f}) but NOT aligned with original header ({original_header_x0:.2f}-{original_header_x1:.2f}). Reassigning to 'Unnamed_Header'. ---")
                col_name = "Unnamed_Header"
                if not unnamed_header_added:
                    user_defined_headers_raw.append("Unnamed_Header")
                    user_defined_headers_upper.append("UNNAMED_HEADER")
                    unnamed_header_added = True

        # Match segment to an existing row or create a new one
        row = match_row(rows, seg_top, seg_bottom)
        if not row:
            row = { 'top': seg_top, 'bottom': seg_bottom }
            # Initialize all columns for the new row based on user-defined headers
            for col_ud_name in user_defined_headers_raw:
                row[col_ud_name] = ''
            rows.append(row)

        actual_output_col_name = ""
        for udh_raw in user_defined_headers_raw:
            if udh_raw.strip().upper() == col_name.strip().upper():
                actual_output_col_name = udh_raw
                break

        if actual_output_col_name:
            # Ensure the column exists in the row before accessing
            if actual_output_col_name not in row:
                row[actual_output_col_name] = '' # Initialize if not present

            # Append segment text to the identified column in the row
            existing = row[actual_output_col_name].strip()
            new_value = f"{existing} {text}".strip() if existing else text
            row[actual_output_col_name] = new_value

    all_rows.extend(rows) # Add processed rows from current page to overall list

# --- CSV Output ---
if not all_rows:
    print("No data rows found.")
else:
    # Define fieldnames for the CSV, ensuring date column is first
    fieldnames_ordered = []
    if output_date_column_name in user_defined_headers_raw:
        fieldnames_ordered.append(output_date_column_name)

    for header in user_defined_headers_raw:
        if header not in fieldnames_ordered:
            fieldnames_ordered.append(header)

    # Ensure "Unnamed_Header" is at the end if it was added
    if "Unnamed_Header" in fieldnames_ordered:
        fieldnames_ordered.remove("Unnamed_Header")
        fieldnames_ordered.append("Unnamed_Header")

    # Write data to CSV file
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_ordered)
        writer.writeheader() # Write header row
        for row in all_rows:
            filtered_row = {}
            for field in fieldnames_ordered:
                filtered_row[field] = row.get(field, '')
            writer.writerow(filtered_row) # Write data row

    print(f" CSV file generated at {csv_output_path}")
