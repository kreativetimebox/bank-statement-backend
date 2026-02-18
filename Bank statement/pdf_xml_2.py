import pdfplumber
import xml.sax.saxutils as saxutils
import fitz  # PyMuPDF for vertical line detection

def get_vertical_line_coordinates(pdf_path):
    """Detect vertical lines in PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    results = {}
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        vertical_segments = []
        
        # Extract drawings
        drawings = page.get_drawings()
        
        # Process each drawing to find vertical line segments
        for d in drawings:
            rect = d["rect"]
            width = abs(rect.x1 - rect.x0)
            height = abs(rect.y1 - rect.y0)
            
            # Identify vertical lines
            if width < 1.0 and height > 5:
                x_mid = (rect.x0 + rect.x1) / 2
                y_min = min(rect.y0, rect.y1)
                y_max = max(rect.y0, rect.y1)
                vertical_segments.append((x_mid, y_min, y_max))
        
        # Group segments by similar x-position
        groups = {}
        for seg in vertical_segments:
            x_mid, y_min, y_max = seg
            key = round(x_mid, 1)
            if key not in groups:
                groups[key] = {"y_mins": [], "y_maxs": []}
            groups[key]["y_mins"].append(y_min)
            groups[key]["y_maxs"].append(y_max)
        
        # Calculate full vertical span for each group
        group_spans = []
        for x, data in groups.items():
            y_min = min(data["y_mins"])
            y_max = max(data["y_maxs"])
            span = y_max - y_min
            group_spans.append((x, span, y_min, y_max))
        
        # Select top 7 longest vertical lines
        group_spans.sort(key=lambda x: x[1], reverse=True)
        selected = group_spans[:7] if len(group_spans) >= 7 else group_spans
        selected.sort(key=lambda x: x[0])  # Sort left-to-right
        
        # Format as (x, y_min, y_max)
        page_lines = []
        for x, span, y_min, y_max in selected:
            page_lines.append((x, y_min, y_max))
        
        results[page_num] = page_lines
    
    doc.close()
    return results

def write_xml_header(file):
    file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    file.write('<Segments>\n')

def write_xml_footer(file):
    file.write('</Segments>\n')

def write_segment(file, segment_id, segment):
    safe_text = saxutils.escape(segment['text'])
    file.write(f'    <Segment id="{segment_id}">\n')
    file.write(f'      <Text>{safe_text}</Text>\n')
    file.write(f'      <x0>{segment["x0"]}</x0>\n')
    file.write(f'      <x1>{segment["x1"]}</x1>\n')
    file.write(f'      <top>{segment["top"]}</top>\n')
    file.write(f'      <bottom>{segment["bottom"]}</bottom>\n')
    file.write(f'    </Segment>\n')

def group_words_by_line(words, line_threshold=3):
    """Group words into lines based on vertical position"""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w['top'], w['x0']))
    lines = []
    current_line = [words[0]]
    for i in range(1, len(words)):
        if abs(words[i]['top'] - current_line[-1]['top']) <= line_threshold:
            current_line.append(words[i])
        else:
            lines.append(current_line)
            current_line = [words[i]]
    lines.append(current_line)
    return lines

def create_horizontal_segments(line_words, gap_threshold=7):
    """Create horizontal segments within a line"""
    if not line_words:
        return []

    segments = []
    current_segment = [line_words[0]]

    for i in range(1, len(line_words)):
        prev_word = current_segment[-1]
        curr_word = line_words[i]

        is_same_segment = (
            curr_word['x0'] <= prev_word['x1'] + gap_threshold
            and abs(curr_word['top'] - prev_word['top']) <= 1
            and abs(curr_word['bottom'] - prev_word['bottom']) <= 1
        )

        if is_same_segment:
            current_segment.append(curr_word)
        else:
            segments.append(current_segment)
            current_segment = [curr_word]

    segments.append(current_segment)
    return segments

def segment_to_dict(segment_words):
    """Convert word list to segment dictionary"""
    text = ' '.join(w['text'] for w in segment_words)
    x0 = min(w['x0'] for w in segment_words)
    x1 = max(w['x1'] for w in segment_words)
    top = min(w['top'] for w in segment_words)
    bottom = max(w['bottom'] for w in segment_words)
    return {'text': text, 'x0': x0, 'x1': x1, 'top': top, 'bottom': bottom}

def process_page(page, vertical_lines):
    """Process a page with vertical line priority"""
    words = page.extract_words()
    if not words:
        return []
    
    # Get sorted vertical line x-coordinates
    vertical_x = sorted([line[0] for line in vertical_lines]) if vertical_lines else []
    
    # Group words into lines
    lines = group_words_by_line(words)
    
    all_segments = []
    
    for line in lines:
        # Create segments based on vertical lines
        vertical_segments = []
        if vertical_x:
            # Group words by vertical columns
            current_col = 0
            current_segment = []
            
            for word in line:
                # Find which column the word belongs to
                while current_col < len(vertical_x) and word['x0'] > vertical_x[current_col]:
                    # Close current segment if we have words
                    if current_segment:
                        vertical_segments.append(current_segment)
                        current_segment = []
                    current_col += 1
                
                # Add word to current segment
                current_segment.append(word)
            
            # Add the last segment
            if current_segment:
                vertical_segments.append(current_segment)
        else:
            vertical_segments = [line]
        
        # Process each vertical segment
        for segment_words in vertical_segments:
            # If between vertical lines, treat as single segment
            if vertical_x:
                seg_dict = segment_to_dict(segment_words)
                all_segments.append(seg_dict)
            else:
                # Use horizontal segmentation when no vertical lines
                horizontal_segments = create_horizontal_segments(segment_words)
                for seg in horizontal_segments:
                    seg_dict = segment_to_dict(seg)
                    all_segments.append(seg_dict)
    
    return all_segments

# Paths
pdf_path = r"pdf/Bank Statement Nov-Jan 2019.pdf"
output_xml_path = r"xml/output_pnb.xml"

# Precompute vertical lines for all pages
vertical_lines_dict = get_vertical_line_coordinates(pdf_path)

with pdfplumber.open(pdf_path) as pdf:
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        write_xml_header(f)
        segment_id = 1
        
        for page_number, page in enumerate(pdf.pages, start=1):
            f.write(f'  <Page id="{page_number}">\n')
            
            # Get vertical lines for this page (0-indexed)
            page_vertical = vertical_lines_dict.get(page_number - 1, [])
            
            # Process page with vertical line priority
            segments = process_page(page, page_vertical)
            
            # Write segments to XML
            for segment in segments:
                write_segment(f, segment_id, segment)
                segment_id += 1
            
            f.write(f'  </Page>\n')
        
        write_xml_footer(f)

print(f"✅ XML written to: {output_xml_path}")