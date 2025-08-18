import json
from bs4 import BeautifulSoup

# Load the JSON data (replace with actual loading from file or string)
# For example: with open('Punjab.json', 'r') as f: data = json.load(f)
# Here, assume 'data' is the list from <DOCUMENT>

def process_table_data(table_data):
    rec_texts = table_data['table_ocr_pred']['rec_texts']
    rec_boxes = table_data['table_ocr_pred']['rec_boxes']
    pred_html = table_data['pred_html']
    
    # Parse pred_html to get non-empty <td> texts
    soup = BeautifulSoup(pred_html, 'html.parser')
    tds = [td.text.strip() for td in soup.find_all('td') if td.text.strip()]
    
    used = set()
    results = []
    
    def find_components(td_text):
        def recurse(remaining, current_seq, start_i):
            if not remaining:
                return current_seq
            for cand_i in range(start_i, len(rec_texts)):
                if cand_i in used:
                    continue
                prefix = rec_texts[cand_i]
                if remaining.startswith(prefix):
                    new_remaining = remaining[len(prefix):].lstrip()  # remove leading spaces
                    new_seq = current_seq + [cand_i]
                    result = recurse(new_remaining, new_seq, cand_i + 1)
                    if result is not None:
                        return result
            return None
        
        return recurse(td_text, [], 0)
    
    for cell_idx, td_text in enumerate(tds):
        # First, try exact non-merged match to an unused rec_text
        matching_indices = [j for j in range(len(rec_texts)) if j not in used and rec_texts[j] == td_text]
        if matching_indices:
            match_idx = min(matching_indices)  # Take the earliest unused
            merged = False
            components = [match_idx]
            used.add(match_idx)
            box = rec_boxes[match_idx]
        else:
            # Merged: find components
            components = find_components(td_text)
            if components:
                merged = True
                for c in components:
                    used.add(c)
                min_x = min(rec_boxes[c][0] for c in components)
                min_y = min(rec_boxes[c][1] for c in components)
                max_x = max(rec_boxes[c][2] for c in components)
                max_y = max(rec_boxes[c][3] for c in components)
                box = [min_x, min_y, max_x, max_y]
            else:
                # If no match, placeholder
                merged = True
                components = []
                box = [0, 0, 0, 0]
        
        results.append({
            'cell_index': cell_idx,
            'text': td_text,
            'merged': merged,
            'components': components,
            'box': box
        })
    
    return results

# To process all tables in the JSON
with open("json/HDFC.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # Load from file

all_processed_data = []

for idx, page_data in enumerate(data):
    if 'table_res_list' in page_data and page_data['table_res_list']:
        results = process_table_data(page_data['table_res_list'][0])
        page_result = {
            "page_number": idx + 1,
            "cells": results
        }
        all_processed_data.append(page_result)
        print(f"### Processed Table for Page {idx+1}")
        print("| Cell Index | Text | Merged | Components | Box |")
        print("|------------|------|--------|------------|-----|")
        for r in results:
            print(f"| {r['cell_index']} | {r['text']} | {r['merged']} | {r['components']} | {r['box']} |")
        print()

# Save all processed data to JSON file
output_file = "processed_HDFC.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_processed_data, f, indent=2, ensure_ascii=False)

print(f"\nProcessed data saved to {output_file}")
