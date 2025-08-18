import json
from bs4 import BeautifulSoup

def process_table_data(table_data):
    rec_texts = table_data['table_ocr_pred']['rec_texts']
    rec_boxes = table_data['table_ocr_pred']['rec_boxes']
    pred_html = table_data['pred_html']

    # Parse pred_html to get non-empty <td> texts
    soup = BeautifulSoup(pred_html, 'html.parser')
    tds = [td for td in soup.find_all('td') if td.text.strip()]  # Keep td objects

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
    
    # Check if pred_html contains any colspan
    has_colspan = any(td.get('colspan') is not None for td in soup.find_all('td'))
    
    cell_idx = 0  # Global cell index counter
    for td in tds:
        td_text = td.text.strip()
        
        if has_colspan and td_text:  # Split all <td> if pred_html has any colspan
            components = find_components(td_text)
            if components:
                # Create a separate cell for each component
                for comp_idx in components:
                    used.add(comp_idx)
                    cell_result = {
                        'cell_index': cell_idx,
                        'text': rec_texts[comp_idx],
                        'merged': False,
                        'components': [comp_idx],
                        'box': rec_boxes[comp_idx]
                    }
                    results.append(cell_result)
                    cell_idx += 1
        else:
            # Original logic for non-colspan pred_html
            cell_result = {
                'cell_index': cell_idx,
                'text': td_text,
                'merged': False,
                'components': [],
                'box': [0, 0, 0, 0]
            }
            matching_indices = [j for j in range(len(rec_texts)) if j not in used and rec_texts[j] == td_text]
            if matching_indices:
                match_idx = min(matching_indices)  # Take the earliest unused
                cell_result['merged'] = False
                cell_result['components'] = [match_idx]
                cell_result['box'] = rec_boxes[match_idx]
                used.add(match_idx)
            else:
                # Merged: find components
                components = find_components(td_text)
                if components:
                    cell_result['merged'] = True
                    for c in components:
                        used.add(c)
                    min_x = min(rec_boxes[c][0] for c in components)
                    min_y = min(rec_boxes[c][1] for c in components)
                    max_x = max(rec_boxes[c][2] for c in components)
                    max_y = max(rec_boxes[c][3] for c in components)
                    cell_result['box'] = [min_x, min_y, max_x, max_y]
                    cell_result['components'] = components
            
            results.append(cell_result)
            cell_idx += 1
    
    # Add unmatched rec_texts as additional cells
    for i in range(len(rec_texts)):
        if i not in used:
            cell_result = {
                'cell_index': cell_idx,
                'text': rec_texts[i],
                'merged': False,
                'components': [i],
                'box': rec_boxes[i]
            }
            results.append(cell_result)
            used.add(i)
            cell_idx += 1
    
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