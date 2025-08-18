import json
import csv
from collections import defaultdict

def ranges_overlap(range1, range2):
    """Check if two ranges [start, end] overlap"""
    start1, end1 = range1
    start2, end2 = range2
    return not (end1 < start2 or end2 < start1)

def find_columns(cells):
    """Group cells into columns based on horizontal overlap"""
    columns = []
    
    for cell in cells:
        left, top, right, bottom = cell['box']
        cell_h_range = (left, right)
        
        # Find which column this cell belongs to
        assigned = False
        for col_idx, column in enumerate(columns):
            for existing_cell in column:
                existing_left, _, existing_right, _ = existing_cell['box']
                existing_h_range = (existing_left, existing_right)
                
                if ranges_overlap(cell_h_range, existing_h_range):
                    columns[col_idx].append(cell)
                    assigned = True
                    break
            if assigned:
                break
        
        if not assigned:
            columns.append([cell])
    
    # Sort columns by their leftmost position
    columns.sort(key=lambda col: min(cell['box'][0] for cell in col))
    
    return columns

def find_rows(cells):
    """Group cells into rows based on vertical overlap"""
    rows = []
    
    for cell in cells:
        left, top, right, bottom = cell['box']
        cell_v_range = (top, bottom)
        
        assigned = False
        for row_idx, row in enumerate(rows):
            for existing_cell in row:
                _, existing_top, _, existing_bottom = existing_cell['box']
                existing_v_range = (existing_top, existing_bottom)
                
                if ranges_overlap(cell_v_range, existing_v_range):
                    rows[row_idx].append(cell)
                    assigned = True
                    break
            if assigned:
                break
        
        if not assigned:
            rows.append([cell])
    
    # Sort rows by their topmost position
    rows.sort(key=lambda row: min(cell['box'][1] for cell in row))
    
    return rows

def create_table_structure(cells):
    """Create a structured table from cells"""
    columns = find_columns(cells)
    rows = find_rows(cells)

    print(f"Found {len(columns)} columns and {len(rows)} rows")

    cell_to_col = {}
    cell_to_row = {}

    for col_idx, column in enumerate(columns):
        for cell in column:
            cell_key = (cell['cell_index'], tuple(cell['box']))
            cell_to_col[cell_key] = col_idx

    for row_idx, row in enumerate(rows):
        for cell in row:
            cell_key = (cell['cell_index'], tuple(cell['box']))
            cell_to_row[cell_key] = row_idx

    # Change the default from `list` to `set` to automatically handle duplicates
    table = defaultdict(lambda: defaultdict(set))

    # Sort cells by cell_index for sequential processing
    sorted_cells = sorted(cells, key=lambda x: x['cell_index'])
    
    print("\n=== CELL PLACEMENT DEBUG ===")
    for cell in sorted_cells:
        cell_key = (cell['cell_index'], tuple(cell['box']))
        if cell_key in cell_to_col and cell_key in cell_to_row:
            col_idx = cell_to_col[cell_key]
            row_idx = cell_to_row[cell_key]
            
            # Use .add() for sets instead of .append() for lists
            table[row_idx][col_idx].add(cell['text'])
            print(f"Cell {cell['cell_index']:2d}: '{cell['text'][:40]:40s}' -> Row {row_idx}, Col {col_idx}")

    return table, len(columns), len(rows)

def save_to_csv(table, num_cols, num_rows, filename):
    """Save the structured table to CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for row_idx in range(num_rows):
            row_data = []
            for col_idx in range(num_cols):
                cell_texts = table[row_idx].get(col_idx, set()) # Use an empty set as default
                if isinstance(cell_texts, set):
                    # Convert set to a list before joining
                    row_data.append(" | ".join(list(cell_texts)))
                else:
                    # This case should not be reached with the fix, but kept for robustness
                    row_data.append(cell_texts)
            writer.writerow(row_data)

def process_page(page_data, page_num):
    """Process a single page"""
    print(f"\n=== Processing Page {page_num} ===")
    cells = page_data['cells']
    
    table, num_cols, num_rows = create_table_structure(cells)
    
    csv_filename = f"structured_table_page_{page_num}.csv"
    save_to_csv(table, num_cols, num_rows, csv_filename)
    
    print(f"Saved structured table to {csv_filename}")
    print(f"Table dimensions: {num_rows} rows × {num_cols} columns")
    
    return table, num_cols, num_rows

def main():
    """Main function to process all pages"""
    # Load your JSON (adjust filename if needed)
    with open('final_with_boxes_starling.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_tables = []
    # Sort pages by key (e.g., Page_1, Page_2, ...)
    sorted_pages = sorted(data.keys(), key=lambda k: int(k.split('_')[1]))
    
    for page_key in sorted_pages:
        page_num = int(page_key.split('_')[1])
        raw_cells = data[page_key]
        
        # Convert to the expected cell format
        cells = []
        for i, item in enumerate(raw_cells):
            cells.append({
                'cell_index': i,
                'text': item['text'],
                'box': item['rec_box']
            })
        
        # Now process as a page_data dict
        page_data = {'cells': cells, 'page_number': page_num}
        table, num_cols, num_rows = process_page(page_data, page_num)
        all_tables.append({
            'page': page_num,
            'table': table,
            'cols': num_cols,
            'rows': num_rows
        })
    
    # Combined CSV across all pages
    print(f"\n=== Creating Combined Table ===")
    combined_table = defaultdict(lambda: defaultdict(list))
    max_cols = 0
    current_row = 0
    
    for table_info in all_tables:
        table = table_info['table']
        num_rows = table_info['rows']
        max_cols = max(max_cols, table_info['cols'])
        
        if current_row > 0:
            combined_table[current_row][0].append(f"=== PAGE {table_info['page']} ===")
            current_row += 1
        
        for row_idx in range(num_rows):
            for col_idx in range(table_info['cols']):
                if col_idx in table[row_idx]:
                    combined_table[current_row][col_idx].extend(table[row_idx][col_idx])
            current_row += 1
    
    combined_filename = "structured_table_all_pages.csv"
    save_to_csv(combined_table, max_cols, current_row, combined_filename)
    print(f"Saved combined table to {combined_filename}")
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(all_tables)} pages")
    print(f"Individual page CSV files created: structured_table_page_1.csv to structured_table_page_{len(all_tables)}.csv")
    print(f"Combined CSV file created: {combined_filename}")

if __name__ == "__main__":
    main()
    