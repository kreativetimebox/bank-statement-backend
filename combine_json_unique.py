import json
import os
from pathlib import Path

def combine_json_files_with_unique_ids():
    """
    Combine all JSON files from the json folder into a single JSON file with unique IDs
    """
    json_folder = Path("json")
    output_file = "combined_invoices_unique_ids.json"
    
    combined_data = []
    current_id = 1
    
    # Get all JSON files and sort them
    json_files = sorted([f for f in json_folder.glob("*.json")])
    
    print(f"Found {len(json_files)} JSON files to combine:")
    for file in json_files:
        print(f"  - {file.name}")
    
    # Process each JSON file
    for json_file in json_files:
        try:
            print(f"Processing {json_file.name}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if data is a list or single object
            if isinstance(data, list):
                for item in data:
                    # Update the ID to ensure uniqueness
                    if isinstance(item, dict) and 'id' in item:
                        original_id = item['id']
                        item['id'] = current_id
                        # Also update inner_id if it exists and matches the original id
                        if 'inner_id' in item and item['inner_id'] == original_id:
                            item['inner_id'] = current_id
                        current_id += 1
                    combined_data.append(item)
                print(f"  Added {len(data)} items from {json_file.name}")
            else:
                # Single object
                if isinstance(data, dict) and 'id' in data:
                    original_id = data['id']
                    data['id'] = current_id
                    if 'inner_id' in data and data['inner_id'] == original_id:
                        data['inner_id'] = current_id
                    current_id += 1
                combined_data.append(data)
                print(f"  Added 1 item from {json_file.name}")
                
        except json.JSONDecodeError as e:
            print(f"  Error reading {json_file.name}: {e}")
        except Exception as e:
            print(f"  Unexpected error with {json_file.name}: {e}")
    
    # Write combined data to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully combined {len(combined_data)} total items into {output_file}")
        print(f"Output file size: {os.path.getsize(output_file):,} bytes")
        print(f"All items now have unique IDs from 1 to {current_id-1}")
        
        # Verify uniqueness
        if combined_data and isinstance(combined_data[0], dict) and 'id' in combined_data[0]:
            ids = [item.get('id') for item in combined_data if isinstance(item, dict)]
            unique_ids = set(ids)
            print(f"Verification: {len(ids)} total IDs, {len(unique_ids)} unique IDs")
            if len(ids) == len(unique_ids):
                print("✅ All IDs are unique!")
            else:
                print("⚠️ Warning: Some IDs are still duplicated!")
        
    except Exception as e:
        print(f"Error writing combined file: {e}")

if __name__ == "__main__":
    combine_json_files_with_unique_ids()
