import json
import os
from pathlib import Path

def combine_json_files():
    """
    Combine all JSON files from the json folder into a single JSON file
    """
    json_folder = Path("json")
    output_file = "combined_invoices.json"
    
    combined_data = []
    
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
                combined_data.extend(data)
                print(f"  Added {len(data)} items from {json_file.name}")
            else:
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
        
    except Exception as e:
        print(f"Error writing combined file: {e}")

if __name__ == "__main__":
    combine_json_files()
