import json
import os

def verify_combined_file():
    """
    Verify the combined JSON file and provide a summary
    """
    combined_file = "combined_invoices.json"
    
    if not os.path.exists(combined_file):
        print(f"Error: {combined_file} not found!")
        return
    
    try:
        print(f"Loading {combined_file}...")
        with open(combined_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded {combined_file}")
        print(f"📊 Total number of items: {len(data)}")
        print(f"💾 File size: {os.path.getsize(combined_file):,} bytes ({os.path.getsize(combined_file)/1024/1024:.1f} MB)")
        
        # Check structure of first item
        if len(data) > 0:
            first_item = data[0]
            print(f"🔍 First item structure:")
            print(f"   - Type: {type(first_item)}")
            print(f"   - Keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dictionary'}")
            if isinstance(first_item, dict) and 'id' in first_item:
                print(f"   - First item ID: {first_item['id']}")
            
        # Check if all items have unique IDs
        if len(data) > 0 and isinstance(data[0], dict) and 'id' in data[0]:
            ids = [item.get('id') for item in data if isinstance(item, dict)]
            unique_ids = set(ids)
            print(f"🆔 Total IDs: {len(ids)}, Unique IDs: {len(unique_ids)}")
            if len(ids) != len(unique_ids):
                print("⚠️  Warning: Some IDs are duplicated!")
            else:
                print("✅ All IDs are unique")
                
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_combined_file()
