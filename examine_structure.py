import json
import os

def examine_json_structure():
    """
    Examine the structure of the JSON file to understand where image references are stored
    """
    json_file = "combined_invoices_unique_ids.json"
    
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found!")
        return
    
    try:
        # Read just a small portion to examine structure
        with open(json_file, 'r', encoding='utf-8') as f:
            # Read first 10000 characters to get structure without loading entire file
            sample = f.read(10000)
        
        print("🔍 Sample of JSON structure (first 10,000 characters):")
        print("=" * 60)
        print(sample)
        print("=" * 60)
        
        # Now load first few complete records
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📊 Total records: {len(data)}")
        
        # Examine first record structure
        if len(data) > 0:
            first_record = data[0]
            print(f"\n🔍 First record keys: {list(first_record.keys())}")
            
            # Look for fields that might contain image references
            for key, value in first_record.items():
                if isinstance(value, str) and ('image' in key.lower() or 'file' in key.lower() or 'data' in key.lower()):
                    print(f"\n📄 {key}: {value}")
                elif isinstance(value, dict):
                    print(f"\n📁 {key} (dict with keys): {list(value.keys())}")
                    # Check if any dict values contain image references
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and ('image' in sub_value.lower() or any(ext in sub_value.lower() for ext in ['.png', '.jpg', '.jpeg'])):
                            print(f"   🖼️  {sub_key}: {sub_value}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"\n📋 {key} (list with {len(value)} items)")
                    if isinstance(value[0], dict):
                        print(f"   First item keys: {list(value[0].keys()) if value[0] else 'empty'}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    examine_json_structure()
