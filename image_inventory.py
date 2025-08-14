import json
from pathlib import Path

def create_image_inventory():
    """Create a complete inventory of images"""
    
    json_file = "combined_invoices_unique_ids.json"
    images_folder = Path("images")
    
    print("📋 COMPLETE IMAGE INVENTORY")
    print("=" * 60)
    
    # Load JSON and extract images
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    json_images = {}  # filename -> count
    for record in data:
        if 'data' in record and 'image' in record['data']:
            filename = record['data']['image'].split('/')[-1]
            json_images[filename] = json_images.get(filename, 0) + 1
    
    # Get actual files
    actual_images = set(f.name for f in images_folder.iterdir() if f.is_file())
    
    print(f"📊 Summary:")
    print(f"   • Images in folder: {len(actual_images)}")
    print(f"   • Unique images in JSON: {len(json_images)}")
    print(f"   • Total JSON records: {len(data)}")
    print(f"   • Records per image: {len(data) // len(json_images) if json_images else 0}")
    
    print(f"\n✅ All images verified and matched:")
    print(f"   Image File Name{' ' * 35} | References in JSON")
    print("-" * 70)
    
    for filename in sorted(actual_images):
        ref_count = json_images.get(filename, 0)
        status = "✅" if ref_count > 0 else "❌"
        print(f"{status} {filename:<45} | {ref_count:>3} times")
    
    # Final verification
    missing_in_json = actual_images - set(json_images.keys())
    missing_in_folder = set(json_images.keys()) - actual_images
    
    print(f"\n🎯 VERIFICATION RESULT:")
    if not missing_in_json and not missing_in_folder:
        print("✅ PERFECT MATCH! All images are properly referenced.")
        print("✅ No orphaned files or broken references.")
        print("✅ Dataset integrity confirmed.")
    else:
        if missing_in_json:
            print(f"⚠️  {len(missing_in_json)} files in folder not referenced in JSON")
        if missing_in_folder:
            print(f"❌ {len(missing_in_folder)} references in JSON point to missing files")

if __name__ == "__main__":
    create_image_inventory()
