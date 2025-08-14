import json
import os
import shutil
from pathlib import Path
import random

def split_dataset_for_donut():
    """
    Split the dataset into training (70%) and validation (30%) for Donut model training
    """
    
    # Paths
    json_file = "combined_invoices_unique_ids.json"
    images_folder = Path("images")
    
    # Create output directories
    train_dir = Path("dataset/train")
    val_dir = Path("dataset/validation")
    
    for split_dir in [train_dir, val_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "images").mkdir(exist_ok=True)
    
    print("🚀 Preparing Donut Dataset Split")
    print("=" * 50)
    
    # Load the combined JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"📊 Loaded {len(all_data)} annotation records")
    
    # Group records by image to ensure we split by image, not by annotation
    image_groups = {}
    for record in all_data:
        if 'data' in record and 'image' in record['data']:
            image_path = record['data']['image']
            image_name = image_path.split('/')[-1]
            
            if image_name not in image_groups:
                image_groups[image_name] = []
            image_groups[image_name].append(record)
    
    print(f"📁 Found {len(image_groups)} unique images")
    print(f"📋 Average annotations per image: {len(all_data) / len(image_groups):.1f}")
    
    # Split images (not individual records) into train/val
    image_names = list(image_groups.keys())
    random.seed(42)  # For reproducible results
    random.shuffle(image_names)
    
    # Calculate split indices
    train_size = int(len(image_names) * 0.7)
    train_images = image_names[:train_size]
    val_images = image_names[train_size:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   • Training images: {len(train_images)} ({len(train_images)/len(image_names)*100:.1f}%)")
    print(f"   • Validation images: {len(val_images)} ({len(val_images)/len(image_names)*100:.1f}%)")
    
    # Create training dataset
    train_records = []
    for image_name in train_images:
        train_records.extend(image_groups[image_name])
        # Copy image to train folder
        src_image = images_folder / image_name
        dst_image = train_dir / "images" / image_name
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
    
    # Create validation dataset
    val_records = []
    for image_name in val_images:
        val_records.extend(image_groups[image_name])
        # Copy image to validation folder
        src_image = images_folder / image_name
        dst_image = val_dir / "images" / image_name
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
    
    print(f"\n📝 Annotation Records:")
    print(f"   • Training records: {len(train_records)}")
    print(f"   • Validation records: {len(val_records)}")
    
    # Update image paths in the records to point to the new locations
    for record in train_records:
        if 'data' in record and 'image' in record['data']:
            image_name = record['data']['image'].split('/')[-1]
            record['data']['image'] = f"images/{image_name}"
    
    for record in val_records:
        if 'data' in record and 'image' in record['data']:
            image_name = record['data']['image'].split('/')[-1]
            record['data']['image'] = f"images/{image_name}"
    
    # Save the split datasets
    train_json = train_dir / "annotations.json"
    val_json = val_dir / "annotations.json"
    
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Files Created:")
    print(f"   • {train_json} ({os.path.getsize(train_json):,} bytes)")
    print(f"   • {val_json} ({os.path.getsize(val_json):,} bytes)")
    
    # Create dataset summary
    summary = {
        "dataset_info": {
            "total_images": len(image_names),
            "total_annotations": len(all_data),
            "train_images": len(train_images),
            "val_images": len(val_images),
            "train_annotations": len(train_records),
            "val_annotations": len(val_records)
        },
        "train_images": sorted(train_images),
        "val_images": sorted(val_images)
    }
    
    with open("dataset/dataset_split_info.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Dataset split completed successfully!")
    print(f"📁 Dataset structure created in 'dataset/' folder")
    
    return len(train_images), len(val_images), len(train_records), len(val_records)

if __name__ == "__main__":
    split_dataset_for_donut()
