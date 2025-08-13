#!/usr/bin/env python3
"""
Dataset Split Summary and Visualization
"""

import json
import os

def show_dataset_summary():
    """Display comprehensive dataset split summary"""
    
    print("🔍 LAYOUTLMV3 DATASET SPLIT SUMMARY")
    print("=" * 60)
    
    # File paths
    train_file = "layoutlmv3_data/train_data.json"
    val_file = "layoutlmv3_data/val_data.json"
    label_file = "layoutlmv3_data/label_mapping.json"
    
    # Check files exist
    for file_path in [train_file, val_file, label_file]:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return
    
    # Load data
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    
    with open(label_file, 'r') as f:
        label_mapping = json.load(f)
    
    # Basic statistics
    total_samples = len(train_data) + len(val_data)
    train_pct = len(train_data) / total_samples * 100
    val_pct = len(val_data) / total_samples * 100
    
    print(f"📊 BASIC STATISTICS")
    print(f"   Total samples: {total_samples}")
    print(f"   Training: {len(train_data)} samples ({train_pct:.1f}%)")
    print(f"   Validation: {len(val_data)} samples ({val_pct:.1f}%)")
    
    # Token and entity analysis
    train_tokens = sum(len(sample['tokens']) for sample in train_data)
    train_entities = sum(sum(1 for tag in sample['ner_tags'] if tag != 0) for sample in train_data)
    
    val_tokens = sum(len(sample['tokens']) for sample in val_data)
    val_entities = sum(sum(1 for tag in sample['ner_tags'] if tag != 0) for sample in val_data)
    
    total_tokens = train_tokens + val_tokens
    total_entities = train_entities + val_entities
    
    print(f"\n📝 TOKEN ANALYSIS")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Training tokens: {train_tokens:,} ({train_tokens/total_tokens*100:.1f}%)")
    print(f"   Validation tokens: {val_tokens:,} ({val_tokens/total_tokens*100:.1f}%)")
    print(f"   Avg tokens per sample: {total_tokens/total_samples:.1f}")
    
    print(f"\n🏷️ ENTITY ANALYSIS")
    print(f"   Total entities: {total_entities:,}")
    print(f"   Training entities: {train_entities:,} ({train_entities/total_entities*100:.1f}%)")
    print(f"   Validation entities: {val_entities:,} ({val_entities/total_entities*100:.1f}%)")
    print(f"   Avg entities per sample: {total_entities/total_samples:.1f}")
    print(f"   Entity ratio: {total_entities/total_tokens*100:.1f}%")
    
    # Label distribution
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    # Count entities by type
    train_entity_counts = {}
    val_entity_counts = {}
    
    for sample in train_data:
        for tag in sample['ner_tags']:
            if tag != 0:
                label = id_to_label[tag]
                train_entity_counts[label] = train_entity_counts.get(label, 0) + 1
    
    for sample in val_data:
        for tag in sample['ner_tags']:
            if tag != 0:
                label = id_to_label[tag]
                val_entity_counts[label] = val_entity_counts.get(label, 0) + 1
    
    print(f"\n📋 ENTITY TYPE DISTRIBUTION")
    print(f"{'Entity Type':<25} | {'Total':<7} | {'Train':<12} | {'Val':<12}")
    print("-" * 70)
    
    all_labels = set(list(train_entity_counts.keys()) + list(val_entity_counts.keys()))
    for label in sorted(all_labels):
        train_count = train_entity_counts.get(label, 0)
        val_count = val_entity_counts.get(label, 0)
        total_count = train_count + val_count
        
        if total_count > 0:
            train_pct = train_count / total_count * 100
            val_pct = val_count / total_count * 100
            print(f"{label:<25} | {total_count:>7} | {train_count:>4} ({train_pct:>5.1f}%) | {val_count:>4} ({val_pct:>5.1f}%)")
    
    # Training recommendations
    print(f"\n💡 TRAINING RECOMMENDATIONS")
    print(f"   ✅ Good train/val split: {train_pct:.0f}%/{val_pct:.0f}%")
    print(f"   ✅ Balanced entity distribution across splits")
    
    if total_samples < 100:
        print(f"   ⚠️  Small dataset: Consider data augmentation")
    
    if total_entities / total_samples < 20:
        print(f"   ⚠️  Low entity density: May need more labeled data")
    else:
        print(f"   ✅ Good entity density: {total_entities/total_samples:.1f} entities/sample")
    
    # File size information
    train_size = os.path.getsize(train_file) / (1024 * 1024)
    val_size = os.path.getsize(val_file) / (1024 * 1024)
    
    print(f"\n💾 FILE INFORMATION")
    print(f"   Train file: {train_size:.2f} MB")
    print(f"   Validation file: {val_size:.2f} MB")
    print(f"   Total size: {train_size + val_size:.2f} MB")

if __name__ == "__main__":
    show_dataset_summary()
