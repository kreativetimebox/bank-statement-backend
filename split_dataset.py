#!/usr/bin/env python3
"""
Dataset Splitter for LayoutLMv3 Training
Splits the training data into train/validation sets
"""

import json
import random
import os
from sklearn.model_selection import train_test_split

def split_dataset(data_file, output_dir, train_ratio=0.7, random_seed=42):
    """
    Split dataset into train and validation sets
    
    Args:
        data_file: Path to the training data JSON file
        output_dir: Directory to save split datasets
        train_ratio: Ratio for training set (0.7 = 70%)
        random_seed: Random seed for reproducible splits
    """
    print(f"📊 Splitting dataset: {data_file}")
    print(f"📈 Train ratio: {train_ratio*100:.0f}%")
    print(f"📈 Validation ratio: {(1-train_ratio)*100:.0f}%")
    print(f"🎲 Random seed: {random_seed}")
    
    # Load the dataset
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"📋 Total samples: {len(data)}")
    
    # Set random seed for reproducible results
    random.seed(random_seed)
    
    # Calculate split sizes
    train_size = int(len(data) * train_ratio)
    val_size = len(data) - train_size
    
    print(f"📊 Train samples: {train_size}")
    print(f"📊 Validation samples: {val_size}")
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Split the data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train set
    train_file = os.path.join(output_dir, 'train_data.json')
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation set
    val_file = os.path.join(output_dir, 'val_data.json')
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✅ Train data saved to: {train_file}")
    print(f"✅ Validation data saved to: {val_file}")
    
    # Analyze the splits
    print(f"\n📊 SPLIT ANALYSIS")
    print("=" * 50)
    
    # Analyze train set
    train_tokens = sum(len(sample['tokens']) for sample in train_data)
    train_entities = sum(sum(1 for tag in sample['ner_tags'] if tag != 0) for sample in train_data)
    
    print(f"📈 TRAIN SET:")
    print(f"   Samples: {len(train_data)}")
    print(f"   Total tokens: {train_tokens:,}")
    print(f"   Total entities: {train_entities:,}")
    print(f"   Avg tokens/sample: {train_tokens/len(train_data):.1f}")
    print(f"   Avg entities/sample: {train_entities/len(train_data):.1f}")
    print(f"   Entity ratio: {train_entities/train_tokens*100:.1f}%")
    
    # Analyze validation set
    val_tokens = sum(len(sample['tokens']) for sample in val_data)
    val_entities = sum(sum(1 for tag in sample['ner_tags'] if tag != 0) for sample in val_data)
    
    print(f"\n📈 VALIDATION SET:")
    print(f"   Samples: {len(val_data)}")
    print(f"   Total tokens: {val_tokens:,}")
    print(f"   Total entities: {val_entities:,}")
    print(f"   Avg tokens/sample: {val_tokens/len(val_data):.1f}")
    print(f"   Avg entities/sample: {val_entities/len(val_data):.1f}")
    print(f"   Entity ratio: {val_entities/val_tokens*100:.1f}%")
    
    # Entity distribution analysis
    print(f"\n🏷️ ENTITY DISTRIBUTION")
    print("=" * 30)
    
    # Load label mapping
    label_file = os.path.join(os.path.dirname(data_file), 'label_mapping.json')
    with open(label_file, 'r') as f:
        label_mapping = json.load(f)
    
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    # Count entities in train set
    train_entity_counts = {}
    for sample in train_data:
        for tag in sample['ner_tags']:
            if tag != 0:
                label = id_to_label[tag]
                train_entity_counts[label] = train_entity_counts.get(label, 0) + 1
    
    # Count entities in validation set
    val_entity_counts = {}
    for sample in val_data:
        for tag in sample['ner_tags']:
            if tag != 0:
                label = id_to_label[tag]
                val_entity_counts[label] = val_entity_counts.get(label, 0) + 1
    
    # Print entity distribution
    all_labels = set(list(train_entity_counts.keys()) + list(val_entity_counts.keys()))
    for label in sorted(all_labels):
        train_count = train_entity_counts.get(label, 0)
        val_count = val_entity_counts.get(label, 0)
        total_count = train_count + val_count
        if total_count > 0:
            train_pct = train_count / total_count * 100
            val_pct = val_count / total_count * 100
            print(f"   {label:<20} | Train: {train_count:>3} ({train_pct:>5.1f}%) | Val: {val_count:>3} ({val_pct:>5.1f}%)")
    
    return train_data, val_data

def main():
    # Configuration
    data_file = "layoutlmv3_data/layoutlmv3_training_data.json"
    output_dir = "layoutlmv3_data"
    train_ratio = 0.7
    random_seed = 42
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file not found: {data_file}")
        return
    
    # Split the dataset
    try:
        train_data, val_data = split_dataset(data_file, output_dir, train_ratio, random_seed)
        print(f"\n✅ Dataset split completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during dataset split: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
