"""
Quick test script to verify LayoutLMv3 setup before full training
"""

import json
import torch
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification

def test_setup():
    print("🧪 Testing LayoutLMv3 Setup...")
    
    # Test 1: Check data loading
    print("\n1️⃣ Testing data loading...")
    try:
        with open('layoutlmv3_data/layoutlmv3_training_data.json', 'r') as f:
            data = json.load(f)
        
        with open('layoutlmv3_data/label_mapping.json', 'r') as f:
            labels = json.load(f)
            
        print(f"✅ Data loaded: {len(data)} samples, {len(labels['label_to_id'])} labels")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test 2: Test model loading
    print("\n2️⃣ Testing model loading...")
    try:
        MODEL_NAME = "microsoft/layoutlmv3-base"
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(MODEL_NAME)
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(labels['label_to_id'])
        )
        print(f"✅ Model loaded: {MODEL_NAME}")
        print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 3: Test tokenization
    print("\n3️⃣ Testing tokenization...")
    try:
        sample = data[0]
        tokens = sample['tokens'][:10]  # First 10 tokens
        bboxes = sample['bboxes'][:10]
        ner_tags = sample['ner_tags'][:10]
        
        encoding = tokenizer(
            tokens,
            boxes=bboxes,
            word_labels=ner_tags,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        print(f"✅ Tokenization successful")
        print(f"   - Input shape: {encoding['input_ids'].shape}")
        print(f"   - Bbox shape: {encoding['bbox'].shape}")
        print(f"   - Labels shape: {encoding['labels'].shape}")
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False
    
    # Test 4: Test forward pass
    print("\n4️⃣ Testing model forward pass...")
    try:
        with torch.no_grad():
            outputs = model(**encoding)
            loss = outputs.loss
            logits = outputs.logits
            
        print(f"✅ Forward pass successful")
        print(f"   - Loss: {loss.item():.4f}")
        print(f"   - Logits shape: {logits.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Ready for training.")
    return True

if __name__ == "__main__":
    success = test_setup()
    if success:
        print("\n▶️  You can now run: python train_layoutlmv3.py")
    else:
        print("\n⚠️  Please fix the issues above before training.")
