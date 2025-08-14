#!/usr/bin/env python3
"""
Test script to check TrainingArguments parameters
"""

try:
    from transformers import TrainingArguments
    import inspect
    
    # Get the signature of TrainingArguments
    sig = inspect.signature(TrainingArguments.__init__)
    
    # Find evaluation-related parameters
    eval_params = [p for p in sig.parameters.keys() if 'eval' in p.lower()]
    print("Available evaluation parameters:", eval_params)
    
    # Test different parameter names
    test_params = [
        'eval_strategy',
        'evaluation_strategy', 
        'eval_steps',
        'eval_accumulation_steps'
    ]
    
    for param in test_params:
        if param in sig.parameters:
            print(f"✅ {param} is valid")
        else:
            print(f"❌ {param} is NOT valid")
    
    # Try to create TrainingArguments with different strategies
    try:
        args = TrainingArguments(
            output_dir="./test",
            eval_strategy="steps",
            num_train_epochs=1
        )
        print("✅ eval_strategy='steps' works")
    except Exception as e:
        print(f"❌ eval_strategy error: {e}")
        
        try:
            args = TrainingArguments(
                output_dir="./test",
                evaluation_strategy="steps",
                num_train_epochs=1
            )
            print("✅ evaluation_strategy='steps' works")
        except Exception as e2:
            print(f"❌ evaluation_strategy error: {e2}")

except Exception as e:
    print(f"Error importing transformers: {e}")
