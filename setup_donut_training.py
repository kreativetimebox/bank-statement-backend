"""
Donut Model Training Configuration for Invoice Processing
"""

import json
import os
from pathlib import Path

# Training configuration for Donut model
DONUT_TRAINING_CONFIG = {
    "model_name_or_path": "naver-clova-ix/donut-base",
    "dataset_name_or_path": "./dataset",
    "output_dir": "./donut_invoice_model",
    "task_name": "invoice_parsing",
    
    # Training parameters
    "num_train_epochs": 30,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-5,
    "warmup_steps": 300,
    "max_length": 1024,
    "image_size": [1280, 960],
    
    # Optimization
    "fp16": True,
    "dataloader_num_workers": 4,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "save_total_limit": 3,
    "evaluation_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    # Early stopping
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.001,
    
    # Regularization
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Special tokens for invoice parsing
    "task_start_token": "<s_invoice>",
    "task_end_token": "</s_invoice>",
    
    # Resume from checkpoint if available
    "resume_from_checkpoint": None,
    
    # Seed for reproducibility
    "seed": 42,
    
    # Push to hub (set to False for local training)
    "push_to_hub": False,
    
    # Preprocessing
    "ignore_mismatched_sizes": True,
    "remove_unused_columns": False,
}

def create_training_script():
    """
    Create the main training script for Donut model
    """
    
    training_script = '''#!/usr/bin/env python3
"""
Donut Model Training Script for Invoice Processing
Based on Hugging Face Transformers and Donut implementation
"""

import os
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from PIL import Image
import datasets
from datasets import Dataset as HFDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceDataset(Dataset):
    """
    Custom dataset for invoice images and annotations
    """
    
    def __init__(self, dataset_path, processor, split="train", max_length=1024):
        self.dataset_path = Path(dataset_path) / split
        self.processor = processor
        self.max_length = max_length
        
        # Load metadata
        metadata_file = self.dataset_path / "metadata.jsonl"
        self.data = []
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        logger.info(f"Loaded {len(self.data)} samples for {split} set")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = self.dataset_path / "images" / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        
        # Prepare target sequence
        target_sequence = f"<s_invoice>{item['ground_truth']}</s_invoice>"
        
        # Process with Donut processor
        processed = self.processor(
            image, 
            text=target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": processed["pixel_values"].squeeze(),
            "labels": processed["input_ids"].squeeze(),
            "attention_mask": processed["attention_mask"].squeeze()
        }

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_pred
    
    # For now, we'll use perplexity as the main metric
    # You can add more sophisticated metrics like exact match, F1, etc.
    
    return {"perplexity": torch.exp(torch.tensor(predictions.mean())).item()}

def main():
    """
    Main training function
    """
    
    # Load configuration
    config_file = Path("donut_training_config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Use default config
        config = {
            "model_name_or_path": "naver-clova-ix/donut-base",
            "dataset_name_or_path": "./dataset",
            "output_dir": "./donut_invoice_model",
            "num_train_epochs": 30,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 4,
            "learning_rate": 3e-5,
            "image_size": [1280, 960],
            "max_length": 1024
        }
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load processor and model
    logger.info("Loading Donut processor and model...")
    processor = DonutProcessor.from_pretrained(config["model_name_or_path"])
    model = VisionEncoderDecoderModel.from_pretrained(config["model_name_or_path"])
    
    # Add special tokens for invoice parsing
    special_tokens = ["<s_invoice>", "</s_invoice>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Update processor image size
    processor.image_processor.size = config.get("image_size", [1280, 960])
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = InvoiceDataset(
        config["dataset_name_or_path"], 
        processor, 
        split="train",
        max_length=config.get("max_length", 1024)
    )
    
    val_dataset = InvoiceDataset(
        config["dataset_name_or_path"], 
        processor, 
        split="validation",
        max_length=config.get("max_length", 1024)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("num_train_epochs", 30),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 3e-5),
        warmup_steps=config.get("warmup_steps", 300),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=config.get("logging_steps", 100),
        save_steps=config.get("save_steps", 500),
        eval_steps=config.get("eval_steps", 500),
        evaluation_strategy="steps",
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.get("fp16", True),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        seed=config.get("seed", 42),
        report_to=None,  # Disable wandb/tensorboard logging for now
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 5),
                early_stopping_threshold=config.get("early_stopping_threshold", 0.001)
            )
        ],
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(config["output_dir"])
    
    # Evaluate
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    with open(Path(config["output_dir"]) / "eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {config['output_dir']}")
    logger.info(f"Final evaluation results: {eval_results}")

if __name__ == "__main__":
    main()
'''
    
    with open("train_donut_model.py", 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    print("✅ Created train_donut_model.py")

def create_config_file():
    """
    Save the training configuration to a JSON file
    """
    with open("donut_training_config.json", 'w', encoding='utf-8') as f:
        json.dump(DONUT_TRAINING_CONFIG, f, indent=2)
    
    print("✅ Created donut_training_config.json")

def create_requirements_file():
    """
    Create requirements.txt for Donut training
    """
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "datasets>=2.0.0",
        "Pillow>=8.0.0",
        "sentencepiece>=0.1.96",
        "protobuf>=3.20.0",
        "accelerate>=0.12.0",
        "evaluate",
        "scikit-learn",
        "nltk",
        "rouge-score"
    ]
    
    with open("requirements.txt", 'w') as f:
        f.write('\n'.join(requirements))
    
    print("✅ Created requirements.txt")

def create_inference_script():
    """
    Create an inference script to test the trained model
    """
    
    inference_script = '''#!/usr/bin/env python3
"""
Donut Model Inference Script for Invoice Processing
"""

import torch
import json
from PIL import Image
from pathlib import Path
from transformers import DonutProcessor, VisionEncoderDecoderModel

class DonutInvoiceParser:
    """
    Donut model for invoice parsing
    """
    
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained model and processor
        self.processor = DonutProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def parse_invoice(self, image_path, max_length=1024):
        """
        Parse an invoice image and extract structured information
        """
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare input
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate
        task_prompt = "<s_invoice>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        
        # Decode output
        decoded_text = self.processor.tokenizer.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )[0]
        
        # Extract JSON from the generated text
        try:
            # Remove task tokens
            cleaned_text = decoded_text.replace("<s_invoice>", "").replace("</s_invoice>", "").strip()
            parsed_data = json.loads(cleaned_text)
            return parsed_data
        except json.JSONDecodeError:
            return {"raw_text": decoded_text, "error": "Failed to parse JSON"}

def main():
    """
    Example usage of the invoice parser
    """
    
    model_path = "./donut_invoice_model"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train_donut_model.py")
        return
    
    # Initialize parser
    parser = DonutInvoiceParser(model_path)
    
    # Test with sample images
    test_images_dir = Path("dataset/validation/images")
    
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))
        
        if image_files:
            # Test with first image
            test_image = image_files[0]
            print(f"\\nTesting with: {test_image.name}")
            
            result = parser.parse_invoice(test_image)
            
            print(f"\\nExtracted information:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("No test images found in validation set")
    else:
        print(f"Validation images directory not found: {test_images_dir}")

if __name__ == "__main__":
    main()
'''
    
    with open("inference_donut_model.py", 'w', encoding='utf-8') as f:
        f.write(inference_script)
    
    print("✅ Created inference_donut_model.py")

def create_setup_complete():
    """
    Create all necessary files for Donut training
    """
    print("🚀 Setting up Donut training environment")
    print("=" * 45)
    
    create_config_file()
    create_training_script()
    create_requirements_file()
    create_inference_script()
    
    print("\\n📁 Training setup completed! Files created:")
    print("   • donut_training_config.json - Training configuration")
    print("   • train_donut_model.py - Main training script")
    print("   • inference_donut_model.py - Inference script")
    print("   • requirements.txt - Python dependencies")
    
    print("\\n🚀 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run training: python train_donut_model.py")
    print("3. Test inference: python inference_donut_model.py")
    
    print("\\n⚡ Training will use:")
    config = DONUT_TRAINING_CONFIG
    print(f"   • Base model: {config['model_name_or_path']}")
    print(f"   • Epochs: {config['num_train_epochs']}")
    print(f"   • Batch size: {config['per_device_train_batch_size']}")
    print(f"   • Learning rate: {config['learning_rate']}")
    print(f"   • Image size: {config['image_size']}")

if __name__ == "__main__":
    create_setup_complete()
