#!/usr/bin/env python3
"""
Donut Model Training Script for Invoice Processing
Based on Hugging Face Transformers and Donut implementation
"""

import os
import json
import torch
import logging
import gc  # Add garbage collection import
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
        
        # Load image with memory optimization
        image_path = self.dataset_path / "images" / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        
        # Resize image to reduce memory usage if it's too large
        if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'size'):
            target_size = self.processor.image_processor.size
            if isinstance(target_size, dict):
                target_width = target_size.get('width', 640)  # Reduced for extra safety
                target_height = target_size.get('height', 480)  # Reduced for extra safety
            else:
                target_width, target_height = 640, 480  # Reduced for extra safety
        else:
            target_width, target_height = 640, 480  # Reduced for extra safety
            
        # Resize if image is larger than target
        if image.size[0] > target_width or image.size[1] > target_height:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Prepare target sequence
        target_sequence = f"<s_invoice>{item['ground_truth']}</s_invoice>"
        
        # Process with Donut processor - use the correct Donut approach
        try:
            # For Donut, we typically process image and text separately
            # Process image
            pixel_values = self.processor.image_processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"]
            
            # Process text for labels
            decoder_input_ids = self.processor.tokenizer(
                target_sequence,
                add_special_tokens=False,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
            
            # Combine for processing output
            processed = {
                "pixel_values": pixel_values,
                "input_ids": decoder_input_ids
            }
            
        except Exception as e:
            logger.warning(f"Separate processing failed: {e}, trying combined approach")
            # Fallback to combined processor call
            try:
                processed = self.processor(
                    images=image, 
                    text=target_sequence,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            except Exception as e2:
                logger.error(f"Both processing approaches failed: {e2}")
                raise
        
        # Debug: print available keys for the first sample only
        if idx == 0:
            logger.info(f"Processor output keys: {list(processed.keys())}")
            for key, value in processed.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  {key}: shape {value.shape}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        # Prepare return dict - Donut doesn't use attention_mask for vision encoder
        result = {
            "pixel_values": processed["pixel_values"].squeeze(),
            "labels": processed["input_ids"].squeeze()
        }
        
        # Clean up to save memory
        del image
        del processed
        
        return result

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    # Handle both tuple and EvalPrediction objects
    if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    elif isinstance(eval_pred, tuple) and len(eval_pred) >= 2:
        predictions, labels = eval_pred[0], eval_pred[1]
    else:
        logger.warning(f"Unexpected eval_pred format: {type(eval_pred)}")
        return {"eval_loss": 0.0}
    
    # Ensure predictions is a tensor and handle potential tuple/list cases
    if isinstance(predictions, (tuple, list)):
        # If predictions is a tuple/list, take the first element (usually logits)
        predictions = predictions[0] if len(predictions) > 0 else predictions
    
    # Convert to tensor if needed
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    
    # For now, we'll use a simple loss-based metric
    # You can add more sophisticated metrics like exact match, F1, etc.
    try:
        if predictions.numel() > 0:
            # Calculate mean loss as a proxy metric
            loss_value = predictions.mean().item() if predictions.requires_grad else predictions.mean().item()
            perplexity = torch.exp(torch.tensor(loss_value)).item()
        else:
            perplexity = float('inf')
    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        perplexity = float('inf')
    
    return {"perplexity": perplexity}

def print_memory_usage():
    """
    Print current memory usage for monitoring
    """
    import psutil
    import gc
    
    # CPU memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    logger.info(f"CPU Memory usage: {cpu_memory:.2f} GB")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB
        logger.info(f"GPU Memory allocated: {gpu_memory:.2f} GB")
        logger.info(f"GPU Memory cached: {gpu_cached:.2f} GB")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        # Use memory-optimized config for 16GB RAM - extra safety
        config = {
            "model_name_or_path": "naver-clova-ix/donut-base",
            "dataset_name_or_path": "./dataset",
            "output_dir": "./donut_invoice_model",
            "num_train_epochs": 30,
            "per_device_train_batch_size": 1,  # Keep at 1 for safety
            "per_device_eval_batch_size": 1,   # Keep at 1 for safety
            "learning_rate": 3e-5,
            "image_size": [640, 480],           # Reduced further from [896, 672] for extra safety
            "max_length": 256                   # Reduced from 512 to 256 for extra safety
        }
    
    # Set up device and memory optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Memory optimization settings - conservative approach
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Use conservative memory settings to avoid FP16 issues
        torch.backends.cuda.enable_math_sdp(True)  # Enable math SDP (more stable)
        torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention for stability
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable for stability
        logger.info("GPU memory optimizations enabled (conservative mode)")
    
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    
    # Load processor and model with memory optimization
    logger.info("Loading Donut processor and model...")
    processor = DonutProcessor.from_pretrained(config["model_name_or_path"])
    
    # Load model with memory optimization but avoid FP16 issues
    model = VisionEncoderDecoderModel.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.float32,  # Use FP32 to avoid gradient scaling issues
        low_cpu_mem_usage=True  # Use less CPU memory during loading
    )
    
    # Add special tokens for invoice parsing
    special_tokens = ["<s_invoice>", "</s_invoice>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Configure model for proper generation
    # Set decoder_start_token_id (required for generation)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    
    # Set other important generation parameters
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Ensure decoder config is properly set
    if hasattr(model.config, 'decoder'):
        model.config.decoder.decoder_start_token_id = model.config.decoder_start_token_id
        model.config.decoder.bos_token_id = model.config.bos_token_id
        model.config.decoder.eos_token_id = model.config.eos_token_id
        model.config.decoder.pad_token_id = model.config.pad_token_id
    
    logger.info(f"Set decoder_start_token_id to: {model.config.decoder_start_token_id}")
    logger.info(f"BOS token ID: {model.config.bos_token_id}")
    logger.info(f"EOS token ID: {model.config.eos_token_id}")
    logger.info(f"PAD token ID: {model.config.pad_token_id}")
    
    # Update processor image size
    image_size = config.get("image_size", [1280, 960])
    if hasattr(processor, 'image_processor'):
        # New transformers version
        processor.image_processor.size = {"height": image_size[1], "width": image_size[0]}
    elif hasattr(processor, 'feature_extractor'):
        # Older transformers version
        processor.feature_extractor.size = image_size
    
    logger.info(f"Set image size to: {image_size}")
    
    # Create datasets
    logger.info("Creating datasets...")
    print_memory_usage()  # Monitor memory before dataset creation
    
    train_dataset = InvoiceDataset(
        config["dataset_name_or_path"], 
        processor, 
        split="train",
        max_length=config.get("max_length", 256)  # Reduced to 256 for extra safety
    )
    
    val_dataset = InvoiceDataset(
        config["dataset_name_or_path"], 
        processor, 
        split="validation",
        max_length=config.get("max_length", 256)  # Reduced to 256 for extra safety
    )
    
    print_memory_usage()  # Monitor memory after dataset creation
    
    # Training arguments - optimized for 16GB RAM
    base_args = {
        "output_dir": config["output_dir"],
        "num_train_epochs": config.get("num_train_epochs", 30),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 1),  # Small batch
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", 1),    # Small batch
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 16),  # Increased to maintain effective batch size
        "learning_rate": config.get("learning_rate", 3e-5),
        "warmup_steps": config.get("warmup_steps", 300),
        "weight_decay": config.get("weight_decay", 0.01),
        "logging_steps": config.get("logging_steps", 100),
        "save_steps": config.get("save_steps", 500),
        "eval_steps": config.get("eval_steps", 500),
        "save_total_limit": config.get("save_total_limit", 2),  # Reduce saved checkpoints
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": False,  # Disable FP16 to avoid gradient scaling issues
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Use BF16 if available (more stable)
        "dataloader_num_workers": config.get("dataloader_num_workers", 0),  # Reduce workers to save memory
        "remove_unused_columns": False,
        "seed": config.get("seed", 42),
        "report_to": None,  # Disable wandb/tensorboard logging for now
        "dataloader_pin_memory": False,  # Disable pin memory to save RAM
        "gradient_checkpointing": True,  # Enable gradient checkpointing to save GPU memory
        "optim": "adamw_torch",  # Use standard optimizer to avoid FP16 issues
        "max_grad_norm": 1.0,  # Add gradient clipping for stability
    }
    
    # Try both parameter names for evaluation strategy
    try:
        training_args = TrainingArguments(
            eval_strategy="steps",  # New parameter name
            **base_args
        )
        logger.info("Using 'eval_strategy' parameter")
    except TypeError:
        try:
            training_args = TrainingArguments(
                evaluation_strategy="steps",  # Old parameter name
                **base_args
            )
            logger.info("Using 'evaluation_strategy' parameter")
        except TypeError as e:
            logger.error(f"Both eval_strategy and evaluation_strategy failed: {e}")
            raise
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # compute_metrics=compute_metrics,  # Disabled to avoid tuple errors
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 5),
                early_stopping_threshold=config.get("early_stopping_threshold", 0.001)
            )
        ],
    )
    
    # Start training
    logger.info("Starting training...")
    print_memory_usage()  # Monitor memory before training
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU out of memory! Try reducing batch size or image size.")
            logger.error("Current settings:")
            logger.error(f"  - Batch size: {config.get('per_device_train_batch_size', 1)}")
            logger.error(f"  - Image size: {config.get('image_size', [640, 480])}")
            logger.error(f"  - Max length: {config.get('max_length', 256)}")
            logger.error("Consider reducing these values further.")
        raise
    except ValueError as e:
        if "fp16" in str(e).lower() or "gradient" in str(e).lower():
            logger.error("Mixed precision training error! This has been disabled in the current configuration.")
            logger.error("If you see this error, it might be due to model compatibility issues.")
            logger.error("The training should work with FP32 (current setting).")
        raise
    except Exception as e:
        logger.error(f"Unexpected training error: {e}")
        logger.error("Check the model configuration and dataset integrity.")
        raise
    
    print_memory_usage()  # Monitor memory after training
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(config["output_dir"])
    
    # Evaluate
    logger.info("Running final evaluation...")
    try:
        eval_results = trainer.evaluate()
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        logger.info("Skipping evaluation and saving model anyway...")
        eval_results = {"eval_loss": "N/A", "note": "Evaluation skipped due to error"}
    
    # Save evaluation results
    with open(Path(config["output_dir"]) / "eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Final memory cleanup
    print_memory_usage()
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {config['output_dir']}")
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Final cleanup
    del trainer
    del model
    del processor
    del train_dataset
    del val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Memory cleanup completed.")

if __name__ == "__main__":
    main()
