"""
LayoutLMv3 Training Script for Receipt Entity Recognition
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3TokenizerFast, 
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from PIL import Image
import os

class ReceiptDataset(Dataset):
    def __init__(self, data_file, label_mapping_file, tokenizer, max_length=512):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        with open(label_mapping_file, 'r') as f:
            self.label_mapping = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = self.label_mapping['label_to_id']
        self.id_to_label = {int(k): v for k, v in self.label_mapping['id_to_label'].items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare inputs
        tokens = item['tokens']
        bboxes = item['bboxes'] 
        ner_tags = item['ner_tags']
        
        # Ensure we don't exceed max length
        if len(tokens) > self.max_length - 2:  # Account for special tokens
            tokens = tokens[:self.max_length - 2]
            bboxes = bboxes[:self.max_length - 2]
            ner_tags = ner_tags[:self.max_length - 2]
        
        # Tokenize with LayoutLMv3
        encoding = self.tokenizer(
            tokens,
            boxes=bboxes,
            word_labels=ner_tags,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bbox': encoding['bbox'].squeeze(),
            'labels': encoding['labels'].flatten()
        }

def compute_metrics(eval_pred):
    """Compute precision, recall, F1 for NER evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Load label mapping for metric computation
    with open('layoutlmv3_data/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    # Remove ignored index (usually -100)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten for sklearn metrics
    flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    
    return {
        "precision": precision_score(flat_true_labels, flat_true_predictions, average='weighted', zero_division=0),
        "recall": recall_score(flat_true_labels, flat_true_predictions, average='weighted', zero_division=0),
        "f1": f1_score(flat_true_labels, flat_true_predictions, average='weighted', zero_division=0),
    }

def train_layoutlmv3():
    # Configuration
    MODEL_NAME = "microsoft/layoutlmv3-base"
    DATA_DIR = "layoutlmv3_data"
    OUTPUT_DIR = "layoutlmv3_receipt_model"
    
    # Load label mapping
    with open(f"{DATA_DIR}/label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    num_labels = len(label_mapping['label_to_id'])
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    # Initialize tokenizer and model
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(MODEL_NAME)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_mapping['label_to_id']
    )
    
    # Load datasets - using separate train and validation sets
    train_dataset = ReceiptDataset(
        f"{DATA_DIR}/train_data.json",
        f"{DATA_DIR}/label_mapping.json", 
        tokenizer
    )
    
    # Load validation dataset
    val_dataset = ReceiptDataset(
        f"{DATA_DIR}/val_data.json",
        f"{DATA_DIR}/label_mapping.json", 
        tokenizer
    )
    
    # Print dataset information
    print(f"📊 DATASET INFORMATION:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"   Train/Val ratio: {len(train_dataset)}/{len(val_dataset)} ({len(train_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}%/{len(val_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}%)")
    print()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=4,  # Increased batch size for better training
        per_device_eval_batch_size=4,   # Increased batch size for faster evaluation
        warmup_steps=50,                # Reduced warmup for smaller dataset
        weight_decay=0.01,
        learning_rate=5e-5,             # Explicit learning rate
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=20,               # More frequent logging
        eval_strategy="steps",
        eval_steps=100,                 # More frequent evaluation
        save_strategy="steps", 
        save_steps=100,                 # More frequent saving
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,                     # Disable mixed precision for stability
        seed=42,                        # Set seed for reproducibility
        data_seed=42,                   # Set data seed for reproducibility
        report_to=None,                 # Disable wandb/tensorboard for now
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to {OUTPUT_DIR}")
    
    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    print("Starting LayoutLMv3 Fine-tuning...")
    print("Dataset: Receipt Entity Recognition")
    print("Model: microsoft/layoutlmv3-base")
    print()
    
    # Start training
    train_layoutlmv3()
