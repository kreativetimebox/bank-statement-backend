import json
import torch
from datasets import load_dataset
from huggingface_hub import HfFolder
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# --- 1. Load and Prepare the Dataset ---

# Load the SROIE dataset from the Hugging Face Hub
dataset = load_dataset("nielsr/donut-base-sroie", split="train")

# --- 2. Initialize Model and Processor ---

# Load the base Donut model and processor from the pretrained checkpoint
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# --- 3. Add Special Tokens for SROIE ---

# Add new tokens for the SROIE dataset fields to the tokenizer.
# This is a crucial step for the model to understand the structure of the data.
sroie_fields = ["company", "date", "address", "total"]
new_special_tokens = [f"<s_{field}>" for field in sroie_fields]
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

# Resize the model's token embeddings to accommodate the new tokens.
model.resize_token_embeddings(len(processor.tokenizer))

# --- 4. Configure the Model ---

# Set Donut-specific configurations
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.decoder.config.vocab_size

# Set the image size for the encoder
image_processor_size = processor.image_processor.size
model.config.encoder.image_size = [image_processor_size["width"], image_processor_size["height"]]

# Dynamically set the max length for the decoder based on the dataset
def get_max_target_length(dataset, processor, fields):
    max_len = 0
    for example in dataset:
        ground_truth = json.loads(example["ground_truth"])
        target_sequence = "".join([f"<s_{field}>{ground_truth.get(field, '').strip()}" for field in fields])
        tokenized_len = len(processor.tokenizer(target_sequence, add_special_tokens=True).input_ids)
        if tokenized_len > max_len:
            max_len = tokenized_len
    return max_len

max_target_length = get_max_target_length(dataset["train"], processor, sroie_fields)
model.config.decoder.max_length = max_target_length
print(f"Set decoder max_length to: {max_target_length}")

# --- 5. Preprocess the Dataset ---

def preprocess_documents(sample, processor, max_length, fields):
    """
    Prepares a single sample for the Donut model by processing the image
    and creating the target sequence from the ground truth JSON.
    """
    # Create the target sequence string in the format Donut expects
    ground_truth = json.loads(sample["ground_truth"])
    target_sequence = "".join([f"<s_{field}>{ground_truth.get(field, '').strip()}" for field in fields])

    # The processor handles image transformations and tokenization
    model_input = processor(
        sample["image"], 
        text=target_sequence, 
        max_length=max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    # The processor's output needs to be squeezed to remove the batch dimension
    return {
        "pixel_values": model_input.pixel_values.squeeze(),
        "labels": model_input.labels.squeeze(),
    }

# Use a lambda to pass additional arguments to the map function
processed_dataset = dataset.map(
    lambda example: preprocess_documents(example, processor, max_target_length, sroie_fields),
    remove_columns=dataset["train"].column_names
)

# --- 6. Set up Training ---

# hyperparameters used for multiple args
hf_repository_id = "donut-base-sroie"

# Arguments for training
training_args = Seq2SeqTrainingArguments(
    output_dir=hf_repository_id,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    eval_strategy="no",  # Use 'eval_strategy' for compatibility with older transformers versions
    save_strategy="epoch",
    predict_with_generate=True,
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=hf_repository_id,
    hub_token=HfFolder.get_token(),
)
 
# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
)

# Start training
trainer.train()

# Save processor and create model card
processor.save_pretrained(hf_repository_id)
trainer.create_model_card()
trainer.push_to_hub()
