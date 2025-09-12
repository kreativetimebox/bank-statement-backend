import torch
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers import DonutProcessor

 
# Initialize processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Load model from huggingface.co
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
 
# The `resize_token_embeddings` method returns the new embedding layer,
# not its size. To print the size, access `new_emb.num_embeddings`.
model.decoder.resize_token_embeddings(len(processor.tokenizer))

print(f"New embedding size: {len(processor.tokenizer)}")

# Adjust our image size and output sequence lengths.
# The model expects the image size in (width, height) format.
image_processor_size = processor.image_processor.size
model.config.encoder.image_size = [image_processor_size["width"], image_processor_size["height"]]

# Define or load processed_dataset before using it
# Example: processed_dataset = {"train": {"labels": [["label1"], ["label2", "label3"]]}}
# Replace the following line with your actual dataset loading code
processed_dataset = {"train": {"labels": [["label1"], ["label2", "label3"]]}}  # TODO: Replace with actual loading

model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))
 
# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]
 
# is done by Trainer
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)