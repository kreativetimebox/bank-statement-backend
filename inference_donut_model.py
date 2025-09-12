#!/usr/bin/env python3
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
            print(f"\nTesting with: {test_image.name}")
            
            result = parser.parse_invoice(test_image)
            
            print(f"\nExtracted information:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("No test images found in validation set")
    else:
        print(f"Validation images directory not found: {test_images_dir}")

if __name__ == "__main__":
    main()
