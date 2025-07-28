import re
import gradio as gr
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load the model and processor for CORD-V2
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_document(image):
    # Prepare encoder input
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Use CORD task prompt (no question needed)
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Generate output
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Post-process output
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    # Convert to structured JSON
    return processor.token2json(sequence)

# Gradio interface
description = "Gradio Demo for Donut (CORD-v2 fine-tuned). Upload a receipt or invoice image to extract structured data like menu, total, date, etc."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2111.15664' target='_blank'>Donut Paper</a> | <a href='https://github.com/clovaai/donut' target='_blank'>GitHub Repo</a></p>"

demo = gr.Interface(
    fn=process_document,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="Donut 🍩 Receipt Parser (CORD-V2)",
    description=description,
    article=article,
    examples=[["misc/sample_image_cord_test_receipt_00004.png"]],
    cache_examples=False
)

demo.queue()
demo.launch()
