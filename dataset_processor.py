import json
from datasets import load_dataset
from transformers import DonutProcessor
 
# --- 1. Functions to process the raw JSON data into Donut-style strings ---
new_special_tokens = [] 
task_start_token = "<s>"
eos_token = "</s>"
 
def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence and discovers special tokens
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"
        return obj
 
def preprocess_documents_for_donut(sample):
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
    image = sample["image"].convert('RGB')
    return {"image": image, "text": d_doc}

# --- 2. Load the base dataset and apply the initial preprocessing ---
print("Loading base dataset...")
base_dataset = load_dataset("imagefolder", data_dir="data/img", split="train")
print("Preprocessing documents...")
proc_dataset = base_dataset.map(preprocess_documents_for_donut)

# --- 3. Load and configure the Donut processor ---
print("Loading and configuring processor...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})
processor.feature_extractor.size = [720,960]
processor.feature_extractor.do_align_long_axis = False

# --- 4. Define the final tokenization function ---
def transform_and_tokenize(sample, processor=processor, split="train", max_length=512, ignore_id=-100):
    try:
        pixel_values = processor(sample["image"], random_padding=split == "train", return_tensors="pt").pixel_values.squeeze()
    except Exception as e:
        print(f"Error processing image: {e}")
        return {}
 
    input_ids = processor.tokenizer(
        sample["text"], add_special_tokens=False, max_length=max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    )["input_ids"].squeeze(0)
 
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}

# --- 5. Apply the final transformation ---
print("Tokenizing and transforming dataset...")
processed_dataset = proc_dataset.map(transform_and_tokenize, remove_columns=["image", "text"])

print("✅ Dataset processing complete.")
print("Original processed dataset info:")
print(processed_dataset)

# --- 6. Split the dataset into training and testing sets ---
print("\nSplitting dataset into training and testing sets (90/10)...")
# The result of train_test_split is a DatasetDict
split_dataset = processed_dataset.train_test_split(test_size=0.1)

print("✅ Splitting complete.")
print("Final dataset structure:")
print(split_dataset)
 
# --- 7. Save the processed and split dataset and processor to disk ---
output_dir = "./processed_sroie_dataset"
print(f"\nSaving processed dataset and processor to {output_dir}...")
try:
    split_dataset.save_to_disk(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Dataset and processor saved successfully.")
except Exception as e:
    print(f"Error saving dataset/processor: {e}")
