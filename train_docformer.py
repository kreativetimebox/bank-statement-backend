import sys, os
sys.path.append(os.path.abspath("src"))

from transformers import TrainingArguments, Trainer, LayoutLMTokenizerFast
from funsd_dataset import FunsdDataset
from src.docformer.modeling import DocFormerForTokenClassification

TRAIN_TXT = "Data/train.txt"
OUTPUT_DIR = "./docformer_outputs"
BASE_TOKENIZER = "microsoft/layoutlm-base-uncased"

label_list = ["O"]  # Extend as needed

tokenizer = LayoutLMTokenizerFast.from_pretrained(BASE_TOKENIZER)
train_dataset = FunsdDataset(TRAIN_TXT, tokenizer=tokenizer, label_list=label_list)

model = DocFormerForTokenClassification.from_pretrained(".", num_labels=len(label_list))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    overwrite_output_dir=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
