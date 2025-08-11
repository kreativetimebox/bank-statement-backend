import os
import json
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
import evaluate
from PIL import Image
import torch


def preprocess_examples(batch, processor, max_length=512):
    """
    Convert raw dataset examples into model-ready format.
    """
    images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    words = batch["words"]
    boxes = batch["bboxes"]
    word_labels = batch["labels"]

    encodings = processor(
        images,
        words,
        boxes=boxes,
        word_labels=word_labels,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Convert tensors to lists for datasets compatibility
    return {k: v.numpy().tolist() for k, v in encodings.items()}


def main(hf_dataset_dir="hf_dataset", model_name="microsoft/layoutlmv3-base", output_dir="runs/layoutlmv3"):
    # Load label mapping
    with open(os.path.join(hf_dataset_dir, "label_list.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    label_list = meta["label_list"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    label2id = {v: int(k) for k, v in meta["id2label"].items()}

    # Load dataset
    ds = load_from_disk(hf_dataset_dir)

    # Processor for LayoutLMv3
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)

    # Encode dataset
    ds_encoded = ds.map(
        lambda x: preprocess_examples(x, processor),
        batched=True,
        remove_columns=ds["train"].column_names
    )

    # Model setup
    num_labels = len(label_list)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=processor.tokenizer)

    # Metric
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels, true_predictions = [], []

        for i in range(len(labels)):
            lab_seq, pred_seq = [], []
            for j in range(len(labels[i])):
                if labels[i][j] != -100:
                    lab_seq.append(id2label[labels[i][j]])
                    pred_seq.append(id2label[preds[i][j]])
            true_labels.append(lab_seq)
            true_predictions.append(pred_seq)

        res = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": res["overall_precision"],
            "recall": res["overall_recall"],
            "f1": res["overall_f1"],
            "accuracy": res["overall_accuracy"]
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",     # modern API
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=6,
        logging_steps=50,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        learning_rate=2e-5,
        weight_decay=0.01,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_encoded["train"],
        eval_dataset=ds_encoded["validation"],
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save final model
    final_model_path = os.path.join(output_dir, "final")
    trainer.save_model(final_model_path)
    print(f"✅ Training finished. Model saved to {final_model_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", default="hf_dataset")
    ap.add_argument("--model_name", default="microsoft/layoutlmv3-base")
    ap.add_argument("--out", default="runs/layoutlmv3")
    args = ap.parse_args()
    main(args.hf_dataset, args.model_name, args.out)