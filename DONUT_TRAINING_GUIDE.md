# Donut Model Training Guide for Invoice Processing

## 🎯 Overview

This guide provides complete setup and training instructions for a Donut model to process invoice images using your labeled dataset.

## 📊 Dataset Summary

- **Total Images**: 48 invoice images
- **Total Annotations**: 432 labeled records (9 annotations per image)
- **Training Split**: 33 images (68.8%) → 297 annotation records
- **Validation Split**: 15 images (31.2%) → 135 annotation records
- **Format**: Label Studio annotations converted to Donut JSONL format

## 🏗️ Setup Status

✅ **All components ready for training!**

### Files Created:
- `dataset/` - Split dataset with train/validation folders
- `train_donut_model.py` - Main training script
- `donut_training_config.json` - Training configuration
- `inference_donut_model.py` - Model inference script
- `requirements.txt` - Python dependencies
- `verify_donut_setup.py` - Setup verification script

## 🚀 Training Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start Training
```bash
python train_donut_model.py
```

### Step 3: Monitor Training
The training will:
- Save checkpoints every 500 steps
- Evaluate every 500 steps
- Use early stopping if no improvement for 5 evaluations
- Save the best model based on validation loss

### Step 4: Test Trained Model
```bash
python inference_donut_model.py
```

## ⚙️ Training Configuration

| Parameter | Value | Description |
|-----------|--------|-------------|
| **Base Model** | naver-clova-ix/donut-base | Pre-trained Donut model |
| **Epochs** | 30 | Maximum training epochs |
| **Batch Size** | 2 | Training batch size per device |
| **Learning Rate** | 3e-5 | Optimizer learning rate |
| **Image Size** | [1280, 960] | Input image dimensions |
| **Max Length** | 1024 | Maximum sequence length |
| **Gradient Accumulation** | 4 | Steps before optimizer update |
| **Warmup Steps** | 300 | Learning rate warmup |
| **Early Stopping** | 5 patience | Stop if no improvement |

## 📈 Expected Training Time

- **Steps per epoch**: ~16 (33 images ÷ 2 batch size)
- **Total steps**: ~480 (16 steps × 30 epochs)
- **Checkpoints**: Every 500 steps (1 checkpoint)
- **Evaluations**: Every 500 steps
- **Estimated time**: 4-8 hours on GPU, 12-24 hours on CPU

## 🎯 Invoice Fields Extracted

The model will learn to extract these invoice fields:

- **supplier_name** - Company/supplier name
- **invoice_date** - Invoice date
- **invoice_id** - Invoice number/ID
- **vat_tax_code** - VAT/tax identification
- And other labeled fields from your annotations

## 📁 Dataset Structure

```
dataset/
├── dataset_split_info.json     # Split metadata
├── train/
│   ├── images/                 # Training images (33 files)
│   ├── annotations.json        # Original Label Studio format
│   └── metadata.jsonl          # Donut training format
└── validation/
    ├── images/                 # Validation images (15 files)
    ├── annotations.json        # Original Label Studio format
    └── metadata.jsonl          # Donut training format
```

## 🔧 Customization Options

### Modify Training Parameters
Edit `donut_training_config.json` to adjust:
- Number of epochs
- Batch size
- Learning rate
- Image size
- Early stopping patience

### GPU Memory Optimization
If you encounter GPU memory issues:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Enable `fp16: true` (already enabled)
- Reduce `image_size` to [1024, 768]

### CPU Training
For CPU-only training:
- Set `fp16: false` in config
- Reduce batch size to 1
- Expect significantly longer training time

## 📊 Monitoring Training

During training, monitor these metrics:
- **Loss**: Should decrease over time
- **Evaluation Loss**: Should decrease and stabilize
- **Perplexity**: Lower is better
- **GPU/CPU Usage**: Should be consistently high
- **Memory Usage**: Should be stable

## 🎯 After Training

### Model Location
Trained model will be saved in: `./donut_invoice_model/`

### Testing Inference
Use the inference script to test on new images:
```python
from inference_donut_model import DonutInvoiceParser

parser = DonutInvoiceParser("./donut_invoice_model")
result = parser.parse_invoice("path/to/invoice.jpg")
print(result)
```

### Expected Output Format
```json
{
  "supplier_name": ["COMPANY NAME"],
  "invoice_date": ["26", "May", "2018"],
  "invoice_id": ["#53073"],
  "vat_tax_code": ["225908010"],
  "total_amount": ["£25.99"]
}
```

## 🚨 Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce batch size
2. **Slow training**: Ensure GPU is being used
3. **Poor results**: Increase training epochs or check data quality
4. **Import errors**: Verify all dependencies are installed

### Check GPU Usage:
```bash
nvidia-smi  # Monitor GPU usage during training
```

### Logs Location:
Training logs will be saved in `./donut_invoice_model/` directory.

## ✅ Pre-Training Checklist

- [x] Dataset split completed (70/30 train/val)
- [x] Images copied to train/validation folders
- [x] Annotations converted to Donut JSONL format
- [x] Training script created and configured
- [x] Dependencies specified in requirements.txt
- [x] Inference script ready for testing
- [x] All files verified and ready

---

**🎉 You're ready to start training your Donut model for invoice processing!**

Run `python train_donut_model.py` to begin training.
