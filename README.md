# LayoutLMv3 Receipt Entity Extraction

A complete end-to-end pipeline for extracting structured information from receipt images using Microsoft's LayoutLMv3 model with custom fine-tuning and DocTR OCR integration.

## 🎯 Overview

This project implements a sophisticated receipt processing system that:
- Extracts text from receipt images using DocTR OCR
- Identifies and classifies entities using fine-tuned LayoutLMv3
- Applies intelligent post-processing for improved accuracy
- Provides structured output for downstream applications

## 🏗️ Architecture

```
Receipt Image → DocTR OCR → LayoutLMv3 → Post-Processing → Structured Entities
```

### Key Components:
- **OCR Engine**: DocTR for robust text extraction
- **ML Model**: Fine-tuned LayoutLMv3 for document understanding
- **Entity Types**: Supplier name, items, prices, quantities, codes, VAT info
- **Post-Processing**: Business rules and confidence filtering

## 📊 Performance Metrics

- **Training F1-Score**: 84.4%
- **Precision**: 82.8%
- **Recall**: 87.8%
- **Entity Coverage**: ~50-75% per receipt
- **Confidence Threshold**: 0.6+ for high-quality extractions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- macOS/Linux (tested on macOS)

### Installation

1. **Clone and Setup**
```bash
git clone <repository-url>
cd Layoutlmv3_1
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify Installation**
```bash
python test_setup.py
```

### Basic Usage

#### 1. Single Receipt Processing
```bash
python improved_inference.py
```

#### 2. Custom Image Processing
```python
from improved_inference import extract_entities

# Process any receipt image
entities = extract_entities("path/to/your/receipt.jpg")
for entity in entities:
    print(f"{entity['label']}: {entity['text']} (confidence: {entity['avg_confidence']:.2f})")
```

## 📁 Project Structure

```
Layoutlmv3_1/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── improved_inference.py               # Main inference script (recommended)
├── inference_receipt.py               # Basic inference script
├── train_layoutlmv3.py               # Model training script
├── convert_to_layoutlmv3.py          # Data format conversion
├── test_setup.py                     # Environment validation
│
├── data/                              # Original datasets
│   ├── 91_image_receipt_min.json
│   └── 71receipt json-min file.json
│
├── images/                            # Receipt images
│   ├── IMG0133_638882809282735163.jpg
│   └── [other receipt images...]
│
├── layoutlmv3_data/                   # Processed training data
│   ├── layoutlmv3_training_data.json
│   └── label_mapping.json
│
├── layoutlmv3_receipt_model/          # Fine-tuned model
│   ├── config.json
│   ├── model.safetensors
│   └── [tokenizer files...]
│
└── combined_receipt_dataset.json     # Merged dataset
```

## 🔧 Configuration

### Entity Types Supported
- **SUPPLIER_NAME**: Store/business name
- **RECEIPT_DATE**: Transaction date
- **RECEIPT_ID**: Receipt/transaction ID
- **ITEM**: Product names and descriptions
- **ITEM_QUANTITY**: Product quantities
- **ITEM_AMOUNT**: Prices and monetary values
- **ITEM_CODE**: Product codes/SKUs
- **VAT_TAX_CODE**: Tax-related information

### Model Configuration
- **Base Model**: microsoft/layoutlmv3-base
- **Max Sequence Length**: 512 tokens
- **Image Size**: Auto-detected, normalized to 1000x1000 scale
- **Batch Size**: 8 (training)
- **Learning Rate**: 5e-5

## 🎯 Advanced Usage

### Custom Training

1. **Prepare Your Data**
```python
# Format: JSON with image_path, tokens, bboxes, labels
{
    "image_path": "path/to/image.jpg",
    "tokens": ["ALDI", "STORES", "10", "X", "BUNS"],
    "bboxes": [[x1, y1, x2, y2], ...],
    "labels": ["B-SUPPLIER_NAME", "I-SUPPLIER_NAME", "B-ITEM_QUANTITY", ...]
}
```

2. **Run Training**
```bash
python train_layoutlmv3.py
```

### Inference Options

#### High Accuracy Mode
```python
# Increase confidence threshold for higher precision
entities = extract_entities(image_path, min_confidence=0.8)
```

#### Batch Processing
```python
import os
from improved_inference import extract_entities

image_dir = "path/to/receipt/images"
results = {}

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        entities = extract_entities(image_path)
        results[filename] = entities
```

## 📈 Results Analysis

### Sample Output
```
📋 Item Amount:
  • 10.78 (confidence: 0.94) █████████
  • 34.17 (confidence: 0.86) ████████
  • 6.45 (confidence: 0.85) ████████

📋 Item Quantity:
  • 70490 (confidence: 0.87) ████████
  • 90 (confidence: 0.82) ████████

📋 Supplier Name:
  • ALDI STORES (confidence: 0.91) █████████
```

### Performance Insights
- **High Confidence (≥0.8)**: Typically accurate extractions
- **Medium Confidence (0.6-0.8)**: Good quality, may need validation
- **Low Confidence (<0.6)**: Filtered out automatically

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Error**
```bash
# Ensure model exists
ls layoutlmv3_receipt_model/
# Re-run training if needed
python train_layoutlmv3.py
```

2. **OCR Failures**
```python
# Check image quality and format
from PIL import Image
img = Image.open("receipt.jpg")
print(f"Image size: {img.size}, Mode: {img.mode}")
```

3. **Low Accuracy**
- Ensure high-quality, well-lit images
- Check if receipt format is similar to training data
- Consider retraining with more diverse data

### Debugging Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

entities = extract_entities(image_path, debug=True)
```

## 🛠️ Development

### Adding New Entity Types

1. **Update Label Mapping**
```json
{
    "label_to_id": {
        "O": 0,
        "B-NEW_ENTITY": 25,
        "I-NEW_ENTITY": 26
    }
}
```

2. **Modify Post-Processing**
```python
def is_new_entity(text):
    # Add detection logic
    return text.startswith("NEW_")
```

3. **Retrain Model**
```bash
python train_layoutlmv3.py
```

### Testing Changes
```bash
python test_setup.py
python improved_inference.py
```

## 📊 Dataset Information

### Training Data
- **Total Samples**: 115 receipts
- **Usable Samples**: 113 (98% success rate)
- **Entity Instances**: 4,351 total
- **Sources**: Combined from multiple receipt datasets

### Data Quality
- **Alignment Check**: Automated validation
- **Format Consistency**: Standardized paths and structures
- **Label Coverage**: All major receipt components

## 🔮 Future Enhancements

### Planned Features
- [ ] Multi-language receipt support
- [ ] Real-time processing API
- [ ] Confidence-based active learning
- [ ] Receipt type classification
- [ ] Batch processing optimization

### Research Directions
- [ ] Few-shot learning for new receipt formats
- [ ] Integration with other document AI models
- [ ] Synthetic data generation
- [ ] Performance optimization for mobile devices

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review the code documentation
- Open an issue with detailed information

## 🙏 Acknowledgments

- **Microsoft Research** for LayoutLMv3
- **Hugging Face** for the Transformers library
- **Mindee** for DocTR OCR
- **PyTorch** team for the deep learning framework

---

**Built with ❤️ for better document understanding**
