# Label Studio to Donut Format Converter

This script converts Label Studio annotations to the format required by the Donut model for document parsing tasks like receipts and invoices.

## Features

- ✅ Converts Label Studio annotations to Donut JSON format
- ✅ Automatically splits dataset into train/validation/test sets (80%/10%/10%)
- ✅ Handles both same-ID and coordinate-based label-text matching
- ✅ Supports all required receipt/invoice fields
- ✅ Reusable script for processing multiple annotation files
- ✅ Proper error handling and progress reporting

## Requirements

- Python 3.6+
- Standard Python libraries (json, os, argparse, shutil, pathlib, random, collections)

## Usage

### Basic Usage

```bash
python3 labelstudio_to_donut_converter.py \
  --annotation_file path/to/annotations.json \
  --image_dir path/to/images \
  --output_dir path/to/output
```

### Example with Manish's Annotations

```bash
python3 labelstudio_to_donut_converter.py \
  --annotation_file manish/Receipt_final_output_labelstudio_updated.json \
  --image_dir images/manish_images \
  --output_dir dataset/donut_converted_manish
```

### Command Line Arguments

- `--annotation_file`: Path to Label Studio annotation JSON file (required)
- `--image_dir`: Path to directory containing images (required)
- `--output_dir`: Path to output directory for Donut dataset (required)
- `--seed`: Random seed for dataset splitting (default: 42)

## Input Format

The script expects Label Studio annotations in JSON format with the following structure:

```json
[
  {
    "data": {
      "image": "./images/image_name.jpg"
    },
    "annotations": [
      {
        "result": [
          {
            "id": "unique-id",
            "type": "rectangle",
            "from_name": "bbox",
            "to_name": "image",
            "value": {
              "x": 10.5,
              "y": 20.3,
              "width": 15.2,
              "height": 5.1
            }
          },
          {
            "id": "unique-id",
            "type": "textarea", 
            "from_name": "transcription",
            "to_name": "image",
            "value": {
              "text": ["Sample Text"]
            }
          },
          {
            "id": "unique-id",
            "type": "labels",
            "from_name": "label", 
            "to_name": "image",
            "value": {
              "labels": ["Supplier Name"]
            }
          }
        ]
      }
    ]
  }
]
```

## Output Format

The script generates Donut-compatible JSON files in the following format:

```json
{
  "gt_parse": {
    "supplier_name": "ABC Supermarket",
    "receipt_no": "12345",
    "receipt_date": "2023-08-21",
    "currency": "USD",
    "transaction_id": "",
    "payment_method": "",
    "special_instructions": "",
    "vat_code": "",
    "vat_percent": "20.0",
    "vat_amount": "5.00",
    "sub_total": "25.00",
    "net_amount": "",
    "total_discount": "",
    "coupon_name": "",
    "coupon_amount": "",
    "total_amount": "30.00",
    "change_amount": "",
    "total_item_count": "2",
    "sale_amount": "",
    "payment_date": "",
    "payment_mode": "",
    "card_details": "",
    "items": [
      {
        "item_name": "Milk",
        "item_quantity": "2",
        "item_unit_price": "1.50",
        "item_amount": "3.00"
      },
      {
        "item_name": "Bread", 
        "item_quantity": "1",
        "item_unit_price": "2.00",
        "item_amount": "2.00"
      }
    ]
  }
}
```

## Supported Labels

The script maps the following Label Studio labels to Donut format fields:

### Receipt-level Fields
- `Supplier Name` → `supplier_name`
- `Receipt No.` → `receipt_no`
- `Receipt Date` → `receipt_date`
- `Currency` → `currency`
- `Transaction ID` → `transaction_id`
- `Payment Method` → `payment_method`
- `Special Instructions` → `special_instructions`
- `VAT Code` → `vat_code`
- `VAT Percent` / `vat_percent` → `vat_percent`
- `VAT Amount` → `vat_amount`
- `Sub Total` → `sub_total`
- `Net Amount` → `net_amount`
- `Total Discount` → `total_discount`
- `Coupon Name` → `coupon_name`
- `Coupon Amount` → `coupon_amount`
- `Total Amount` → `total_amount`
- `Change Amount` → `change_amount`
- `Total Item Count` → `total_item_count`
- `Sale Amount` → `sale_amount`
- `Payment Date` → `payment_date`
- `Payment Mode` → `payment_mode`
- `Card Details` → `card_details`

### Item-level Fields
- `Item Code` → `item_code`
- `Item Name` → `item_name`
- `Item Quantity` → `item_quantity`
- `Item Unit Price` → `item_unit_price`
- `Item Discount` → `item_discount`
- `Item Amount` → `item_amount`
- `Item VAT Code` → `item_vat_code`

## Output Directory Structure

The script creates the following directory structure:

```
output_dir/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.json
│       ├── image2.json
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Label-Text Matching Strategies

The script uses two strategies to match labels with transcribed text:

### 1. ID-based Matching (Preferred)
When the same ID is used for bbox, transcription, and label annotations, they are automatically matched together.

### 2. Coordinate-based Matching (Fallback)
When IDs don't match (common in some Label Studio exports), the script falls back to matching labels and transcriptions based on bounding box proximity. This uses the center coordinates of bounding boxes and finds the closest matches.

## Example Results

After running the converter on Manish's annotations:

```
Starting Label Studio to Donut conversion...
Input annotation file: manish/Receipt_final_output_labelstudio_updated.json
Input image directory: images/manish_images
Output directory: dataset/donut_converted_manish
Processing 91 annotations...
Successfully converted 83 annotations
Dataset split - Train: 66, Val: 8, Test: 9

Summary:
Total processed: 83
Train set: 66 (79.5%)
Validation set: 8 (9.6%)
Test set: 9 (10.8%)
```

Sample extracted data:
- Supplier names: "STORES", "Favell"
- Receipt dates: "27/03/25", "2025"
- Currency: "GBP"
- Total amounts: "35.16", "£9.90", "15.93"
- Item quantities: "10", "2"
- Item codes: "728670", "830712"

## Troubleshooting

### No Data Extracted
- Check if your Label Studio annotations include label information (`from_name: "label"`)
- Verify that label names match the supported labels list
- Ensure images exist in the specified image directory

### Coordinate Matching Issues
- The script uses a distance threshold of 10 units for coordinate matching
- Adjust the threshold in the code if needed for your specific annotation style

### Missing Images
- The script will skip annotations for images that don't exist
- Check the image paths in your annotation file match the actual file locations

## Processing Multiple Annotation Files

To process multiple annotation files, run the script separately for each:

```bash
# Process Manish's annotations
python3 labelstudio_to_donut_converter.py \
  --annotation_file manish/Receipt_final_output_labelstudio_updated.json \
  --image_dir images/manish_images \
  --output_dir dataset/donut_manish

# Process other annotations (if they have labels)
python3 labelstudio_to_donut_converter.py \
  --annotation_file other/annotations.json \
  --image_dir images/other_images \
  --output_dir dataset/donut_other
```

Then combine the datasets if needed or use them separately for training.

## Notes

- The script preserves the original image files by copying them to the output directories
- JSON files are saved with UTF-8 encoding to handle special characters
- Empty fields are represented as empty strings in the output JSON
- The random seed ensures reproducible dataset splits
- Progress is reported every 10 processed annotations
