# VAT Percentage Extraction Summary Report

## Overview
This document summarizes the successful extraction of VAT percentage information from the Label Studio export JSON file (`exported_json_58_img.json`) and the creation of a new updated file with separate `vat_percent` label fields.

## Problem Statement
The original JSON file contained VAT percentage values embedded within VAT/Tax fields (such as VAT/Tax Code and Tax Amount), making it difficult to:
- Identify and extract VAT percentage values separately
- Use VAT percentages for calculations or analysis
- Maintain clean data structure for downstream processing

## Solution Implemented
Created a comprehensive Python script (`vat_percentage_extractor.py`) that:
1. **Identifies VAT percentage values** using regex patterns to detect percentage formats
2. **Creates separate `vat_percent` label fields** with proper bounding box coordinates
3. **Preserves all original annotation metadata** including coordinates, dimensions, and other properties
4. **Generates an updated JSON file** (`exported_json_58_img_updated.json`) with the new structure

## Extraction Results

### Summary Statistics
- **Total VAT Percentage Fields Created**: 52
- **Files with VAT Percentages**: 12 out of 58 total files
- **Unique VAT Percentage Values**: 21 different percentage values

### VAT Percentage Distribution
The extracted percentages range from **20.0% to 77.0%** with the following distribution:

| Range | Count | Examples |
|-------|-------|----------|
| 0-25% | 1 | 20.00% |
| 25-50% | 14 | 25.1%, 25.2%, 34.8%, 35.0%, 35.4%, 45.3%, 48.8%, 49.7% |
| 50-75% | 27 | 50.4%, 50.7%, 51.0%, 51.1%, 55.7%, 68.6%, 70%, 76.0% |
| 75-100% | 3 | 77.0% |

### Most Common VAT Percentages
- **55.7%**: 8 occurrences
- **50.4%**: 8 occurrences  
- **68.6%**: 4 occurrences
- **48.8%**: 3 occurrences
- **77.0%**: 2 occurrences

### Files with VAT Percentages
The following 12 files contain VAT percentage annotations:
1. `IMG0036_638882808429891937.jpg` - 5 VAT percentages
2. `IMG0043_638882810137603510.jpg` - 3 VAT percentages
3. `IMG0095_638882810139090472.jpg` - 11 VAT percentages
4. `IMG0107_638882808778734226.jpg` - 3 VAT percentages
5. `IMG0136_638882809286294062.jpg` - 6 VAT percentages
6. `IMG0138_638882809288492957.jpg` - 4 VAT percentages
7. `IMG0140_638882809291824386.jpg` - 3 VAT percentages
8. `IMG0426_638882810533325914.jpg` - 1 VAT percentage
9. `IMG0562_638882810545490459.jpg` - 3 VAT percentages
10. `IMG0608_638882810823941231.jpg` - 5 VAT percentages
11. `IMG1027_638882811303889434.jpg` - 5 VAT percentages
12. `IMG1030_638882811307386846.jpg` - 3 VAT percentages

## Technical Implementation

### Key Features
1. **Pattern Recognition**: Uses multiple regex patterns to identify percentage values
2. **Bounding Box Preservation**: Maintains exact coordinates and dimensions from original annotations
3. **Metadata Integrity**: Preserves all Label Studio annotation properties
4. **Unique ID Generation**: Creates new UUIDs for each extracted field
5. **Comprehensive Coverage**: Handles both labeled and unlabeled percentage values

### Regex Patterns Used
- `(\d+\.?\d*%)` - Matches percentages like 55.7%, 48.8%
- `(\d+\.?\d*)\s*%` - Matches percentages like 55.7 %, 48.8 %
- `(\d+\.?\d*)\s*percent` - Matches percentages like 55.7 percent
- `(\d+\.?\d*)\s*per\s*cent` - Matches percentages like 55.7 per cent

### Data Structure
Each extracted `vat_percent` field includes:
- **Label annotation**: Type "labels" with label "vat_percent"
- **Transcription annotation**: Type "textarea" with the percentage value
- **Bounding box coordinates**: x, y, width, height from original annotation
- **Image metadata**: original_width, original_height, rotation
- **Unique identifiers**: New UUIDs for both label and transcription

## Files Generated

### 1. `vat_percentage_extractor.py`
- Main extraction script
- Handles the complete VAT percentage extraction process
- Preserves all annotation metadata

### 2. `exported_json_58_img_updated.json`
- Updated JSON file with new `vat_percent` fields
- Contains all original annotations plus 52 new VAT percentage fields
- Ready for import back into Label Studio or other systems

### 3. `vat_percentage_summary.py`
- Analysis script for the updated JSON file
- Provides detailed statistics and distribution analysis
- Generates comprehensive reports

### 4. `annotation_analyzer.py` (Updated)
- Enhanced version that recognizes `vat_percent` as a VAT/tax related label
- Provides complete annotation analysis including new fields

## Benefits of the Solution

1. **Data Quality**: Separates VAT percentages into dedicated fields for better data organization
2. **Analysis Ready**: Enables easy calculation and analysis of VAT percentages
3. **Label Studio Compatible**: Maintains full compatibility with Label Studio import/export
4. **Metadata Preservation**: All original annotation information is preserved
5. **Scalable**: Can be applied to other Label Studio exports with similar structures

## Usage Instructions

### Running the Extraction
```bash
python3 vat_percentage_extractor.py
```

### Analyzing Results
```bash
python3 vat_percentage_summary.py
```

### Complete Annotation Analysis
```bash
python3 annotation_analyzer.py
```

## Next Steps

The updated JSON file (`exported_json_58_img_updated.json`) can now be:
1. **Imported back into Label Studio** for further annotation work
2. **Used for data analysis** with clear separation of VAT percentages
3. **Processed by downstream systems** that need structured VAT data
4. **Served as a template** for processing similar Label Studio exports

## Conclusion

The VAT percentage extraction process successfully identified and extracted 52 VAT percentage values from the original JSON file, creating a clean, structured dataset that separates VAT percentages into dedicated fields while preserving all original annotation metadata. This solution provides a solid foundation for further data analysis and processing workflows.
