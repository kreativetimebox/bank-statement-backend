#!/usr/bin/env python3
"""
Standalone OCR script using docTR for text extraction from images.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class DocTR_OCR:
    def __init__(self, det_arch: str = 'db_resnet50', reco_arch: str = 'crnn_vgg16_bn'):
        """Initialize the docTR OCR model."""
        print(f"Initializing docTR OCR with {det_arch} detector and {reco_arch} recognizer...")
        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
        print("OCR model loaded successfully")

    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image from file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        print(f"Loaded image: {img.shape[1]}x{img.shape[0]} (HxW), {img.shape[2]} channels")
        return img

    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Process an image and extract text with bounding boxes."""
        print(f"\nProcessing image: {os.path.basename(image_path)}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Use DocumentFile to load the image directly
            print("Loading image with DocumentFile...")
            doc = DocumentFile.from_images(image_path)
            print("Image loaded successfully")
            
            # Run OCR
            print("Running OCR...")
            result = self.model(doc)
            print("OCR completed")
            
            # Extract results
            ocr_results = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            ocr_results.append({
                                'text': word.value,
                                'confidence': float(word.confidence or 0.0),
                                'bbox': [
                                    float(word.geometry[0][0]),  # x1
                                    float(word.geometry[0][1]),  # y1
                                    float(word.geometry[1][0]),  # x2
                                    float(word.geometry[1][1])   # y2
                                ]
                            })
            
            print(f"Found {len(ocr_results)} text elements")
            return ocr_results
            
        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def print_results(results: List[Dict[str, Any]]):
    """Print OCR results in a readable format."""
    if not results:
        print("No text detected in the image.")
        return
        
    print("\n=== OCR Results ===")
    for i, res in enumerate(results, 1):
        print(f"{i:3d}. {res['text']:30s} | Confidence: {res['confidence']:.2f}")
    
    # Print full text
    full_text = " ".join([r['text'] for r in results])
    print("\nFull text:")
    print("-" * 80)
    print(full_text)
    print("-" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text from images using docTR')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--det', default='db_resnet50', 
                       help='Detection architecture (default: db_resnet50)')
    parser.add_argument('--reco', default='crnn_vgg16_bn',
                       help='Recognition architecture (default: crnn_vgg16_bn)')
    
    args = parser.parse_args()
    # current_directory = os.getcwd()
    # print("Current Working Directory:", current_directory)
    try:
        # Initialize OCR
        ocr = DocTR_OCR(det_arch=args.det, reco_arch=args.reco)
        
        # Process image
        results = ocr.process_image(args.image_path)
        
        # Print results
        print_results(results)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
