#!/usr/bin/env python3
"""
Test script to verify that the VAT percentage extractor works with both file formats
"""

import os
from vat_percentage_extractor import VATPercentageExtractor

def test_file_extraction(input_file: str, output_file: str, description: str):
    """Test VAT percentage extraction on a specific file"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    try:
        # Initialize extractor
        extractor = VATPercentageExtractor(input_file, output_file)
        
        # Run the extraction
        extractor.run_extraction()
        
        print(f"✅ Successfully processed: {input_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False

def main():
    """Test both file formats"""
    print("VAT Percentage Extractor - Multi-format Test")
    
    # Test 1: Original file format (himanshi directory)
    test1_success = test_file_extraction(
        "himanshi/exported_json_58_img.json",
        "himanshi/exported_json_58_img_updated.json",
        "Original file format (himanshi directory)"
    )
    
    # Test 2: New file format (manish directory)
    test2_success = test_file_extraction(
        "manish/Receipt_final_output_labelstudio.json",
        "manish/Receipt_final_output_labelstudio_updated.json",
        "New file format (manish directory)"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Original format test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"New format test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The extractor works with both file formats.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
