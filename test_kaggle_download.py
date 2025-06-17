#!/usr/bin/env python3
"""
Test script for the enhanced download_from_kaggle() functionality.

This script demonstrates:
- Credential handling from multiple sources
- Retry logic with exponential backoff
- ZIP file extraction and cleanup
- Graceful error handling

Usage:
    python test_kaggle_download.py
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.load_data import DataLoader

def test_kaggle_download():
    """
    Test the enhanced download_from_kaggle() method.
    """
    print("🔍 Testing enhanced Kaggle download functionality...\n")
    
    # Initialize DataLoader
    loader = DataLoader()
    
    print(f"📁 Project root: {loader.project_root}")
    print(f"📁 Raw data directory: {loader.raw_data_dir}")
    print(f"📁 Interim data directory: {loader.interim_data_dir}\n")
    
    # Test 1: Download with default parameters
    print("🧪 Test 1: Download with default parameters")
    print("Dataset: mashlyn/customer-churn-modeling")
    print("Expected file: Churn_Modelling.csv")
    print("Max retries: 3\n")
    
    try:
        file_path = loader.download_from_kaggle()
        print(f"✅ Download completed")
        print(f"📄 File path: {file_path}")
        print(f"📊 File exists: {file_path.exists()}")
        
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"📏 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Test credential setup
    print("🧪 Test 2: Test credential setup")
    creds_setup = loader._setup_kaggle_credentials()
    print(f"🔑 Credentials setup successful: {creds_setup}\n")
    
    # Test 3: Load and validate data if available
    expected_file = loader.raw_data_dir / "Churn_Modelling.csv"
    if expected_file.exists():
        print("🧪 Test 3: Load and validate downloaded data")
        try:
            df = loader.load_csv_data("Churn_Modelling.csv")
            print(f"📊 Data shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Validate schema
            is_valid, issues = loader.validate_schema(df)
            print(f"✅ Schema validation: {'PASSED' if is_valid else 'FAILED'}")
            if issues:
                print(f"⚠️  Issues found: {issues}")
            
            # Get data summary
            summary = loader.get_data_summary(df)
            print(f"📈 Data summary generated with {summary['total_rows']} rows")
            
        except Exception as e:
            print(f"❌ Data loading/validation failed: {e}")
    else:
        print("🧪 Test 3: Skipped (no data file available)")
    
    print("\n" + "="*60 + "\n")
    print("🎉 Testing completed!")

def main():
    """
    Main function to run the test.
    """
    # Configure logging for better visibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Enhanced Kaggle Download Test")
    print("=" * 40)
    print()
    
    test_kaggle_download()

if __name__ == "__main__":
    main()