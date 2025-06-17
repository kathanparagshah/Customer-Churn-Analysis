#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.load_data import DataLoader

def main():
    loader = DataLoader()
    print(f"Testing download from Kaggle...")
    
    result = loader.download_from_kaggle()
    print(f"Result: {result}")
    print(f"File exists: {result.exists()}")
    
    if result.exists():
        print(f"File size: {result.stat().st_size} bytes")
    else:
        print("File was not downloaded successfully")

if __name__ == "__main__":
    main()