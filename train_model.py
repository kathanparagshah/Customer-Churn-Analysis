#!/usr/bin/env python3

from src.models.train_churn import run_full_pipeline
from pathlib import Path

def main():
    # Set data path
    data_path = Path('data/raw/Churn_Modelling.csv')
    
    print(f"Training model with data from: {data_path}")
    print(f"Data file exists: {data_path.exists()}")
    
    # Train the model using the full pipeline
    try:
        predictor = run_full_pipeline(str(data_path))
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()