"""ShieldFlow Data Preprocessing Pipeline.

Loads raw login events, cleans unresolved coordinates and impossible travel outliers,
sorts chronologically per user, deduplicates concurrent logins, and outputs the clean dataset.
"""

import argparse
from pathlib import Path
import pandas as pd

from clean import clean_data
from dedup import deduplicate_data

# Input and Output paths
DEFAULT_INPUT_PATH = Path("/home/dhruv/Documents/fraud-detector/data/raw/synthetic_logins.csv")
DEFAULT_OUTPUT_PATH = Path("/home/dhruv/Documents/fraud-detector/data/processed/clean_logins.csv")

def preprocess_pipeline(input_file: str, output_file: str):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input raw dataset not found at: {input_path}")
        
    print(f"Loading raw logins from: {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} initial login rows.")
    
    # 1. Temporal Sorting: Group by user_id and sort strictly ascending by timestamp
    print("Sorting logins chronologically per user...")
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    # 2. Cleaning: Drop null coordinates and impossible travel outliers
    print("Applying data cleaning step...")
    df_cleaned = clean_data(df)
    print(f"Rows after cleaning: {len(df_cleaned)}")
    
    # 3. Deduplication: Merge rapid-fire same-device logins within 10 seconds
    print("Applying deduplication step...")
    df_processed = deduplicate_data(df_cleaned)
    print(f"Rows after deduplication: {len(df_processed)}")
    
    # Ensure directory exists and write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Clean dataset saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShieldFlow Data Preprocessing Pipeline")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Raw logins CSV path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Clean logins CSV path")
    args = parser.parse_args()
    
    preprocess_pipeline(args.input, args.output)
