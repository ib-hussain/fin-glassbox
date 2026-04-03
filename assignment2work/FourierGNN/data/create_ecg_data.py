"""
This script converts the ECG5000 dataset files into a single CSV file
that the FourierGNN data loader expects.

The script should be run from the fin-glassbox directory or FourierGNN directory.
"""

import pandas as pd
import numpy as np
import os

def create_ecg_data():
    """
    Converts ECG5000_TRAIN.txt and ECG5000_TEST.txt into ECG_data.csv
    """
    # Path to your ECG5000 data
    data_path = "assignment2work/FourierGNN/data/ECG5000/"
    
    # Output path for the combined CSV
    output_path = "assignment2work/FourierGNN/data/ECG_data.csv"
    
    print(f"Reading training data from: {data_path}ECG5000_TRAIN.txt")
    print(f"Reading test data from: {data_path}ECG5000_TEST.txt")
    
    # Load the data files
    # The UCR format: first column is the label (1,2,3 for ECG), remaining columns are time series values
    train_data = pd.read_csv(
        os.path.join(data_path, "ECG5000_TRAIN.txt"), 
        sep='\s+',  # whitespace separator
        header=None
    )
    
    test_data = pd.read_csv(
        os.path.join(data_path, "ECG5000_TEST.txt"), 
        sep='\s+', 
        header=None
    )
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Drop the label column (first column) as we only need the time series values
    # The FourierGNN model expects only the multivariate time series data
    train_values = train_data.iloc[:, 1:]
    test_values = test_data.iloc[:, 1:]
    
    # Combine training and test data
    combined_data = pd.concat([train_values, test_values], axis=0, ignore_index=True)
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Number of time series (variables): {combined_data.shape[1]}")
    print(f"Total time steps: {combined_data.shape[0]}")
    
    # Save to CSV without headers (as expected by the data loader)
    combined_data.to_csv(output_path, index=False, header=False)
    
    print(f"\n✅ Successfully created: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Print a small preview
    print("\nFirst 5 rows, first 10 columns:")
    print(combined_data.iloc[:5, :10].to_string())
    
    return combined_data

def verify_dataset():
    """
    Quick verification that the dataset matches the paper's description.
    Paper says ECG has 140 variables (features).
    """
    import pandas as pd
    
    data = pd.read_csv("assignment2work/FourierGNN/data/ECG_data.csv", header=None)
    
    print("\n" + "="*50)
    print("DATASET VERIFICATION")
    print("="*50)
    print(f"Number of time series (N): {data.shape[1]}")
    print(f"Number of timesteps (T): {data.shape[0]}")
    
    # Paper says ECG has feature_size 140
    if data.shape[1] == 140:
        print("✅ Matches paper: 140 variables (features)")
    else:
        print(f"⚠️ Paper says 140 variables, but found {data.shape[1]}")
    
    # Check for missing values
    missing = data.isnull().sum().sum()
    print(f"Missing values: {missing}")
    
    if missing == 0:
        print("✅ No missing values")
    else:
        print(f"⚠️ Found {missing} missing values")
    
    return data

if __name__ == "__main__":
    print("="*50)
    print("ECG DATA CONVERTER FOR FOURIERGNN")
    print("="*50)
    
    # Create the dataset
    create_ecg_data()
    
    # Verify the dataset
    verify_dataset()
    
    print("\n✅ You can now run: python main.py --data ECG")