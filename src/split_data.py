"""
    USAGE: python3 split_data.py ../data/historicalData_IE00B5BMR087_clean.csv 2024-09-30 --train_output ../data/train.csv --test_output ../data/test.csv
"""

import argparse
import pandas as pd

def split_train_test(file_path, split_date):
    """
    Splits data into training and testing sets based on a given split date.

    Args:
        file_path (str): Path to the input CSV file.
        split_date (str): Date to split the data on (inclusive in the training set).

    Returns:
        pd.DataFrame, pd.DataFrame: Training and testing DataFrames.
    """
    # Read the CSV file
    data = pd.read_csv(file_path, parse_dates=['Date'])
    
    # Sort by date and set the index
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)

    # Split into training and testing
    train_data = data.loc[:split_date]
    test_data = data.loc[split_date:]

    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description="Split a dataset into training and testing sets.")
    parser.add_argument("file_path", help="Path to the input CSV file.")
    parser.add_argument("split_date", help="Date to split the data on (inclusive in the training set).")
    parser.add_argument("--train_output", default="train_data.csv", help="Output file for training data (default: train_data.csv).")
    parser.add_argument("--test_output", default="test_data.csv", help="Output file for testing data (default: test_data.csv).")

    args = parser.parse_args()

    # Split the data
    train_data, test_data = split_train_test(args.file_path, args.split_date)

    # Save the outputs
    train_data.to_csv(args.train_output)
    test_data.to_csv(args.test_output)

    print(f"Training data saved to {args.train_output}")
    print(f"Testing data saved to {args.test_output}")

if __name__ == "__main__":
    main()