# SP500-Price-Prediction

## Overview
This repository contains the solution for the MACLE2425 course assignment, focused on developing a time-series prediction model to forecast the closing prices of an S&P 500 ETF. The project includes a Python implementation and a detailed report explaining the methodology, results, and insights.

## Assignment Details
- **Objective:** Predict the closing price of an S&P 500 ETF for a specific number of future dates (20-30 points, typically one month) based on historical data.
- **Dataset:** The dataset includes the following features:
  - Date
  - Opening Price
  - High Price of the Day
  - Low Price of the Day
- **Target Variable:** Closing Price
- **Evaluation Metric:** Mean Absolute Percentage Error (MAPE)

## Requirements
The Python code should:
1. Accept two CSV files: one for training data and one for testing data.
2. Output the MAPE of the predictions and generate any relevant plots for analysis.
3. Be robust and error-free.

## Deliverables
The deliverable includes:
- **Python Script:** Implements the model and evaluation.
- **IEEE-format Report:** Explains the methodology, results, and insights in a 6-page document, including citations, using the IEEE double column format.

Please zip the Python script and the report, and upload the zip file. Each team member should upload the same zip file.

## Allowed Libraries:

-  `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pytorch`, `jax`, `pytest`

## Project Structure
```
├── data/               # Training and testing datasets
├── src/                # Python implementation
│   ├── model.py        # Main script to train and evaluate the model
│   ├── utils.py        # Helper functions
│   ├── tests.py        # Unit tests
├── report/             # Final report in IEEE format
├── plots/              # Output plots for analysis
├── requirements.txt    # List of required libraries
├── README.md           # Project overview
```

## Instructions for Running
1. Ensure you have the required libraries installed. You can install them using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```
2. Run the `model.py` script with the paths to the training and testing data:
   ```bash
   python src/model.py --train data/train.csv --test data/test.csv
   ```
3. The script will output the MAPE score and generate relevant plots in the `plots/` directory.

## Evaluation
The grading is based on two components:
1. **Report (66%):** Quality of the report, including clarity, methodology, and insights.
2. **Performance (34%):** Accuracy of the model compared to other teams.

### Final grade formula:
```
(0.66 * Report Score + 0.34 * Performance Score) * 6
```
