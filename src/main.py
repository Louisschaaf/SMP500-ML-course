#!/usr/bin/env python3
"""
Module Docstring
USAGE: python3 main.py ../data/train.csv ../data/test.csv
"""

__author__ = "Leander Winters, Louis Schaaf"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import Hyperparameter_tuning as hyperparam

scaler = MinMaxScaler(feature_range=(0, 1))

def preprocess_data(file_path):
    """
    Preprocess the input CSV data.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)
    
    return data

# -----------------------------------------------
# Additional Indicator Functions
# -----------------------------------------------
def compute_rsi(series, period=14):
    """
    Returns RSI over a given period.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    # tiny epsilon to avoid division by zero
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(price, window=20, num_std=2):
    """
    Returns (middle, upper, lower) Bollinger Bands based on 'window' SMA and 'num_std' standard deviations.
    """
    sma = price.rolling(window=window).mean()
    std = price.rolling(window=window).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    return sma, upper, lower

# -----------------------------------------------
# LSTM Model
# -----------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# -----------------------------------------------
# Prepare Data Function
# -----------------------------------------------
def prepare_data(data, look_back):
    """
    Prepares sliding window data for LSTM input.

    Args:
        data (pd.DataFrame): DataFrame with multiple features (including 'Returns').
        look_back (int): Number of past days to include in the input window.

    Returns:
        np.ndarray: x (input features) of shape (samples, look_back, features).
        np.ndarray: y (output target) of shape (samples, 30).
    """
    if len(data) < look_back + 30:
        raise ValueError("Not enough data to prepare the required sequences.")

    x, y = [], []
    for i in range(len(data) - look_back - 30 + 1):
        x_window = data.iloc[i : i + look_back].values
        y_future = data['Returns'].iloc[i + look_back : i + look_back + 30].values
        x.append(x_window)
        y.append(y_future)

    return np.array(x), np.array(y)

# -----------------------------------------------
# Train and Predict
# -----------------------------------------------
def train_and_predict(train_data, look_back=10):
    """
    Trains the LSTM model on `train_data` and returns
    a 30-day forecast from the last window in the training set.
    """

    # ------------------------------------------------
    # 1. Feature Engineering
    # ------------------------------------------------
    # (A) Compute Returns
    train_data['Returns'] = train_data['Last Close'] / train_data['Open']

    # (B) 10-day MA of the closing price
    train_data['10d_MA'] = train_data['Last Close'].rolling(window=10).mean()

    # (C) 20-day MA
    train_data['20d_MA'] = train_data['Last Close'].rolling(window=20).mean()

    # (D) 10-day STD of daily returns
    train_data['10d_STD'] = train_data['Returns'].rolling(window=10).std()

    # (E) RSI (14) of Last Close
    train_data['RSI_14'] = compute_rsi(train_data['Last Close'], period=14)

    # (F) Bollinger Bands of Last Close
    bb_mid, bb_up, bb_low = compute_bollinger_bands(train_data['Last Close'], window=20, num_std=2)
    train_data['BB_MID'] = bb_mid
    train_data['BB_UP'] = bb_up
    train_data['BB_LOW'] = bb_low

    # (G) Day of Week
    if hasattr(train_data.index, 'dayofweek'):
        train_data['DayOfWeek'] = train_data.index.dayofweek
    else:
        train_data['DayOfWeek'] = 0

    # Fill missing values
    train_data.bfill(inplace=True)

    # ------------------------------------------------
    # 2. Select and Scale Features
    # ------------------------------------------------
    features = [
        'Returns',
        '10d_MA',
        '20d_MA',
        '10d_STD',
        'RSI_14',
        'BB_MID',
        'BB_UP',
        'BB_LOW',
        'DayOfWeek'
    ]

    # Fit scaler on the selected features
    scaler.fit(train_data[features])
    scaled_data = scaler.transform(train_data[features])
    scaled_train_data = pd.DataFrame(scaled_data, columns=features, index=train_data.index)

    # ------------------------------------------------
    # 3. Prepare Training Sequences
    # ------------------------------------------------
    x_train, y_train = prepare_data(scaled_train_data, look_back)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # ------------------------------------------------
    # 4. Initialize and Train the Model
    # ------------------------------------------------
    input_size = len(features)
    hidden_size = 64
    output_size = 30 

    model = LSTMModel(input_size, hidden_size, output_size, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 75
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
    # Save the trained model state
    torch.save(model.state_dict(), "good_model.pth")

    # ------------------------------------------------
    # 5. Generate Forecast from Last Window
    # ------------------------------------------------
    model.eval()
    with torch.no_grad():
        # Last `look_back` rows of the scaled features
        last_window = scaled_train_data.iloc[-look_back:].values
        last_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

        # Predict the next 30 returns (scaled)
        predictions_scaled = model(last_window).numpy().flatten()

    # ------------------------------------------------
    # 6. Inverse Transform (only 'Returns' column)
    # ------------------------------------------------
    placeholder = np.zeros((len(predictions_scaled), len(features)))
    placeholder[:, 0] = predictions_scaled
    predictions_unscaled = scaler.inverse_transform(placeholder)[:, 0]

    # ------------------------------------------------
    # 7. Convert Predicted Returns -> Predicted Prices
    # ------------------------------------------------
    predictions_prices = []
    last_price = train_data['Last Close'].iloc[-1]
    for prediction in predictions_unscaled:
        predicted_price = last_price * prediction
        predictions_prices.append(predicted_price)
        last_price = predicted_price

    return predictions_prices, train_data

def load_and_predict(model_path, train_data, look_back=10):
    """
    Loads a weights file and returns a 30-day forecast
    from the last window in the training set.
    """

    # ------------------------------------------------
    # 1. Feature Engineering
    # ------------------------------------------------
    # (A) Compute Returns
    train_data['Returns'] = train_data['Last Close'] / train_data['Open']
    
    # (B) 10-day MA of the closing price
    train_data['10d_MA'] = train_data['Last Close'].rolling(window=10).mean()

    # (C) 20-day MA
    train_data['20d_MA'] = train_data['Last Close'].rolling(window=20).mean()

    # (D) 10-day STD of daily returns
    train_data['10d_STD'] = train_data['Returns'].rolling(window=10).std()

    # (E) RSI (14) of Last Close
    train_data['RSI_14'] = compute_rsi(train_data['Last Close'], period=14)

    # (F) Bollinger Bands of Last Close
    bb_mid, bb_up, bb_low = compute_bollinger_bands(train_data['Last Close'], window=20, num_std=2)
    train_data['BB_MID'] = bb_mid
    train_data['BB_UP'] = bb_up
    train_data['BB_LOW'] = bb_low

    # (G) Day of Week
    if hasattr(train_data.index, 'dayofweek'):
        train_data['DayOfWeek'] = train_data.index.dayofweek
    else:
        train_data['DayOfWeek'] = 0

    # Fill missing values
    train_data.bfill(inplace=True)

    # ------------------------------------------------
    # 2. Select and Scale Features
    # ------------------------------------------------
    features = [
        'Returns',
        '10d_MA',
        '20d_MA',
        '10d_STD',
        'RSI_14',
        'BB_MID',
        'BB_UP',
        'BB_LOW',
        'DayOfWeek'
    ]

    # Fit scaler on the selected features
    scaler.fit(train_data[features])
    scaled_data = scaler.transform(train_data[features])
    scaled_train_data = pd.DataFrame(scaled_data, columns=features, index=train_data.index)

    # ------------------------------------------------
    # 3. Prepare Training Sequences
    # ------------------------------------------------
    x_train, y_train = prepare_data(scaled_train_data, look_back)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # ------------------------------------------------
    # 4. Load the Model
    # ------------------------------------------------
    input_size = len(features)
    hidden_size = 64
    output_size = 30 

    model = LSTMModel(input_size, hidden_size, output_size, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model weights loaded from {model_path}")

    # ------------------------------------------------
    # 5. Generate Forecast from Last Window
    # ------------------------------------------------
    with torch.no_grad():
        # Last `look_back` rows of the scaled features
        last_window = scaled_train_data.iloc[-look_back:].values
        last_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

        # Predict the next 30 returns (scaled)
        predictions_scaled = model(last_window).numpy().flatten()

    # ------------------------------------------------
    # 6. Inverse Transform (only 'Returns' column)
    # ------------------------------------------------
    placeholder = np.zeros((len(predictions_scaled), len(features)))
    placeholder[:, 0] = predictions_scaled
    predictions_unscaled = scaler.inverse_transform(placeholder)[:, 0]

    # ------------------------------------------------
    # 7. Convert Predicted Returns -> Predicted Prices
    # ------------------------------------------------
    predictions_prices = []
    last_price = train_data['Last Close'].iloc[-1]
    for prediction in predictions_unscaled:
        predicted_price = last_price * prediction
        predictions_prices.append(predicted_price)
        last_price = predicted_price

    return predictions_prices, train_data

# -----------------------------------------------
# Evaluate / Plot
# -----------------------------------------------
def calculate_mape(test_data, predictions_prices):
    # Align
    n_points = min(len(test_data), len(predictions_prices))
    predictions_prices_trimmed = predictions_prices[:n_points]
    test_data_trimmed = test_data.iloc[:n_points]

    # MAPE
    mape = mean_absolute_percentage_error(test_data_trimmed['Last Close'], predictions_prices_trimmed) * 100
    
    return mape, test_data_trimmed, predictions_prices_trimmed

def plot_predictions(train_data, test_data_trimmed, predictions_prices, mape):
    """
    Plots:
      - A continuous Bollinger Band for the entire timeframe (train + test)
      - The train close prices
      - The test close prices
      - The LSTM predictions on test
      - Shows MAPE as the plot title
    """

    # --------------------------------------------
    # 1) Combine Train + Test for Bollinger
    # --------------------------------------------
    # We only need the 'Last Close' column to compute Bollinger.
    df_combined = pd.concat([
        train_data[['Last Close']], 
        test_data_trimmed[['Last Close']]
    ], axis=0)

    # Compute Bollinger Bands on the entire merged dataset
    df_combined['BB_MID'], df_combined['BB_UP'], df_combined['BB_LOW'] = compute_bollinger_bands(
        df_combined['Last Close'], window=20, num_std=2
    )

    # --------------------------------------------
    # 2) Plot Continuous Bollinger for All Data
    # --------------------------------------------
    plt.figure(figsize=(12, 6))

    # Entire Bollinger Band from start of train to end of test
    plt.plot(df_combined.index, df_combined['BB_UP'], label='Bollinger Upper (All Data)', 
             color='orange', linestyle='--')
    plt.plot(df_combined.index, df_combined['BB_LOW'], label='Bollinger Lower (All Data)', 
             color='orange', linestyle='--')
    plt.fill_between(df_combined.index, df_combined['BB_LOW'], df_combined['BB_UP'], 
                     color='orange', alpha=0.08)  # shading

    # --------------------------------------------
    # 3) Plot Train Segment
    # --------------------------------------------
    plt.plot(train_data.index, train_data['Last Close'], label='Train Close', color='blue')

    # --------------------------------------------
    # 4) Plot Test Segment
    # --------------------------------------------
    plt.plot(test_data_trimmed.index, test_data_trimmed['Last Close'], 
             label='Test Close', color='green')

    # --------------------------------------------
    # 5) Plot Predictions
    # --------------------------------------------
    plt.plot(test_data_trimmed.index, predictions_prices, 
             label='LSTM Prediction', color='red', linestyle='--')

    # --------------------------------------------
    # 6) Finalize Figure
    # --------------------------------------------
    plt.title(f"Bollinger Bands (Continuous) | MAPE: {mape:.2f}%")
    plt.legend()
    plt.show()

def main(args):
    #######################
    ## HYPERPARAM TUNING ##
    #######################
    # df = pd.read_csv('../data/train.csv', index_col='Date', parse_dates=True)

    # # Reindex to daily frequency
    # df = df.asfreq('D')

    # # Forward-fill ALL columns to avoid NaNs
    # df.ffill(inplace=True)

    # # If there's a first row(s) were NaN
    # df.bfill(inplace=True)

    # result_df = hyperparam.run_hyperparam_tuning_wf(df)
    # print("\nHyperparam Tuning Results:")
    # print(result_df)

    #######################
    ## RUN THE MODEL     ##
    #######################
    train_data_path = args.training_file
    test_data_path = args.testing_file

    train_data = preprocess_data(train_data_path)
    test_data = preprocess_data(test_data_path)

    # RUN THIS TO TRAIN AND RUN THE MODEL
    #predictions_prices, train_data = train_and_predict(train_data, look_back=10)

    # RUN THIS TO RUN THE MODEL WITH A WEIGHTS FILE
    predictions_prices, train_data = load_and_predict("final_weights.pth", train_data, look_back=10)

    # CALCULATE MAPE
    mape, test_data_trimmed, predictions_prices_trimmed = calculate_mape(test_data, predictions_prices)

    # PLOT PREDICTIONS
    plot_predictions(train_data, test_data_trimmed, predictions_prices_trimmed, mape)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
