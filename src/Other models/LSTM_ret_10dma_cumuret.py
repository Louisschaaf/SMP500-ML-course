import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# ------------------------------
# Define the LSTM Model
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Predict the next 30 days

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ------------------------------
# Prepare Data Function
# ------------------------------
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
    # Need at least look_back days + 30 days for future
    if len(data) < look_back + 30:
        raise ValueError("Not enough data to prepare the required sequences.")

    x, y = [], []
    # Sliding window
    for i in range(len(data) - look_back - 30 + 1):
        # Input: the past `look_back` days
        x_window = data.iloc[i:i + look_back].values  # shape: (look_back, num_features)

        # Output: the next 30 days of 'Returns'
        y_future = data['Returns'].iloc[i + look_back : i + look_back + 30].values  # shape: (30,)

        x.append(x_window)
        y.append(y_future)

    return np.array(x), np.array(y)

# ------------------------------
# Train and Predict
# ------------------------------ 
def train_and_predict(train_data, look_back=15):
    """
    Trains the LSTM model on `train_data` and returns
    a 30-day forecast from the last window in the training set.

    Args:
        train_data (pd.DataFrame): Training data with 'Last Close' column.
        look_back (int): The length of history used for each training sample.

    Returns:
        (predictions_prices, train_data) tuple
    """

    # ------------------------------------------------
    # 1. Feature Engineering
    # ------------------------------------------------
    # 1) 10-day moving average of the closing price
    train_data['10d_MA'] = train_data['Last Close'].rolling(window=10).mean()

    # 2) Cumulative returns
    train_data['Cumulative_Returns'] = (train_data['Last Close'] / train_data['Last Close'].iloc[0]).cumprod()

    # Fill missing values
    train_data.bfill(inplace=True)

    # ------------------------------------------------
    # 2. Select and Scale Features
    # ------------------------------------------------
    features = ['Returns', '10d_MA', 'Cumulative_Returns']
    scaler.fit(train_data[features])
    scaled_data = scaler.transform(train_data[features])

    # Create a scaled DataFrame
    scaled_train_data = pd.DataFrame(scaled_data, columns=features, index=train_data.index)

    # ------------------------------------------------
    # 3. Prepare Training Sequences
    # ------------------------------------------------
    x_train, y_train = prepare_data(scaled_train_data, look_back)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # ------------------------------------------------
    # 4. Initialize and Train the Model
    # ------------------------------------------------
    input_size = len(features)
    hidden_size = 100
        
    # 30-day forecast
    output_size = 30
    model = LSTMModel(input_size, hidden_size, output_size, num_layers=2)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 75
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # ------------------------------------------------
    # 5. Generate Forecast from Last Window
    # ------------------------------------------------
    model.eval()
    with torch.no_grad():
        # Get the last `look_back` rows of the scaled features
        last_window = scaled_train_data.iloc[-look_back:].values
        last_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)
        
        # Predict the next 30 returns
        predictions_scaled = model(last_window).numpy().flatten()

    # Placeholder so we can inverse_transform just the 'Returns' column
    placeholder = np.zeros((len(predictions_scaled), len(features)))

    # Insert the predictions into the 'Returns' column (assumed to be the first feature)
    placeholder[:, 0] = predictions_scaled
    
    # Inverse transform using the scaler
    predictions_unscaled = scaler.inverse_transform(placeholder)[:, 0]

    # ------------------------------------------------
    # 6. Convert Predicted Returns -> Predicted Prices
    # ------------------------------------------------
    predictions_prices = []
    last_price = train_data['Last Close'].iloc[-1]

    for prediction in predictions_unscaled:
        predicted_price = last_price * prediction
        predictions_prices.append(predicted_price)
        last_price = predicted_price

    return predictions_prices, train_data

# ------------------------------
# Evaluate / Plot
# ------------------------------
def calculate_mape_plot(train_data, test_data, predictions_prices):
    # Trim predictions and test data
    n_points = min(len(test_data), len(predictions_prices))
    predictions_prices = predictions_prices[:n_points]
    test_data_trimmed = test_data.iloc[:n_points]

    # Print predictions and test data side by side
    # print(pd.DataFrame({'Predictions': predictions_prices, 'Actual': test_data_trimmed['Last Close']}))

    # Plot predictions
    plot_predictions(train_data, test_data_trimmed, predictions_prices)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_data_trimmed['Last Close'], predictions_prices) * 100

    return mape

def plot_predictions(train_data, test_data_trimmed, predictions_prices):
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot train data
    plt.plot(train_data.index, train_data['Last Close'], label='Train Data (Close)', color='blue')

    # Plot trimmed test data
    plt.plot(test_data_trimmed.index, test_data_trimmed['Last Close'], label='Actual Test Data (Close)', color='green')

    # Plot predictions
    plt.plot(test_data_trimmed.index, predictions_prices, label='LSTM Prediction', color='red', linestyle='dashed')

    plt.legend()
    #plt.show()
