import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from dateutil.relativedelta import relativedelta

####################################################
# GLOBALS & HELPER FUNCTIONS
####################################################

FORECAST_DAYS = 20
LOOK_BACK = 10
scaler = MinMaxScaler(feature_range=(0, 1))

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(price, window=20, num_std=2):
    sma = price.rolling(window=window).mean()
    std = price.rolling(window=window).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    return sma, upper, lower


####################################################
# LSTM MODEL
####################################################
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


####################################################
# PREPARE DATA (FIXED MULTI-STEP)
####################################################
def prepare_data(data, look_back=LOOK_BACK):
    needed = look_back + FORECAST_DAYS
    if len(data) < needed:
        raise ValueError(f"Not enough data to form sequences (need {needed} rows).")
    x, y = [], []
    for i in range(len(data) - needed + 1):
        x_window = data.iloc[i : i + look_back].values
        y_future = data['Returns'].iloc[i + look_back : i + look_back + FORECAST_DAYS].values
        x.append(x_window)
        y.append(y_future)
    return np.array(x), np.array(y)


####################################################
# FEATURE ENGINEERING
####################################################
def feature_engineering(df):
    df = df.copy()
    # 'Returns' = 'Last Close' / 'Open' (adapt if needed)
    df['Returns'] = df['Last Close'] / df['Open']
    df['10d_MA'] = df['Last Close'].rolling(window=10).mean()
    df['20d_MA'] = df['Last Close'].rolling(window=20).mean()
    df['10d_STD'] = df['Returns'].rolling(window=10).std()
    df['RSI_14'] = compute_rsi(df['Last Close'], period=14)
    bb_mid, bb_up, bb_low = compute_bollinger_bands(df['Last Close'], window=20, num_std=2)
    df['BB_MID'] = bb_mid
    df['BB_UP'] = bb_up
    df['BB_LOW'] = bb_low
    if hasattr(df.index, 'dayofweek'):
        df['DayOfWeek'] = df.index.dayofweek
    else:
        df['DayOfWeek'] = 0
    df.bfill(inplace=True)
    return df


####################################################
# TRAIN + FORECAST ONE MONTH
####################################################
def train_and_forecast_month(train_data, test_data, look_back=LOOK_BACK, hidden_size=100, num_layers=2, lr=0.001, epochs=50, batch_size=32):
    """
    1) Train on train_data (feature engineering + LSTM)
    2) Forecast FORECAST_DAYS from the last window
    3) Compare to first FORECAST_DAYS of test_data
    Returns MAPE, or None if not enough data
    """
    if len(test_data) < FORECAST_DAYS:
        return None  # skip if month < FORECAST_DAYS

    # 1) FEATURE ENG on train
    df_train = feature_engineering(train_data)
    features = [
        'Returns','10d_MA','20d_MA','10d_STD','RSI_14',
        'BB_MID','BB_UP','BB_LOW','DayOfWeek'
    ]
    # Fit scaler on train only
    scaler.fit(df_train[features])
    scaled_train = scaler.transform(df_train[features])
    scaled_train_df = pd.DataFrame(scaled_train, columns=features, index=df_train.index)

    try:
        x_train, y_train = prepare_data(scaled_train_df, look_back=look_back)
    except ValueError:
        return None

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define model
    input_size = len(features)
    output_size = FORECAST_DAYS
    model = LSTMModel(input_size, hidden_size, output_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 2) Training
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # 3) Forecast from last train window
    model.eval()
    with torch.no_grad():
        last_window_scaled = scaled_train_df.iloc[-look_back:].values
        last_window_tensor = torch.tensor(last_window_scaled, dtype=torch.float32).unsqueeze(0)
        pred_returns_scaled = model(last_window_tensor).numpy().flatten()

    # Inverse transform
    placeholder = np.zeros((len(pred_returns_scaled), len(features)))
    placeholder[:, 0] = pred_returns_scaled
    pred_returns_unscaled = scaler.inverse_transform(placeholder)[:, 0]

    # Convert returns->prices
    last_known_price = df_train['Last Close'].iloc[-look_back - 1]
    predicted_prices = []
    running_price = last_known_price
    for ret in pred_returns_unscaled:
        next_price = running_price * ret
        predicted_prices.append(next_price)
        running_price = next_price

    # Compare to first FORECAST_DAYS of test_data
    df_test = feature_engineering(test_data)
    actual_prices = df_test['Last Close'].iloc[:FORECAST_DAYS].values
    if len(actual_prices) < FORECAST_DAYS:
        return None

    mape_month = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
    return mape_month


####################################################
# MONTHLY WALK-FORWARD
####################################################
def monthly_walk_forward(df_full, start_date='2024-01-01', end_date='2024-09-30', look_back=LOOK_BACK, hidden_size=100, num_layers=2, lr=0.001, epochs=50, batch_size=32):
    df_full = df_full.sort_index()
    # df_full = df_full.asfreq('D')
    # df_full.ffill(inplace=True)

    # We'll assume you have from 2018 onward
    df_full = df_full.loc['2018-01-01':]

    current_start = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)
    monthly_results = []

    while current_start <= final_end:
        month_start = current_start
        next_month_start = month_start + relativedelta(months=1)
        month_end = next_month_start - pd.Timedelta(days=1)
        if month_end > final_end:
            month_end = final_end

        # Train = up to day before month_start
        train_data = df_full.loc[:(month_start - pd.Timedelta(days=1))]
        # Test = month_start->month_end
        test_data = df_full.loc[month_start: month_end]

        # Enough train data?
        if len(train_data) < (look_back + FORECAST_DAYS):
            print(f"{month_start.strftime('%Y-%m')} - Not enough train data.")
            current_start = next_month_start
            continue

        if len(test_data) < 1:
            print(f"{month_start.strftime('%Y-%m')} - No test data.")
            current_start = next_month_start
            continue

        mape_month = train_and_forecast_month(
            train_data=train_data,
            test_data=test_data,
            look_back=look_back,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
        )

        month_label = month_start.strftime('%Y-%m')
        if mape_month is not None:
            monthly_results.append({
                'Month': month_label,
                'Start': month_start,
                'End': month_end,
                'MAPE': mape_month
            })
            print(f"{month_label} - MAPE: {mape_month:.2f}%")
        else:
            print(f"{month_label} - Not enough data for forecast or other issue.")

        current_start = next_month_start

    results_df = pd.DataFrame(monthly_results)
    avg_mape = results_df['MAPE'].mean() if not results_df.empty else None
    return results_df, avg_mape


####################################################
# HYPERPARAM TUNING (monthly walk-forward)
####################################################
def run_hyperparam_tuning_wf(df):
    """
    Example hyperparam grid for monthly_walk_forward.
    Each iteration:
      1) Run monthly_walk_forward with that param set.
      2) Store the final average MAPE.
    """
    hyperparam_grid = [
        # 1) Variation in hidden_size
        {'hidden_size': 32,  'num_layers': 1, 'lr': 0.001,  'epochs': 30,  'batch_size': 16},
        {'hidden_size': 32,  'num_layers': 1, 'lr': 0.001,  'epochs': 30,  'batch_size': 32},
        {'hidden_size': 32,  'num_layers': 2, 'lr': 0.001,  'epochs': 50,  'batch_size': 32},
        
        # 2) Variation in learning rate
        {'hidden_size': 64,  'num_layers': 2, 'lr': 0.0005, 'epochs': 50,  'batch_size': 32},
        {'hidden_size': 64,  'num_layers': 2, 'lr': 0.001,  'epochs': 50,  'batch_size': 32},
        {'hidden_size': 64,  'num_layers': 2, 'lr': 0.005,  'epochs': 50,  'batch_size': 32},
        
        # 3) Vary epochs
        {'hidden_size': 100, 'num_layers': 2, 'lr': 0.001,  'epochs': 30,  'batch_size': 16},
        {'hidden_size': 100, 'num_layers': 2, 'lr': 0.001,  'epochs': 50,  'batch_size': 32},
        {'hidden_size': 100, 'num_layers': 2, 'lr': 0.001,  'epochs': 75,  'batch_size': 32},
        
        # 4) More layers
        {'hidden_size': 128, 'num_layers': 3, 'lr': 0.001,  'epochs': 50,  'batch_size': 32},
        {'hidden_size': 128, 'num_layers': 3, 'lr': 0.0005, 'epochs': 75,  'batch_size': 32},
        
        # 5) Variation in batch sizes
        {'hidden_size': 64,  'num_layers': 2, 'lr': 0.001,  'epochs': 50,  'batch_size': 16},
        {'hidden_size': 64,  'num_layers': 2, 'lr': 0.001,  'epochs': 50,  'batch_size': 64},
        
        # 6) Even larger hidden size
        {'hidden_size': 200, 'num_layers': 2, 'lr': 0.001,  'epochs': 50,  'batch_size': 32},
        {'hidden_size': 200, 'num_layers': 2, 'lr': 0.0005, 'epochs': 75,  'batch_size': 32},
        
        # 7) A “big” combination
        {'hidden_size': 128, 'num_layers': 3, 'lr': 0.0005, 'epochs': 100, 'batch_size': 16},
    ]

    results = []
    for i, params in enumerate(hyperparam_grid, start=1):
        print(f"\n--- Running experiment {i} with {params} ---")
        # Run the monthly walk-forward for each param set
        results_df, avg_mape = monthly_walk_forward(
            df,
            start_date='2024-01-01',
            end_date='2024-09-30',
            look_back=LOOK_BACK,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            lr=params['lr'],
            epochs=params['epochs'],
            batch_size=params['batch_size']
        )
        results.append({
            'Experiment': i,
            'Hidden Size': params['hidden_size'],
            'Num Layers': params['num_layers'],
            'Learning Rate': params['lr'],
            'Epochs': params['epochs'],
            'Batch Size': params['batch_size'],
            'Average MAPE': avg_mape
        })
    
    return pd.DataFrame(results)


####################################################
# MAIN (EXAMPLE)
####################################################
if __name__ == "__main__":
    # 1) Load your CSV
    df = pd.read_csv("../data/train.csv", index_col="Date", parse_dates=True)

    # 2) Optionally reindex to daily freq if you want weekends included
    #    Then forward-fill all columns to remove NaNs on weekends:
    # df = df.asfreq('D')
    # df.ffill(inplace=True)

    # 3) Run hyperparam tuning
    result_df = run_hyperparam_tuning_wf(df)
    print("\nHyperparam Tuning Results:")
    print(result_df)
