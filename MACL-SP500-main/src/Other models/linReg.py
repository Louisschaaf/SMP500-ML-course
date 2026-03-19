import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def train_and_predict_lin(train_data, test_data):
    """
    Trains a linear regression model on the training data and predicts for the test data.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.

    Returns:
        float: MAPE of the predictions on the test data.
    """
    # Prepare features for linear regression
    full_features = np.arange(len(train_data) + len(test_data)).reshape(-1, 1)
    train_features = np.arange(len(train_data)).reshape(-1, 1)
    train_target = train_data['Last Close'].values

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(train_features, train_target)

    # Predict for both train and test data
    full_predictions = model.predict(full_features)
    test_features = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
    test_predictions = model.predict(test_features)

    # Calculate MAPE for the test data
    mape = mean_absolute_percentage_error(test_data['Last Close'].values, test_predictions) * 100

    # Plot the results
    plot_predictions_with_lin(train_data, test_data, full_predictions)

    return mape

def plot_predictions_with_lin(train_data, test_data, full_predictions):
    """
    Plots train data, test data, and linear regression predictions.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        full_predictions (np.ndarray): Predicted closing prices for the full range (train + test).
    """
    plt.figure(figsize=(12, 6))

    # Plot train data
    plt.plot(train_data.index, train_data['Last Close'], label='Train Data (Close)', color='blue')

    # Plot test data
    plt.plot(test_data.index, test_data['Last Close'], label='Actual Test Data (Close)', color='green')

    # Plot full linear regression predictions
    full_index = pd.concat([train_data, test_data]).index
    plt.plot(full_index, full_predictions, label='Linear Regression Prediction', color='red', linestyle='dashed')

    plt.title('Train, Test, and Linear Regression Predictions (Closing Prices)')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_predict_poly(train_data, test_data, degree=10):
    """
    Trains a polynomial regression model on the training data and predicts for the test data.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        degree (int): Degree of the polynomial.

    Returns:
        float: MAPE of the predictions on the test data.
    """
    # Prepare features for polynomial regression
    full_features = np.arange(len(train_data) + len(test_data))
    train_features = np.arange(len(train_data))
    train_target = train_data['Last Close'].values

    # Fit the polynomial regression model
    poly_coeffs = np.polyfit(train_features, train_target, deg=degree)
    poly_model = np.poly1d(poly_coeffs)

    # Predict for both train and test data
    full_predictions = poly_model(full_features)
    test_features = np.arange(len(train_data), len(train_data) + len(test_data))
    test_predictions = poly_model(test_features)

    # Calculate MAPE for the test data
    mape = mean_absolute_percentage_error(test_data['Last Close'].values, test_predictions) * 100

    # Plot the results
    plot_predictions_with_poly(train_data, test_data, full_predictions, degree)

    return mape

def plot_predictions_with_poly(train_data, test_data, full_predictions, degree):
    """
    Plots train data, test data, and polynomial regression predictions.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        full_predictions (np.ndarray): Predicted closing prices for the full range (train + test).
        degree (int): Degree of the polynomial.
    """
    plt.figure(figsize=(12, 6))

    # Plot train data
    plt.plot(train_data.index, train_data['Last Close'], label='Train Data (Close)', color='blue')

    # Plot test data
    plt.plot(test_data.index, test_data['Last Close'], label='Actual Test Data (Close)', color='green')

    # Plot full polynomial regression predictions
    full_index = pd.concat([train_data, test_data]).index
    plt.plot(full_index, full_predictions, label=f'Polynomial Regression (Degree {degree})', color='orange', linestyle='dashed')

    plt.title(f'Train, Test, and Polynomial Regression Predictions (Closing Prices)')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_predict_combined(train_data, test_data, poly_degree=10):
    """
    Trains both linear and polynomial regression models, and combines their predictions
    by averaging them for the final prediction.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        poly_degree (int): Degree of the polynomial for the regression.

    Returns:
        float: MAPE of the combined predictions on the test data.
    """
    # Prepare features for linear regression
    full_features = np.arange(len(train_data) + len(test_data)).reshape(-1, 1)
    train_features = np.arange(len(train_data)).reshape(-1, 1)
    train_target = train_data['Last Close'].values

    # Linear regression
    linear_model = LinearRegression()
    linear_model.fit(train_features, train_target)
    full_linear_predictions = linear_model.predict(full_features)
    test_features = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
    test_linear_predictions = linear_model.predict(test_features)

    # Polynomial regression
    poly_coeffs = np.polyfit(np.arange(len(train_data)), train_target, deg=poly_degree)
    poly_model = np.poly1d(poly_coeffs)
    full_poly_predictions = poly_model(np.arange(len(train_data) + len(test_data)))
    test_poly_predictions = poly_model(np.arange(len(train_data), len(train_data) + len(test_data)))

    # Combine predictions by averaging
    test_combined_predictions = (test_linear_predictions + test_poly_predictions) / 2
    full_combined_predictions = (full_linear_predictions + full_poly_predictions) / 2

    # Calculate MAPE for the combined predictions
    mape = mean_absolute_percentage_error(test_data['Last Close'].values, test_combined_predictions) * 100

    # Plot the results
    plot_predictions_combined(train_data, test_data, full_combined_predictions, poly_degree)

    return mape

def plot_predictions_combined(train_data, test_data, full_combined_predictions, poly_degree):
    """
    Plots train data, test data, and combined regression predictions.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        full_combined_predictions (np.ndarray): Combined predictions for the full range (train + test).
        poly_degree (int): Degree of the polynomial for reference in the plot title.
    """
    plt.figure(figsize=(12, 6))

    # Plot train data
    plt.plot(train_data.index, train_data['Last Close'], label='Train Data (Close)', color='blue')

    # Plot test data
    plt.plot(test_data.index, test_data['Last Close'], label='Actual Test Data (Close)', color='green')

    # Plot combined regression predictions
    full_index = pd.concat([train_data, test_data]).index
    plt.plot(full_index, full_combined_predictions, label=f'Combined Regression (Linear + Polynomial Degree {poly_degree})', color='purple', linestyle='dashed')

    plt.title('Train, Test, and Combined Regression Predictions (Closing Prices)')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()