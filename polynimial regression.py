import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# File paths
fpt_path = 'D:/FPT.csv'
msn_path = 'D:/MSN.csv'
pnj_path = 'D:/PNJ.csv'
vic_path = 'D:/VIC.csv'

# Perform polynomial regression and plot for each file path
file_paths = [fpt_path, msn_path, pnj_path, vic_path]

for path in file_paths:
    # Read data from CSV file into DataFrame
    data_df = pd.read_csv(path)

    # Rename columns to match the desired format
    data_df.columns = ['Ticker', 'Date/Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest']

    # Convert 'Date/Time' column to standard format
    data_df['Date/Time'] = pd.to_datetime(data_df['Date/Time'])

    # Select the desired features and target variable
    X = data_df['Date/Time'].values.reshape(-1, 1)
    y = data_df['Close'].values

    # Define the degree of the polynomial
    degree = 5  # You can adjust the degree as needed

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Make predictions
    y_pred = model.predict(X_poly)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("RMSE -", path, ":", rmse)
    coefficients = model.coef_
    print("Coefficients:", coefficients)

    # Sort the data for plotting
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Plot the data and predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Polynomial Regression')
    ax.plot(X, y, color='blue', linewidth=2, label='Actual Data')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Close Price')
    ax.set_title('Polynomial Regression - ' + path)
    ax.legend()
    plt.show()