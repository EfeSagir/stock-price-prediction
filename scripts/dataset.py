import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def get_amazon_data(csv_path, lookback=20):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    prices = df['Price']
    if prices.dtype == object:
        prices = prices.str.replace(',', '').astype(float)
        
    prices = prices.values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(prices)
    
    X, y = create_sliding_windows(scaled_data, lookback)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, scaler

def create_sliding_windows(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)