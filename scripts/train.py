import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error

def train_model(model, X_train, y_train, epochs=60, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
    
    preds_real = scaler.inverse_transform(preds)
    y_test_real = scaler.inverse_transform(y_test.numpy())
    
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    
    return rmse, preds_real, y_test_real