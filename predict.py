from scripts.dataset import get_amazon_data
from scripts.model import LSTMModel, GRUModel
from scripts.train import train_model, evaluate_model
from results.results import plot_results
import os
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, 'data', 'raw', 'AMZN[01.04.2021-12.31.2025].csv')    
    X_train, y_train, X_test, y_test, scaler = get_amazon_data(csv_path, lookback=20)
    
    print("--- Training LSTM Model ---")
    lstm_model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    lstm_model = train_model(lstm_model, X_train, y_train, epochs=60, lr=0.01)
    lstm_rmse, lstm_preds, y_test_real = evaluate_model(lstm_model, X_test, y_test, scaler)
    
    print("\n--- Training GRU Model ---")
    gru_model = GRUModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    gru_model = train_model(gru_model, X_train, y_train, epochs=60, lr=0.01)
    gru_rmse, gru_preds, _ = evaluate_model(gru_model, X_test, y_test, scaler)
    
    print(f"\n[Results Summary]")
    print(f"LSTM RMSE: {lstm_rmse:.4f}")
    print(f"GRU RMSE: {gru_rmse:.4f}")
    
    plot_results(y_test_real, lstm_preds, gru_preds)

if __name__ == "__main__":
    main()
