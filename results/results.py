import matplotlib.pyplot as plt

def plot_results(y_test_real, lstm_preds, gru_preds):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label='Actual Prices', color='black', linewidth=2)
    plt.plot(lstm_preds, label='LSTM Predictions', color='blue', linestyle='--')
    plt.plot(gru_preds, label='GRU Predictions', color='orange', linestyle='--')
    
    plt.title('Amazon Stock Price Prediction: LSTM vs. GRU')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()