import tensorflow as tf
import numpy as np
import joblib
from data_loader import extract_and_load_data

full_df = extract_and_load_data()
features = full_df[['Load']]
targets = full_df[['Voltage']]

scaler_X = joblib.load("models/scaler_X.save")
scaler_y = joblib.load("models/scaler_y.save")
X_scaled = scaler_X.transform(features)
y_scaled = scaler_y.transform(targets)
mlp_model = tf.keras.models.load_model("models/mlp_forecast_model.h5")
mlp_predictions = mlp_model.predict(X_scaled)
mlp_mse = np.mean((mlp_predictions - y_scaled)**2)
print(f"MLP Test MSE: {mlp_mse:.4f}")

lstm_scaler_X = joblib.load("models/lstm_scaler_X.save")
lstm_scaler_y = joblib.load("models/lstm_scaler_y.save")
X_scaled_lstm = lstm_scaler_X.transform(features)
y_scaled_lstm = lstm_scaler_y.transform(targets)
sequence_length = 10
X_seq = []
for i in range(len(X_scaled_lstm) - sequence_length):
    X_seq.append(X_scaled_lstm[i:i+sequence_length])
X_seq = np.array(X_seq)
lstm_model = tf.keras.models.load_model("models/lstm_forecast_model.h5")
lstm_predictions = lstm_model.predict(X_seq)
lstm_mse = np.mean((lstm_predictions - y_scaled_lstm[sequence_length:])**2)
print(f"LSTM Test MSE: {lstm_mse:.4f}") 