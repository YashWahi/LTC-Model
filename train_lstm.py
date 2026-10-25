import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from data_loader import extract_and_load_data

full_df = extract_and_load_data()
full_df = full_df.sort_values(by='Time')
features = full_df[['Load']]
targets = full_df[['Voltage']]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(targets)
sequence_length = 10
X_seq, y_seq = [], []
for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y_scaled[i+sequence_length])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
model = models.Sequential([
    layers.Input(shape=(sequence_length, X_train.shape[2])),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
model.save("models/lstm_forecast_model.h5")
joblib.dump(scaler_X, "models/lstm_scaler_X.save")
joblib.dump(scaler_y, "models/lstm_scaler_y.save") 