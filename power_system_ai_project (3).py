# Full Project Structure - Complete Code

# File: data_loader.py

import pandas as pd
import os
import zipfile

def extract_and_load_data(zip_path='dt_health.zip', extract_path='data/'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    all_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.xlsx'):
                all_files.append(os.path.join(root, file))
    full_df = pd.DataFrame()
    for file_path in all_files:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        full_df = pd.concat([full_df, df], ignore_index=True)
    full_df.dropna(inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    return full_df

if __name__ == "__main__":
    df = extract_and_load_data()


# File: train_mlp.py

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from data_loader import extract_and_load_data

full_df = extract_and_load_data()
features = full_df[['Load']]
targets = full_df[['Voltage']]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(targets)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
model.save("models/mlp_forecast_model.h5")
joblib.dump(scaler_X, "models/scaler_X.save")
joblib.dump(scaler_y, "models/scaler_y.save")


# File: train_lstm.py

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


# File: train_rl_tap_control.py

import gym
import numpy as np
from stable_baselines3 import PPO
from gym import spaces

class LTCEnv(gym.Env):
    def __init__(self):
        super(LTCEnv, self).__init__()
        self.observation_space = spaces.Box(low=np.array([0.8, 0]), high=np.array([1.2, 2000]), dtype=np.float32)
        self.action_space = spaces.Discrete(33)
        self.reset()
    def step(self, action):
        tap_change = action - 16
        self.tap_position += tap_change
        self.tap_position = np.clip(self.tap_position, -16, 16)
        base_voltage = 1.0 + self.tap_position * 0.00625
        voltage_drop = (self.load / 2000.0) * 0.05
        self.voltage = base_voltage - voltage_drop + np.random.normal(0, 0.003)
        self.load += np.random.uniform(-50, 50)
        self.load = np.clip(self.load, 100, 2000)
        voltage_penalty = abs(self.voltage - 1.0)
        overload_penalty = max(0, self.load - 1500) / 500.0
        tap_penalty = abs(tap_change) * 0.01
        reward = -(voltage_penalty + overload_penalty + tap_penalty)
        obs = np.array([self.voltage, self.load], dtype=np.float32)
        done = False
        return obs, reward, done, {}
    def reset(self):
        self.voltage = np.random.uniform(0.95, 1.05)
        self.load = np.random.uniform(500, 1500)
        self.tap_position = 0
        return np.array([self.voltage, self.load], dtype=np.float32)

if __name__ == "__main__":
    env = LTCEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("models/ppo_tap_controller")


# File: train_autoencoder_anomaly.py

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from data_loader import extract_and_load_data

full_df = extract_and_load_data()
features = full_df[['Voltage', 'Load']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
X_train, X_test = train_test_split(features_scaled, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(2, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, validation_split=0.2, epochs=50, batch_size=32)
autoencoder.save("models/autoencoder_anomaly_model.h5")
joblib.dump(scaler, "models/autoencoder_scaler.save")


# File: test_models.py

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

features_full = full_df[['Voltage', 'Load']]
auto_scaler = joblib.load("models/autoencoder_scaler.save")
features_scaled_auto = auto_scaler.transform(features_full)
auto_model = tf.keras.models.load_model("models/autoencoder_anomaly_model.h5")
reconstructions = auto_model.predict(features_scaled_auto)
reconstruction_errors = np.mean((reconstructions - features_scaled_auto)**2, axis=1)
thresh = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > thresh
print(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} samples")


# File: visualize.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
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
mlp_predictions_inverse = scaler_y.inverse_transform(mlp_predictions)
plt.figure(figsize=(12,6))
plt.plot(targets[:200].values, label="Actual Voltage", color='blue')
plt.plot(mlp_predictions_inverse[:200], label="MLP Predicted", color='red', linestyle='--')
plt.xlabel("Samples")
plt.ylabel("Voltage (p.u.)")
plt.title("Voltage Forecasting - Actual vs MLP Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("results/voltage_forecast_demo.png")
plt.show()


# File: requirements.txt

pandas
numpy
scikit-learn
matplotlib
tensorflow
stable-baselines3
joblib
gym
openpyxl
