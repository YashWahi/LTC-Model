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