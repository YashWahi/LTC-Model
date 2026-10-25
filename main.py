import os
import subprocess

# Ensure required folders exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Step 1: Data Loading
print('Extracting and loading data...')
subprocess.run(['python', 'data_loader.py'], check=True)

# Step 2: Train MLP
print('Training MLP model...')
subprocess.run(['python', 'train_mlp.py'], check=True)

# Step 3: Train LSTM
print('Training LSTM model...')
subprocess.run(['python', 'train_lstm.py'], check=True)

# Step 4: Train RL Tap Changer
print('Training RL Tap Changer...')
subprocess.run(['python', 'train_rl_tap_control.py'], check=True)

# Step 5: Train Autoencoder for Anomaly Detection
print('Training Autoencoder for anomaly detection...')
subprocess.run(['python', 'train_autoencoder_anomaly.py'], check=True)

# Step 6: Test Models
print('Testing models...')
subprocess.run(['python', 'test_models.py'], check=True)

print('All steps completed!') 