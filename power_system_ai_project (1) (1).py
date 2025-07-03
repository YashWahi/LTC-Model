# This is the master README file for your fully runnable project

## Power System AI Project: Load Tap Changer Control & Load Forecasting

### Folder Structure

```
power_system_ai_project/
|├─ data_loader.py  # Dataset extraction & preprocessing
|├─ train_mlp.py  # Supervised MLP model for voltage prediction
|├─ train_lstm.py  # Time-series LSTM model
|├─ train_rl_tap_control.py  # Reinforcement Learning tap changer
|├─ train_autoencoder_anomaly.py  # Anomaly detection via Autoencoder
|├─ test_models.py  # Evaluate all models
|├─ visualize.py  # Visualization for your report
|├─ requirements.txt  # All Python dependencies
|├─ models/  # Saved trained models
|└─ results/  # Saved generated plots
```

### How to Run

1. **Extract your original BSES dataset** (`dt health.zip`) into this directory.

2. **Install Dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run Training Modules Sequentially:**
```bash
python data_loader.py
python train_mlp.py
python train_lstm.py
python train_rl_tap_control.py
python train_autoencoder_anomaly.py
```

4. **Test Models:**
```bash
python test_models.py
```

5. **Visualize Outputs:**
```bash
python visualize.py
```

---

✅ This project is fully ready for your research paper & demo.

✅ It includes all zones and fully uses your provided data.

✅ Produces demo image suitable for report.

---

*You are now fully ready to proceed!*
