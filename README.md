# Glucose Level Forecasting (CGM) — LSTM + Classical ML

Summary: Forecast future glucose (30 minutes ahead) from past CGM sensor data. Compares RandomForest and LSTM.

Tech: Python, Pandas, Scikit-learn, PyTorch, Matplotlib

How to run:
1. Activate virtualenv
2. `pip install -r requirements.txt`
3. Start Jupyter: `jupyter lab` and open `notebooks/01_cgm_forecast.ipynb`

Results:
Final LSTM MAE: 1.3550435304641724
Final LSTM RMSE: 7.616977160557403

Extensions: Add attention, deploy as API, anomaly detection, train on real CGM dataset.
