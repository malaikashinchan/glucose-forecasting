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


#DataFrame
<img width="523" height="255" alt="image" src="https://github.com/user-attachments/assets/7200436e-afc0-49c1-b5e0-a05a116c75c2" />

#Sample Glucose Trace(after cleaning)
<img width="1171" height="463" alt="image" src="https://github.com/user-attachments/assets/2b96f7b2-6a8d-485c-a749-3cefcfbd5e10" />

#Random Forest Prediction
<img width="1161" height="488" alt="image" src="https://github.com/user-attachments/assets/81fb690d-fd20-43b1-9b07-168ee683507c" />

#LSTM Prediction
<img width="1188" height="559" alt="image" src="https://github.com/user-attachments/assets/99941b14-0048-447d-81dc-a15685dc6c1b" />
