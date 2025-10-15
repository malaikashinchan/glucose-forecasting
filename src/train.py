import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

from models import SimpleLSTM
from utils import create_lag_features, split_train_test

# --- SETTINGS ---
DATA_PATH = "data/ohio_cgm_combined.csv"
os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD & PREP DATA ---
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values(["patient", "timestamp"]).reset_index(drop=True)
df["glucose_smooth"] = df["glucose"].rolling(window=3, min_periods=1).mean()
df = create_lag_features(df)
train_df, test_df, features, target_col = split_train_test(df)

# --- RANDOM FOREST BASELINE ---
X_train, y_train = train_df[features].values, train_df[target_col].values
X_test, y_test = test_df[features].values, test_df[target_col].values

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
print(f"RF MAE={mae_rf:.3f}  RMSE={rmse_rf:.3f}")

# --- LSTM TRAINING ---
class SequenceDataset(Dataset):
    def __init__(self, df, features, target_col):
        self.X = df[features].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx].reshape(-1, 1)
        y = np.array([self.y[idx]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

train_loader = DataLoader(SequenceDataset(train_df, features, target_col), batch_size=64, shuffle=True)
test_loader = DataLoader(SequenceDataset(test_df, features, target_col), batch_size=64, shuffle=False)

model = SimpleLSTM(input_size=1, hidden_size=64, num_layers=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(1, 11):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch}: TrainLoss={np.mean(losses):.5f}")

torch.save(model.state_dict(), "models/lstm_cgm.pth")
print("✅ Saved model to models/lstm_cgm.pth")
