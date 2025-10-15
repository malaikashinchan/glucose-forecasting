import pandas as pd
import numpy as np

def create_lag_features(df, n_lags=12, horizon=6):
    def make_feats(g):
        g = g.sort_values("timestamp").copy()
        for lag in range(1, n_lags + 1):
            g[f"lag_{lag}"] = g["glucose_smooth"].shift(lag)
        g[f"target_h{horizon}"] = g["glucose_smooth"].shift(-horizon)
        return g
    df = df.groupby("patient", group_keys=False).apply(make_feats)
    df = df.dropna().reset_index(drop=True)
    return df

def split_train_test(df, n_lags=12, horizon=6, ratio=0.8):
    features = [f"lag_{i}" for i in range(1, n_lags + 1)]
    target = f"target_h{horizon}"
    train_rows, test_rows = [], []
    for pid, g in df.groupby("patient"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(g) * ratio)
        train_rows.append(g.iloc[:split_idx])
        test_rows.append(g.iloc[split_idx:])
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    return train_df, test_df, features, target
