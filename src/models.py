import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, out_size=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)
