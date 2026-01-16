# model.py
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        out, (hn, _) = self.lstm(x)
        h = hn[-1]                # final layer hidden state
        return self.fc(h).squeeze()
