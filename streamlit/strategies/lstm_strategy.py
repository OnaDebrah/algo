"""
LSTM Neural Network Strategy using PyTorch
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from streamlit.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM Model"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class LSTMStrategy(BaseStrategy):
    """LSTM Strategy for Time Series Forecasting"""

    def __init__(
        self,
        name: str = "LSTM Strategy",
        lookback: int = 10,
        hidden_dim: int = 32,
        num_layers: int = 1,
        epochs: int = 20,
        learning_rate: float = 0.01,
        classes: int = 2,
    ):
        """
        Initialize LSTM strategy

        Args:
            name: Strategy name
            lookback: Sequence length for LSTM
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            epochs: Training epochs
            learning_rate: Training learning rate
            classes: Number of output classes
        """
        params = {
            "name": name,
            "lookback": lookback,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "classes": classes,
        }
        super().__init__(name, params)

        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.classes = classes

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = []

    def prepare_data(self, data: pd.DataFrame):
        """Prepare data for LSTM (create sequences)"""
        df = data.copy()

        # Simple features: Returns, High/Low range, Volatility
        df["returns"] = df["Close"].pct_change()
        df["range"] = (df["High"] - df["Low"]) / df["Close"]
        df["volatility"] = df["returns"].rolling(10).std()

        df = df.dropna()
        self.feature_cols = ["returns", "range", "volatility"]

        return df[self.feature_cols].values, df["returns"].shift(-1).dropna().values  # Predicting next return

    def create_sequences(self, data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length - 1):
            x = data[i : (i + seq_length)]
            y = data[i + seq_length]  # We are predicting the move at the END of the sequence?
            # Actually, standard is: inputs [t-N...t], predict t+1.
            # In prepare_data, I aligned 'returns' to be current returns.
            # The target should be *future* returns.
            # Let's adjust slightly:
            # We want to use X[t-N : t] to predict Y[t+1]
            xs.append(x)
            ys.append(y)  # y is already shifted in prepare_data if we did it right

        return np.array(xs), np.array(ys)

    def train(self, data: pd.DataFrame, test_size: float = 0.2):
        """Train the LSTM model"""
        features, targets = self.prepare_data(data)

        # Scale features
        # ... (reuse existing logic but split first?)
        # Let's keep it simple for now: Train on ALL, test on same (or split if strictly needed)
        # To match MLStrategy, we should split.

        # Redoing data prep locally to control split
        df = data.copy()
        df["returns"] = df["Close"].pct_change().dropna()
        df = df.dropna()

        dataset_values = df["returns"].values.reshape(-1, 1)
        self.feature_cols = ["returns"]

        # Scale
        scaled_data = self.scaler.fit_transform(dataset_values)

        X, y = [], []
        for i in range(len(scaled_data) - self.lookback):
            X.append(scaled_data[i : i + self.lookback])
            next_ret = scaled_data[i + self.lookback]
            label = 1 if next_ret > 0 else 0
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Split into train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train_np, X_test_np = X[:split_idx], X[split_idx:]
        y_train_np, y_test_np = y[:split_idx], y[split_idx:]

        X_train = torch.from_numpy(X_train_np).float()
        y_train = torch.from_numpy(y_train_np).long()

        # Model
        self.model = LSTMModel(input_dim=1, hidden_dim=self.hidden_dim, output_dim=self.classes, num_layers=self.num_layers)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for i in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        self.is_trained = True
        logger.info("LSTM Model Trained")

        # Calculate scores
        self.model.eval()
        with torch.no_grad():
            # Train score
            train_out = self.model(X_train)
            _, train_pred = torch.max(train_out.data, 1)
            train_score = (train_pred == y_train).sum().item() / len(y_train)

            # Test score
            if len(X_test_np) > 0:
                X_test = torch.from_numpy(X_test_np).float()
                y_test = torch.from_numpy(y_test_np).long()
                test_out = self.model(X_test)
                _, test_pred = torch.max(test_out.data, 1)
                test_score = (test_pred == y_test).sum().item() / len(y_test)
            else:
                test_score = 0.0

        return train_score, test_score

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate signal"""
        if not self.is_trained or len(data) < self.lookback + 2:
            return 0

        # Prepare latest sequence
        last_returns = data["Close"].pct_change().tail(self.lookback).values.reshape(-1, 1)

        if len(last_returns) < self.lookback:
            return 0

        scaled_seq = self.scaler.transform(last_returns)
        seq_tensor = torch.from_numpy(scaled_seq).float().unsqueeze(0)  # Batch size 1

        self.model.eval()
        with torch.no_grad():
            output = self.model(seq_tensor)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()

        # Prediction 1 = Up, 0 = Down/Hold
        return 1 if prediction == 1 else -1
