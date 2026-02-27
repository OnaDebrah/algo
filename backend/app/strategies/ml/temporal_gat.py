"""
Temporal Graph Attention Network for volatility prediction [citation:4]
Captures cross-market spillovers and volatility transmission
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for capturing inter-market dependencies"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # Learnable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Graph Attention Layer with proper use of N

        Args:
            h: Node features [N, in_features]
            adj: Adjacency matrix [N, N]

        Returns:
            Updated node features [N, out_features]
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size()[0]  # Number of nodes

        # Compute attention coefficients
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])  # [N, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])  # [N, 1]

        # Broadcast for pairwise attention
        e = Wh1 + Wh2.T  # [N, N]
        e = self.leaky_relu(e)

        # Create mask from adjacency matrix
        mask = (adj > 0).float()

        # Apply mask - set unconnected to very negative values
        e = e * mask + (-9e15) * (1 - mask)

        # Calculate node degrees (number of connections)
        node_degrees = adj.sum(dim=1, keepdim=True)  # [N, 1]

        # Normalize attention by degree (optional, helps with scale)
        # This prevents nodes with many connections from dominating
        if hasattr(self, "normalize_by_degree") and self.normalize_by_degree:
            e = e / (node_degrees + 1e-6)

        if hasattr(self, "adaptive_temperature") and self.adaptive_temperature:
            # Larger graphs need sharper attention
            temperature = torch.log(torch.tensor(N, dtype=torch.float))
            e = e / temperature

        attention = F.softmax(e, dim=1)

        if hasattr(self, "use_structural_bias") and self.use_structural_bias:
            # Nodes with higher degree get more attention
            degree_bias = torch.log(node_degrees + 1)  # [N, 1]
            attention = attention * degree_bias
            # Renormalize
            attention = attention / attention.sum(dim=1, keepdim=True)

        attention = self.dropout_layer(attention)

        h_prime = torch.matmul(attention, Wh)  # [N, out_features]

        if hasattr(self, "use_residual") and self.use_residual:
            # Scale residual by 1/sqrt(N) to maintain variance
            residual_scale = 1.0 / torch.sqrt(torch.tensor(N, dtype=torch.float))
            h_prime = h_prime + residual_scale * h

        output = F.elu(h_prime)

        if hasattr(self, "use_layer_norm") and self.use_layer_norm:
            mean = output.mean(dim=0, keepdim=True)
            var = output.var(dim=0, keepdim=True)
            output = (output - mean) / torch.sqrt(var + 1e-6)
            output = output * self.gamma + self.beta  # Learnable scale and shift

        return output


class TemporalGAT(nn.Module):
    """
    Temporal Graph Attention Network for volatility prediction [citation:4]

    Combines:
    - LSTM for temporal dynamics
    - Graph Attention for cross-market dependencies
    """

    def __init__(self, num_nodes: int, in_features: int, hidden_dim: int = 64, num_heads: int = 4, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=in_features, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout, bidirectional=True
        )

        # Graph Attention Layers
        self.gat_layers = nn.ModuleList([GraphAttentionLayer(hidden_dim * 2, hidden_dim, dropout) for _ in range(num_heads)])

        # Output layer
        self.fc = nn.Linear(hidden_dim * num_heads, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len, num_nodes, features]
        adj: torch.Tensor,  # [num_nodes, num_nodes]
    ) -> torch.Tensor:
        """
        Forward pass
        """
        batch_size, seq_len, num_nodes, features = x.shape

        # Reshape for LSTM: [batch * num_nodes, seq_len, features]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, features)

        # LSTM encoding
        lstm_out, _ = self.lstm(x_reshaped)  # [batch * num_nodes, seq_len, hidden*2]

        # Take last timestep
        lstm_last = lstm_out[:, -1, :]  # [batch * num_nodes, hidden*2]

        # Reshape back: [batch, num_nodes, hidden*2]
        temporal_features = lstm_last.reshape(batch_size, num_nodes, -1)

        # Apply GAT for each batch
        gat_outputs = []
        for i in range(batch_size):
            node_features = temporal_features[i]  # [num_nodes, hidden*2]

            # Multi-head attention
            head_outputs = []
            for gat in self.gat_layers:
                head_out = gat(node_features, adj)  # [num_nodes, hidden]
                head_outputs.append(head_out)

            # Concatenate heads
            multi_head = torch.cat(head_outputs, dim=1)  # [num_nodes, hidden * num_heads]
            gat_outputs.append(multi_head)

        # Stack batch
        gat_out = torch.stack(gat_outputs, dim=0)  # [batch, num_nodes, hidden * num_heads]

        # Final prediction (volatility for each node)
        predictions = self.fc(gat_out).squeeze(-1)  # [batch, num_nodes]

        return predictions


class VolatilitySpilloverIndex:
    """
    Diebold-Yilmaz volatility spillover index [citation:4]

    Measures directional shock transmission between markets
    """

    def __init__(self, horizon: int = 10):
        self.horizon = horizon

    def calculate_spillover_matrix(self, returns: pd.DataFrame, var_lags: int = 2) -> Tuple[pd.DataFrame, Dict]:
        """
        Calculate volatility spillover matrix using Diebold-Yilmaz method
        """
        from statsmodels.tsa.api import VAR

        # Fit VAR model
        model = VAR(returns)
        results = model.fit(var_lags)

        # Get forecast error variance decomposition
        fevd = results.fevd(self.horizon)

        n = len(returns.columns)
        spillover_matrix = pd.DataFrame(np.zeros((n, n)), index=returns.columns, columns=returns.columns)

        # Fill spillover matrix
        for i, name_i in enumerate(returns.columns):
            for j, name_j in enumerate(returns.columns):
                # Contribution of shock j to variance of i
                contribution = fevd.decomp[i, j]
                spillover_matrix.loc[name_i, name_j] = contribution

        # Calculate directional spillovers
        from_others = spillover_matrix.sum(axis=1) - np.diag(spillover_matrix)
        to_others = spillover_matrix.sum(axis=0) - np.diag(spillover_matrix)
        net = to_others - from_others
        total = spillover_matrix.sum().sum() - np.trace(spillover_matrix)

        metrics = {
            "total_spillover": total / n,
            "directional_from": from_others.to_dict(),
            "directional_to": to_others.to_dict(),
            "net_spillover": net.to_dict(),
            "pairwise": spillover_matrix.to_dict(),
        }

        return spillover_matrix, metrics

    def build_adjacency_matrix(self, spillover_matrix: pd.DataFrame, threshold: float = 0.1) -> np.ndarray:
        """Build adjacency matrix from spillover matrix"""
        adj = (spillover_matrix.values > threshold).astype(float)
        # Add self-loops
        np.fill_diagonal(adj, 1)
        return adj


class TemporalGATService:
    """
    Service for TemporalGAT volatility prediction
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.num_nodes = len(symbols)
        self.model = None
        self.spillover = VolatilitySpilloverIndex()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def prepare_data(
        self, price_data: Dict[str, pd.DataFrame], lookback: int = 60, forecast_horizon: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for TemporalGAT

        Returns:
            X: [num_samples, lookback, num_nodes, features]
            y: [num_samples, num_nodes] (future volatility)
            adj: [num_nodes, num_nodes] (spillover matrix)
        """
        # Calculate returns and volatility for each symbol
        returns_dict = {}
        volatility_dict = {}

        for symbol, df in price_data.items():
            returns = df["Close"].pct_change().dropna()
            returns_dict[symbol] = returns

            # Realized volatility
            vol = returns.rolling(22).std() * np.sqrt(252)  # Annualized
            volatility_dict[symbol] = vol

        # Create returns DataFrame for spillover calculation
        returns_df = pd.DataFrame(returns_dict)

        # Calculate spillover matrix (updated periodically)
        spillover_matrix, _ = self.spillover.calculate_spillover_matrix(
            returns_df.iloc[-252:]  # Use last year
        )
        adj = self.spillover.build_adjacency_matrix(spillover_matrix)

        # Prepare feature matrix
        features = []
        targets = []

        # Features: returns, volume, volatility, etc.
        for i in range(lookback, len(returns_df) - forecast_horizon):
            feature_window = []
            for symbol in self.symbols:
                symbol_features = []

                # Returns over different horizons
                for lag in [1, 5, 10, 20]:
                    ret = returns_df[symbol].iloc[i - lag : i].mean()
                    symbol_features.append(ret)

                # Recent volatility
                recent_vol = volatility_dict[symbol].iloc[i - 20 : i].values[-5:].tolist()
                symbol_features.extend(recent_vol)

                # Volume (if available)
                if "Volume" in price_data[symbol].columns:
                    volume = price_data[symbol]["Volume"].iloc[i - 5 : i].values / 1e6
                    symbol_features.extend(volume.tolist())

                feature_window.append(symbol_features)

            features.append(feature_window)

            # Target: future volatility for each symbol
            future_vol = []
            for symbol in self.symbols:
                fut_vol = volatility_dict[symbol].iloc[i + forecast_horizon]
                future_vol.append(fut_vol)
            targets.append(future_vol)

        X = np.array(features)
        y = np.array(targets)

        return X, y, adj

    def build_model(self, in_features: int):
        """Build TemporalGAT model"""
        self.model = TemporalGAT(num_nodes=self.num_nodes, in_features=in_features, hidden_dim=64, num_heads=4, lstm_layers=2, dropout=0.1)
        return self.model

    def train(self, X: np.ndarray, y: np.ndarray, adj: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the model"""

        # Scale features
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_X.fit_transform(X_flat)
        X = X_scaled.reshape(original_shape)

        # Scale targets
        y_scaled = self.scaler_y.fit_transform(y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y_scaled)
        adj_tensor = torch.FloatTensor(adj)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build model
        if self.model is None:
            self.build_model(X.shape[-1])

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_X, adj_tensor)
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")

    def predict(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Predict future volatility"""
        self.model.eval()

        # Scale features
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_X.transform(X_flat)
        X = X_scaled.reshape(original_shape)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        adj_tensor = torch.FloatTensor(adj)

        with torch.no_grad():
            predictions = self.model(X_tensor, adj_tensor)

        # Inverse scale
        pred_np = predictions.numpy()
        pred_original = self.scaler_y.inverse_transform(pred_np)

        return pred_original
