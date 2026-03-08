"""
TNCM-VAE: Time-series Neural Causal Model for Counterfactual Market Simulation
Based on 2025 research achieving L1 distances of 0.03-0.10 from ground truth
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from causallearn.search.ConstraintBased.PC import pc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class CausalGraph:
    """Causal graph structure for market variables"""

    nodes: List[str]  # Variable names
    edges: List[Tuple[str, str, float]]  # (from, to, strength)
    adjacency_matrix: np.ndarray
    lagged_effects: Dict[int, List[Tuple[str, str, float]]]  # Lagged causal effects


class TemporalCausalEncoder(nn.Module):
    """
    Encoder that captures temporal causal relationships
    """

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2, max_lag: int = 5):
        super().__init__()

        self.max_lag = max_lag
        self.latent_dim = latent_dim

        # Temporal convolution for lagged effects
        self.temporal_conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=max_lag, padding=max_lag - 1)

        # Causal attention mechanism
        self.causal_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.1)

        # Variational encoding
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            causal_mask: Causal adjacency matrix [input_dim, input_dim]

        Returns:
            mu, logvar: Latent distribution parameters
        """
        batch_size, seq_len, _ = x.shape

        # Apply temporal convolution (capture lagged effects)
        x_t = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        temporal_features = self.temporal_conv(x_t)
        temporal_features = temporal_features.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Apply causal attention if mask provided
        if causal_mask is not None:
            # Expand mask for attention
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            causal_features, _ = self.causal_attention(temporal_features, temporal_features, temporal_features, attn_mask=causal_mask)
        else:
            causal_features = temporal_features

        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(causal_features)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim * 2]

        # Variational parameters
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)

        return mu, logvar


class CausalDecoder(nn.Module):
    """
    Decoder that generates counterfactual sequences respecting causal constraints
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128, seq_len: int = 60, max_lag: int = 5):
        super().__init__()

        self.seq_len = seq_len
        self.max_lag = max_lag

        # Latent to hidden
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)

        # Causal LSTM for generation
        self.causal_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

        # Causal constraint projection
        self.causal_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Lagged effect generator
        self.lag_generator = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(max_lag)])

    def forward(
        self, z: torch.Tensor, causal_graph: Optional[torch.Tensor] = None, intervention: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate sequence with causal constraints

        Args:
            z: Latent vector [batch, latent_dim]
            causal_graph: Causal adjacency matrix [output_dim, output_dim]
            intervention: Dict of {timestep: intervention_values}

        Returns:
            Generated sequence [batch, seq_len, output_dim]
        """
        batch_size = z.shape[0]
        device = z.device

        # ========== 1. Project latent to hidden ==========
        h = self.fc_hidden(z)  # [batch, hidden_dim]

        # Add positional encoding
        positions = torch.arange(self.seq_len, device=device)
        pos_encoding = self.pos_encoder(positions)  # [seq_len, hidden_dim]
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # Expand and add position info
        h = h.unsqueeze(1).expand(-1, self.seq_len, -1)  # [batch, seq_len, hidden_dim]
        h = h + pos_encoding

        # ========== 2. Generate with causal LSTM ==========
        lstm_out, (hidden, cell) = self.causal_lstm(h)  # [batch, seq_len, hidden_dim*2]

        # ========== 3. Apply causal attention ==========
        attended = self._apply_causal_attention(lstm_out, causal_graph, batch_size)

        # ========== 4. Project to causal space ==========
        causal_hidden = self.causal_proj(attended)  # [batch, seq_len, hidden_dim]

        # ========== 5. Generate base output ==========
        base_output = self.fc_out(causal_hidden)  # [batch, seq_len, output_dim]

        # ========== 6. Apply lagged effects ==========
        output = self._apply_lagged_effects(causal_hidden, base_output)

        # ========== 7. Apply interventions ==========
        output = self._apply_interventions(output, intervention)

        # ========== 8. Ensure output is in valid range ==========
        # For financial data, we want to allow negative returns but keep prices positive
        # This is a soft constraint - actual bounds depend on your data
        output = torch.tanh(output) * 5  # Bound to [-5, 5] for returns

        return output


class TNCM_VAE(nn.Module):
    """
    Time-series Neural Causal Model Variational Autoencoder

    Learns causal structure from market data and generates counterfactual scenarios
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        seq_len: int = 60,
        max_lag: int = 5,
        beta: float = 1.0,  # KL divergence weight
        learning_rate: float = 0.001,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.max_lag = max_lag
        self.beta = beta

        self.encoder = TemporalCausalEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, max_lag=max_lag)

        self.decoder = CausalDecoder(latent_dim=latent_dim, output_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len, max_lag=max_lag)

        self.causal_graph = None
        self.causal_mask = None

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None, intervention: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional intervention

        Args:
            x: Input sequence [batch, seq_len, input_dim]
            causal_mask: Causal adjacency matrix
            intervention: Dict of {timestep: intervention_values}

        Returns:
            Dictionary with reconstruction, latent params, and generated samples
        """
        # Encode
        mu, logvar = self.encoder(x, causal_mask)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode (with intervention if specified)
        x_recon = self.decoder(z, causal_mask, intervention)

        return {"reconstruction": x_recon, "mu": mu, "logvar": logvar, "z": z}

    def loss_function(self, x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss with causal regularization

        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def discover_causal_structure(self, data: np.ndarray, variable_names: List[str], alpha: float = 0.05) -> CausalGraph:
        """
        Discover causal structure from observational data using PC algorithm

        Args:
            data: Observational data [n_samples, n_variables]
            variable_names: Names of variables
            alpha: Significance level for conditional independence tests

        Returns:
            CausalGraph object
        """

        cg = pc(data, alpha, indep_test="fisherz", stable=True)

        # Extract adjacency matrix
        adj_matrix = np.zeros((len(variable_names), len(variable_names)))
        edges = []

        for i in range(cg.G.get_num_edges()):
            edge = cg.G.get_edge(i)
            from_node = edge.get_node1().get_name()
            to_node = edge.get_node2().get_name()

            # Parse node names (remove 'X' prefix)
            from_idx = int(from_node.replace("X", "")) - 1
            to_idx = int(to_node.replace("X", "")) - 1

            # Check edge direction
            if edge.get_direction() == "-->":
                adj_matrix[from_idx, to_idx] = 1
                edges.append((variable_names[from_idx], variable_names[to_idx], 1.0))
            elif edge.get_direction() == "<--":
                adj_matrix[to_idx, from_idx] = 1
                edges.append((variable_names[to_idx], variable_names[from_idx], 1.0))
            else:  # '---' bidirectional
                adj_matrix[from_idx, to_idx] = 1
                adj_matrix[to_idx, from_idx] = 1
                edges.append((variable_names[from_idx], variable_names[to_idx], 1.0))
                edges.append((variable_names[to_idx], variable_names[from_idx], 1.0))

        # Discover lagged effects (simplified - using correlation at different lags)
        lagged_effects = defaultdict(list)
        max_lag = self.max_lag

        for lag in range(1, max_lag + 1):
            for i, var1 in enumerate(variable_names):
                for j, var2 in enumerate(variable_names):
                    if i != j:
                        # Calculate cross-correlation at this lag
                        corr = np.corrcoef(data[:-lag, i], data[lag:, j])[0, 1]

                        if abs(corr) > 0.3:  # Threshold for significance
                            lagged_effects[lag].append((var1, var2, corr))

        self.causal_graph = CausalGraph(nodes=variable_names, edges=edges, adjacency_matrix=adj_matrix, lagged_effects=dict(lagged_effects))

        # Create causal mask for attention
        self.causal_mask = torch.FloatTensor(adj_matrix)

        return self.causal_graph


class CausalMarketSimulator:
    """
    Main class for causal market simulation and counterfactual analysis
    """

    def __init__(
        self, symbol_mappings: Dict[str, str], seq_len: int = 60, latent_dim: int = 32, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            symbol_mappings: Dict mapping symbols to display names
            seq_len: Sequence length for training
            latent_dim: Latent space dimension
            device: Computing device
        """
        self.symbol_mappings = symbol_mappings
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.device = device

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.variable_names = list(symbol_mappings.keys())
        self.n_vars = len(self.variable_names)

    def prepare_data(self, data_dict: Dict[str, pd.DataFrame], features=None) -> np.ndarray:
        """
        Prepare multi-variable time series data

        Args:
            data_dict: Dict mapping symbol -> price DataFrame
            features: List of features to include

        Returns:
            Array of shape [n_samples, n_vars * n_features]
        """
        if features is None:
            features = ["Close", "Volume", "Returns"]
        aligned_data = []

        for symbol in self.variable_names:
            df = data_dict.get(symbol)
            if df is None:
                raise ValueError(f"No data for symbol {symbol}")

            symbol_features = []

            if "Close" in features:
                symbol_features.append(df["Close"].values)

            if "Volume" in features and "Volume" in df.columns:
                symbol_features.append(df["Volume"].values)

            if "Returns" in features:
                returns = df["Close"].pct_change().values
                symbol_features.append(returns)

            # Stack features for this symbol
            symbol_array = np.column_stack(symbol_features)
            aligned_data.append(symbol_array)

        # Concatenate all symbols
        data = np.column_stack(aligned_data)

        # Remove NaN rows
        data = data[~np.isnan(data).any(axis=1)]

        return data

    def fit(
        self, data_dict: Dict[str, pd.DataFrame], epochs: int = 1000, batch_size: int = 32, learning_rate: float = 0.001, discover_causal: bool = True
    ) -> Dict:
        """
        Train the causal model on market data
        """
        # Prepare data
        raw_data = self.prepare_data(data_dict)
        n_samples, n_features = raw_data.shape

        self.input_dim = n_features

        # Scale data
        scaled_data = self.scaler_X.fit_transform(raw_data)

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_len):
            X.append(scaled_data[i : i + self.seq_len])
            y.append(scaled_data[i + self.seq_len])

        X = np.array(X)
        y = np.array(y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Discover causal structure if requested
        if discover_causal:
            logger.info("Discovering causal structure from data...")
            self.model = TNCM_VAE(input_dim=n_features, latent_dim=self.latent_dim, seq_len=self.seq_len, learning_rate=learning_rate).to(self.device)

            causal_graph = self.model.discover_causal_structure(scaled_data, self._get_variable_names())
            logger.info(f"Discovered {len(causal_graph.edges)} causal edges")
        else:
            self.model = TNCM_VAE(input_dim=n_features, latent_dim=self.latent_dim, seq_len=self.seq_len, learning_rate=learning_rate).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                self.model.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X, self.model.causal_mask)

                # Compute loss
                losses = self.model.loss_function(
                    batch_X,  # Use full sequence for reconstruction
                    outputs["reconstruction"],
                    outputs["mu"],
                    outputs["logvar"],
                )

                # Backward pass
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.model.optimizer.step()

                epoch_loss += losses["loss"].item()

            avg_loss = epoch_loss / len(loader)
            train_losses.append(avg_loss)

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        self.model.eval()

        return {"final_loss": train_losses[-1], "train_losses": train_losses, "causal_graph": self.model.causal_graph}

    def _get_variable_names(self) -> List[str]:
        """Get list of all variable names with features"""
        var_names = []
        for symbol in self.variable_names:
            var_names.append(f"{symbol}_Close")
            var_names.append(f"{symbol}_Volume")
            var_names.append(f"{symbol}_Returns")
        return var_names

    def generate_counterfactual(self, base_sequence: np.ndarray, intervention: Dict[str, Dict[int, float]], n_samples: int = 1) -> np.ndarray:
        """
        Generate counterfactual scenario with interventions

        Args:
            base_sequence: Base market sequence [seq_len, n_features]
            intervention: Dict of {variable_name: {timestep: value}}
            n_samples: Number of counterfactual samples to generate

        Returns:
            Counterfactual sequences [n_samples, seq_len, n_features]
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale base sequence
        base_scaled = self.scaler_X.transform(base_sequence)

        # Convert to tensor
        base_tensor = torch.FloatTensor(base_scaled).unsqueeze(0).to(self.device)

        # Encode base sequence to get latent
        with torch.no_grad():
            mu, logvar = self.model.encoder(base_tensor, self.model.causal_mask)

        counterfactuals = []

        for _ in range(n_samples):
            # Sample from latent distribution
            z = self.model.reparameterize(mu, logvar)

            # Prepare intervention tensor
            intervention_dict = {}
            var_indices = self._get_variable_indices()

            for var_name, timestep_values in intervention.items():
                if var_name in var_indices:
                    idx = var_indices[var_name]
                    for t, value in timestep_values.items():
                        if t not in intervention_dict:
                            intervention_dict[t] = torch.zeros(1, self.input_dim).to(self.device)
                        # Scale intervention value
                        scaled_value = self.scaler_X.transform([[value] * self.input_dim])[0, idx]
                        intervention_dict[t][0, idx] = scaled_value

            # Generate counterfactual
            with torch.no_grad():
                cf_scaled = self.model.decoder(z, self.model.causal_mask, intervention_dict)

            # Inverse transform
            cf = self.scaler_X.inverse_transform(cf_scaled.squeeze(0).cpu().numpy())
            counterfactuals.append(cf)

        return np.array(counterfactuals)

    def _get_variable_indices(self) -> Dict[str, int]:
        """Get mapping from variable name to feature index"""
        indices = {}
        var_names = self._get_variable_names()
        for i, name in enumerate(var_names):
            indices[name] = i
        return indices

    def ask_what_if(self, historical_data: Dict[str, pd.DataFrame], what_if_question: str, visualization: bool = True) -> Dict:
        """
        High-level interface for answering "what if" questions

        Args:
            historical_data: Historical market data
            what_if_question: Natural language question
            visualization: Whether to generate plots

        Returns:
            Dict with counterfactual results and analysis
        """
        # Parse the question (simplified - in production use LLM)
        parsed = self._parse_what_if_question(what_if_question)

        if not parsed:
            return {"error": "Could not parse question"}

        # Get base sequence (last seq_len days)
        base_sequence = self._get_base_sequence(historical_data)

        # Generate counterfactual
        counterfactuals = self.generate_counterfactual(
            base_sequence=base_sequence, intervention=parsed["intervention"], n_samples=parsed.get("n_samples", 100)
        )

        # Compare with historical
        comparison = self._compare_scenarios(base_sequence, counterfactuals, parsed["variable_of_interest"])

        result = {
            "question": what_if_question,
            "interpretation": parsed["interpretation"],
            "counterfactual_mean": comparison["mean"],
            "counterfactual_std": comparison["std"],
            "effect_size": comparison["effect_size"],
            "probability_better": comparison["prob_better"],
            "samples": counterfactuals if not visualization else None,
        }

        if visualization:
            result["visualization"] = self._create_comparison_plot(
                base_sequence, counterfactuals, parsed["variable_of_interest"], parsed["interpretation"]
            )

        return result

    def _parse_what_if_question(self, question: str) -> Dict:
        """
        Parse natural language what-if question
        In production, this would use an LLM
        """
        question_lower = question.lower()

        # Simplified parsing - in production use LLM
        intervention = {}
        interpretation = ""
        variable_of_interest = self.variable_names[0]  # Default

        # Example patterns
        if "fed raises rates" in question_lower:
            # Find interest rate variable
            rate_var = next((v for v in self.variable_names if "rate" in v.lower()), None)
            if rate_var:
                intervention[rate_var] = {0: 0.05}  # 5% rate at t=0
                interpretation = "Fed raises rates by 25bps"
                variable_of_interest = self.variable_names[0]  # Stock index

        elif "oil price shock" in question_lower:
            oil_var = next((v for v in self.variable_names if "oil" in v.lower() or "CL" in v), None)
            if oil_var:
                intervention[oil_var] = {0: 100}  # Oil at $100
                interpretation = "Oil price jumps to $100"

        elif "market crash" in question_lower:
            # Crash at time 0
            for var in self.variable_names:
                intervention[var] = {0: -0.2}  # 20% drop
            interpretation = "20% market crash at t=0"

        return {"intervention": intervention, "interpretation": interpretation, "variable_of_interest": variable_of_interest, "n_samples": 100}

    def _get_base_sequence(self, data_dict: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Get the most recent sequence for counterfactual generation"""
        # Get last seq_len days of data
        sequences = []
        for symbol in self.variable_names:
            df = data_dict.get(symbol)
            if df is not None:
                seq = df[["Close", "Volume"]].iloc[-self.seq_len :].values
                if len(seq) == self.seq_len:
                    sequences.append(seq)

        # Stack all symbols
        base = np.column_stack(sequences)
        return base

    def _compare_scenarios(self, base: np.ndarray, counterfactuals: np.ndarray, variable: str) -> Dict:
        """Compare base scenario with counterfactuals"""
        var_idx = self._get_variable_indices().get(f"{variable}_Close", 0)

        # Get final values
        base_final = base[-1, var_idx]
        cf_finals = counterfactuals[:, -1, var_idx]

        # Calculate statistics
        mean_cf = np.mean(cf_finals)
        std_cf = np.std(cf_finals)

        # Effect size (Cohen's d)
        effect_size = (mean_cf - base_final) / std_cf if std_cf > 0 else 0

        # Probability that counterfactual is better
        prob_better = np.mean(cf_finals > base_final)

        return {
            "base": base_final,
            "mean": mean_cf,
            "std": std_cf,
            "effect_size": effect_size,
            "prob_better": prob_better,
            "percentiles": {
                "5th": np.percentile(cf_finals, 5),
                "25th": np.percentile(cf_finals, 25),
                "50th": np.percentile(cf_finals, 50),
                "75th": np.percentile(cf_finals, 75),
                "95th": np.percentile(cf_finals, 95),
            },
        }

    def _create_comparison_plot(self, base: np.ndarray, counterfactuals: np.ndarray, variable: str, interpretation: str):
        """Create visualization comparing scenarios"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        var_idx = self._get_variable_indices().get(f"{variable}_Close", 0)

        fig = make_subplots(
            rows=2, cols=2, subplot_titles=("Counterfactual Trajectories", "Distribution of Outcomes", "Effect Over Time", "Probability Analysis")
        )

        # 1. Trajectories
        time_points = np.arange(len(base))

        # Plot base
        fig.add_trace(go.Scatter(x=time_points, y=base[:, var_idx], name="Historical", line=dict(color="blue", width=3)), row=1, col=1)

        # Plot sample of counterfactuals
        for i in range(min(20, len(counterfactuals))):
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=counterfactuals[i, :, var_idx],
                    name=f"CF {i + 1}" if i == 0 else None,
                    line=dict(color="red", width=1, dash="dash"),
                    opacity=0.3,
                    showlegend=(i == 0),
                ),
                row=1,
                col=1,
            )

        # 2. Distribution of outcomes
        cf_finals = counterfactuals[:, -1, var_idx]

        fig.add_trace(go.Histogram(x=cf_finals, name="Final Values", nbinsx=30, marker="red", opacity=0.7), row=1, col=2)

        # Add vertical line for historical
        fig.add_vline(x=base[-1, var_idx], line_dash="dash", line_color="blue", annotation_text="Historical", row=1, col=2)

        # 3. Effect over time
        mean_cf = np.mean(counterfactuals[:, :, var_idx], axis=0)
        std_cf = np.std(counterfactuals[:, :, var_idx], axis=0)

        fig.add_trace(go.Scatter(x=time_points, y=mean_cf, name="Mean CF", line=dict(color="red", width=2)), row=2, col=1)

        fig.add_trace(go.Scatter(x=time_points, y=mean_cf + std_cf, name="+1 Std", line=dict(color="red", width=0), showlegend=False), row=2, col=1)

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=mean_cf - std_cf,
                name="-1 Std",
                fill="tonexty",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="red", width=0),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(go.Scatter(x=time_points, y=base[:, var_idx], name="Historical", line=dict(color="blue", width=2)), row=2, col=1)

        # 4. Probability analysis
        prob_over_time = []
        for t in range(len(base)):
            prob = np.mean(counterfactuals[:, t, var_idx] > base[t, var_idx])
            prob_over_time.append(prob)

        fig.add_trace(
            go.Scatter(x=time_points, y=prob_over_time, name="P(CF > Historical)", line=dict(color="green", width=2), fill="tozeroy"), row=2, col=2
        )

        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=2)

        # Update layout
        fig.update_layout(title_text=f"What-If Analysis: {interpretation}", template="plotly_dark", height=800, showlegend=True)

        fig.update_xaxes(title_text="Days", row=1, col=1)
        fig.update_xaxes(title_text="Days", row=2, col=1)
        fig.update_xaxes(title_text="Days", row=2, col=2)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Price", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=2)

        return fig
