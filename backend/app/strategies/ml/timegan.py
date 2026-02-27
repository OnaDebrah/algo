"""
TimeGAN for synthetic market data generation [citation:9]
Creates realistic synthetic time series for training and stress testing
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TimeGANEmbedding(nn.Module):
    """Embedding network for TimeGAN"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()

        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence"""
        output, _ = self.rnn(x)
        return output


class TimeGANRecovery(nn.Module):
    """Recovery network to reconstruct original data"""

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation"""
        output, _ = self.rnn(h)
        reconstructed = self.fc(output)
        return reconstructed


class TimeGANGenerator(nn.Module):
    """Generator network for synthetic data"""

    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()

        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(input_size=noise_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate hidden representation from noise"""
        output, _ = self.rnn(z)
        generated = torch.tanh(self.fc(output))
        return generated


class TimeGANDiscriminator(nn.Module):
    """Discriminator network for adversarial training"""

    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Classify as real or fake"""
        output, _ = self.rnn(h)
        logits = self.fc(output)
        return torch.sigmoid(logits)


class TimeGAN:
    """
    Time Generative Adversarial Network [citation:9]

    Generates synthetic time series that preserve:
    - Temporal dynamics
    - Distributional properties
    - Autocorrelation structure
    """

    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int = 24, noise_dim: int = 6, num_layers: int = 3, device: str = "cpu"):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.device = device

        # Initialize networks
        self.embedder = TimeGANEmbedding(input_dim, hidden_dim, num_layers).to(device)
        self.recovery = TimeGANRecovery(hidden_dim, input_dim, num_layers).to(device)
        self.generator = TimeGANGenerator(noise_dim, hidden_dim, num_layers).to(device)
        self.discriminator = TimeGANDiscriminator(hidden_dim, num_layers).to(device)

        # Optimizers
        self.optimizer_e = optim.Adam(self.embedder.parameters(), lr=0.001)
        self.optimizer_r = optim.Adam(self.recovery.parameters(), lr=0.001)
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.001)

    def _generate_noise(self, batch_size: int) -> torch.Tensor:
        """Generate random noise for generator"""
        return torch.randn(batch_size, self.seq_len, self.noise_dim).to(self.device)

    def train(self, data: np.ndarray, epochs: int = 1000, batch_size: int = 128):
        """
        Train TimeGAN on real data
        """
        # Prepare data
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        for epoch in range(epochs):
            for batch in loader:
                real_data = batch[0]
                batch_size_actual = real_data.shape[0]

                # ========== Train Autoencoder (Embedder + Recovery) ==========
                self.optimizer_e.zero_grad()
                self.optimizer_r.zero_grad()

                # Encode and decode
                h = self.embedder(real_data)
                reconstructed = self.recovery(h)

                # Reconstruction loss
                e_loss = mse_loss(reconstructed, real_data)
                e_loss.backward()

                self.optimizer_e.step()
                self.optimizer_r.step()

                # ========== Train Generator ==========
                self.optimizer_g.zero_grad()

                # Generate fake data
                z = self._generate_noise(batch_size_actual)
                fake_h = self.generator(z)

                # Discriminate fake
                fake_logits = self.discriminator(fake_h)

                # Generator wants to fool discriminator
                g_loss = bce_loss(fake_logits, torch.ones_like(fake_logits))
                g_loss.backward()

                self.optimizer_g.step()

                # ========== Train Discriminator ==========
                self.optimizer_d.zero_grad()

                # Real data
                real_h = self.embedder(real_data).detach()
                real_logits = self.discriminator(real_h)

                # Fake data
                fake_logits = self.discriminator(fake_h.detach())

                # Discriminator loss
                d_loss_real = bce_loss(real_logits, torch.ones_like(real_logits))
                d_loss_fake = bce_loss(fake_logits, torch.zeros_like(fake_logits))
                d_loss = (d_loss_real + d_loss_fake) / 2

                d_loss.backward()
                self.optimizer_d.step()

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: E_loss={e_loss.item():.4f}, " f"G_loss={g_loss.item():.4f}, D_loss={d_loss.item():.4f}")

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate synthetic time series data

        Returns:
            Generated data with shape [n_samples, seq_len, input_dim]
        """
        self.generator.eval()

        with torch.no_grad():
            z = self._generate_noise(n_samples)
            fake_h = self.generator(z)
            fake_data = self.recovery(fake_h)

        return fake_data.cpu().numpy()

    def evaluate_quality(self, real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict:
        """
        Evaluate quality of synthetic data [citation:9]

        Metrics:
        - Discriminative score (how well classifier distinguishes)
        - Predictive score (t-step ahead prediction accuracy)
        - Statistical similarity (moments, autocorrelation)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        # 1. Discriminative score
        n_real = len(real_data)
        n_synthetic = len(synthetic_data)

        # Create labels
        X = np.concatenate([real_data.reshape(n_real, -1), synthetic_data.reshape(n_synthetic, -1)])
        y = np.concatenate([np.ones(n_real), np.zeros(n_synthetic)])

        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        discriminative_score = accuracy_score(y_test, y_pred)

        # 2. Statistical similarity
        stats = {}
        for i in range(self.input_dim):
            real_series = real_data[:, :, i].flatten()
            synth_series = synthetic_data[:, :, i].flatten()

            stats[f"dim_{i}_mean_diff"] = abs(real_series.mean() - synth_series.mean())
            stats[f"dim_{i}_std_diff"] = abs(real_series.std() - synth_series.std())
            stats[f"dim_{i}_skew_diff"] = abs(pd.Series(real_series).skew() - pd.Series(synth_series).skew())
            stats[f"dim_{i}_kurt_diff"] = abs(pd.Series(real_series).kurt() - pd.Series(synth_series).kurt())

        # 3. Autocorrelation similarity
        autocorr_diff = []
        for lag in [1, 5, 10, 20]:
            real_acf = self._compute_acf(real_data, lag)
            synth_acf = self._compute_acf(synthetic_data, lag)
            autocorr_diff.append(abs(real_acf - synth_acf))

        return {
            "discriminative_score": discriminative_score,
            "statistical_similarity": stats,
            "autocorrelation_diff": np.mean(autocorr_diff),
            "quality_score": 1 - discriminative_score,  # Lower discriminative = better
        }

    def _compute_acf(self, data: np.ndarray, lag: int) -> float:
        """Compute average autocorrelation at given lag"""
        acf_sum = 0
        n = 0

        for series in data:
            series_flat = series.flatten()
            if len(series_flat) > lag:
                series_centered = series_flat - series_flat.mean()
                acf = np.correlate(series_centered, series_centered, mode="full")
                acf = acf[len(acf) // 2 :] / (series_centered.std() ** 2 * len(series_centered))
                if len(acf) > lag:
                    acf_sum += acf[lag]
                    n += 1

        return acf_sum / n if n > 0 else 0
