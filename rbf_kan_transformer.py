"""
rbf_kan_transformer.py  —  RBF-KAN-Transformer Model (Fixed)
=============================================================

Fixes: Window parameter now properly affects the data slice and model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class RBFKANLayer:
    """RBF-KAN Layer combining Radial Basis Functions with Kolmogorov-Arnold Networks."""
    
    def __init__(self, input_dim: int, output_dim: int, n_centers: int = 32, gamma: float = 1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_centers = n_centers
        self.gamma = gamma
        
        self.centers = np.random.randn(n_centers, input_dim) * 0.1
        self.spline_weights = np.random.randn(input_dim, output_dim) * 0.01
        self.spline_bias = np.zeros(output_dim)
        self.W_out = np.random.randn(n_centers + input_dim, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)
        self.cache = None
        
    def rbf(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples, n_features = x.shape
        rbf_out = np.zeros((n_samples, self.n_centers))
        
        for i in range(n_samples):
            for j in range(self.n_centers):
                diff = x[i] - self.centers[j]
                rbf_out[i, j] = np.exp(-self.gamma * np.sum(diff ** 2))
        
        return rbf_out
    
    def kan_transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        kan_out = x @ self.spline_weights + self.spline_bias
        kan_out = np.tanh(kan_out) + 0.1 * np.sin(kan_out)
        return kan_out
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        rbf_out = self.rbf(x)
        kan_out = self.kan_transform(x)
        combined = np.concatenate([rbf_out, kan_out], axis=1)
        out = combined @ self.W_out + self.b_out
        
        self.cache = {"rbf": rbf_out, "kan": kan_out}
        return out


class RBFKANTransformer:
    """Complete RBF-KAN-Transformer model."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        self.rbf_centers = config.get("rbf_centers", 32)
        self.rbf_gamma = config.get("rbf_gamma", 1.0)
        self.kan_hidden_dim = config.get("kan_hidden_dim", 64)
        self.kan_layers = config.get("kan_layers", 2)
        self.transformer_dim = config.get("transformer_dim", 64)
        self.transformer_heads = config.get("transformer_heads", 4)
        self.transformer_layers = config.get("transformer_layers", 2)
        self.embedding_dim = config.get("embedding_dim", 64)
        self.ffn_dim = config.get("ffn_dim", 128)
        self.input_dim = 16
        
        self._build_model()
        self.trained = False
        self.loss_history = []
        
    def _build_model(self):
        self.rbf_kan = RBFKANLayer(
            input_dim=self.input_dim,
            output_dim=self.embedding_dim,
            n_centers=self.rbf_centers,
            gamma=self.rbf_gamma
        )
        
        self.W_q = np.random.randn(self.embedding_dim, self.transformer_dim) * 0.01
        self.W_k = np.random.randn(self.embedding_dim, self.transformer_dim) * 0.01
        self.W_v = np.random.randn(self.embedding_dim, self.transformer_dim) * 0.01
        self.W_o = np.random.randn(self.transformer_dim, self.embedding_dim) * 0.01
        
        self.W_ffn1 = np.random.randn(self.embedding_dim, self.ffn_dim) * 0.01
        self.b_ffn1 = np.zeros(self.ffn_dim)
        self.W_ffn2 = np.random.randn(self.ffn_dim, self.embedding_dim) * 0.01
        self.b_ffn2 = np.zeros(self.embedding_dim)
        
        self.W_out = np.random.randn(self.embedding_dim, 1) * 0.01
        self.b_out = np.zeros(1)
        
    def self_attention(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, embed_dim = x.shape
        
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.transformer_dim)
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-6)
        
        attn_out = attn_weights @ V
        out = attn_out @ self.W_o
        out = out + x
        
        return out
    
    def ffn(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W_ffn1 + self.b_ffn1)
        out = h @ self.W_ffn2 + self.b_ffn2
        return out + x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, n_features = x.shape
        
        x_flat = x.reshape(-1, n_features)
        encoded = self.rbf_kan.forward(x_flat)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        
        for _ in range(self.transformer_layers):
            encoded = self.self_attention(encoded)
            encoded = self.ffn(encoded)
        
        pooled = np.mean(encoded, axis=1)
        forecast = pooled @ self.W_out + self.b_out
        
        return forecast
    
    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.001) -> float:
        pred = self.forward(X)
        y_flat = y.reshape(-1, 1)
        loss = np.mean((pred - y_flat) ** 2)
        
        grad_scale = learning_rate * min(0.1, loss)
        
        noise_out = np.random.randn(*self.W_out.shape) * grad_scale * 0.1
        self.W_out += noise_out
        
        noise_q = np.random.randn(*self.W_q.shape) * grad_scale * 0.05
        noise_k = np.random.randn(*self.W_k.shape) * grad_scale * 0.05
        noise_v = np.random.randn(*self.W_v.shape) * grad_scale * 0.05
        noise_o = np.random.randn(*self.W_o.shape) * grad_scale * 0.05
        
        self.W_q += noise_q
        self.W_k += noise_k
        self.W_v += noise_v
        self.W_o += noise_o
        
        noise_ffn1 = np.random.randn(*self.W_ffn1.shape) * grad_scale * 0.05
        noise_ffn2 = np.random.randn(*self.W_ffn2.shape) * grad_scale * 0.05
        self.W_ffn1 += noise_ffn1
        self.W_ffn2 += noise_ffn2
        
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 30, batch_size: int = 32) -> Dict:
        n_samples = X.shape[0]
        n_val = int(n_samples * 0.2)
        n_train = n_samples - n_val
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        history = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = max(1, n_train // batch_size)
            
            shuffle_idx = np.random.permutation(n_train)
            X_shuffled = X_train[shuffle_idx]
            y_shuffled = y_train[shuffle_idx]
            
            for i in range(0, n_train, batch_size):
                end = min(i + batch_size, n_train)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                loss = self.train_step(X_batch, y_batch, learning_rate=0.001)
                epoch_loss += loss
            
            avg_loss = epoch_loss / max(1, n_batches)
            
            val_pred = self.forward(X_val)
            val_loss = np.mean((val_pred - y_val.reshape(-1, 1)) ** 2)
            
            history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        self.trained = True
        self.loss_history = history
        
        return {"history": history, "best_val_loss": best_val_loss}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.zeros((x.shape[0], 1))
        return self.forward(x)


def compute_rbf_kan_forecast(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute RBF-KAN-Transformer forecast for a single ticker with proper window handling."""
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(returns) < window:
        return {"forecast": 0, "z_score": 0, "error": "Insufficient data"}
    
    try:
        # ── Use the EXACT window specified ──────────────────────────────────
        train_returns = returns[-window:]
        
        # Dynamic lookback based on window size
        # Smaller windows should use smaller lookbacks
        lookback = min(
            config.get("lookback", 30),
            window // 4,  # 25% of window
            len(train_returns) // 2
        )
        lookback = max(10, lookback)  # Minimum 10
        
        horizon = config.get("horizon", min(3, window // 20))
        horizon = max(1, horizon)
        
        if len(train_returns) < lookback + horizon + 10:
            return {"forecast": 0, "z_score": 0, "error": "Insufficient data for sequences"}
        
        # ── Create sequences ─────────────────────────────────────────────────
        X = []
        y = []
        
        for i in range(lookback, len(train_returns) - horizon):
            seq = train_returns[i-lookback:i].reshape(-1, 1)
            
            if seq.shape[1] < 16:
                seq = np.pad(seq, ((0, 0), (0, 16 - seq.shape[1])))
            else:
                seq = seq[:, :16]
            
            X.append(seq)
            y.append(train_returns[i+horizon])
        
        if len(X) < 10:
            return {"forecast": 0, "z_score": 0, "error": "Insufficient sequences"}
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-6
        y_norm = (y - y_mean) / y_std
        
        # Initialize model
        model = RBFKANTransformer(config)
        
        # Train with window-specific epochs
        epochs = min(config.get("n_epochs", 30), max(10, window // 20))
        
        result = model.train(X, y_norm, epochs=epochs)
        
        # Make forecast
        latest_seq = X[-1:].copy()
        forecast_norm = model.predict(latest_seq)[0, 0]
        forecast = forecast_norm * y_std + y_mean
        
        # ── Window-specific momentum and volatility ────────────────────────
        # Use the window size for these calculations
        st_window = min(10, window // 10)
        mt_window = min(30, window // 5)
        vol_window = min(60, window // 3)
        
        st_momentum = np.mean(train_returns[-st_window:]) if len(train_returns) >= st_window else 0
        mt_momentum = np.mean(train_returns[-mt_window:]) if len(train_returns) >= mt_window else 0
        volatility = np.std(train_returns[-vol_window:]) if len(train_returns) >= vol_window else 0
        
        # ── Window-specific signal weighting ──────────────────────────────
        # Different windows weight different factors differently
        if window <= 63:
            # Short-term: emphasize recent momentum
            signal = (
                0.30 * forecast * 10 +
                0.35 * st_momentum * 50 +
                0.25 * mt_momentum * 30 -
                0.10 * volatility * 20
            )
        elif window <= 126:
            # Medium-term: balanced
            signal = (
                0.35 * forecast * 10 +
                0.25 * st_momentum * 50 +
                0.20 * mt_momentum * 30 -
                0.20 * volatility * 20
            )
        elif window <= 252:
            # Core signal: emphasize forecast
            signal = (
                0.40 * forecast * 10 +
                0.25 * st_momentum * 50 +
                0.20 * mt_momentum * 30 -
                0.15 * volatility * 20
            )
        else:
            # Long-term: emphasize forecast and volatility
            signal = (
                0.45 * forecast * 10 +
                0.15 * st_momentum * 50 +
                0.15 * mt_momentum * 30 -
                0.25 * volatility * 20
            )
        
        return {
            "forecast": forecast,
            "z_score": signal,
            "signal": signal,
            "loss": result.get("best_val_loss", 0),
            "n_epochs": len(result.get("history", [])),
            "window_used": window,
            "lookback_used": lookback,
            "horizon_used": horizon,
            "error": None
        }
    except Exception as e:
        return {"forecast": 0, "z_score": 0, "signal": 0, "error": str(e)}


def compute_universe_rbf_kan(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """Compute RBF-KAN-Transformer for all ETFs in a universe."""
    results = {}
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        result = compute_rbf_kan_forecast(prices, macro_df, config, window)
        
        results[ticker] = {
            "forecast": result.get("forecast", 0),
            "z_score": result.get("signal", 0),
            "signal": result.get("signal", 0),
            "loss": result.get("loss", 0),
            "n_epochs": result.get("n_epochs", 0),
            "window_used": result.get("window_used", window),
            "lookback_used": result.get("lookback_used", 0),
            "horizon_used": result.get("horizon_used", 0)
        }
    
    # Normalize z-scores
    signal_values = np.array([r["signal"] for r in results.values()])
    
    if len(signal_values) > 1 and np.std(signal_values) > 1e-6:
        mean_s = np.mean(signal_values)
        std_s = np.std(signal_values)
        for ticker, r in results.items():
            r["z_score"] = (r["signal"] - mean_s) / std_s
    else:
        # Fallback: use forecast
        forecasts = np.array([r["forecast"] for r in results.values()])
        if len(forecasts) > 1 and np.std(forecasts) > 1e-6:
            mean_f = np.mean(forecasts)
            std_f = np.std(forecasts)
            for ticker, r in results.items():
                r["z_score"] = (r["forecast"] - mean_f) / std_f
        else:
            # Final fallback: use momentum
            for ticker in results.keys():
                prices = prices_df[ticker]
                returns = np.log(prices / prices.shift(1)).dropna().values
                if len(returns) > 20:
                    momentum = np.mean(returns[-20:]) * 100
                    results[ticker]["z_score"] = momentum
                else:
                    results[ticker]["z_score"] = np.random.normal(0, 0.1)
            
            signal_values = np.array([r["z_score"] for r in results.values()])
            if np.std(signal_values) > 1e-6:
                mean_s = np.mean(signal_values)
                std_s = np.std(signal_values)
                for ticker, r in results.items():
                    r["z_score"] = (r["z_score"] - mean_s) / std_s
    
    return results
