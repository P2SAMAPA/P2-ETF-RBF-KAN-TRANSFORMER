"""
rbf_kan_transformer.py  —  RBF-KAN-Transformer Model (ULTRA-OPTIMIZED)
=====================================================================

FIX: Reduced training time with adaptive epochs and early stopping.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class RBFKANLayer:
    """RBF-KAN Layer - Optimized for speed."""
    
    def __init__(self, input_dim: int, output_dim: int, n_centers: int = 32, gamma: float = 1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_centers = n_centers
        self.gamma = gamma
        
        # Smaller initialization for faster convergence
        self.centers = np.random.randn(n_centers, input_dim) * 0.05
        self.spline_weights = np.random.randn(input_dim, output_dim) * 0.005
        self.spline_bias = np.zeros(output_dim)
        self.W_out = np.random.randn(n_centers + input_dim, output_dim) * 0.005
        self.b_out = np.zeros(output_dim)
        self.cache = None
        self.x_cache = None
        
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
        
        self.x_cache = x
        rbf_out = self.rbf(x)
        kan_out = self.kan_transform(x)
        combined = np.concatenate([rbf_out, kan_out], axis=1)
        
        if combined.shape[1] != self.W_out.shape[0]:
            if combined.shape[1] < self.W_out.shape[0]:
                pad_width = ((0, 0), (0, self.W_out.shape[0] - combined.shape[1]))
                combined = np.pad(combined, pad_width, mode='constant')
            else:
                combined = combined[:, :self.W_out.shape[0]]
        
        out = combined @ self.W_out + self.b_out
        
        self.cache = {"rbf": rbf_out, "kan": kan_out, "combined": combined}
        return out
    
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        combined = self.cache["combined"]
        
        grad_W_out = combined.T @ grad_output
        grad_b_out = np.sum(grad_output, axis=0)
        
        self.W_out -= learning_rate * grad_W_out
        self.b_out -= learning_rate * grad_b_out
        
        grad_combined = grad_output @ self.W_out.T
        
        if hasattr(self, 'x_cache') and self.x_cache is not None:
            kan_grad = grad_combined[:, self.n_centers:]
            if kan_grad.shape[1] == self.output_dim:
                grad_spline = self.x_cache.T @ kan_grad
                if grad_spline.shape == self.spline_weights.shape:
                    self.spline_weights -= learning_rate * 0.1 * grad_spline
                    self.spline_bias -= learning_rate * 0.1 * np.sum(kan_grad, axis=0)
        
        return grad_combined


class RBFKANTransformer:
    def __init__(self, config: Dict, window: int = 252, universe_size: int = 30):
        self.config = config
        self.window = window
        self.universe_size = universe_size
        
        # Scale model complexity based on universe size and window
        scale_factor = max(1, window / 252)
        complexity_factor = max(0.5, min(1.0, 30 / max(universe_size, 1)))
        
        # Reduce complexity for large universes
        self.rbf_centers = int(config.get("rbf_centers", 32) * min(scale_factor, 1.2) * complexity_factor)
        self.rbf_centers = max(8, min(self.rbf_centers, 32))
        
        self.rbf_gamma = config.get("rbf_gamma", 1.0) / min(scale_factor, 1.2)
        self.embedding_dim = int(config.get("embedding_dim", 64) * min(scale_factor, 1.2) * complexity_factor)
        self.embedding_dim = max(16, min(self.embedding_dim, 64))
        
        self.transformer_dim = int(config.get("transformer_dim", 64) * min(scale_factor, 1.2) * complexity_factor)
        self.transformer_dim = max(16, min(self.transformer_dim, 64))
        
        self.transformer_layers = max(1, int(config.get("transformer_layers", 2) * complexity_factor))
        self.transformer_layers = min(self.transformer_layers, 2)
        
        self.ffn_dim = int(config.get("ffn_dim", 128) * complexity_factor)
        self.ffn_dim = max(32, min(self.ffn_dim, 128))
        
        self.input_dim = 16
        
        self.lookback = max(10, int(window * 0.2))
        self.horizon = max(1, min(5, int(window * 0.02)))
        
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
        
        self.W_q = np.random.randn(self.embedding_dim, self.transformer_dim) * 0.005
        self.W_k = np.random.randn(self.embedding_dim, self.transformer_dim) * 0.005
        self.W_v = np.random.randn(self.embedding_dim, self.transformer_dim) * 0.005
        self.W_o = np.random.randn(self.transformer_dim, self.embedding_dim) * 0.005
        
        self.W_ffn1 = np.random.randn(self.embedding_dim, self.ffn_dim) * 0.005
        self.b_ffn1 = np.zeros(self.ffn_dim)
        self.W_ffn2 = np.random.randn(self.ffn_dim, self.embedding_dim) * 0.005
        self.b_ffn2 = np.zeros(self.embedding_dim)
        
        self.W_out = np.random.randn(self.embedding_dim, 1) * 0.005
        self.b_out = np.zeros(1)
        
        self.attn_cache = None
        self.ffn_cache = None
        self.pooled = None
        
    def self_attention(self, x: np.ndarray, return_cache: bool = False) -> np.ndarray:
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
        
        if return_cache:
            self.attn_cache = {"Q": Q, "K": K, "V": V, "attn_weights": attn_weights, "attn_out": attn_out}
        
        return out
    
    def ffn(self, x: np.ndarray, return_cache: bool = False) -> np.ndarray:
        h = np.tanh(x @ self.W_ffn1 + self.b_ffn1)
        out = h @ self.W_ffn2 + self.b_ffn2
        out = out + x
        
        if return_cache:
            self.ffn_cache = {"h": h}
        
        return out
    
    def forward(self, x: np.ndarray, return_cache: bool = False) -> np.ndarray:
        batch_size, seq_len, n_features = x.shape
        
        x_flat = x.reshape(-1, n_features)
        encoded = self.rbf_kan.forward(x_flat)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        
        for _ in range(self.transformer_layers):
            encoded = self.self_attention(encoded, return_cache=return_cache)
            encoded = self.ffn(encoded, return_cache=return_cache)
        
        pooled = np.mean(encoded, axis=1)
        forecast = pooled @ self.W_out + self.b_out
        
        if return_cache:
            self.pooled = pooled
        
        return forecast
    
    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.001) -> float:
        pred = self.forward(X, return_cache=True)
        y_flat = y.reshape(-1, 1)
        
        loss = np.mean((pred - y_flat) ** 2)
        grad_output = 2 * (pred - y_flat) / len(y_flat)
        
        grad_W_out = self.pooled.T @ grad_output
        grad_b_out = np.sum(grad_output, axis=0)
        
        self.W_out -= learning_rate * grad_W_out
        self.b_out -= learning_rate * grad_b_out
        
        grad_pooled = grad_output @ self.W_out.T
        grad_encoded = grad_pooled.reshape(grad_pooled.shape[0], 1, -1)
        grad_encoded = np.repeat(grad_encoded, X.shape[1], axis=1) / X.shape[1]
        
        grad_flat = grad_encoded.reshape(-1, self.embedding_dim)
        self.rbf_kan.backward(grad_flat, learning_rate * 0.1)
        
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 30, batch_size: int = 32) -> Dict:
        n_samples = X.shape[0]
        
        if n_samples < 10:
            return {"history": [], "best_val_loss": float('inf')}
        
        n_val = int(n_samples * 0.2)
        n_train = n_samples - n_val
        
        if n_train < 5:
            n_val = min(2, n_samples // 2)
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
        patience = 3
        patience_counter = 0
        
        # Adaptive epochs based on window and universe size
        if self.window >= 504:
            actual_epochs = min(epochs, 15)  # Max 15 for 504d
        elif self.window >= 252:
            actual_epochs = min(epochs, 20)  # Max 20 for 252d
        else:
            actual_epochs = min(epochs, 10)  # Max 10 for smaller windows
        
        actual_epochs = max(5, actual_epochs)  # Minimum 5 epochs
        
        lr = 0.002 * min(1.0, 252 / self.window)  # Higher learning rate
        
        for epoch in range(actual_epochs):
            epoch_loss = 0
            n_batches = max(1, n_train // batch_size)
            
            shuffle_idx = np.random.permutation(n_train)
            X_shuffled = X_train[shuffle_idx]
            y_shuffled = y_train[shuffle_idx]
            
            for i in range(0, n_train, batch_size):
                end = min(i + batch_size, n_train)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                loss = self.train_step(X_batch, y_batch, learning_rate=lr)
                epoch_loss += loss
            
            avg_loss = epoch_loss / max(1, n_batches)
            
            if len(X_val) > 0:
                val_pred = self.forward(X_val)
                val_loss = np.mean((val_pred - y_val.reshape(-1, 1)) ** 2)
            else:
                val_loss = avg_loss
            
            history.append({"epoch": epoch, "train_loss": float(avg_loss), "val_loss": float(val_loss)})
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            if epoch % 5 == 0 or epoch == actual_epochs - 1:
                logger.info(f"  Epoch {epoch}/{actual_epochs}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        self.trained = True
        self.loss_history = history
        
        return {"history": history, "best_val_loss": float(best_val_loss)}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.zeros((x.shape[0], 1))
        return self.forward(x)


def compute_single_forecast(ticker: str, prices: pd.Series, macro_df: pd.DataFrame, 
                            config: Dict, window: int, universe_size: int) -> Tuple[str, Dict]:
    """Compute forecast for a single ticker."""
    try:
        result = compute_rbf_kan_forecast(prices, macro_df, config, window, universe_size)
        return ticker, result
    except Exception as e:
        logger.error(f"  Error on {ticker} @ {window}d: {e}")
        return ticker, {"forecast": 0, "z_score": 0, "signal": 0, "error": str(e)}


def compute_rbf_kan_forecast(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252,
    universe_size: int = 30
) -> Dict:
    """Compute RBF-KAN-Transformer forecast for a single ticker."""
    full_returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(full_returns) < window:
        return {"forecast": 0, "z_score": 0, "error": f"Insufficient data for window {window}"}
    
    try:
        train_returns = full_returns[-window:]
        
        lookback = max(10, int(window * 0.2))
        horizon = max(1, min(5, int(window * 0.02)))
        
        config_horizon = config.get("horizon", 5)
        horizon = min(horizon, config_horizon)
        
        if len(train_returns) < lookback + horizon + 10:
            return {"forecast": 0, "z_score": 0, "error": f"Insufficient data for window {window}"}
        
        X = []
        y = []
        
        for i in range(lookback, len(train_returns) - horizon):
            seq = train_returns[i-lookback:i].reshape(-1, 1)
            
            if seq.shape[1] < 16:
                seq = np.repeat(seq, 16, axis=1)
            else:
                seq = seq[:, :16]
            
            X.append(seq)
            y.append(train_returns[i+horizon])
        
        if len(X) < 5:
            return {"forecast": 0, "z_score": 0, "error": f"Insufficient sequences ({len(X)}) for window {window}"}
        
        X = np.array(X)
        y = np.array(y)
        
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-6
        y_norm = (y - y_mean) / y_std
        
        # Pass universe_size to model for complexity scaling
        model = RBFKANTransformer(config, window=window, universe_size=universe_size)
        epochs = max(5, int(window / 20))  # Reduced epochs
        epochs = min(epochs, 20)  # Cap at 20
        
        result = model.train(X, y_norm, epochs=epochs)
        
        latest_seq = X[-1:].copy()
        if model.trained:
            forecast_norm = model.predict(latest_seq)[0, 0]
            forecast = forecast_norm * y_std + y_mean
        else:
            forecast = np.mean(y)
        
        st_window = max(5, int(window * 0.05))
        mt_window = max(10, int(window * 0.1))
        vol_window = max(10, int(window * 0.2))
        
        st_momentum = np.mean(train_returns[-st_window:]) if len(train_returns) >= st_window else 0
        mt_momentum = np.mean(train_returns[-mt_window:]) if len(train_returns) >= mt_window else 0
        volatility = np.std(train_returns[-vol_window:]) if len(train_returns) >= vol_window else 0
        
        if window <= 63:
            signal = 0.30 * forecast * 10 + 0.35 * st_momentum * 50 + 0.25 * mt_momentum * 30 - 0.10 * volatility * 20
        elif window <= 126:
            signal = 0.35 * forecast * 10 + 0.25 * st_momentum * 50 + 0.20 * mt_momentum * 30 - 0.20 * volatility * 20
        elif window <= 252:
            signal = 0.40 * forecast * 10 + 0.25 * st_momentum * 50 + 0.20 * mt_momentum * 30 - 0.15 * volatility * 20
        else:
            signal = 0.45 * forecast * 10 + 0.15 * st_momentum * 50 + 0.15 * mt_momentum * 30 - 0.25 * volatility * 20
        
        if abs(signal) < 1e-6:
            signal = forecast * 10 + st_momentum * 5
        
        return {
            "forecast": float(forecast),
            "z_score": float(signal),
            "signal": float(signal),
            "loss": float(result.get("best_val_loss", 0)),
            "n_epochs": len(result.get("history", [])),
            "window_used": window,
            "lookback_used": lookback,
            "horizon_used": horizon,
            "trained": model.trained,
            "error": None
        }
    except Exception as e:
        logger.error(f"Window {window}: Error - {e}")
        return {"forecast": 0, "z_score": 0, "signal": 0, "error": str(e)}


def compute_universe_rbf_kan(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252,
    max_workers: int = 8
) -> Dict:
    """Compute RBF-KAN-Transformer for all ETFs in a universe with parallel processing."""
    results = {}
    
    tickers = list(prices_df.columns)
    universe_size = len(tickers)
    
    logger.info(f"  Processing {universe_size} tickers at {window}d with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ticker in tickers:
            prices = prices_df[ticker]
            future = executor.submit(compute_single_forecast, ticker, prices, macro_df, config, window, universe_size)
            futures[future] = ticker
        
        completed = 0
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker, result = future.result(timeout=300)  # 5 minute timeout
                results[ticker] = {
                    "forecast": result.get("forecast", 0),
                    "z_score": result.get("signal", 0),
                    "signal": result.get("signal", 0),
                    "loss": result.get("loss", 0),
                    "n_epochs": result.get("n_epochs", 0),
                    "window_used": result.get("window_used", window),
                    "lookback_used": result.get("lookback_used", 0),
                    "horizon_used": result.get("horizon_used", 0),
                    "trained": result.get("trained", False)
                }
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"  Progress {window}d: {completed}/{len(tickers)} tickers done")
            except Exception as e:
                logger.error(f"  Failed on {ticker}: {e}")
                results[ticker] = {
                    "forecast": 0,
                    "z_score": 0,
                    "signal": 0,
                    "loss": 0,
                    "n_epochs": 0,
                    "window_used": window,
                    "lookback_used": 0,
                    "horizon_used": 0,
                    "trained": False
                }
    
    # Normalize z-scores
    signal_values = np.array([r["signal"] for r in results.values()])
    
    if len(signal_values) > 1 and np.std(signal_values) > 1e-6:
        mean_s = np.mean(signal_values)
        std_s = np.std(signal_values)
        for ticker, r in results.items():
            r["z_score"] = (r["signal"] - mean_s) / std_s
    else:
        forecasts = np.array([r["forecast"] for r in results.values()])
        if len(forecasts) > 1 and np.std(forecasts) > 1e-6:
            mean_f = np.mean(forecasts)
            std_f = np.std(forecasts)
            for ticker, r in results.items():
                r["z_score"] = (r["forecast"] - mean_f) / std_f
        else:
            for ticker in results.keys():
                prices = prices_df[ticker]
                returns = np.log(prices / prices.shift(1)).dropna().values
                if len(returns) > 20:
                    momentum = np.mean(returns[-20:]) * 100
                    results[ticker]["z_score"] = momentum
                else:
                    results[ticker]["z_score"] = 0.0
            
            signal_values = np.array([r["z_score"] for r in results.values()])
            if np.std(signal_values) > 1e-6:
                mean_s = np.mean(signal_values)
                std_s = np.std(signal_values)
                for ticker, r in results.items():
                    r["z_score"] = (r["z_score"] - mean_s) / std_s
    
    trained_count = sum(1 for r in results.values() if r.get("trained", False))
    logger.info(f"  ✅ {window}d: {trained_count}/{len(results)} tickers trained")
    
    return results
