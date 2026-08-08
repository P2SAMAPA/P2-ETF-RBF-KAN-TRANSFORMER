"""
config.py  —  Configuration for RBF-KAN-Transformer Engine
===========================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - MODEL: RBF-KAN-Transformer architecture parameters
  - TRAINING: Training parameters
  - FORECAST: Forecasting parameters
  - WINDOWS: Time windows for analysis
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-rbf-kan-transformer-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 126, 252, 504]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    126: "126d (~6 months) — Medium-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Long-term",
}
PRIMARY_WINDOW = 252


# ── RBF-KAN-Transformer Architecture ──────────────────────────────────────

MODEL = {
    # RBF Layer
    "rbf_centers": 32,         # Number of RBF centers
    "rbf_gamma": 1.0,          # RBF width parameter
    
    # KAN Layer
    "kan_hidden_dim": 64,      # KAN hidden dimension
    "kan_grid_size": 5,        # KAN grid size (spline resolution)
    "kan_degree": 3,           # KAN spline degree
    "kan_layers": 2,           # Number of KAN layers
    
    # Transformer Layer
    "transformer_dim": 64,     # Transformer dimension
    "transformer_heads": 4,    # Number of attention heads
    "transformer_layers": 2,   # Number of transformer layers
    "transformer_dropout": 0.1,# Dropout rate
    
    # General
    "embedding_dim": 64,       # Overall embedding dimension
    "ffn_dim": 128,            # Feed-forward network dimension
}


# ── Training Parameters ─────────────────────────────────────────────────────

TRAINING = {
    "learning_rate": 0.001,    # Learning rate
    "n_epochs": 100,           # Training epochs
    "batch_size": 32,          # Batch size
    "weight_decay": 0.01,      # Weight decay
    "early_stopping": True,    # Early stopping
    "patience": 10,            # Patience for early stopping
    "val_split": 0.2,          # Validation split
}


# ── Forecasting Parameters ──────────────────────────────────────────────────

FORECAST = {
    "horizon": 5,              # Forecast horizon (days ahead)
    "lookback": 60,            # Lookback window for features
    "n_forecasts": 100,        # Number of forecasts
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
