import os
from datetime import datetime, timedelta

class Config:
    # Model parameters
    MODEL_PARAMS = {
        'max_features': 5000,
        'max_length': 200,
        'embedding_dim': 100,
        'epochs': 10,
        'batch_size': 32,
        'cv_folds': 5
    }
    
    # Backtesting parameters
    BACKTEST_PARAMS = {
        'initial_capital': 10000,
        'position_size': 0.1  # 10% of capital per trade
    }