"""
Crypto Prediction - Configuration Module
========================================

This module defines global configuration parameters, hyperparameters,
file paths, and feature lists used throughout the project.
"""

import torch

class Config:
    """
    Configuration class for the Crypto Prediction project.
    """
    # Data Settings
    TICKER = "ETH-USD"
    START_DATE = "2018-02-01" # Changed to align with Fear & Greed Index availability
    END_DATE = "2026-01-06"   # Fixed end date for reproducibility. Set to None to use latest data.
    SEQ_LENGTH = 60       # Increased to 60 days (2 months) to capture longer trends
    PREDICTION_HORIZON = 1 # Prediction horizon in days (1 = Daily, 7 = Weekly)
    TEST_SPLIT = 0.15     # Final Unseen Test Set
    VAL_SPLIT = 0.15      # Validation Set for Early Stopping
    
    # Model Settings
    MODEL_ARCH = 'lstm'   # Options: 'lstm' (LSTM), 'rf' (Random Forest), 'linear' (Logistic Regression)
    MODEL_TYPE = 'classification' 
    INPUT_SIZE = None # Will be determined dynamically
    HIDDEN_SIZE = 64      # Reduced capacity (Restored Sniper Config)
    NUM_LAYERS = 2        # Shallower network (Restored Sniper Config)
    DROPOUT = 0.5         # High dropout (Restored Sniper Config)
    
    # Training Settings
    TRAIN_MODEL = True # Enable training to populate new output folder
    EPOCHS = 200          
    BATCH_SIZE = 64       
    LEARNING_RATE = 0.0005 # Restored Lower LR
    WEIGHT_DECAY = 1e-3   # Restored Stronger L2 Regularization (was 1e-4)
    PATIENCE = 20         # Restored Lower Patience
    
    USE_PROFIT_WEIGHTS = True  # Enable profit-based sample weighting
    USE_CLASS_WEIGHTS = True   # Enable class-imbalance weighting

    # Strategy Settings
    TRANSACTION_COST = 0.001  # 0.1% per trade (Realistic Exchange Fee)
    CONFIDENCE_THRESHOLD = 0.52  # Slightly above 0.5 for margin of safety
    
    # Adaptive Strategy
    USE_ADAPTIVE_THRESHOLD = True
    OPTIMIZE_STRATEGY_PARAMS = True 
    ADAPTIVE_WINDOW = 25 
    ADAPTIVE_STD = 0.10
    
    # Benchmarks
    INCLUDE_SMA_BENCHMARK = True
    SMA_WINDOW = 50
    
    # Paths
    MODEL_FILE = 'eth_lstm_classifier.pth'
    SCALER_FILE = 'eth_scaler_cls.pkl'
    OUTPUT_DIR = 'output'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def print_config():
        print("\n" + "="*30)
        print("        CONFIGURATION        ")
        print("="*30)
        for key, value in Config.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key:<15} : {value}")
        print("="*30 + "\n")
