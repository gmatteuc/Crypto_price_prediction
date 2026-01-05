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
    SEQ_LENGTH = 14       # Reduced to 14 days (2 weeks) to capture recent momentum and reduce noise
    PREDICTION_HORIZON = 1 # Prediction horizon in days (1 = Daily, 7 = Weekly)
    TEST_SPLIT = 0.2
    
    # Model Settings
    MODEL_TYPE = 'regression' # 'classification' or 'regression'
    INPUT_SIZE = None # Will be determined dynamically
    HIDDEN_SIZE = 64      # Reduced capacity to prevent overfitting
    NUM_LAYERS = 2        # Two layers to capture more complex patterns
    DROPOUT = 0.2         # Increased dropout for better regularization
    
    # Training Settings
    TRAIN_MODEL = True
    EPOCHS = 200          
    BATCH_SIZE = 64       
    LEARNING_RATE = 0.001 # Increased learning rate for GRU
    WEIGHT_DECAY = 1e-4   # Standard L2 Regularization
    PATIENCE = 20         # Increased patience
    DIRECTIONAL_PENALTY_WEIGHT = 1.0 # Reduced penalty to prevent model collapse to zero
    
    # Strategy Settings
    TRANSACTION_COST = 0.001  # 0.1% per trade (Realistic Exchange Fee)
    CONFIDENCE_THRESHOLD = 0.60  # Decreased for more aggressive fixed threshold
    
    # Adaptive Strategy
    USE_ADAPTIVE_THRESHOLD = True
    ADAPTIVE_WINDOW = 10 # Reduced to 10 days for faster reaction to trend changes
    ADAPTIVE_STD = 1.0 # Decreased from 1.5 to 1.0 to be more aggressive
    
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
