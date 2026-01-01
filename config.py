import torch

class Config:
    """
    Configuration class for the Crypto Prediction project.
    """
    # Data Settings
    TICKER = "ETH-USD"
    START_DATE = "2018-02-01" # Changed to align with Fear & Greed Index availability
    SEQ_LENGTH = 30       # Increased to 30 days to capture longer trends
    PREDICTION_HORIZON = 1 # Prediction horizon in days (1 = Daily, 7 = Weekly)
    TEST_SPLIT = 0.2
    
    # Model Settings
    INPUT_SIZE = None # Will be determined dynamically
    HIDDEN_SIZE = 64      # Medium capacity
    NUM_LAYERS = 1        # Single layer to prevent overfitting
    DROPOUT = 0.2         # Low dropout
    
    # Training Settings
    TRAIN_MODEL = True
    EPOCHS = 200          
    BATCH_SIZE = 64       
    LEARNING_RATE = 0.001 # Increased learning rate for GRU
    WEIGHT_DECAY = 1e-4   # Standard L2 Regularization
    PATIENCE = 20         # Increased patience
    
    # Strategy Settings
    TRANSACTION_COST = 0.001  # 0.1% per trade (Realistic Exchange Fee)
    CONFIDENCE_THRESHOLD = 0.65  # Increased for more conservative fixed threshold
    
    # Adaptive Strategy
    USE_ADAPTIVE_THRESHOLD = True
    ADAPTIVE_WINDOW = 20
    ADAPTIVE_STD = 1.5 # Increased from 1.0 to 1.5 to be more conservative (wider bands)
    
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
