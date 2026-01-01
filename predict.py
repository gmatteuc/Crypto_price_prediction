"""
Crypto Prediction - Inference Script
====================================

This script is designed for daily usage. It loads the trained model and latest data
to generate a trading signal (BUY/SELL/HOLD) for the next market day.
"""

import logging
import sys
import os

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from config import Config
from main import main

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    print("==================================================")
    print("   CRYPTO PREDICTION: DAILY INFERENCE MODE")
    print("==================================================")
    print("1. Downloading latest data from Yahoo Finance...")
    print("2. Loading pre-trained model (skipping training)...")
    print("3. Generating prediction for the next trading day...")
    print("==================================================\n")

    # Force training to False to use the existing model
    Config.TRAIN_MODEL = False
    
    # Run the main pipeline
    try:
        main()
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Make sure 'eth_lstm_classifier.pth' and 'eth_scaler_cls.pkl' exist.")
