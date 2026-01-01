import numpy as np
import joblib
import os
import torch
import logging
from config import Config
from utils import setup_logging, set_seeds, ensure_directories
from data_loader import download_data
from preprocessor import add_technical_indicators, prepare_features, scale_data, create_sequences
from model import CryptoLSTMClassifier
from trainer import Trainer
from evaluator import Evaluator
import visualization

def main():
    # 1. Setup
    ensure_directories([Config.OUTPUT_DIR])
    logger = setup_logging(os.path.join(Config.OUTPUT_DIR, "app.log"))
    set_seeds(42)
    visualization.set_plot_style()
    Config.print_config()
    
    # 2. Data Acquisition
    df = download_data(Config.TICKER)
    
    # Filter to take data from START_DATE onwards
    df = df[df.index >= Config.START_DATE]

    # 3. Preprocessing and Feature Engineering
    logger.info("Calculating technical indicators...")
    df_processed = add_technical_indicators(df)
    
    # Select features (removes initial NaNs and adds Target_Class)
    df_features = prepare_features(df_processed)
    
    logger.info("--- Target Class Distribution ---")
    dist = df_features['Target_Class'].value_counts(normalize=True)
    logger.info(f"Up (1): {dist.get(1.0, 0):.2%}")
    logger.info(f"Down (0): {dist.get(0.0, 0):.2%}")
    logger.info("---------------------------------")
    
    # Scaling
    target_col_name = 'Target_Class'
    y_raw = df_features[target_col_name].values
    X_df = df_features.drop(columns=[target_col_name])
    
    if Config.TRAIN_MODEL:
        scaled_X, scaler = scale_data(X_df)
        joblib.dump(scaler, Config.SCALER_FILE)
        logger.info(f"Scaler saved in {Config.SCALER_FILE}")
    else:
        if os.path.exists(Config.SCALER_FILE):
            scaler = joblib.load(Config.SCALER_FILE)
            scaled_X = scaler.transform(X_df)
            logger.info(f"Scaler loaded from {Config.SCALER_FILE}")
        else:
            logger.error("Scaler not found. Train the model first.")
            return
            
    # Create sequences
    # Pass features and targets separately to avoid target leakage in X and ensure correct alignment
    X, y = create_sequences(scaled_X, y_raw, Config.SEQ_LENGTH)
    
    # Split data
    split_index = int(len(X) * (1 - Config.TEST_SPLIT))
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    logger.info(f"Total data: {len(X)}")
    logger.info(f"Training Set: {len(X_train)} samples")
    logger.info(f"Test Set: {len(X_test)} samples")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float().to(Config.DEVICE)
    y_train_tensor = torch.from_numpy(y_train).float().to(Config.DEVICE)
    X_test_tensor = torch.from_numpy(X_test).float().to(Config.DEVICE)
    y_test_tensor = torch.from_numpy(y_test).float().to(Config.DEVICE)
    
    # 4. Modeling
    input_size = X_train.shape[2]
    Config.INPUT_SIZE = input_size # Update config
    
    model = CryptoLSTMClassifier(
        input_size=input_size, 
        hidden_size=Config.HIDDEN_SIZE, 
        num_layers=Config.NUM_LAYERS, 
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    trainer = Trainer(model, Config, Config.DEVICE)
    
    if Config.TRAIN_MODEL:
        trainer.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        # Load best model for evaluation
        model.load(Config.MODEL_FILE, Config.DEVICE)
    else:
        model.load(Config.MODEL_FILE, Config.DEVICE)
    
    # 5. Evaluation
    evaluator = Evaluator(model, Config, Config.DEVICE)
    predictions = evaluator.evaluate_test_set(X_test_tensor, y_test)
    
    # Calculate Returns for Equity Curve
    # We need to align returns with the test set predictions
    # X[k] corresponds to y[k]
    # y[k] is y_raw[i] where i is the index in the original data
    # The first sequence ends at index (seq_length - 1)
    # So X[0] corresponds to index (seq_length - 1)
    # The test set starts at split_index relative to X
    # So X_test[0] corresponds to X[split_index]
    # Which corresponds to original index (seq_length - 1 + split_index)
    # We want the return that corresponds to the target y_test[0]
    # y_test[0] is target at original index (seq_length - 1 + split_index)
    # Target at index i is derived from Log_Ret at i+1 (shifted -1)
    # So we need Log_Ret at (seq_length - 1 + split_index + 1) = (seq_length + split_index)
    
    test_start_idx = Config.SEQ_LENGTH + split_index
    # Note: df_features has dropped NaNs, so indices are reset? No, df_features index is preserved if not reset.
    # But y_raw is numpy array from df_features.
    # We should use .values from df_features['Log_Ret']
    
    # Wait, create_sequences loop:
    # i starts at seq_length - 1.
    # X[0] is features[0:seq_length]. Ends at index seq_length-1.
    # y[0] is targets[seq_length-1].
    # targets[k] comes from df_features['Target_Class'].iloc[k]
    # df_features['Target_Class'] at k is derived from Log_Ret at k+1.
    # So y[0] predicts return at index seq_length.
    
    # X_test starts at split_index.
    # X_test[0] is X[split_index].
    # Corresponds to i = (seq_length - 1) + split_index.
    # y_test[0] is targets[i].
    # Predicts return at index i+1 = seq_length + split_index.
    
    # So we need Log_Ret starting from index (seq_length + split_index).
    
    test_log_rets = df_features['Log_Ret'].values[test_start_idx:]
    
    # Ensure lengths match
    min_len = min(len(predictions), len(test_log_rets))
    predictions = predictions[:min_len]
    test_log_rets = test_log_rets[:min_len]
    
    evaluator.calculate_financial_metrics(test_log_rets, predictions)

    # 6. Prediction for Tomorrow
    logger.info("\n--- Prediction for Tomorrow ---")
    # Last sequence should be the last available window
    # scaled_X is the full feature set
    last_sequence = scaled_X[-Config.SEQ_LENGTH:]
    last_sequence = last_sequence.reshape(1, Config.SEQ_LENGTH, input_size)
    last_sequence_tensor = torch.from_numpy(last_sequence).float().to(Config.DEVICE)
    
    model.eval()
    with torch.no_grad():
        prediction_logit = model(last_sequence_tensor)
        prediction_prob = torch.sigmoid(prediction_logit).item()
        prediction_class = 1 if prediction_prob > 0.5 else 0
    
    logger.info(f"Last Date in Data: {df.index[-1]}")
    horizon_str = "Day" if Config.PREDICTION_HORIZON == 1 else f"{Config.PREDICTION_HORIZON} Days"
    logger.info(f"Prediction for Next {horizon_str}: {'UP' if prediction_class == 1 else 'DOWN'}")
    logger.info(f"Confidence (Probability of UP): {prediction_prob:.4f}")
    logger.info("-------------------------------")

if __name__ == "__main__":
    main()
