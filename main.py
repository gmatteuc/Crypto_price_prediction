"""
Crypto Prediction - Main Execution Pipeline
===========================================

This is the entry point of the application. It orchestrates the entire workflow:
data loading, preprocessing, model training, evaluation, and visualization.
"""

import numpy as np
import joblib
import os
import torch
import logging
from config import Config
from utils import setup_logging, set_seeds, ensure_directories
from data_loader import download_data
from preprocessor import add_technical_indicators, prepare_features, create_sequences
from sklearn.preprocessing import StandardScaler
from model import CryptoLSTMClassifier, RandomForestWrapper, CryptoLinearClassifier

from trainer import Trainer
from evaluator import Evaluator
from strategy_optimizer import optimize_strategy_params
import visualization
from sklearn.ensemble import RandomForestClassifier



def main():
    # 1. Setup
    # --- Dynamic Output Directory based on Model Type ---
    # Include Architecture in path to separate LSTM/RF results
    model_arch = getattr(Config, 'MODEL_ARCH', 'lstm')
    Config.OUTPUT_DIR = os.path.join("output", f"{Config.MODEL_TYPE}_{model_arch}")
    
    # Update model and scaler paths to be inside the specific output folder
    ext = ".joblib" if model_arch == 'rf' else ".pth"
    Config.MODEL_FILE = os.path.join(Config.OUTPUT_DIR, f"eth_model_{Config.MODEL_TYPE}{ext}")
    Config.SCALER_FILE = os.path.join(Config.OUTPUT_DIR, f"eth_scaler_{Config.MODEL_TYPE}.pkl")

    ensure_directories([Config.OUTPUT_DIR])
    logger = setup_logging(os.path.join(Config.OUTPUT_DIR, "app.log"))
    set_seeds(42)
    visualization.set_plot_style()
    Config.print_config()
    
    # 2. Data Acquisition
    df = download_data(Config.TICKER)
    
    # Filter to take data from START_DATE onwards
    df = df[df.index >= Config.START_DATE]
    if getattr(Config, 'END_DATE', None):
        df = df[df.index <= Config.END_DATE]

    # 3. Preprocessing and Feature Engineering
    logger.info("Calculating technical indicators...")
    df_processed = add_technical_indicators(df)
    
    # Select features (removes initial NaNs and adds Target_Class or Target_Return)
    df_features = prepare_features(df_processed)
    
    target_col_name = 'Target_Class'
    logger.info("--- Target Class Distribution ---")
    dist = df_features['Target_Class'].value_counts(normalize=True)
    logger.info(f"Up (1): {dist.get(1.0, 0):.2%}")
    logger.info(f"Down (0): {dist.get(0.0, 0):.2%}")
    logger.info("---------------------------------")
    
    # Scaling
    y_raw = df_features[target_col_name].values
    y_scaled_input = y_raw

    X_df = df_features.drop(columns=[target_col_name])
    
    # --- NO LEAKAGE SCALING ---
    # Define split point on RAW data to fit scaler only on training portion
    raw_train_size = int(len(X_df) * (1 - Config.TEST_SPLIT))
    
    if Config.TRAIN_MODEL:
        scaler = StandardScaler()
        # Fit ONLY on the training portion of the data
        scaler.fit(X_df.iloc[:raw_train_size])
        joblib.dump(scaler, Config.SCALER_FILE)
        logger.info(f"Scaler fitted on first {raw_train_size} samples and saved.")
    else:
        if os.path.exists(Config.SCALER_FILE):
            scaler = joblib.load(Config.SCALER_FILE)
            logger.info(f"Scaler loaded from {Config.SCALER_FILE}")
        else:
            logger.error("Scaler not found. Train the model first.")
            return

    # Transform BOTH Train and Test using the scaler fitted on Train
    scaled_X = scaler.transform(X_df)

    # Create sequences
    # Pass features and targets separately to avoid target leakage in X and ensure correct alignment
    # Check if we should pass weights
    
    # Use magnitude of return as weight
    # Target_Return_Mag was added in preprocessor.py if classification
    weights = df_features['Target_Return_Mag'].values
    # Normalize weights to avoid exploding gradients: Scale to mean 1.0
    weights = weights / (np.mean(weights) + 1e-8)
    # Clip extreme weights
    weights = np.clip(weights, 0.1, 5.0) # Downweight noise, cap outliers (5x normal)
    
    X, y, w = create_sequences(scaled_X, y_raw, Config.SEQ_LENGTH, weights)
    
    # Split data using a 3-way split: Train, Validation (Early Stopping), Test (Unseen)
    total_len = len(X)
    test_size = int(total_len * Config.TEST_SPLIT)
    val_size = int(total_len * Config.VAL_SPLIT)
    train_size = total_len - val_size - test_size
    
    # Indices
    start_val = train_size
    start_test = train_size + val_size
    
    # Splitting
    X_train = X[:train_size]
    X_val = X[start_val:start_test]
    X_test = X[start_test:]
    
    y_train = y[:train_size]
    y_val = y[start_val:start_test]
    y_test = y[start_test:]
    
    # Split Weights if classification
    w_train = None
    if w is not None:
         w_train = w[:train_size]
         # We don't necessarily need w_val and w_test for training logic, but good for completeness
         w_val = w[start_val:start_test]
         w_test = w[start_test:]

    # --- Class Balance Correction ---
    class_weights_tensor = None
    
    # Calculate Class Imbalance in Training Set
    targets = y_train.flatten()
    n_down = len(targets[targets == 0])
    n_up = len(targets[targets == 1])
    logger.info(f"Training Class Balance: UP={n_up}, DOWN={n_down}")
    
    if n_down > 0 and n_up > 0:
        # Inverse Frequency Weights
        w0 = 1.0 / n_down
        w1 = 1.0 / n_up
        # Normalize to sum to 2
        total_w = w0 + w1
        w0 = (w0 / total_w) * 2
        w1 = (w1 / total_w) * 2
        class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float32).to(Config.DEVICE)
        logger.info(f"Calculated Class Weights: DOWN={w0:.4f}, UP={w1:.4f}")
    
    # --- Date Alignment for Logging ---
    # Retrieve dates corresponding to the LAST day of each sequence
    # Sequence X[i] ends at index (seq_len - 1 + i) in the original dataframe
    # We need the index *after clean up* (df_features).
    
    # Indices in df_features corresponding to the end of sequences:
    # Train range: [seq_len-1, seq_len-1 + train_size]
    # Val range: [seq_len-1 + start_val, seq_len-1 + start_test]
    # Test range: [seq_len-1 + start_test, seq_len-1 + total_len]
    
    def get_date_range(start_idx, size, base_idx):
        if size == 0: return "N/A"
        date_start = df_features.index[base_idx + start_idx]
        date_end = df_features.index[base_idx + start_idx + size - 1]
        return f"{date_start.date()} -> {date_end.date()}"
        
    start_seq_offset = Config.SEQ_LENGTH - 1
    
    logger.info(f"Total data: {total_len}")
    logger.info(f"Training Set: {len(X_train)} samples | Period: {get_date_range(0, len(X_train), start_seq_offset)}")
    logger.info(f"Validation Set: {len(X_val)} samples  | Period: {get_date_range(start_val, len(X_val), start_seq_offset)}")
    logger.info(f"Test Set: {len(X_test)} samples        | Period: {get_date_range(start_test, len(X_test), start_seq_offset)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float().to(Config.DEVICE)
    y_train_tensor = torch.from_numpy(y_train).float().to(Config.DEVICE)
    
    # Process Sample Weights (Profit Weights)
    w_train_tensor = None
    if getattr(Config, 'USE_PROFIT_WEIGHTS', False) and w_train is not None:
         logger.info("Converting Profit Weights to Tensor...")
         w_train_tensor = torch.from_numpy(w_train).float().to(Config.DEVICE)
         
    # Need Val tensor for Trainer
    X_val_tensor = torch.from_numpy(X_val).float().to(Config.DEVICE)
    y_val_tensor = torch.from_numpy(y_val).float().to(Config.DEVICE)

    # Need Test tensor for Evaluator
    X_test_tensor = torch.from_numpy(X_test).float().to(Config.DEVICE)
    y_test_tensor = torch.from_numpy(y_test).float().to(Config.DEVICE)
    
    # 4. Modeling
    input_size = X_train.shape[2]
    Config.INPUT_SIZE = input_size # Update config
    
    model_arch = getattr(Config, 'MODEL_ARCH', 'lstm')
    
    if model_arch == 'rf':
         logger.info(f"--- Initializing Random Forest {Config.MODEL_TYPE.capitalize()} ---")
         # Flatten for RF: (Samples, Seq_Len, Features) -> (Samples, Seq_Len*Features)
         X_train_flat = X_train.reshape(X_train.shape[0], -1)
         
         # Use balanced class weights to handle imbalance
         rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1)
         
         if Config.TRAIN_MODEL:
             logger.info(f"Training Random Forest on {len(X_train_flat)} samples...")
             rf.fit(X_train_flat, y_train.ravel())
             logger.info(f"Saving RF model to {Config.MODEL_FILE}")
             joblib.dump(rf, Config.MODEL_FILE)
             
             # Log Feature Importances (Top 5)
             importances = rf.feature_importances_
             indices = np.argsort(importances)[-5:]
             logger.info(f"Top 5 Feature Indices: {indices}")
         else:
             if os.path.exists(Config.MODEL_FILE):
                 logger.info(f"Loading RF model from {Config.MODEL_FILE}")
                 rf = joblib.load(Config.MODEL_FILE)
             else:
                 raise FileNotFoundError(f"Model file {Config.MODEL_FILE} not found.")

         model = RandomForestWrapper(rf, Config.MODEL_TYPE)
         
    else:
        # --- Deep Learning (LSTM/GRU) Path ---
        if model_arch == 'linear':
             logger.info(f"--- Initializing Linear {Config.MODEL_TYPE.capitalize()} Model ---")
             model = CryptoLinearClassifier(input_size=input_size, seq_length=Config.SEQ_LENGTH).to(Config.DEVICE)
        else:
            model = CryptoLSTMClassifier(
                input_size=input_size, 
                hidden_size=Config.HIDDEN_SIZE, 
                num_layers=Config.NUM_LAYERS, 
                dropout=Config.DROPOUT
            ).to(Config.DEVICE)
        
        trainer = Trainer(model, Config, Config.DEVICE)
        
        if Config.TRAIN_MODEL:
            # Pass Validation Set for Early Stopping
            # Updated to use class_weights AND/OR sample weights
            cw = class_weights_tensor if getattr(Config, 'USE_CLASS_WEIGHTS', True) else None
            sw = w_train_tensor if getattr(Config, 'USE_PROFIT_WEIGHTS', False) else None
            
            trainer.train(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, class_weights=cw, sample_weights=sw)
            # Load best model for evaluation
            if hasattr(model, 'load'):
                model.load(Config.MODEL_FILE, Config.DEVICE)
            else:
                model.load_state_dict(torch.load(Config.MODEL_FILE, map_location=Config.DEVICE, weights_only=True))
        else:
            if hasattr(model, 'load'):
                model.load(Config.MODEL_FILE, Config.DEVICE)
            else:
                model.load_state_dict(torch.load(Config.MODEL_FILE, map_location=Config.DEVICE, weights_only=True))
    
    # --- Strategy Optimization (Auto-Tuning) ---
    if Config.OPTIMIZE_STRATEGY_PARAMS and Config.USE_ADAPTIVE_THRESHOLD:
        # Optimization still uses Training Set (strict)
        b_win, b_std = optimize_strategy_params(model, X_train_tensor, df_features, train_size, Config, Config.DEVICE)
        Config.ADAPTIVE_WINDOW = b_win
        Config.ADAPTIVE_STD = b_std
        logger.info(f"Updated Config with Optimized Params: Window={Config.ADAPTIVE_WINDOW}, Std={Config.ADAPTIVE_STD}")
    
    # 5. Evaluation
    evaluator = Evaluator(model, Config, Config.DEVICE)
    predictions, raw_predictions = evaluator.evaluate_test_set(X_test_tensor, y_test)
    
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
    
    test_start_idx = Config.SEQ_LENGTH + start_test
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
    raw_predictions = raw_predictions[:min_len]
    test_log_rets = test_log_rets[:min_len]
    
    # Calculate SMA Benchmark Returns
    sma_net_rets = None
    if getattr(Config, 'INCLUDE_SMA_BENCHMARK', False):
        logger.info("Calculating SMA Benchmark Returns...")
        try:
            # Get target dates from df_features
            target_dates = df_features.index[test_start_idx : test_start_idx + min_len]
            
            # Calculate SMA logic on full history (df_processed)
            close_prices = df_processed['Close']
            sma_50 = close_prices.rolling(window=Config.SMA_WINDOW).mean() # Config.SMA_WINDOW defaults to 50
            
            # Signal at T: Close[T] > SMA[T]
            sma_signal = (close_prices > sma_50).astype(int)
            
            # We trade at T+1 based on Signal at T. 
            # So Signal at T aligns with Return at T+1.
            # Shift signal forward by 1 to align with Returns.
            sma_signal_shifted = sma_signal.shift(1)
            
            # Extract signals for the test period
            aligned_modes = sma_signal_shifted.loc[target_dates]
            
            # Helper to handle missing indices (if any mismatch)
            aligned_modes = aligned_modes.reindex(target_dates, method='ffill').fillna(0)
            
            sma_positions = aligned_modes.values
            
            # Calculate Returns
            sma_gross_rets = sma_positions * test_log_rets
            
            # SMA Transaction Costs
            sma_trades = np.abs(np.diff(np.concatenate(([0], sma_positions))))
            sma_costs = sma_trades * Config.TRANSACTION_COST
            
            sma_net_rets = sma_gross_rets - sma_costs
            
        except Exception as e:
            logger.error(f"Failed to calculate SMA Benchmark: {e}")
            sma_net_rets = None
            
    evaluator.calculate_financial_metrics(test_log_rets, predictions, sma_returns=sma_net_rets)
    
    # 5b. Event Study Analysis (New)
    # Verify if "Up" predictions actually lead to "Up" moves on average
    evaluator.perform_event_study(test_log_rets, predictions, raw_predictions=raw_predictions, window_back=3, window_fwd=5)

    # 6. Prediction for Tomorrow
    logger.info("\n--- Prediction for Tomorrow ---")
    # Last sequence should be the last available window
    # scaled_X is the full feature set
    last_sequence = scaled_X[-Config.SEQ_LENGTH:]
    last_sequence = last_sequence.reshape(1, Config.SEQ_LENGTH, input_size)
    last_sequence_tensor = torch.from_numpy(last_sequence).float().to(Config.DEVICE)
    
    model.eval()
    with torch.no_grad():
        output = model(last_sequence_tensor)
        
        # Classification
        prediction_prob = torch.sigmoid(output).item()
        
        # Determining Threshold
        threshold = 0.5
        if hasattr(evaluator, 'best_thresh'):
            threshold = evaluator.best_thresh
            
        prediction_class = 1 if prediction_prob > threshold else 0
        confidence_str = f"Confidence (Probability of UP): {prediction_prob:.4f} (Threshold: {threshold:.4f})"
        
        if Config.USE_ADAPTIVE_THRESHOLD:
             # Just showing simple probability for now in main.py output, 
             # as adaptive threshold relies on rolling window of TEST set probabilities which we have in evaluator.
             pass

    logger.info(f"Last Date in Data: {df.index[-1]}")
    horizon_str = "Day" if Config.PREDICTION_HORIZON == 1 else f"{Config.PREDICTION_HORIZON} Days"
    logger.info(f"Prediction for Next {horizon_str}: {'UP' if prediction_class == 1 else 'DOWN'}")
    logger.info(confidence_str)
    logger.info("-------------------------------")

if __name__ == "__main__":
    main()
