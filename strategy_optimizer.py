"""
Crypto Prediction - Strategy Optimization
=========================================

This module implements grid search optimization for the adaptive strategy parameters
(Window and Standard Deviation) to maximize Sharpe Ratio on the training set.
"""

import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger("CryptoPrediction")

def optimize_strategy_params(model, X_tensor, df_features, split_index, config, device):
    """
    Optimizes Adaptive Strategy parameters (Window, Std) using the Training Set.
    
    Args:
        model: Trained PyTorch model
        X_tensor: Training set features tensor
        df_features: DataFrame containing features and targets (for return calculation)
        split_index: Index separating Train and Test
        config: Config object
        device: Torch device (cuda/cpu)
        
    Returns:
        tuple: (best_window, best_std)
    """
    logger.info("Starting Strategy Hyperparameter Optimization on Training Set...")
    
    # 1. Generate Model Probabilities for Training Set
    model.eval()
    
    batch_size = 512 # Process in chunks to save memory
    probs_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            
            if config.MODEL_TYPE == 'classification':
                p = torch.sigmoid(outputs).cpu().numpy().flatten()
            else:
                p = outputs.cpu().numpy().flatten()
            probs_list.append(p)
            
    probs = np.concatenate(probs_list)
    
    # 2. Extract Aligned Returns
    # We need the "Future Return" that the model was trying to predict.
    # Logic matches main.py/optimize_hyperparameters.py data alignment.
    
    # Re-calculate Target Return (Shifted) if not present
    # We need the return from t to t+Horizon. 
    # Log_Ret at index k is return from k-1 to k.
    # Prediction at index i (time t) targets return at t+Horizon.
    # So we want Log_Ret.shift(-Horizon).
    
    # But wait, df_features has already dropped NaNs.
    # And we need to match the "create_sequences" alignment.
    # X[0] (seq 0) uses features[0:seq_len].
    # Target y[0] is typically target[seq_len-1].
    # If using prepare_features logic:
    # df['Log_Ret'] is return.
    # We want return[seq_len + horizon]?
    
    # Let's use the robust alignment derived in optimize_hyperparameters.py (Attempt 5)
    # 1. Get raw Log Returns from df_features
    # 2. Shift them by -Horizon to get "Future Return"
    # 3. Align with Sequences
    
    # Calculate shifted returns on the full df_features
    # df_features has 'Log_Ret'.
    # Note: 'Log_Ret' is past return.
    # Return to Capture = Log_Ret of (t + Horizon).
    # Since df_features index i corresponds to time t.
    # We want Log_Ret at index i + Horizon.
    
    future_returns = df_features['Log_Ret'].shift(-config.PREDICTION_HORIZON).values
    
    # Align with sequences
    # X corresponds to sequences ending at i = seq_len-1, seq_len, ...
    # So for X[k], the corresponding time index in df_features is `seq_len - 1 + k`.
    # We want future_return at that index.
    
    # Start index in df_features for X[0]
    start_idx_df = config.SEQ_LENGTH - 1
    
    # Slice matched returns for the Training Set
    # We only need as many returns as we have training samples (len(probs))
    train_returns = future_returns[start_idx_df : start_idx_df + len(probs)]
    
    # Handle NaNs (last few values might be NaN due to shift, but Training set is at the beginning, so should be fine)
    # Replace any NaNs with 0 to avoid breaking calc
    train_returns = np.nan_to_num(train_returns)
    
    logger.info(f"Optimizing on {len(probs)} samples. Mean Return: {np.mean(train_returns)*100:.4f}%")
    
    # 3. Grid Search
    windows = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    stds = [0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    
    best_sharpe = -100
    best_params = {'Window': config.ADAPTIVE_WINDOW, 'Std': config.ADAPTIVE_STD} # Default fallback
    
    # Store results for logging
    top_results = []
    
    # Benchmark (Buy & Hold or Fixed Threshold)
    # Fixed 0.5
    bench_pos = (probs > 0.5).astype(float)
    bench_rets = bench_pos * train_returns - (np.abs(np.diff(bench_pos, prepend=0)) * config.TRANSACTION_COST)
    if np.std(bench_rets) == 0: bench_sharpe = 0
    else: bench_sharpe = np.mean(bench_rets) / np.std(bench_rets) * np.sqrt(365)
    
    logger.info(f"Benchmark (Fixed 0.5) Sharpe: {bench_sharpe:.4f}")
    
    for w in windows:
        for s in stds:
            # Calculate signals
            preds_series = pd.Series(probs)
            
            # Rolling Stats (Shift 1 to use PAST data only)
            rolling_mean = preds_series.rolling(window=w, min_periods=1).mean().shift(1)
            rolling_std = preds_series.rolling(window=w, min_periods=1).std().shift(1).fillna(1e-6)
            rolling_mean = rolling_mean.fillna(0.5)
            
            upper = rolling_mean + (s * rolling_std)
            lower = rolling_mean - (s * rolling_std)
            
            # Apply Strategy
            # Vectorized implementation of:
            # if p > u: 1, elif p < l: 0, else: hold
            
            # Since "hold" is recursive, pure vectorization is hard. 
            # But we can approximate or use a loop. loop for 2000 items is fast.
            positions = np.zeros(len(probs))
            curr = 0 # Start Flat
            
            # Optimization: Extract numpy arrays for speed
            p_arr = probs
            u_arr = upper.values
            l_arr = lower.values
            
            for i in range(len(p_arr)):
                if p_arr[i] > u_arr[i]:
                    curr = 1
                elif p_arr[i] < l_arr[i]:
                    curr = 0
                positions[i] = curr
                
            # Calcuclate Returns
            strat_rets = positions * train_returns - (np.abs(np.diff(positions, prepend=0)) * config.TRANSACTION_COST)
            
            if np.std(strat_rets) == 0:
                sharpe = 0
            else:
                sharpe = np.mean(strat_rets) / np.std(strat_rets) * np.sqrt(365)
                
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'Window': w, 'Std': s}
                
            if sharpe > (bench_sharpe * 1.05): # Only track if better than bench
                 top_results.append((w, s, sharpe))
                 
    # Sort top results
    top_results.sort(key=lambda x: x[2], reverse=True)
    
    if len(top_results) > 0:
        logger.info(f"Top 3 Configs found: {top_results[:3]}")
    
    logger.info(f"Optimization Complete. Best Params: Window={best_params['Window']}, Std={best_params['Std']} (Sharpe: {best_sharpe:.4f})")
    
    return best_params['Window'], best_params['Std']
