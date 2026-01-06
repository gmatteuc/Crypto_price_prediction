
"""
=============================================================================
INTEGRATION TEST: ML PIPELINE VALIDATION
=============================================================================

This script verifies the integrity of the Crypto Prediction Pipeline by attempting
to learn a known signal from synthetic data.

RATIONALE:
The production model shows neutral performance (Acc ~50%) on real market data.
To determine if this is due to 'Market Efficiency' (no signal exists) or 'Code Defects' 
(pipeline broken), we test the exact same pipeline on synthetic data where we KNOW 
a signal exists.

METHODOLOGY:
1.  Generate Synthetic Price Data: 
    - Construct an 'ETH' price series that is mathematically dependent on lagged 
      market regressors (BTC, SP500, DXY) + Noise.
    - Formula: ETH_Ret(t+1) = f(BTC_Ret(t), SP500_Ret(t)) + Noise.
2.  Pipeline Reuse:
    - Reuse `preprocessor.py` to generate features.
    - Reuse `trainer.py` to train the production LSTM architecture.
3.  Success Metric:
    - If the code works, the model should achieve High Accuracy (>70%) on this task.
    - This proves the LSTM is capable of learning complex non-linear dependencies 
      when they actually exist.

OUTPUT:
- Prints training logs to terminal.
- Saves `test_pipeline_results.png` to output/ folder evaluating performance.
=============================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from preprocessor import add_technical_indicators, prepare_features, create_sequences, scale_data
from model import CryptoLSTMClassifier
from trainer import Trainer
from data_loader import download_data
import visualization

class TestPredictionPipeline:
    """
    Test Suite for the Crypto Prediction Pipeline.
    Encapsulates synthetic data generation and full integration testing.
    """
    
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generates synthetic data using REAL market regressors (BTC, SP500, etc.).
        The Synthetic ETH price is constructed as a deterministic response to these 
        lagged market factors.
        """
        print("--- Downloading Real Market Data for Synthetic Construction ---")
        # Reuse the main data loader to get the exact same structure (SP500, BTC, Gold...)
        # Fetching a long enough history to mimic the real dataset
        start_date = "2018-01-01" 
        df = download_data("ETH-USD", start_date=start_date)
        
        # Ensure we have enough data
        if len(df) > n_samples:
            df = df.iloc[-n_samples:].copy() # Use recent data
        
        print(f"Constructing Synthetic 'ETH' from {len(df)} days of real auxiliary data...")
        
        # 1. Calculate Returns of the real drivers
        # These features will be re-calculated by preprocessor later, but we need them now for construction
        btc_ret = df['BTC'].pct_change().fillna(0)
        sp500_ret = df['SP500'].pct_change().fillna(0)
        dxy_ret = df['DXY'].pct_change().fillna(0) # Dollar Index
        
        # 2. Define the Synthetic Causal Logic
        # Target(t+1) responds to Drivers(t)
        # ETH moves with BTC and SP500, inverses to DXY, with a lag of 1 day (predictable)
        signal = 0.6 * btc_ret + 0.3 * sp500_ret - 0.3 * dxy_ret
        
        # Shift causality: Yesterday's signal drives Today's return
        # So we can predict Today's return using Yesterday's data
        synthetic_log_ret = signal.shift(1).fillna(0)
        
        # 3. Add Noise (to make it non-trivial but learnable)
        # Reduced noise to avoid overfitting issues on small dataset
        noise = np.random.normal(0, 0.0025, len(df))
        synthetic_log_ret = synthetic_log_ret + noise
        
        # 4. Reconstruct Price Path
        # Start at 1000
        # cumsum of log returns = log(price/start) -> price = start * exp(cumsum)
        # We handle NaN at first element
        price_path = 1000 * np.exp(synthetic_log_ret.cumsum())
        
        # 5. Overwrite the ETH columns in the dataframe
        df['Close'] = price_path
        df['Open'] = price_path # Simplified
        df['High'] = price_path * 1.02
        df['Low'] = price_path * 0.98
        df['Adj Close'] = price_path
        # Keep original Volume or randomize it
        # df['Volume'] = df['Volume'] 
        
        # Identify correlations for verification
        corr_btc = np.corrcoef(synthetic_log_ret[2:], btc_ret[1:-1])[0,1]
        print(f"Synthetic Data Construction Stats:")
        print(f"  - Lagged Correlation with BTC: {corr_btc:.4f}")
        
        return df

    def run_integration_test(self):
        """
        Main Test Execution:
        1. Generate Synthetic Data
        2. Run Full Processing Pipeline
        3. Train Model
        4. Evaluate & Visualize
        """
        print("\n--- Generating Synthetic Data ---")
        df = self.generate_synthetic_data(1000)
        
        # 1. Add Indicators
        # This will calculate Volume_Log_Change
        print("--- Adding Technical Indicators ---")
        df_processed = add_technical_indicators(df)
        
        # Debug: Check correlation
        # Volume_Log_Change[t] vs Log_Ret[t+1] (Target)
        # Note: Log_Ret in df is current return.
        # Future Target is derived later.
        
        # 2. Select Features & Create Target
        print("--- Preparing Features ---")
        # Ensure Config is set correctly for the test
        Config.PREDICTION_HORIZON = 1
        Config.SEQ_LENGTH = 10 # Short sequence for faster training
        
        df_features = prepare_features(df_processed)
        
        # Debug: Check columns
        print(f"Features: {df_features.columns.tolist()}")
        
        # Check if we have a signal
        # Target_Class is 1 if Close[t+1] > Close[t]
        # We constructed data such that if Vol_Change[t] > 0 -> Close[t+1] > Close[t]
        # So Vol_Change[t] should be positively correlated with Target_Class[t]
        
        # Debug: Check correlation
        # Check signal correlation with BTC (as proxy for signal)
        # In our construction: Close(t+1) moves with BTC(t)
        # So Target_Class(t) (which is direction of t -> t+1) should corr with BTC_Ret(t)
        
        # Verify columns exist
        print(f"Features in df: {df_features.columns.tolist()}")
        if 'BTC_Ret' in df_features.columns:
            correlation = df_features['BTC_Ret'].corr(df_features['Target_Class'])
            print(f"Signal Correlation (BTC_Ret vs Target): {correlation:.4f}")
        else:
            correlation = 0
            print("BTC_Ret not found in features, skipping simple correlation check.")
        
        # 3. Scale and Create Sequences
        target_col = 'Target_Class'
        y_raw = df_features[target_col].values
        X_df = df_features.drop(columns=[target_col, 'Target_Return_Mag']) # Drop auxiliary target columns
        
        # Use project's scale_data function to ensure identical preprocessing
        scaled_X, scaler = scale_data(X_df)
        
        # Create Sequences
        # We need to reshape for 3D input first? 
        # No, create_sequences takes 2D scaled_X and returns 3D X
        
        X, y = create_sequences(scaled_X, y_raw, Config.SEQ_LENGTH)
        
        # 4. Split Train/Test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        
        # 5. Train Model using Project Trainer
        # Set Config params for this test
        Config.BATCH_SIZE = 32
        Config.EPOCHS = 30
        Config.LEARNING_RATE = 0.005
        Config.MODEL_TYPE = 'classification'
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device) # Trainer expects 1D or logic in trainer handles unsqueeze
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        input_size = X_train.shape[2]
        model = CryptoLSTMClassifier(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.0)
        model.to(device)
        
        # Initialize Trainer
        trainer = Trainer(model, Config, device)
        
        print(f"--- Training LSTM using Project Pipeline ({Config.EPOCHS} epochs) ---")
        trainer.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        
        # 6. Evaluate
        model.eval()
        with torch.no_grad():
            # Trainer handles batching, but for eval small data we can do full batch
            train_outputs = model(X_train_tensor)
            test_outputs = model(X_test_tensor)
            
            # Helper for probability
            train_probs = torch.sigmoid(train_outputs).cpu().numpy()
            test_probs = torch.sigmoid(test_outputs).cpu().numpy()
            
            y_pred_train = (train_probs > 0.5).astype(int).flatten()
            y_pred_test = (test_probs > 0.5).astype(int).flatten()
            
            # Use probabilities for consistency with visualization code
            y_probs = test_probs.flatten()
            y_pred = y_pred_test

        train_score = (y_pred_train == y_train).mean()
        test_score = (y_pred == y_test).mean()
        
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy:  {test_score:.4f}")

        # --- Visualization ---
        # --- Visualization ---
        self.visualize_results(y_test, y_pred, y_probs, test_score, df, X_test, df_features, target_col)
        
        # Assertions
        # With real market noise and a complex mixture signal, accuracy won't be as high as simple signals.
        # But it should be significantly better than random guessing (0.50).
        # We observed ~0.70+ with real BTC/SP500 mixture.
        assert test_score > 0.65, f"Pipeline failed to learn complex mixture signal! Test Acc: {test_score:.4f}"
        print("TEST PASSED: Pipeline correctly learned the complex market mixture signal.")

    def visualize_results(self, y_test, y_pred, y_probs, test_score, df, X_test, df_features, target_col):
        """Generates and saves validation plots."""
        print("--- Generating Visualizations ---")
        visualization.set_plot_style()
        
        # Create a figure with layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax1, cbar=False)
        ax1.set_title('Confusion Matrix (LSTM on Synthetic)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_xticklabels(['Down', 'Up'])
        ax1.set_yticklabels(['Down', 'Up'])
        
        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Use Theme Colors
        main_purple = "#4B0082"
        main_orange = "#FF8C00"
        sec_purple = "#9370DB"
        
        ax2.plot(fpr, tpr, color=main_orange, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color=main_purple, lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve (Synthetic Data)')
        ax2.legend(loc="lower right")

        # 3. Combined Price + Signals (Bottom Row - Spanning Full Width)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Plot Price History (Focus on Test Period for clarity)
        # Identify Test Indices in original DF
        test_start_idx = len(df) - len(y_test)
        test_dates = df.index[test_start_idx:]
        test_prices = df['Close'].iloc[test_start_idx:]
        
        # Limit to 150 points for readability if large
        subset_len = min(150, len(test_dates))
        plot_dates = test_dates[:subset_len]
        plot_prices = test_prices[:subset_len]
        plot_y_test = y_test[:subset_len]
        plot_y_pred = y_pred[:subset_len]

        ax3.plot(plot_dates, plot_prices, color='#333333', lw=2, label='Price')
        ax3.set_title(f'Test Period: Price vs Predictions (LSTM) - Acc: {test_score:.1%} - Driven by Market Regressors')
        ax3.set_ylabel('Price ($)')
        ax3.grid(True, alpha=0.3)
        
        # Overlay Trend Background
        # Plot BTC trend as background driver
        ax3b = ax3.twinx()
        
        feature_cols = df_features.drop(columns=[target_col, 'Target_Return_Mag']).columns.tolist()
        
        if 'BTC_Ret' in feature_cols:
             btc_idx = feature_cols.index('BTC_Ret')
             btc_signal = X_test[:subset_len, -1, btc_idx]
             ax3b.plot(plot_dates, btc_signal, color=sec_purple, alpha=0.3, lw=1.5, ls='-', label='BTC Return (Driver)')
             ax3b.set_ylabel('BTC Return (Std)', color=sec_purple)
        
        # Plot Targets and Predictions on Twin Axis (Higher zorder)
        # Actual Targets (Circles)
        ax3b.scatter(plot_dates, plot_y_test, color=main_orange, label='Actual Next Move', marker='o', s=40, zorder=5)
        # Predictions (Xs)
        ax3b.scatter(plot_dates, plot_y_pred, color=main_purple, label='Predicted Next Move', marker='x', s=60, lw=2, zorder=6)
        
        # Hide BTC axis ticks to avoid confusion, or keep them on right
        ax3b.tick_params(axis='y', labelcolor=sec_purple)
        
        # Legends
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
        
        plt.tight_layout()
        
        # Save plot to output directory
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, "test_pipeline_results.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    tester = TestPredictionPipeline()
    tester.run_integration_test()
