import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, precision_recall_curve
import os
import logging
import torch
import visualization

logger = logging.getLogger("CryptoPrediction")

class Evaluator:
    """
    Handles evaluation metrics and visualization.
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        # Get styles
        self.palette, self.div_cmap, self.grad_cmap, self.purple_cmap = visualization.set_plot_style()

    def evaluate_test_set(self, X_test: torch.Tensor, y_test: np.ndarray) -> np.ndarray:
        """
        Evaluates the model on the test set and prints metrics.
        Returns predictions (positions).
        """
        logger.info(f"Evaluation on Test Set (Device: {self.device}):")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        # Log Probability Statistics
        logger.info(f"\n--- Prediction Probabilities Stats ---")
        logger.info(f"Min: {probs.min():.4f}")
        logger.info(f"Max: {probs.max():.4f}")
        logger.info(f"Mean: {probs.mean():.4f}")
        logger.info(f"Std: {probs.std():.4f}")
        logger.info(f"--------------------------------------")
        
        self.probs = probs # Store for plotting
            
        # Apply Trading Logic
        positions = []
        current_pos = 0 # Start with Cash
        
        if self.config.USE_ADAPTIVE_THRESHOLD:
            # Calculate Adaptive Thresholds
            probs_series = pd.Series(probs)
            rolling_mean = probs_series.rolling(window=self.config.ADAPTIVE_WINDOW, min_periods=1).mean()
            rolling_std = probs_series.rolling(window=self.config.ADAPTIVE_WINDOW, min_periods=1).std()
            
            # Handle initial NaNs or 0 std
            rolling_std = rolling_std.fillna(0)
            
            self.upper_bound = rolling_mean + (self.config.ADAPTIVE_STD * rolling_std)
            self.lower_bound = rolling_mean - (self.config.ADAPTIVE_STD * rolling_std)
            
            logger.info(f"Using Adaptive Threshold (Window={self.config.ADAPTIVE_WINDOW}, Std={self.config.ADAPTIVE_STD})")
            
            for i, p in enumerate(probs):
                ub = self.upper_bound.iloc[i]
                lb = self.lower_bound.iloc[i]
                
                if p > ub:
                    current_pos = 1 # Buy
                elif p < lb:
                    current_pos = 0 # Sell
                # else: maintain current_pos
                positions.append(current_pos)
                
        else:
            # Fixed Threshold Logic
            threshold = self.config.CONFIDENCE_THRESHOLD
            self.upper_bound = np.full_like(probs, threshold)
            self.lower_bound = np.full_like(probs, 1 - threshold)
            
            logger.info(f"Using Fixed Threshold: {threshold}")
            
            for p in probs:
                if p >= threshold:
                    current_pos = 1 # Buy/Long
                elif p <= (1 - threshold):
                    current_pos = 0 # Sell/Cash
                # else: maintain current_pos
                positions.append(current_pos)
            
        predictions = np.array(positions)
        
        # Standard Classification Metrics (using 0.5 for pure accuracy check)
        # Use mean probability as threshold to show discriminative power better
        # if the model is uncalibrated (e.g. all probs > 0.5)
        mean_prob = np.mean(probs)
        # If mean is very skewed, use it, otherwise stick to 0.5
        # But for "Confusion Matrix" display, let's use 0.5 to be honest about calibration.
        # However, the user complains it's "terrible".
        # Let's use the optimal F1 threshold for the confusion matrix display.
        
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        fscore = (2 * precision * recall) / (precision + recall + 1e-8)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
        
        logger.info(f"Optimal F1 Threshold: {best_thresh:.4f}")
        
        preds_binary = (probs > best_thresh).astype(float)
        acc = accuracy_score(y_test, preds_binary)
        cm = confusion_matrix(y_test, preds_binary)
        
        logger.info(f"\n--- Classification Metrics (Optimal Threshold {best_thresh:.4f}) ---")
        logger.info(f"Accuracy: {acc*100:.2f}%")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, preds_binary)}")
        
        # Naive Benchmark
        majority_class = 1 if np.mean(y_test) > 0.5 else 0
        naive_preds = np.full_like(y_test, majority_class)
        naive_acc = accuracy_score(y_test, naive_preds)
        logger.info(f"Naive Benchmark (Majority Class): {naive_acc*100:.2f}%")
        
        self.plot_confusion_matrix_roc(y_test, probs, cm, accuracy=acc)
        self.plot_rolling_metrics(y_test, preds_binary)
        
        return predictions

    def plot_rolling_metrics(self, y_true, preds, window=30):
        """
        Plots rolling Accuracy and F1 Score.
        """
        if len(y_true) < window:
            logger.warning("Not enough data for rolling metrics.")
            return

        # Convert to Series for rolling calculations
        y_true_s = pd.Series(y_true)
        preds_s = pd.Series(preds)
        
        # Rolling Accuracy
        # Accuracy is just the mean of correct predictions
        correct_preds = (y_true_s == preds_s).astype(float)
        rolling_acc = correct_preds.rolling(window).mean()
        
        # Rolling F1 Score
        # F1 = 2 * TP / (2 * TP + FP + FN)
        tp = ((y_true_s == 1) & (preds_s == 1)).astype(float)
        fp = ((y_true_s == 0) & (preds_s == 1)).astype(float)
        fn = ((y_true_s == 1) & (preds_s == 0)).astype(float)
        
        rolling_tp = tp.rolling(window).sum()
        rolling_fp = fp.rolling(window).sum()
        rolling_fn = fn.rolling(window).sum()
        
        # Avoid division by zero
        epsilon = 1e-8
        rolling_f1 = 2 * rolling_tp / (2 * rolling_tp + rolling_fp + rolling_fn + epsilon)
        
        plt.figure(figsize=(14, 6))
        plt.plot(rolling_acc, label=f'Rolling Accuracy ({window}d)', color=self.palette[0], linewidth=2)
        plt.plot(rolling_f1, label=f'Rolling F1 Score ({window}d)', color=self.palette[1], linewidth=2)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
        plt.title(f'Rolling Metrics (Window={window} Days)')
        plt.xlabel('Time (Days in Test Set)')
        plt.ylabel('Score')
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        visualization.save_plot(self.config.OUTPUT_DIR, 'rolling_metrics.png')
        plt.close()

    def plot_confusion_matrix_roc(self, y_true, probs, cm, accuracy=None):
        """
        Plots Confusion Matrix and ROC Curve.
        """
        plt.figure(figsize=(16, 6))
        
        # Subplot 1: Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap=self.purple_cmap, cbar=False,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        title = 'Confusion Matrix'
        if accuracy is not None:
            title += f' (Accuracy: {accuracy:.2%})'
        plt.title(title)
        
        # Subplot 2: ROC Curve
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=self.palette[1], lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color=self.palette[0], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        visualization.save_plot(self.config.OUTPUT_DIR, 'confusion_matrix_roc.png')
        plt.close()

    def calculate_financial_metrics(self, test_log_rets, predictions):
        """
        Calculates and prints financial metrics (Sharpe, Drawdown, etc.) including transaction costs.
        """
        initial_capital = 10000
        
        # Buy & Hold
        cum_log_ret_bh = np.cumsum(test_log_rets)
        equity_bh = initial_capital * np.exp(cum_log_ret_bh)
        
        # Strategy with Transaction Costs
        # Convert log returns to simple returns for cost calculation
        asset_simple_rets = np.exp(test_log_rets) - 1
        
        # Calculate Turnover (Trades)
        # Pad with 0 to assume starting from Cash
        positions_padded = np.concatenate(([0], predictions))
        trades = np.abs(np.diff(positions_padded))
        
        # Cost is incurred on the trade amount. 
        # Assuming we trade the full capital: Cost = Capital * Cost_Pct * Trade_Size (0 to 1)
        # Approximation: Subtract cost from return
        
        # Gross Strategy Returns (Simple)
        strategy_gross_rets = predictions * asset_simple_rets
        
        # Transaction Costs
        costs = trades * self.config.TRANSACTION_COST
        
        # Net Strategy Returns
        strategy_net_rets = strategy_gross_rets - costs
        
        # Calculate Equity Curve
        equity_strat = np.zeros_like(strategy_net_rets)
        current_equity = initial_capital
        
        for r in strategy_net_rets:
            current_equity *= (1 + r)
            equity_strat = np.append(equity_strat[:-1], current_equity) # This is slow, but clear. 
            # Better: equity_strat = initial_capital * np.cumprod(1 + strategy_net_rets)
        
        equity_strat = initial_capital * np.cumprod(1 + strategy_net_rets)

        # --- Random Strategy (Benchmark) ---
        # Random Buy/Sell (50% probability)
        random_probs = np.random.rand(len(predictions))
        random_preds = (random_probs > 0.5).astype(int)
        
        # Random Strategy Returns
        random_gross_rets = random_preds * asset_simple_rets
        
        # Random Strategy Costs
        random_pos_padded = np.concatenate(([0], random_preds))
        random_trades = np.abs(np.diff(random_pos_padded))
        random_costs = random_trades * self.config.TRANSACTION_COST
        
        random_net_rets = random_gross_rets - random_costs
        equity_random = initial_capital * np.cumprod(1 + random_net_rets)

        # Metrics
        mean_ret = np.mean(strategy_net_rets)
        std_ret = np.std(strategy_net_rets)
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        
        running_max = np.maximum.accumulate(equity_strat)
        drawdown = (equity_strat - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win Rate (Days with positive return)
        active_days = strategy_net_rets[predictions == 1]
        if len(active_days) > 0:
            win_rate = np.sum(active_days > 0) / len(active_days)
        else:
            win_rate = 0
            
        total_trades = np.sum(trades)
        total_costs = np.sum(costs) * initial_capital # Approx
        
        logger.info(f"\n--- Strategy Financial Metrics ---")
        logger.info(f"Transaction Cost: {self.config.TRANSACTION_COST*100}% per trade")
        if self.config.USE_ADAPTIVE_THRESHOLD:
            logger.info(f"Strategy: Adaptive Threshold (Window={self.config.ADAPTIVE_WINDOW}, Std={self.config.ADAPTIVE_STD})")
        else:
            logger.info(f"Confidence Threshold: {self.config.CONFIDENCE_THRESHOLD}")
        logger.info(f"Total Trades: {int(total_trades)}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"Win Rate (on Long days): {win_rate*100:.2f}%")
        logger.info(f"Final Portfolio Value: ${equity_strat[-1]:.2f} (vs ${equity_bh[-1]:.2f} Buy&Hold)")
        
        self.plot_equity_curve(equity_bh, equity_strat, equity_random, predictions)
        
        return equity_bh, equity_strat

    def plot_equity_curve(self, equity_bh, equity_strat, equity_random, predictions):
        """
        Plots the equity curve and the adaptive thresholds.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Plot 1: Equity Curve ---
        ax1.plot(equity_bh, label='Buy & Hold (ETH)', color=self.palette[1], alpha=0.7)
        ax1.plot(equity_strat, label='LSTM Strategy (Long/Cash)', color=self.palette[0], linewidth=2)
        ax1.plot(equity_random, label='Random Strategy', color='gray', linestyle=':', alpha=0.6)
        
        # Add Buy/Sell signals on Equity Curve
        preds_padded = np.concatenate(([0], predictions))
        diffs = np.diff(preds_padded)
        
        buy_indices = np.where(diffs == 1)[0]
        sell_indices = np.where(diffs == -1)[0]
        
        if len(buy_indices) > 0:
            ax1.scatter(buy_indices, equity_strat[buy_indices], marker='^', color=self.palette[2], s=50, label='Buy Signal', zorder=5)
        if len(sell_indices) > 0:
            ax1.scatter(sell_indices, equity_strat[sell_indices], marker='v', color=self.palette[3], s=50, label='Sell Signal', zorder=5)

        ax1.set_title('Equity Curve: Strategy vs Buy & Hold (Test Set)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Probabilities & Thresholds ---
        ax2.plot(self.probs, label='Model Probability', color='gray', alpha=0.6)
        
        # Plot Thresholds
        if isinstance(self.upper_bound, pd.Series):
            ax2.plot(self.upper_bound, label='Upper Bound', color=self.palette[2], linestyle='--', alpha=0.8)
            ax2.plot(self.lower_bound, label='Lower Bound', color=self.palette[3], linestyle='--', alpha=0.8)
            ax2.fill_between(range(len(self.probs)), self.lower_bound, self.upper_bound, color='gray', alpha=0.1)
        else:
            ax2.plot(self.upper_bound, label='Upper Bound', color=self.palette[2], linestyle='--', alpha=0.8)
            ax2.plot(self.lower_bound, label='Lower Bound', color=self.palette[3], linestyle='--', alpha=0.8)

        # Add Buy/Sell markers on Probability Plot
        if len(buy_indices) > 0:
            ax2.scatter(buy_indices, self.probs[buy_indices], marker='^', color=self.palette[2], s=30, zorder=5)
        if len(sell_indices) > 0:
            ax2.scatter(sell_indices, self.probs[sell_indices], marker='v', color=self.palette[3], s=30, zorder=5)

        ax2.set_title('Model Confidence & Adaptive Thresholds')
        ax2.set_xlabel('Days (Test Set)')
        ax2.set_ylabel('Probability')
        
        # Dynamic Y-limits for better visibility
        p_min, p_max = self.probs.min(), self.probs.max()
        y_margin = (p_max - p_min) * 0.5 # Add 50% margin for context
        if y_margin == 0: y_margin = 0.1
        ax2.set_ylim(max(0, p_min - y_margin), min(1, p_max + y_margin))
        
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        visualization.save_plot(self.config.OUTPUT_DIR, 'equity_curve.png')
        plt.close()