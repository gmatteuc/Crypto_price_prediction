"""
Crypto Prediction - Evaluation Module
=====================================

This module contains functions for evaluating model performance, backtesting
trading strategies, and calculating financial metrics (Sharpe, Sortino, etc.).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, precision_recall_curve, r2_score
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
            outputs = self.model(X_test)
            
        # Classification Logic
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
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
            # Shift(1) to ensure we use ONLY past data for current threshold (Strictly No Leakage)
            rolling_mean = probs_series.rolling(window=self.config.ADAPTIVE_WINDOW, min_periods=1).mean().shift(1)
            rolling_std = probs_series.rolling(window=self.config.ADAPTIVE_WINDOW, min_periods=1).std().shift(1)
            
            # Handle initial NaNs (First sample has no history)
            rolling_mean = rolling_mean.fillna(0.5) 
            rolling_std = rolling_std.fillna(0.0)
            
            self.upper_bound = rolling_mean + (self.config.ADAPTIVE_STD * rolling_std)
            self.lower_bound = rolling_mean - (self.config.ADAPTIVE_STD * rolling_std)
            
            logger.info(f"Using Adaptive Threshold (Window={self.config.ADAPTIVE_WINDOW}, Std={self.config.ADAPTIVE_STD}) with Shift(1) for robustness")
            
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
        
        # Standard Classification Metrics
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        fscore = (2 * precision * recall) / (precision + recall + 1e-8)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
        
        logger.info(f"Optimal F1 Threshold: {best_thresh:.4f}")
        self.best_thresh = best_thresh # Store for Event Study
        
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
        # self.plot_rolling_metrics(y_test, preds_binary)
        
        return predictions, probs

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
        
        visualization.save_plot(self.config.OUTPUT_DIR, 'rolling_metrics_classification.png')
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
        
        visualization.save_plot(self.config.OUTPUT_DIR, 'confusion_matrix_roc_classification.png')
        plt.close()

    def calculate_financial_metrics(self, test_log_rets, predictions, sma_returns=None):
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
        current_equity_strat = initial_capital # Use separate variable for cumulative product

        # Vectorized cumprod is safer for performance
        # equity_strat = initial_capital * np.cumprod(1 + strategy_net_rets)
        
        # Iterative approach (if we want to handle drawdown-based risk management or stop-loss in future)
        # But for now, vectorized:
        equity_strat = initial_capital * np.cumprod(1 + strategy_net_rets)

        # --- SMA Benchmark Strategy ---
        equity_sma = None
        sma_net_rets = None
        if sma_returns is not None:
            # SMA returns passed from main.py (calculated from Open/Close/SMA)
            # Assuming they are already net returns or gross returns?
            # main.py will calculate gross returns. We should apply costs here if main.py passed gross returns.
            # But main.py likely passes raw returns of the SMA strategy.
            # Let's assume sma_returns are GROSS returns (Position * Asset_Ret)
            
            # We need to calculate SMA Costs?
            # To do that, we need SMA Positions.
            # It's cleaner if main.py passes `sma_net_rets`.
            # Let's assume sma_returns ARE `sma_net_rets` for simplicity, as calculating trades 
            # requires positions which we might not want to pass around.
            # OR we pass `sma_positions` and calculate costs here.
            # Decision: Let's assume main.py passes `sma_net_rets` directly.
            
            sma_net_rets = sma_returns
            equity_sma = initial_capital * np.cumprod(1 + sma_net_rets)
            
            # SMA Metrics
            sma_mean = np.mean(sma_net_rets)
            sma_std = np.std(sma_net_rets)
            sma_sharpe = (sma_mean / sma_std) * np.sqrt(252) if sma_std > 0 else 0
            sma_total_ret = (equity_sma[-1] / initial_capital) - 1

        # --- Random Strategy (Benchmark) ---
        # Match Model Exposure (Time in Market) to ensure fair comparison
        model_exposure = np.mean(predictions)
        logger.info(f"Model Exposure (Time in Market): {model_exposure*100:.1f}%")
        
        # Generate random preds with approximately same probability of being 1
        random_probs = np.random.rand(len(predictions))
        random_preds = (random_probs < model_exposure).astype(int)
        
        # Random Strategy Returns
        random_gross_rets = random_preds * asset_simple_rets
        
        # Random Strategy Costs
        random_pos_padded = np.concatenate(([0], random_preds))
        random_trades = np.abs(np.diff(random_pos_padded))
        random_costs = random_trades * self.config.TRANSACTION_COST
        
        random_net_rets = random_gross_rets - random_costs
        equity_random = initial_capital * np.cumprod(1 + random_net_rets)
        
        # Note: Random Strategy equity curve is one realization, 
        # but strategy comparison bar chart uses the passed random_net_rets which is this one realization.
        # Ideally, bar chart should re-sample random strategy to be robust, but matching exposure here helps.

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
        if equity_sma is not None:
             logger.info(f"SMA Benchmark Value: ${equity_sma[-1]:.2f} (Sharpe: {sma_sharpe:.2f})")
        
        self.plot_equity_curve(equity_bh, equity_strat, equity_random, predictions, equity_sma=equity_sma)
        
        # Plot Strategy Comparison Bar Chart (Random Sampling)
        self.plot_strategy_comparison_bar(asset_simple_rets, strategy_net_rets, random_net_rets, sma_net_rets=sma_net_rets)
        
        # Perform Robust Bootstrap Analysis
        # Increased block_size to 60 to preserve longer-term market regimes (volatility clustering)
        self.perform_bootstrap_analysis(strategy_net_rets, asset_simple_rets, block_size=60, sma_returns=sma_net_rets)
        
        return equity_bh, equity_strat

    def perform_bootstrap_analysis(self, strategy_rets, benchmark_rets, num_samples=1000, block_size=60, sma_returns=None):
        """
        Performs a Block Bootstrap analysis to test the statistical significance of the strategy's performance.
        Generates synthetic equity curves by resampling blocks of returns.
        """
        logger.info(f"\n--- Robust Bootstrap Analysis ({num_samples} samples, block_size={block_size}) ---")
        
        n = len(strategy_rets)
        if n < block_size:
            logger.warning("Not enough data for block bootstrap.")
            return

        strat_sharpes = []
        bench_sharpes = []
        sma_sharpes = [] if sma_returns is not None else None
        
        strat_total_rets = []
        bench_total_rets = []
        sma_total_rets = [] if sma_returns is not None else None
        
        np.random.seed(42)
        possible_starts = np.arange(n - block_size + 1)
        
        for _ in range(num_samples):
            num_blocks_needed = int(np.ceil(n / block_size))
            chosen_starts = np.random.choice(possible_starts, size=num_blocks_needed, replace=True)
            
            synth_strat = []
            synth_bench = []
            synth_sma = [] if sma_returns is not None else None
            
            for start in chosen_starts:
                end = start + block_size
                synth_strat.extend(strategy_rets[start:end])
                synth_bench.extend(benchmark_rets[start:end])
                if sma_returns is not None:
                    synth_sma.extend(sma_returns[start:end])
            
            # Trim
            synth_strat = np.array(synth_strat[:n])
            synth_bench = np.array(synth_bench[:n])
            if sma_returns is not None:
                synth_sma = np.array(synth_sma[:n])
            
            # Metrics
            # --- Strategy ---
            s_mean, s_std = np.mean(synth_strat), np.std(synth_strat)
            strat_sharpes.append((s_mean / s_std * np.sqrt(252)) if s_std > 0 else 0)
            strat_total_rets.append(np.prod(1 + synth_strat) - 1)
            
            # --- Benchmark (B&H) ---
            b_mean, b_std = np.mean(synth_bench), np.std(synth_bench)
            bench_sharpes.append((b_mean / b_std * np.sqrt(252)) if b_std > 0 else 0)
            bench_total_rets.append(np.prod(1 + synth_bench) - 1)
            
            # --- SMA ---
            if sma_returns is not None:
                sma_mean, sma_std = np.mean(synth_sma), np.std(synth_sma)
                sma_sharpes.append((sma_mean / sma_std * np.sqrt(252)) if sma_std > 0 else 0)
                sma_total_rets.append(np.prod(1 + synth_sma) - 1)

        # Convert to arrays
        strat_sharpes = np.array(strat_sharpes)
        bench_sharpes = np.array(bench_sharpes)
        strat_total_rets = np.array(strat_total_rets)
        bench_total_rets = np.array(bench_total_rets)
        
        # Stats vs B&H
        sharpe_diff = strat_sharpes - bench_sharpes
        win_rate = np.mean(sharpe_diff > 0)
        median_return_diff = np.median(strat_total_rets - bench_total_rets) * 100
        
        logger.info(f"Bootstrap Win Rate (Strategy Sharpe > B&H Sharpe): {win_rate*100:.2f}%")
        logger.info(f"Median Return Diff (Strategy - B&H): {median_return_diff:.2f}%")
        
        # Stats vs SMA
        if sma_returns is not None:
            sma_sharpes = np.array(sma_sharpes)
            sma_total_rets = np.array(sma_total_rets)
            sharpe_diff_sma = strat_sharpes - sma_sharpes
            win_rate_sma = np.mean(sharpe_diff_sma > 0)
            median_return_diff_sma = np.median(strat_total_rets - sma_total_rets) * 100
            
            logger.info(f"Bootstrap Win Rate (Strategy Sharpe > SMA Sharpe): {win_rate_sma*100:.2f}%")
            logger.info(f"Median Return Diff (Strategy - SMA): {median_return_diff_sma:.2f}%")

        # Plotting
        # If SMA present, use 2x2 grid. Else 1x2.
        rows = 2 if sma_returns is not None else 1
        plt.figure(figsize=(12, 6 * rows))
        
        # --- Row 1: Strategy vs B&H ---
        plt.subplot(rows, 2, 1)
        sns.histplot(strat_sharpes, color=self.palette[0], label='Strategy', kde=True, alpha=0.5)
        sns.histplot(bench_sharpes, color=self.palette[1], label='Buy & Hold', kde=True, alpha=0.5)
        plt.axvline(np.mean(strat_sharpes), color=self.palette[0], linestyle='--')
        plt.title(f'Sharpe: Strategy vs B&H\n(Win Rate: {win_rate*100:.1f}%)')
        plt.legend()
        
        plt.subplot(rows, 2, 2)
        sns.histplot(strat_total_rets*100, color=self.palette[0], label='Strategy', kde=True, alpha=0.5)
        sns.histplot(bench_total_rets*100, color=self.palette[1], label='Buy & Hold', kde=True, alpha=0.5)
        plt.axvline(np.mean(strat_total_rets)*100, color=self.palette[0], linestyle='--')
        plt.title(f'Return: Strategy vs B&H\n(Median Diff: {median_return_diff:+.1f}%)')
        plt.legend()
        
        # --- Row 2: Strategy vs SMA ---
        if sma_returns is not None:
            plt.subplot(rows, 2, 3)
            # Find common range
            # common_range = (min(min(strat_sharpes), min(sma_sharpes)), max(max(strat_sharpes), max(sma_sharpes)))
            sns.histplot(strat_sharpes, color=self.palette[0], label='LSTM Model', kde=True, alpha=0.5)
            sns.histplot(sma_sharpes, color='#C4A4E0', label='SMA Strategy', kde=True, alpha=0.5)
            plt.axvline(np.mean(strat_sharpes), color=self.palette[0], linestyle='--')
            plt.axvline(np.mean(sma_sharpes), color='#C4A4E0', linestyle='--')
            plt.title(f'Sharpe: LSTM vs SMA\n(Win Rate: {win_rate_sma*100:.1f}%)')
            plt.xlabel('Sharpe Ratio')
            plt.legend()

            plt.subplot(rows, 2, 4)
            sns.histplot(strat_total_rets*100, color=self.palette[0], label='LSTM Model', kde=True, alpha=0.5)
            sns.histplot(sma_total_rets*100, color='#C4A4E0', label='SMA Strategy', kde=True, alpha=0.5)
            plt.axvline(np.mean(strat_total_rets)*100, color=self.palette[0], linestyle='--')
            plt.axvline(np.mean(sma_total_rets)*100, color='#C4A4E0', linestyle='--')
            plt.title(f'Return: LSTM vs SMA\n(Median Diff: {median_return_diff_sma:+.1f}%)')
            plt.xlabel('Total Return (%)')
            plt.legend()

        plt.tight_layout()
        visualization.save_plot(self.config.OUTPUT_DIR, 'bootstrap_analysis_classification.png')
        plt.close()

    def plot_strategy_comparison_bar(self, asset_rets, strategy_rets, random_rets, sma_net_rets=None, window_days=30, num_samples=200):
        """
        Plots a bar chart comparing average returns over a fixed window (e.g., 1 month)
        by sampling random starting points from the test set.
        """
        valid_length = len(asset_rets) - window_days
        if valid_length <= 0:
            logger.warning("Not enough data for strategy comparison bar chart.")
            return

        # Lists to store cumulative returns for each sample
        bh_samples = []
        strat_samples = []
        rand_samples = []
        sma_samples = [] if sma_net_rets is not None else None

        # Set seed for reproducibility of the sampling
        np.random.seed(42)
        
        # Generate random starting indices
        start_indices = np.random.randint(0, valid_length, size=num_samples)

        for start_idx in start_indices:
            end_idx = start_idx + window_days
            
            # Calculate cumulative return for this window: (1+r1)*(1+r2)*... - 1
            # Buy & Hold
            bh_ret = np.prod(1 + asset_rets[start_idx:end_idx]) - 1
            bh_samples.append(bh_ret)
            
            # Strategy
            strat_ret = np.prod(1 + strategy_rets[start_idx:end_idx]) - 1
            strat_samples.append(strat_ret)
            
            # Random Strategy
            rand_ret = np.prod(1 + random_rets[start_idx:end_idx]) - 1
            rand_samples.append(rand_ret)
            
            # SMA Strategy
            if sma_samples is not None:
                sma_ret = np.prod(1 + sma_net_rets[start_idx:end_idx]) - 1
                sma_samples.append(sma_ret)

        # Calculate Means and SEMs
        means = [np.mean(strat_samples), np.mean(bh_samples), np.mean(rand_samples)]
        sems = [np.std(strat_samples) / np.sqrt(num_samples), 
                np.std(bh_samples) / np.sqrt(num_samples), 
                np.std(rand_samples) / np.sqrt(num_samples)]
                
        model_name = self.config.MODEL_ARCH.upper()
        labels = [f'{model_name} Strategy', 'Buy & Hold', 'Random Strategy']
        colors = [self.palette[0], self.palette[1], 'gray']
        
        if sma_samples is not None:
            means.insert(1, np.mean(sma_samples))
            sems.insert(1, np.std(sma_samples) / np.sqrt(num_samples))
            labels.insert(1, 'SMA Strategy')
            colors.insert(1, '#C4A4E0')

        # Plot
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(labels))
        
        # Convert to percentage for display
        means_pct = np.array(means) * 100
        sems_pct = np.array(sems) * 100
        
        # Create bars with error bars (Standard Error)
        # ecolor='gray' makes error bars gray
        bars = plt.bar(x_pos, means_pct, yerr=sems_pct, align='center', alpha=0.7, color=colors, capsize=10, ecolor='gray')
        
        # Add value labels on top of bars
        for bar, mean_val, sem_val in zip(bars, means_pct, sems_pct):
            height = bar.get_height()
            # Position text above positive bars and below negative bars, accounting for error bar
            if height >= 0:
                y_pos = height + sem_val + (abs(mean_val) * 0.05) # dynamic offset
                va = 'bottom'
            else:
                y_pos = height - sem_val - (abs(mean_val) * 0.05)
                va = 'top'
                
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{mean_val:.2f}%',
                     ha='center', va=va, color='gray', fontweight='bold')

        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.xticks(x_pos, labels)
        plt.ylabel(f'Average Return (%) over {window_days} Days')
        plt.title(f'Strategy Performance Comparison (Avg over {window_days}-Day Windows)\nSampled {num_samples} times (Error Bars = Standard Error)')
        
        # Symmetrize Y-axis
        # Find the maximum absolute extent including error bars
        max_extent = np.max(np.abs(means_pct) + sems_pct)
        # Add some padding (e.g. 20%)
        limit = max_extent * 1.2
        # Ensure limit is not zero
        if limit == 0: limit = 1.0
        plt.ylim(-limit, limit)
        
        plt.grid(axis='y', alpha=0.3)
        
        visualization.save_plot(self.config.OUTPUT_DIR, 'strategy_comparison_bar_classification.png')
        plt.close()

    def plot_equity_curve(self, equity_bh, equity_strat, equity_random, predictions, equity_sma=None):
        """
        Plots the equity curve and the adaptive thresholds.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Plot 1: Equity Curve ---
        model_name = self.config.MODEL_ARCH.upper() if hasattr(self.config, 'MODEL_ARCH') else 'LSTM'
        ax1.plot(equity_bh, label=f'Buy & Hold (ETH) ({(equity_bh[-1]/equity_bh[0]-1)*100:.1f}%)', color=self.palette[1], alpha=0.7)
        ax1.plot(equity_strat, label=f'{model_name} Strategy (Long/Cash) ({(equity_strat[-1]/equity_strat[0]-1)*100:.1f}%)', color=self.palette[0], linewidth=2)
        ax1.plot(equity_random, label=f'Random Strategy ({(equity_random[-1]/equity_random[0]-1)*100:.1f}%)', color='gray', linestyle=':', alpha=0.6)
        
        if equity_sma is not None:
             ax1.plot(equity_sma, label=f'SMA Strategy ({(equity_sma[-1]/equity_sma[0]-1)*100:.1f}%)', color='#C4A4E0', linewidth=2, linestyle='-')
        
        # Add Buy/Sell signals on Equity Curve
        preds_padded = np.concatenate(([0], predictions))
        diffs = np.diff(preds_padded)
        
        buy_indices = np.where(diffs == 1)[0]
        sell_indices = np.where(diffs == -1)[0]
        
        if len(buy_indices) > 0:
            ax1.scatter(buy_indices, equity_strat[buy_indices], marker='^', color=self.palette[2], s=50, label='Buy Signal', zorder=5)
        if len(sell_indices) > 0:
            ax1.scatter(sell_indices, equity_strat[sell_indices], marker='v', color=self.palette[3], s=50, label='Sell Signal', zorder=5)

        # Calculate total returns for title
        ret_strat = (equity_strat[-1] - equity_strat[0]) / equity_strat[0] * 100
        ret_bh = (equity_bh[-1] - equity_bh[0]) / equity_bh[0] * 100
        ret_rand = (equity_random[-1] - equity_random[0]) / equity_random[0] * 100

        ax1.set_title(f'Equity Curve: Strategy ({ret_strat:.1f}%) vs Buy & Hold ({ret_bh:.1f}%) vs Random ({ret_rand:.1f}%)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Probabilities & Thresholds ---
        ax2.plot(self.probs, label='Model Probability', color='gray', alpha=0.6)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='0.5 Threshold')
        
        # Plot Thresholds
        if isinstance(self.upper_bound, pd.Series):
            ax2.plot(self.upper_bound, label='Upper Bound', color=self.palette[2], linestyle='--', alpha=0.8)
            ax2.plot(self.lower_bound, label='Lower Bound', color=self.palette[3], linestyle='--', alpha=0.8)
            ax2.fill_between(range(len(self.probs)), self.lower_bound, self.upper_bound, color='gray', alpha=0.1)
        else:
            ax2.plot(self.upper_bound, label='Upper Bound', color=self.palette[2], linestyle='--', alpha=0.8)
            ax2.plot(self.lower_bound, label='Lower Bound', color=self.palette[3], linestyle='--', alpha=0.8)
        
        ax2.set_ylabel('Probability')
        ax2.set_title('Model Confidence & Adaptive Thresholds')

        # Add Buy/Sell markers on Probability Plot
        if len(buy_indices) > 0:
            ax2.scatter(buy_indices, self.probs[buy_indices], marker='^', color=self.palette[2], s=30, zorder=5)
        if len(sell_indices) > 0:
            ax2.scatter(sell_indices, self.probs[sell_indices], marker='v', color=self.palette[3], s=30, zorder=5)

        ax2.set_xlabel('Days (Test Set)')
        
        # Dynamic Y-limits for better visibility
        p_min, p_max = self.probs.min(), self.probs.max()
        y_margin = (p_max - p_min) * 0.5 # Add 50% margin for context
        if y_margin == 0: y_margin = 0.1
        
        ax2.set_ylim(max(0, p_min - y_margin), min(1, p_max + y_margin))
        
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        visualization.save_plot(self.config.OUTPUT_DIR, 'equity_curve_classification.png')
        plt.close()

    def perform_event_study(self, returns: np.ndarray, predictions: np.ndarray, raw_predictions: np.ndarray = None, window_back: int = 3, window_fwd: int = 5):
        """
        Performs an Event Study Analysis to verify if model predictions lead to actual price moves.
        Plots the average cumulative return path for 'Up' predictions vs 'Down' predictions.
        
        Args:
            returns (np.ndarray): Actual log returns.
            predictions (np.ndarray): Predicted positions/signals.
            raw_predictions (np.ndarray, optional): Raw predicted returns/probs.
            window_back (int): Days to look back before signal.
            window_fwd (int): Days to look forward after signal.
        """
        logger.info(f"\n--- Event Study Analysis (Back={window_back}, Fwd={window_fwd}) ---")
        
        # Identify Events
        # For regression: Up = Pred > 0 (or Threshold), Down = Pred < 0
        # For classification: Up = Prob > 0.5, Down = Prob < 0.5
        
        up_events = []
        down_events = []
        
        # We need enough data before and after
        start_idx = window_back
        end_idx = len(returns) - window_fwd
        
        if start_idx >= end_idx:
            logger.warning("Not enough data for Event Study.")
            return

        # Determine Thresholds (Use Adaptive if available, else 0)
        # For simplicity in this specific analysis, we use the raw sign of the prediction
        # to test the fundamental "directionality" of the model, independent of the threshold strategy.
        # This answers: "When the model thinks it's going up, does it go up?"
        
        for i in range(start_idx, end_idx):
            pred = predictions[i]
            
            # Extract window of ACTUAL returns
            # returns[i] is the return at t+1 (tomorrow).
            # We want the path centered at signal time t.
            # Signal at t predicts returns[i].
            # So t=0 in the plot corresponds to the return realized at t+1?
            # Standard Event Study: t=0 is the event day.
            # Here event is "Prediction Made".
            # Path:
            # t-2: returns[i-2]
            # t-1: returns[i-1]
            # t+1: returns[i]   (The return we are predicting)
            # t+2: returns[i+1]
            
            # Slice: from i - window_back to i + window_fwd
            # Length: window_back + window_fwd
            
            # Note: returns array is already aligned such that returns[i] is the target for predictions[i]
            # So predictions[i] says "returns[i] will be X".
            
            window_returns = returns[i - window_back : i + window_fwd]
            
            # Calculate Cumulative Return Path relative to t=0 (Signal Time)
            # We want the price at "Signal Time" to be 0%.
            # The return at 'i' is the return occurring AFTER the signal.
            # So the cumulative sum should start adding from index 'window_back'.
            
            cum_path = np.cumsum(window_returns)
            # Normalize so that the value BEFORE the predicted return (t=0) is 0.
            # The predicted return is at index `window_back`.
            # So we subtract the cumulative sum up to `window_back - 1`.
            
            # Let's do it simpler: Reconstruct Price Path
            price_path = np.exp(np.cumsum(window_returns))
            # Normalize: Price at signal time (just before return[i]) should be 1.0
            # return[i] is the change from Price[t] to Price[t+1].
            # So Price[t] is the baseline.
            # The cumulative sum includes return[i-back]...
            # Let's say window_back=2. Indices: i-2, i-1, i, i+1, i+2.
            # i-2: t-2 to t-1
            # i-1: t-1 to t
            # i  : t   to t+1 (Target)
            
            # We want Price[t] = 1.0.
            # Price[t-1] = Price[t] / exp(ret[i-1])
            # Price[t+1] = Price[t] * exp(ret[i])
            
            # Construct path relative to index `window_back` (which is 'i')
            # 0: i-2
            # 1: i-1
            # 2: i (Target)
            
            # Base = 0.0
            path = np.zeros(len(window_returns) + 1)
            path[window_back] = 0.0 # t=0
            
            # Backward fill
            current = 0.0
            for k in range(window_back - 1, -1, -1):
                current -= window_returns[k]
                path[k] = current
                
            # Forward fill
            current = 0.0
            for k in range(window_back, len(window_returns)):
                current += window_returns[k]
                path[k+1] = current
                
            # Remove the last point if it exceeds our window logic or just keep it.
            # path has length window_back + window_fwd + 1
            
            if pred > 0.5:
                up_events.append(path)
            else:
                down_events.append(path)


                    
        # Convert to arrays
        up_events = np.array(up_events)
        down_events = np.array(down_events)
        
        logger.info(f"Up Events: {len(up_events)}, Down Events: {len(down_events)}")
        
        # --- Reverse Event Study: Model Anticipation ---
        # Analyze: When the Market moves Big, did the Model predict it?
        pred_given_up = []
        pred_given_down = []
        std_ret = np.std(returns)
        
        # Use raw predictions if available (for better scale), else use signals
        study_preds = raw_predictions if raw_predictions is not None else predictions
        
        # Determine classification threshold
        cls_thresh = getattr(self, 'best_thresh', 0.5)

        for i in range(start_idx, end_idx):
            act_ret = returns[i]
            # Window check
            if i - window_back < 0 or i + window_fwd >= len(study_preds):
                continue
                
            # For this study, we store the PREDICTIONS path, not the returns path
            pred_window = study_preds[i - window_back : i + window_fwd + 1]
            if len(pred_window) != window_back + window_fwd + 1:
                continue
            
            # For Classification: Use probabilities directly (converted to %)
            # This allows us to calculate MEAN prediction (Prop of UP) which visualizes conviction.
            if self.config.MODEL_TYPE == 'classification' and raw_predictions is not None:
                # Do NOT binarize. Show average Probability of UP.
                # Center around 50%? Or just show raw %. 
                # Raw % is 0 to 100.
                pred_window = pred_window * 100.0 

            if act_ret > std_ret: # Real UP Move
                 pred_given_up.append(pred_window)
            elif act_ret < -std_ret: # Real DOWN Move
                 pred_given_down.append(pred_window)

        pred_given_up = np.array(pred_given_up)
        pred_given_down = np.array(pred_given_down)
        # --- Subplot 3: Signal Logic (Z-Score of Prediction) ---
        # Analyze: How many Standard Deviations is the prediction away from the Rolling Mean?
        # This is the actual "Signal" used by the Adaptive Strategy.
        
        signal_given_up = []
        signal_given_down = []
        
        # Calculate full series signal first
        if self.config.USE_ADAPTIVE_THRESHOLD:
            preds_series = pd.Series(study_preds)
            rolling_mean = preds_series.rolling(window=self.config.ADAPTIVE_WINDOW, min_periods=1).mean()
            rolling_std = preds_series.rolling(window=self.config.ADAPTIVE_WINDOW, min_periods=1).std().fillna(1e-6)
            
            # This is the "Conviction" index
            # If > 1.0 -> Buy Signal. If < -1.0 -> Sell Signal.
            z_scores = (preds_series - rolling_mean.shift(1)) / (rolling_std.shift(1) + 1e-9)
            # Clip outliers to prevent visualization artifacts
            z_scores = z_scores.fillna(0).clip(-5, 5).values
        else:
            # If no adaptive, signal is just Pred - 0.5 (scaled?) 
            # Not really defined, but let's mock it
            z_scores = (study_preds - 0.5) * 10 
        
        for i in range(start_idx, end_idx):
            act_ret = returns[i]
            if i - window_back < 0 or i + window_fwd >= len(z_scores):
                continue
            
            sig_window = z_scores[i - window_back : i + window_fwd + 1]
            if len(sig_window) != window_back + window_fwd + 1:
                continue
                
            if act_ret > std_ret:
                signal_given_up.append(sig_window)
            elif act_ret < -std_ret:
                signal_given_down.append(sig_window)
        
        signal_given_up = np.array(signal_given_up)
        signal_given_down = np.array(signal_given_down)
        
        # Plotting (3 Subplots)
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        x_axis = np.arange(-window_back, window_fwd + 1)
        
        # --- Subplot 1: Market Response to Model (Forward) ---
        ax1 = axes[0]
        # Plot Up Events
        if len(up_events) > 0:
            mean_up = np.mean(up_events, axis=0) * 100 # %
            std_up = np.std(up_events, axis=0) * 100
            sem_up = std_up / np.sqrt(len(up_events)) # Standard Error
            
            ax1.plot(x_axis, mean_up, color=self.palette[0], linewidth=3, label=f'Avg Market Path when Model Predicts UP (N={len(up_events)})')
            ax1.fill_between(x_axis, mean_up - sem_up, mean_up + sem_up, color=self.palette[0], alpha=0.2)
            
        # Plot Down Events
        if len(down_events) > 0:
            mean_down = np.mean(down_events, axis=0) * 100 # %
            std_down = np.std(down_events, axis=0) * 100
            sem_down = std_down / np.sqrt(len(down_events))
            
            ax1.plot(x_axis, mean_down, color=self.palette[3], linewidth=3, label=f'Avg Market Path when Model Predicts DOWN (N={len(down_events)})')
            ax1.fill_between(x_axis, mean_down - sem_down, mean_down + sem_down, color=self.palette[3], alpha=0.2)
            
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(0, color='gray', linestyle=':', alpha=0.8, label='Signal Time')
        ax1.set_title(f'Study 1: Market Response to Signals\n(Price Path relative to Signal Day t=0)')
        ax1.set_ylabel('Price Change vs Signal Day (%)')
        ax1.legend(fontsize='small', loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # --- Subplot 2: Model Anticipation of Market (Reverse) ---
        ax2 = axes[1]
        
        # Plot Preds for Real UP
        if len(pred_given_up) > 0:
            # If classification, pred_window is already %, so don't multiply by 100 again.
            multiplier = 1.0 
            
            mean_p_up = np.mean(pred_given_up, axis=0) * multiplier
            std_p_up = np.std(pred_given_up, axis=0) * multiplier
            sem_p_up = std_p_up / np.sqrt(len(pred_given_up))
            
            ax2.plot(x_axis, mean_p_up, color=self.palette[0], linewidth=3, linestyle='-', label=f'Avg Prediction when Market Actually Jumps UP (N={len(pred_given_up)})')
            ax2.fill_between(x_axis, mean_p_up - sem_p_up, mean_p_up + sem_p_up, color=self.palette[0], alpha=0.1)

        # Plot Preds for Real DOWN
        if len(pred_given_down) > 0:
            multiplier = 1.0
            
            mean_p_down = np.mean(pred_given_down, axis=0) * multiplier
            std_p_down = np.std(pred_given_down, axis=0) * multiplier
            sem_p_down = std_p_down / np.sqrt(len(pred_given_down))
            
            ax2.plot(x_axis, mean_p_down, color=self.palette[1], linewidth=3, linestyle='-', label=f'Avg Prediction when Market Actually Crashes (N={len(pred_given_down)})')
            ax2.fill_between(x_axis, mean_p_down - sem_p_down, mean_p_down + sem_p_down, color=self.palette[1], alpha=0.1)

        ax2.set_ylabel('Avg Pred Probability of UP (%)')
        ax2.axhline(50, color='gray', linestyle='--', alpha=0.8, label='Neutral (50%)')

        ax2.axvline(0, color='gray', linestyle=':', alpha=0.8)
        ax2.set_title(f'Study 2: Model Anticipation of Significant Moves\n(Does the model predict consistently BEFORE the event?)')
        ax2.set_xlabel('Days Relative to Event (0 = Day of Big Move)')
        ax2.legend(fontsize='small', loc='lower left', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # --- Subplot 3: Adaptive Signal Z-Score ---
        ax3 = axes[2]
        
        if len(signal_given_up) > 0:
            mean_s_up = np.mean(signal_given_up, axis=0)
            std_s_up = np.std(signal_given_up, axis=0)
            sem_s_up = std_s_up / np.sqrt(len(signal_given_up))
            
            # Use Palette[0] (Deep Purple) to match Subplot 2 "Real UP"
            ax3.plot(x_axis, mean_s_up, color=self.palette[0], linewidth=3, linestyle='-', label='Avg Signal Z-Score (Real UP)')
            ax3.fill_between(x_axis, mean_s_up - sem_s_up, mean_s_up + sem_s_up, color=self.palette[0], alpha=0.2)
            
        if len(signal_given_down) > 0:
            mean_s_down = np.mean(signal_given_down, axis=0)
            std_s_down = np.std(signal_given_down, axis=0)
            sem_s_down = std_s_down / np.sqrt(len(signal_given_down))
            
            # Use Palette[1] (Dark Orange) to match Subplot 2 "Real DOWN"
            ax3.plot(x_axis, mean_s_down, color=self.palette[1], linewidth=3, linestyle='-', label='Avg Signal Z-Score (Real DOWN)')
            ax3.fill_between(x_axis, mean_s_down - sem_s_down, mean_s_down + sem_s_down, color=self.palette[1], alpha=0.2)
            
        ax3.axhline(self.config.ADAPTIVE_STD, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
        ax3.axhline(-self.config.ADAPTIVE_STD, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
        ax3.axhline(0, color='brown', linestyle='--', alpha=0.5, label='Neutral')
        ax3.axvline(0, color='gray', linestyle=':', alpha=0.8)
        
        ax3.set_title(f'Study 3: Adaptive Signal Strength (Z-Score)\n(Normalized Distance from Rolling Mean)')
        ax3.set_ylabel('Signal Strength (Std Devs)')
        ax3.set_xlabel('Days Relative to Event (0 = Day of Big Move)')
        # Fix scale to prevent artifacts from distorting the view
        ax3.set_ylim(-3, 3) 
        ax3.legend(fontsize='small', loc='upper right', framealpha=0.9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        visualization.save_plot(self.config.OUTPUT_DIR, f'event_study_classification.png')
        plt.close()
        
        # Statistical Test (T-test on t+1 return)
        if len(up_events) > 0 and len(down_events) > 0:
            # t+1 return is at index window_back + 1 in the path array
            # path[window_back] is 0. path[window_back+1] is the return of the target day.
            idx_target = window_back + 1
            
            ret_up = up_events[:, idx_target]
            ret_down = down_events[:, idx_target]
            
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(ret_up, ret_down, equal_var=False)
            
            logger.info(f"Statistical Significance (t+1 Return):")
            logger.info(f"Avg Return (UP Signal): {np.mean(ret_up)*100:.4f}%")
            logger.info(f"Avg Return (DOWN Signal): {np.mean(ret_down)*100:.4f}%")
            logger.info(f"Difference: {(np.mean(ret_up) - np.mean(ret_down))*100:.4f}%")
            logger.info(f"T-Statistic: {t_stat:.4f}")
            logger.info(f"P-Value: {p_val:.6f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")