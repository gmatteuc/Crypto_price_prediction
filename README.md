# Crypto Price Prediction Playground

<img src="misc/dataset-cover.png" width="800">

## Project Overview

This project aimed to build a predictive model for cryptocurrency markets (Ethereum) using advanced machine learning techniques (LSTM, Random Forest) and rigorous backtesting. 

**The Conclusion:** While Neural Networks are powerful, our extensive testing reveals that for this specific timeframe and feature set, **they offer no significant "Alpha" over simple Trend Following strategies.**

Instead of forcing a "black box" to work, this project serves as a **transparent, rigorous study** demonstrating that market complexity does not always require model complexity.

| Architecture | Strategy Return | Sharpe Ratio | Win Rate vs SMA* |
| :--- | :--- | :--- | :--- |
| **LSTM (Deep Learning)** | +47.8% | 1.10 | 30.6% |
| **Simple SMA (Benchmark)** | **+76.2%** | **1.25** | - |
| **Buy & Hold** | +4.0% | 0.08 | - |

*> "Win Rate vs SMA" indicates how often the ML Model beat the Simple Moving Average in 1,000 bootstrap simulations.*

*> Note: The results presented above were generated on **January 6, 2026**. Since the pipeline continually fetches the latest data from Yahoo Finance, future executions may yield different numerical results.*

---

## Methodology & Rigor

The value of this project lies in the **integrity of the pipeline**, designed to prevent common pitfalls in financial ML:

1.  **Strict Leakage Prevention**: 
    *   StandardScaler fitted *only* on training data.
    *   Lagged technical indicators to ensure no look-ahead.
    *   Adaptive Thresholds calculated using only past information.
2.  **Robust Evaluation**:
    *   **Block Bootstrap Analysis**: Re-sampling data blocks to preserve volatility correlation while testing statistical significance.
    *   **Transaction Costs**: All backtests include a realistic 0.1% fee per trade.
    *   **Strategic Benchmarks**: Comparing not just to "Buy & Hold", but to a "Simple Moving Average" strategy to isolate the ML contribution.

---

## The "Deep Learning Trap"

We tested three increasing levels of model complexity:

1.  **Linear Models**: Logistic Regression.
2.  **Ensemble Models**: Random Forest.
3.  **Deep Learning**: LSTM.

### Findings
All three models converged to similar behavior. They essentially learned to be **"Momentum Filters"**.
*   The models learned that "if price went up recently, it might keep going up".
*   This imitation of momentum is indistinguishable from a simple Moving Average.
*   The "Adaptive Threshold" we implemented to filter weak signals acted exactly like a volatility filter on a Moving Average.

<p align="center">
  <img src="misc/equity_curve_classification_example.png" width="800" alt="Equity Curve Comparison">
  <br>
  <em>Figure 1: The ML Model (Dark Purple) tracks the SMA Benchmark (Light Purple) almost perfectly, but with more noise and lower total return due to over-trading.</em>
</p>

---
## 🔍 Event Study: Searching for Latent Alpha

Despite the fact that the **Validation Loss** curve showed signs of overfitting (flatlining or increasing while training loss decreased), the model predictions were extremely biased towards predicting UP rather than DOWN (despite attempts to balance), and the accuracy hovered very close to 50% (apprent predictive failure), we investigated whether the model captured *any* true predictive signal.

We conducted an **Event Study Analysis** to visualize the market's behavior around the model's signals:

<p align="center">
  <img src="misc/event_study_classification_example.png" width="800" alt="Event Study Analysis">
  <br>
  <em>Figure 2: Event Study Analysis verifying signal quality.</em>
</p>

*   **Study 1 (Market Response)**: When the model predicts "UP" (Purple line), the average price path *does* trend positively over the next 5 days. However, the wide standard deviation (shaded areas) indicate high variance.
*   **Study 2 (Anticipation)**: We checked if the model predicts consistently *before* large market moves. There is a slight increase in probability leading up to big jumps, suggesting some sensitivity to volatility clustering.
*   **Conclusion**: There is a "faint" predictive signal, but it is likely detecting the onset of volatility trends rather than "predicting" price direction, confirming the Momentum Filter hypothesis.

---
## Statistical Validation

We utilized a **Bootstrap Analysis** to verify if the model's performance was due to skill or luck.

<p align="center">
  <img src="misc/strategy_comparison_bar_classification_example.png" width="800" alt="Bootstrap Analysis">
  <br>
  <em>Figure 3: In rigorous random resampling 30-day sampling windows, the ML Strategy (Purple) performs identically to the Simple Momentum Strategy (Pink), while both massively outperform Buy & Hold (Orange) and Random Trading (Grey).</em>
</p>

**Interpretation:**
*   The ML model is **robust**: It consistently beats random trading and Buy & Hold.
*   The ML model is **redundant**: It fails to consistently beat the simpler SMA strategy. 

## Project Structure

This codebase provides a professional framework for anyone wishing to test their own signals:

```
├── misc/                       # Images and assets
├── output/                     # Generated plots, models, and logs
├── config.py                   # Centralized configuration for reproducible experiments
├── data_loader.py              # Data fetching module
├── evaluator.py                # Advanced plotting (Equity Curves, Event Studies, Bootstrap)
├── main.py                     # Orchestrator for Training, Optimization, and Evaluation
├── model.py                    # Implementations of LSTM, Linear, and RF wrappers
├── predict.py                  # Inference script for daily prediction
├── preprocessor.py             # Feature engineering pipeline
├── strategy_optimizer.py       # Methods to find optimal trading thresholds
├── trainer.py                  # Model training loop
└── requirements.txt            # Project dependencies
```

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Experiment**:
    Edit `config.py` to switch between models:
    ```python
    MODEL_ARCH = 'lstm'   # or 'rf', 'linear'
    ```
3.  **Run Pipeline**:
    ```bash
    python main.py
    ```

## Final Thoughts

Predicting cryptocurrency price movements based solely on historical price and volume data is notoriously difficult. This project demonstrates that in efficient or trend-driven markets, simpler approaches often outperform complex ones.

Our analysis revealed that a basic **Momentum strategy (SMA) outperformed sophisticated Machine Learning models** (Linear, Random Forest, LSTM) on this dataset. This finding underscores a critical lesson in quantitative analysis: **complexity does not automatically yield better predictive and financial results**.

Most importantly, this project highlights the absolute necessity of **rigorous statistical validation** and strict **measures against data leakage**. Without safeguards like the Bootstrap analysis and separate scaling used here, it is dangerously easy to be misled by backtests that look promising but fail to generalize.

From a predictive standpoint, the results reported here represent a **negative result**, demonstrating **market efficiency** in this context. While the system can still be engineered to financially outperform a passive Buy & Hold strategy (similar to the SMA benchmark), the core finding stands: price history alone did not yield a significant predictive power. However, in professional data science, **a rigorous, validated negative result is far more valuable than an unreliable positive one**.

