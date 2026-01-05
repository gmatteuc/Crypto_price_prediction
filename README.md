# Crypto Prediction with LSTM

<img src="misc/dataset-cover.png" width="800">

## Project Overview
This project implements a professional-grade **Quantitative Trading Strategy** for Ethereum (ETH), utilizing a **GRU (Gated Recurrent Unit) with Attention Mechanism**.

Unlike typical retail "price prediction" bots that attempt to forecast exact prices (often resulting in overfitting), this project focuses on **Regime Identification** and **Risk Management**. The core hypothesis is that while exact price prediction in crypto is nearly impossible due to high noise, identifying **periods of market weakness** (negative skew) is statistically feasible.

As a Financial Data Science portfolio project, it showcases a complete "Quant Ops" workflow: from data ingestion and feature engineering to building a custom Deep Learning model and implementing a rigorous **Statistical Validation** framework.

The goal is to address two critical challenges in Algo-Trading:
1.  **Can we extract a statistically significant signal from noisy financial data without overfitting?**
2.  **Can we build a "Defensive Alpha" strategy that preserves capital during market crashes?**

**Financial use case scenario:**
*Imagine an automated trading system that needs to navigate a highly volatile crypto market. The goal is not to trade every single day, but to identify high-probability opportunities and, crucially, to sit in cash when the market direction is uncertain or bearish. This tool acts as a risk-managed portfolio allocator.*

Created by Giulio Matteucci in 2026 as a Financial Data Science portfolio project.

## Dataset
The data is fetched dynamically using the `yfinance` API.
- **Source**: Yahoo Finance.
- **Asset**: Ethereum (ETH-USD).
- **Period**: Daily data from 2018 to present.
- **Features**:
  - **Market Data**: OHLCV (Open, High, Low, Close, Volume).
  - **Macro Indicators**: S&P 500 (`^GSPC`), 10-Year Treasury Yield (`^TNX`), VIX (`^VIX`), DXY (`DX-Y.NYB`).
  - **Technical Factors**: Garman-Klass Volatility, RSI, MACD, Bollinger Bands.

## Methodology

### 1. Data Engineering & Preprocessing
- **Feature Engineering**: Calculation of advanced technical indicators and integration of macro-economic data to provide market context.
- **Scaling**: Robust scaling to handle outliers common in crypto data.
- **Sequence Generation**: Creation of rolling time-window sequences (14 days) for GRU input.

### 2. Model Architecture
- **Core**: 2-layer **GRU (Gated Recurrent Unit)** to capture temporal dependencies with lower computational cost than LSTM.
- **Attention Mechanism**: A custom Attention layer allows the model to weigh specific past days more heavily, improving interpretability and signal extraction.
- **Loss Function**: **Directional Huber Loss**.
  $$Loss = Huber(y, \hat{y}) + \lambda \cdot |y - \hat{y}| \cdot \mathbb{1}_{sign(y) \neq sign(\hat{y})}$$
  This custom loss penalizes "wrong direction" errors more severely than magnitude errors, aligning the optimization objective with the trading objective.

### 3. Signal Generation (Regime Filtering)
Instead of acting on raw predictions, the strategy uses an **Adaptive Threshold** mechanism:
- **Buy Signal**: Prediction > $RollingMean + \sigma$ (Strong Momentum)
- **Sell Signal**: Prediction < $RollingMean - \sigma$ (Weak Momentum/Downside Risk)
- **Hold/Neutral**: Prediction within bounds (Noise)

This filters out low-confidence predictions, trading only when the signal-to-noise ratio is favorable.

### 4. Statistical Validation (The "Why it works" section)
To prove the model isn't just overfitting noise, we employ rigorous statistical tests:
- **Event Study**: We analyze the average cumulative return of the asset $N$ days after a signal.
    - *Result*: "Sell" signals are followed by negative average returns, confirming the model successfully identifies danger zones.
- **Bootstrap Analysis**: We simulate 1,000 alternative realities (block bootstrapping) to compare the strategy's Sharpe Ratio against a random distribution.
    - *Result*: The strategy outperforms the benchmark in >75% of bootstrap samples.

## Key Findings
- **Asymmetric Performance**: The model excels at **Capital Preservation**. By moving to cash during "Down" predictions, it avoids the steep drawdowns characteristic of Crypto winters.
- **Low Correlation to Price**: The model's raw predictions may look like a "flat line" (low variance), but this is a feature, not a bug. It represents a conservative estimate of the *drift*, and the **Adaptive Threshold** successfully extracts the actionable signal from this conservative baseline.
- **Real-World Viability**: Even after accounting for transaction costs (0.1% per trade), the strategy demonstrates a positive expectancy and a superior risk-adjusted return (Sharpe Ratio) compared to a passive Buy & Hold approach.

<p align="center">
  <img src="misc/equity_curve_regression_example.png" width="80%">
</p>
<p align="center">
  <img src="misc/event_study_regression_example.png" width="80%">
</p>

*Figure: (Top) Equity curve showing the "Defensive" nature of the strategy (flat during crashes). (Bottom) Event Study showing the clear divergence in future returns following Buy vs. Sell signals.*

## 💻 Project Structure
```
├── misc/                                   # Images and assets
├── output/                                 # Generated plots and logs
├── check_gpu.py                            # Utility to verify CUDA
├── config.py                               # Hyperparameters and settings
├── data_loader.py                          # Data fetching (yfinance)
├── evaluator.py                            # Backtesting, Event Studies, Bootstrap
├── main.py                                 # Main execution entry point
├── predict.py                              # Daily inference script
├── model.py                                # PyTorch GRU + Attention definition
├── preprocessor.py                         # Feature engineering
├── trainer.py                              # Training loop with DirectionalHuberLoss
├── utils.py                                # Logging and seeding utilities
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Crypto_prediction
   ```

2. **Set up the environment**:
   It is recommended to use Conda.
   ```bash
   conda create -n crypto_prediction python=3.10 -y
   conda activate crypto_prediction
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   ```bash
   python main.py
   ```

## Dependencies
- **Python 3.10+**
- **PyTorch** (with CUDA support)
- **Pandas** & **NumPy**
- **TA-Lib** (Technical Analysis Library)
- **Matplotlib** & **Seaborn**
- **yfinance**

