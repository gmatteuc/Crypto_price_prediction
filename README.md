![Project Banner](dataset-cover.png)

# Crypto Prediction with LSTM (Regression Strategy)

This project implements a Deep Learning model (LSTM) to predict the daily returns of Ethereum (ETH). Unlike traditional classifiers, this model uses a **Regression** approach with a custom **Directional Huber Loss** to prioritize the *direction* of the price movement over the exact magnitude.

The trading strategy employs an **Adaptive Threshold** mechanism to filter out low-confidence predictions, resulting in a robust, defensive trading algorithm.

## Key Results (Jan 2026)

Despite a challenging market period where Buy & Hold resulted in a loss, the LSTM strategy remained profitable by effectively managing risk and staying in cash during downturns.

*   **LSTM Strategy**: **+21.9%**
*   **Buy & Hold**: **-14.8%**
*   **Random Strategy**: **-45.0%**

![Equity Curve](output/equity_curve_regression.png)

### Why it works (The "Defensive" Paradox)
Interestingly, the model's raw predictive metrics (R2 Score, Directional Accuracy) are low. However, the strategy succeeds because:
1.  **Adaptive Threshold**: It only trades when the predicted return exceeds a dynamic volatility-based threshold.
2.  **Risk Management**: It effectively identifies and avoids periods of high downside volatility (as seen in the equity curve, where the strategy line stays flat while the market crashes).

## Project Structure

-   `main.py`: The main entry point. Handles data downloading, preprocessing, model training, and evaluation.
-   `model.py`: Defines the `CryptoLSTMRegressor` PyTorch model.
-   `config.py`: Centralized configuration for hyperparameters (Window size, Loss weights, etc.).
-   `evaluator.py`: Advanced metrics and plotting (Equity Curves, Bar Charts with Standard Error).
-   `trainer.py`: Custom training loop implementing the Directional Huber Loss.
-   `data_loader.py`: Handles downloading historical data using `yfinance`.
-   `preprocessor.py`: Feature engineering (MACD, RSI, Bollinger Bands, Macro Indicators).

## Features

-   **Regression Model**: Predicts continuous returns rather than binary classes.
-   **Directional Huber Loss**: A custom loss function that penalizes sign errors (wrong direction) more heavily than magnitude errors.
    $$ Loss = Huber(y, \hat{y}) + \lambda \cdot |y - \hat{y}| \cdot \mathbb{1}_{sign(y) \neq sign(\hat{y})} $$
-   **Macro-Economic Context**: Incorporates S&P 500, 10-Year Treasury Yields, VIX, and Bitcoin data.
-   **Statistical Validation**: Includes Monte Carlo-style sampling to compare strategy performance against random baselines over 30-day windows with Standard Error bars.

## Setup and Usage

1.  **Create Environment (Conda)**:
    It is highly recommended to use Conda to manage the environment and Python version.
    ```bash
    conda create -n crypto_prediction python=3.10 -y
    conda activate crypto_prediction
    ```

2.  **Install PyTorch with CUDA Support (CRITICAL STEP)**:
    To use your NVIDIA GPU, you **MUST** install the CUDA-enabled version of PyTorch using the following command. Do not rely on the default `pip install torch`.
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
    *(Note: If you do not have a GPU, you can skip this step and let `requirements.txt` install the CPU version, or run `pip install torch`)*

3.  **Install Other Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Code**:
    ```bash
    python main.py
    ```

## Dependencies

-   Python 3.8+
-   torch
-   pandas
-   numpy
-   yfinance
-   scikit-learn
-   matplotlib
-   seaborn
-   ta
-   joblib
