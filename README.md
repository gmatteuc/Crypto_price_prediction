# Crypto Prediction with LSTM

This project implements a Deep Learning model (LSTM) to predict the daily price movement (Up/Down) of Ethereum (ETH). It uses historical price data along with technical indicators and macroeconomic data (S&P 500, Treasury Yields, VIX, Bitcoin).

## Project Structure

-   `main.py`: The main entry point. Handles data downloading, preprocessing, model training, and evaluation.
-   `model.py`: Defines the `CryptoLSTMClassifier` PyTorch model.
-   `data_loader.py`: Handles downloading historical data using `yfinance`.
-   `preprocessor.py`: Contains functions for feature engineering (technical indicators) and data scaling.
-   `check_gpu.py`: Utility script to check for CUDA availability.
-   `requirements.txt`: List of Python dependencies.

## Features

-   **Data Acquisition**: Fetches daily OHLCV data for ETH-USD from Yahoo Finance.
-   **External Data**: Incorporates S&P 500, 10-Year Treasury Yield (^TNX), Volatility Index (^VIX), and Bitcoin (BTC-USD) data to capture broader market trends.
-   **Feature Engineering**: Calculates technical indicators:
    -   SMA (20, 50)
    -   EMA (12, 26)
    -   MACD & Signal
    -   Bollinger Bands
    -   RSI
    -   ATR
-   **Model**: Long Short-Term Memory (LSTM) neural network for sequence classification.
-   **Evaluation**: Accuracy, Confusion Matrix, Classification Report, ROC Curve.

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
