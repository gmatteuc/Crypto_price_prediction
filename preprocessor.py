"""
Crypto Prediction - Data Preprocessing
======================================

This module handles feature engineering, technical indicator calculation,
data scaling, and sequence generation for LSTM input.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import logging
from typing import Tuple, List, Optional
from config import Config

logger = logging.getLogger("CryptoPrediction")

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    # --- 1. Price & Volume Changes (Stationary) ---
    # Log Returns (The most important feature)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Intra-day volatility (High-Low range relative to Close)
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    
    # Close-Open change (Body size relative to Open)
    df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
    
    # Volume Change (Log change to handle spikes)
    # Add 1e-8 to avoid log(0)
    df['Volume_Log_Change'] = np.log((df['Volume'] + 1e-8) / (df['Volume'].shift(1) + 1e-8))

    # --- 2. Trend Indicators (Relative) ---
    # Distance from SMAs (Percentage)
    sma_20 = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    sma_50 = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['SMA_20_Dist'] = (df['Close'] - sma_20) / sma_20
    df['SMA_50_Dist'] = (df['Close'] - sma_50) / sma_50
    
    # Distance from EMAs (Percentage)
    ema_12 = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    ema_26 = EMAIndicator(close=df['Close'], window=26).ema_indicator()
    df['EMA_12_Dist'] = (df['Close'] - ema_12) / ema_12
    df['EMA_26_Dist'] = (df['Close'] - ema_26) / ema_26
    
    # MACD (Normalized by Close to be price-agnostic)
    macd = MACD(close=df['Close'])
    df['MACD_Norm'] = macd.macd() / df['Close']
    df['MACD_Signal_Norm'] = macd.macd_signal() / df['Close']
    df['MACD_Diff_Norm'] = macd.macd_diff() / df['Close']
    
    # --- 3. Volatility Indicators (Relative) ---
    # Garman-Klass Volatility (More efficient estimator than Close-to-Close)
    # GK = 0.5 * ln(High/Low)^2 - (2*ln(2)-1) * ln(Close/Open)^2
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    df['Garman_Klass_Vol'] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    
    # Bollinger Bands %B (Position within bands, 0=Low, 1=High)
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_PctB'] = bb.bollinger_pband()
    # Band Width (Volatility measure normalized)
    df['BB_Width'] = bb.bollinger_wband()
    
    # ATR (Normalized by Close)
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['ATR_Norm'] = atr / df['Close']
    
    # --- 4. Momentum Indicators (Oscillators are already stationary) ---
    # RSI (0-100)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # ROC (Rate of Change) - Momentum
    df['ROC_7'] = ROCIndicator(close=df['Close'], window=7).roc()
    df['ROC_14'] = ROCIndicator(close=df['Close'], window=14).roc()
    df['ROC_30'] = ROCIndicator(close=df['Close'], window=30).roc() # Added Monthly Momentum

    # Distance from SMA (Trend)
    sma_50 = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['Dist_SMA_50'] = (df['Close'] - sma_50) / sma_50
    
    # --- 5. Volume Indicators ---
    # MFI (Money Flow Index) - Volume-weighted RSI (0-100)
    df['MFI'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).money_flow_index()
    
    # OBV (On Balance Volume) - We need to make it stationary. 
    # OBV itself is cumulative. We use OBV Slope or Change.
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['OBV_Slope'] = obv.pct_change() # Change in OBV

    # --- 5b. Fractal & Efficiency Indicators ---
    # Kaufman Efficiency Ratio (KER)
    # KER = |Change(N)| / Sum(|Change(1)| for N periods)
    # Measures trend efficiency. 1 = Straight line, 0 = Random noise.
    ker_window = 14
    direction = df['Close'].diff(ker_window).abs()
    volatility = df['Close'].diff().abs().rolling(window=ker_window).sum()
    df['KER'] = direction / volatility

    # Fractal Dimension (Simplified via Volatility)
    # We can use the relationship between Range and Volatility to estimate Fractal Dimension
    # But KER is a good proxy for "Trendiness" vs "Choppiness".
    
    # Rolling Hurst Exponent (Simplified Proxy)
    # A true Hurst calculation is slow. We can use a proxy:
    # H ~ log(Range) / log(Time)
    # We'll use a standardized version of Range/StdDev (Rescaled Range Analysis)
    # This is computationally expensive for rolling windows in pure pandas.
    # We will stick to KER as our primary "Regime" filter.

    # --- 6. Macro Indicators (Transformed) ---
    # Absolute prices of SP500 and Rates are not as predictive as their changes
    if 'SP500' in df.columns:
        # Daily return S&P 500
        df['SP500_Ret'] = df['SP500'].pct_change()
        
    if 'TNX' in df.columns:
        # Daily change in Rates (e.g. +0.05%)
        df['TNX_Change'] = df['TNX'].diff()
        
    if 'VIX' in df.columns:
        # VIX is already a volatility measure, use absolute value and change
        df['VIX'] = df['VIX'] # VIX level is mean-reverting, so it's okay-ish, but change is better
        df['VIX_Change'] = df['VIX'].diff()

    if 'BTC' in df.columns:
        # Daily return Bitcoin
        df['BTC_Ret'] = df['BTC'].pct_change()

    if 'Gold' in df.columns:
        # Daily return Gold
        df['Gold_Ret'] = df['Gold'].pct_change()
        
    if 'DXY' in df.columns:
        # Daily return DXY
        df['DXY_Ret'] = df['DXY'].pct_change()

    if 'NASDAQ' in df.columns:
        # Daily return NASDAQ
        df['NASDAQ_Ret'] = df['NASDAQ'].pct_change()
        
    if 'Oil' in df.columns:
        # Daily return Oil
        df['Oil_Ret'] = df['Oil'].pct_change()
        
    if 'COIN' in df.columns:
        # Daily return Coinbase
        df['COIN_Ret'] = df['COIN'].pct_change()
        
    if 'NVDA' in df.columns:
        # Daily return NVIDIA
        df['NVDA_Ret'] = df['NVDA'].pct_change()
        
    if 'EURUSD' in df.columns:
        # Daily return EUR/USD
        df['EURUSD_Ret'] = df['EURUSD'].pct_change()
        
    if 'FNG' in df.columns:
        # Fear & Greed Index (0-100)
        # We can use it as is (0-100) or normalize it. 
        # Since we use StandardScaler later, raw value is fine.
        # But maybe change is also useful?
        df['FNG'] = df['FNG'] # Keep raw level
        
    if 'BTC' in df.columns and 'Close' in df.columns:
        # ETH/BTC Ratio (Relative Strength)
        # If ETH outperforms BTC, ratio goes up.
        df['ETH_BTC_Ratio'] = df['Close'] / df['BTC']
        df['ETH_BTC_Ret'] = df['ETH_BTC_Ratio'].pct_change()
        
    # --- 7. Cyclical Time Features ---
    # Day of Week (0-6) -> Sine/Cosine
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Month of Year (1-12) -> Sine/Cosine
    df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # --- Cleanup ---
    # Drop raw non-stationary columns to force model to use relative features
    # We keep 'Close' only for calculating targets later (if needed outside), 
    # but we should drop it from the feature set used for scaling/training.
    # For now, we'll drop the original OHLCV columns from the dataframe 
    # EXCEPT 'Close' which might be needed for target calculation in main.py before scaling.
    # Actually, main.py calculates target based on 'Close' BEFORE calling scale_data? 
    # Let's check main.py. 
    # Usually data_loader loads data, then preprocessor adds indicators.
    # We should drop the raw columns here to be safe, but keep Close for target gen.
    
    cols_to_drop = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 
                    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 
                    'BB_High', 'BB_Low', 'ATR', 'SP500', 'TNX', 'BTC', 'Gold', 'DXY', 'NASDAQ', 'Oil', 'ETH_BTC_Ratio']
    
    # Only drop if they exist
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')
    
    return df

def scale_data(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scales data using StandardScaler (better for stationary features).
    
    Args:
        df (pd.DataFrame): DataFrame to scale.
        
    Returns:
        Tuple[np.ndarray, StandardScaler]: Scaled data and the fitted scaler.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int, weights: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Creates time sequences for LSTM.
    
    Args:
        features (np.ndarray): Scaled feature data.
        targets (np.ndarray): Target data.
        seq_length (int): Length of the sequence.
        weights (np.ndarray, optional): Sample weights.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: X (sequences), y (targets), w (weights - if provided).
    """
    X = []
    y = []
    w = [] if weights is not None else None
    
    # We want to predict target at index i using features up to index i
    # (Since target[i] is already shifted to be return[i+1])
    # So X should be features[i-seq_length+1 : i+1]
    # Loop starts at seq_length-1 to have enough data for first sequence
    
    for i in range(seq_length - 1, len(features)):
        X.append(features[i - seq_length + 1 : i + 1])
        y.append(targets[i])
        if weights is not None:
            w.append(weights[i])
        
    if weights is not None:
        return np.array(X), np.array(y), np.array(w)
    else:
        return np.array(X), np.array(y)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects columns to use as features for the model.
    
    Args:
        df (pd.DataFrame): DataFrame with all indicators.
        
    Returns:
        pd.DataFrame: DataFrame with selected features and target.
    """
    # Use only stationary features
    # Expanded Feature List (The "Core 14" with Alternative Data)
    feature_cols = [
        'Log_Ret',              # Momentum (Price)
        'Volume_Log_Change',    # Momentum (Volume)
        'RSI',                  # Overbought/Oversold
        'MACD_Norm',            # Trend Strength
        'BB_PctB',              # Mean Reversion
        'ATR_Norm',             # Volatility
        'ROC_30',               # Monthly Momentum (New)
        'Dist_SMA_50',          # Trend Deviation (New)
        'SP500_Ret',            # Macro Correlation
        'BTC_Ret',              # Crypto Market Correlation
        'Gold_Ret',             # Safe Haven Correlation
        'DXY_Ret',              # Currency Strength Correlation
        'NASDAQ_Ret',           # Tech Sector Correlation
        'Oil_Ret',              # Commodity/Inflation Correlation
        'FNG',                  # Sentiment (Fear & Greed)
        'ETH_BTC_Ret'           # Relative Strength vs BTC        'KER',                  # Kaufman Efficiency Ratio (Trend Quality)        # 'COIN_Ret',             # Crypto Equity Correlation (Removed: Noise)
        # 'NVDA_Ret',             # Tech/AI Correlation (Removed: Noise)
        # 'EURUSD_Ret',           # Forex Correlation (Removed: Noise)
        # 'Garman_Klass_Vol',     # Advanced Volatility (Removed: Redundant)
        # 'TNX_Change',           # Interest Rate Change (Removed: Noise)
        # 'VIX_Change',           # Volatility Index Change (Removed: Noise)
        # 'Day_Sin', 'Day_Cos',   # Cyclical Time (Weekly) (Removed: Noise)
        # 'Month_Sin', 'Month_Cos'# Cyclical Time (Yearly) (Removed: Noise)
    ]
    
    # Filter only columns that actually exist in the dataframe
    existing_cols = [col for col in feature_cols if col in df.columns]
    
    # We might need 'Close' or 'Log_Ret' to create the target if not already there
    # But Log_Ret is in feature_cols.
    
    df_features = df[existing_cols].copy()
    
    # Create Binary Target based on Prediction Horizon
    horizon = Config.PREDICTION_HORIZON
    
    # Calculate N-day future return: (Close[t+N] - Close[t]) / Close[t]
    # pct_change(N) gives (Close[t] - Close[t-N]) / Close[t-N]
    # Shift(-N) brings that value to index t-N? No.
    # Let's do it explicitly to be safe and clear.
    
    # Future Close price N days from now
    future_close = df['Close'].shift(-horizon)
    
    # Target is 1 if Future Close > Current Close
    df_features['Target_Class'] = (future_close > df['Close']).astype(float)
    # Also store return magnitude for sample weighting
    df_features['Target_Return_Mag'] = np.abs(np.log(future_close / df['Close']))
    
    # Drop NaNs created by indicators (rolling windows) and shifting
    df_features.dropna(inplace=True)
    
    return df_features
