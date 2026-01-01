import yfinance as yf
import pandas as pd
import logging
import requests
from typing import Optional

logger = logging.getLogger("CryptoPrediction")

def fetch_fear_and_greed_index() -> pd.Series:
    """
    Fetches historical Crypto Fear & Greed Index data.
    Returns a pandas Series with the index values.
    """
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['data']
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.normalize()
        df['value'] = pd.to_numeric(df['value'])
        
        # Set index and sort
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        series = df['value']
        series.name = 'FNG'
        return series
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed Index: {e}")
        return pd.Series(name='FNG', dtype=float)

def download_data(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Downloads historical data for a given ticker and merges it with macroeconomic indicators.
    
    Args:
        ticker (str): The ticker symbol (e.g., "ETH-USD").
        start_date (Optional[str]): Start date in "YYYY-MM-DD" format.
        end_date (Optional[str]): End date in "YYYY-MM-DD" format.
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data and macro indicators.
    """
    logger.info(f"Downloading data for {ticker}...")
    stock = yf.Ticker(ticker)
    
    if start_date and end_date:
        df = stock.history(start=start_date, end=end_date)
    else:
        df = stock.history(period="max")
    
    # Remove columns we don't need (Dividends, Stock Splits) if present
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Filter only existing columns
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep]
    
    # --- Add Macroeconomic and Market Data ---
    logger.info("Downloading S&P 500, NASDAQ, Treasury Yield, VIX, Bitcoin, Gold, Oil, and DXY data...")
    
    # Helper function to fetch and normalize auxiliary data
    def fetch_aux_data(aux_ticker: str, name: str) -> pd.Series:
        try:
            aux_df = yf.Ticker(aux_ticker).history(period="max")
            if 'Close' in aux_df.columns:
                series = aux_df['Close']
                series.name = name
                # Normalize index
                series.index = series.index.normalize().tz_localize(None)
                return series
            else:
                logger.warning(f"Could not find 'Close' column for {aux_ticker}")
                return pd.Series(name=name, dtype=float)
        except Exception as e:
            logger.error(f"Error downloading {aux_ticker}: {e}")
            return pd.Series(name=name, dtype=float)

    sp500 = fetch_aux_data("^GSPC", 'SP500')
    nasdaq = fetch_aux_data("^IXIC", 'NASDAQ')
    tnx = fetch_aux_data("^TNX", 'TNX')
    vix = fetch_aux_data("^VIX", 'VIX')
    btc = fetch_aux_data("BTC-USD", 'BTC')
    gold = fetch_aux_data("GC=F", 'Gold')
    oil = fetch_aux_data("CL=F", 'Oil')
    dxy = fetch_aux_data("DX-Y.NYB", 'DXY')
    
    # Fetch Fear & Greed Index
    fng = fetch_fear_and_greed_index()
    
    # Normalize main dataframe index
    df.index = df.index.normalize().tz_localize(None)
    
    # Merge data
    # Use left join to keep the main ticker index (Crypto/Stock)
    df = df.join(sp500, how='left')
    df = df.join(nasdaq, how='left')
    df = df.join(tnx, how='left')
    df = df.join(vix, how='left')
    df = df.join(btc, how='left')
    df = df.join(gold, how='left')
    df = df.join(oil, how='left')
    df = df.join(dxy, how='left')
    df = df.join(fng, how='left')
    
    # Fill any gaps (forward fill then backward fill)
    df = df.ffill().bfill()
    
    logger.debug("Data Loader Head:\n" + str(df[['Close', 'SP500', 'NASDAQ', 'FNG']].head()))
    
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN count after filling:\n{nan_counts}")
    
    logger.info(f"Downloaded {len(df)} days of data (with macro indicators).")
    return df
