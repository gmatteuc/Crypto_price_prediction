"""
Crypto Prediction - Fear & Greed Fetcher
========================================

This script retrieves historical Fear & Greed Index data from the alternative.me API
and saves it to a CSV file for use as a sentiment feature.
"""

import requests
import pandas as pd
from datetime import datetime

def fetch_fng_data():
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['data']
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by date just in case, though API usually returns sorted
        df = df.sort_values('date')
        
        print("First 5 entries:")
        print(df.head(5)[['value', 'value_classification', 'date']])
        print("\nLast 5 entries:")
        print(df.tail(5)[['value', 'value_classification', 'date']])
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        print(f"\nData range: {min_date} to {max_date}")
        
        start_check = datetime(2015, 1, 1)
        if min_date <= start_check:
            print("Data covers range from 2015.")
        else:
            print(f"Data starts from {min_date}, which is after 2015-01-01.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_fng_data()
