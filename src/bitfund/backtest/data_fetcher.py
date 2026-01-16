import pandas as pd
import requests
from ib_insync import IB, Contract, util
import datetime
import logging

async def get_historical_data_ib(ib: IB, contract: Contract, duration: str, bar_size: str, what_to_show: str = 'TRADES', use_rth: bool = True, formatDate: str = '1', keepUpToDate: bool = False) -> pd.DataFrame:
    """
    Requests historical market data for a given contract from Interactive Brokers (IB) using ib_insync.

    Args:
        ib (IB): The ib_insync IB client instance.
        contract (Contract): The IB Contract object specifying the security.
        duration (str): The duration string for the historical data request (e.g., '1 Y', '3 M', '1 W', '1 D', '30 D', '1 H', '30 mins').
        bar_size (str): The bar size for the historical data (e.g., '1 secs', '5 secs', '1 min', '5 mins', '1 hour', '1 day').
        what_to_show (str, optional): The type of data to retrieve (e.g., 'TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'). Defaults to 'TRADES'.
        use_rth (bool, optional): Whether to retrieve data only during regular trading hours. Defaults to True.
        formatDate (str, optional): The format of the date in the returned data ('1' for YYYYMMDD HH:MM:SS, '2' for system time in seconds). Defaults to '1'.
        keepUpToDate (bool, optional): If True, the historical data will be updated in real-time. Note: This requires the function to run continuously. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical market data with columns:
                          ['date', 'open', 'high', 'low', 'close', 'volume', 'average', 'barCount'].
                          Returns an empty DataFrame if no data is received or if there's an error.
    """
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=formatDate,
            keepUpToDate=keepUpToDate
        )
        if bars:
            df = pd.DataFrame([vars(bar) for bar in bars])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            print(f"No historical data received for {contract.symbol} with duration '{duration}' and bar size '{bar_size}'.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error requesting historical data for {contract.symbol}: {e}")
        return pd.DataFrame()

def get_bitcoin_data_coincodex(days: int = 365) -> pd.DataFrame:
    """
    Fetches Bitcoin daily price data from CoinCodex API.
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # CoinCodex API might limit the number of samples returned in a single call.
    # If the requested range is large, we might need to make multiple calls or use a different endpoint.
    # However, for simplicity and robustness, let's try to fetch year by year if the range is large.
    
    all_records = []
    
    current_start = start_date
    while current_start < end_date:
        # Fetch in chunks of 1 year (365 days) to avoid sampling limits
        current_end = min(current_start + datetime.timedelta(days=365), end_date)
        
        s_str = current_start.strftime("%Y-%m-%d")
        e_str = current_end.strftime("%Y-%m-%d")
        
        # Request slightly more samples than days to ensure daily resolution
        chunk_days = (current_end - current_start).days
        samples = chunk_days + 5 
        
        url = f"https://coincodex.com/api/coincodex/get_coin_history/BTC/{s_str}/{e_str}/{samples}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'BTC' in data:
                for point in data['BTC']:
                    date = pd.to_datetime(point[0], unit='s')
                    price = float(point[1])
                    all_records.append({'date': date, 'close': price})
            else:
                print(f"Warning: No data for range {s_str} to {e_str}")
                
        except Exception as e:
            print(f"Error fetching chunk {s_str} to {e_str}: {e}")
        
        # Move to next chunk
        current_start = current_end
        
    if all_records:
        df = pd.DataFrame(all_records)
        # Remove duplicates if any (due to overlapping or API behavior)
        df = df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    print("Fetching Bitcoin data...")
    btc_df = get_bitcoin_data_coincodex(days=365*5)

    if btc_df.empty:
        print("Failed to fetch Bitcoin data.")
    else:
        print(f"Bitcoin data shape: {btc_df.shape}")
