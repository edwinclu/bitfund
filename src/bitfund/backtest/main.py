import asyncio
import pandas as pd
from ib_insync import IB, Stock
from bitfund.backtest.data_fetcher import get_historical_data_ib, get_bitcoin_data_coincodex
from bitfund.backtest.engine import compute_compound_value, BacktestEngine
import matplotlib.pyplot as plt

async def main():
    # 1. Connect to IB
    ib = IB()
    try:
        print("Connecting to IB...")
        # clientId=2 to avoid conflict with other clients
        await ib.connectAsync('127.0.0.1', 7497, clientId=2)
    except Exception as e:
        print(f"Could not connect to IB: {e}")
        return

    # 2. Fetch Stock Data (e.g., MSTR)
    print("Fetching MSTR data...")
    contract = Stock('MSTR', 'SMART', 'USD')
    mstr_df = await get_historical_data_ib(ib, contract, duration='2 Y', bar_size='1 day')

    ib.disconnect()

    if mstr_df.empty:
        print("Failed to fetch MSTR data.")
        return

    # 3. Fetch Bitcoin Data
    print("Fetching Bitcoin data...")
    btc_df = get_bitcoin_data_coincodex(days=365*2)

    if btc_df.empty:
        print("Failed to fetch Bitcoin data.")
        return

    # 4. Compute Compound Value
    print("Computing compound value...")
    compound_df = compute_compound_value(mstr_df, btc_df, stock_col='close', crypto_col='close')

    if compound_df.empty:
        print("No overlapping data found.")
        return

    print(f"Data prepared. {len(compound_df)} points.")

    # 5. Run Backtest
    print("Running backtest...")
    engine = BacktestEngine(compound_df, initial_batch_size=100, penalty=1.0)
    engine.run()

    # 6. Results
    results = engine.get_results()
    print("Backtest complete.")
    print(results.tail())

    # Plotting
    if not results.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(results['date'], results['value'], label='Portfolio Value')
        plt.title('Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Value (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    asyncio.run(main())
