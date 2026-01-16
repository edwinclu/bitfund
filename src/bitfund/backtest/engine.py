import pandas as pd
import numpy as np
from bitfund.utils import segmentation_r1 as sgr

def compute_compound_value(stock_df: pd.DataFrame, crypto_df: pd.DataFrame, stock_col='close', crypto_col='close') -> pd.DataFrame:
    """
    Computes the quotient (compound value) of stock price divided by crypto price.
    Ensures alignment on date.
    """
    # Ensure dates are datetime objects and set as index for easy merging
    stock_df = stock_df.copy()
    crypto_df = crypto_df.copy()
    
    if 'date' in stock_df.columns:
        stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    if 'date' in crypto_df.columns:
        crypto_df['date'] = pd.to_datetime(crypto_df['date'])
        
    # Merge on date
    # We use inner join to ensure we have data for both on the same days
    merged_df = pd.merge(stock_df, crypto_df, on='date', how='inner', suffixes=('_stock', '_crypto'))
    
    # Compute quotient
    merged_df['quotient'] = merged_df[f'{stock_col}_stock'] / merged_df[f'{crypto_col}_crypto']
    
    # Normalize (as per mstr.ipynb example: overprice_norm = overprice * 1000)
    merged_df['quotient_norm'] = merged_df['quotient'] * 1000
    
    return merged_df[['date', f'{stock_col}_stock', f'{crypto_col}_crypto', 'quotient', 'quotient_norm']]

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_batch_size: int = 100, penalty: float = 1.0, mono_n: int = 10, drastic_factor: float = 1.3, threshold: float = 0.05):
        self.data = data.sort_values('date').reset_index(drop=True)
        self.initial_batch_size = initial_batch_size
        self.penalty = penalty
        self.mono_n = mono_n
        self.drastic_factor = drastic_factor
        self.threshold = threshold
        
        self.segmenter = None
        self.signals = [] # List of (date, signal, price)
        self.portfolio_value = [] # Track value over time
        
        # Trading state
        self.position = 0 # 0: neutral, 1: long, -1: short (simplified)
        self.cash = 100000.0 # Initial cash
        self.holdings = 0.0
        
    def run(self):
        if len(self.data) < self.initial_batch_size:
            print("Not enough data for initial batch.")
            return

        # Prepare data for segmenter (needs numpy array of [index, value] or [time, value])
        # We use index as x-axis for segmentation to keep it simple and consistent with the util
        # The value we segment is 'quotient_norm'
        
        values = self.data['quotient_norm'].values
        indices = np.arange(len(values))
        data_array = np.column_stack((indices, values))
        
        # Warm up
        initial_data = data_array[:self.initial_batch_size]
        self.segmenter = sgr.ConsistentRunningSegmenter(
            initial_data_array=initial_data,
            penalty_lambda=self.penalty,
            condition_penalty=2*self.penalty,
            mono_n=self.mono_n,
            drastic_factor=self.drastic_factor
        )
        
        print(f"Initial batch processed. Starting backtest loop from index {self.initial_batch_size}...")
        
        # Store previous number of segments
        prev_num_segments = len(self.segmenter.get_segments())
        
        # Loop through subsequent data points
        # We start from initial_batch_size.
        # At step i, we are at day T (current_date).
        # We have data up to T-1 already processed in segmenter (from previous iteration or warmup).
        # Wait, the warmup includes data up to initial_batch_size-1.
        # So at i=initial_batch_size, we are processing the point at index i.
        
        # To avoid look-ahead bias as requested:
        # "at day T, run segmenter.process_new_point with closing compound price of day T-1"
        # This implies we make decisions for day T using information available up to T-1.
        
        # However, the loop iterates over the data available.
        # Let's align indices:
        # i is the index of the current day T we are simulating.
        # We want to trade at the open/close of day T.
        # The decision should be based on segments formed by data up to T-1.
        
        # But the prompt says: "run segmenter.process_new_point with closing compound price of day T-1".
        # This means we update the model with yesterday's data, check the state change, and then trade today.
        
        # In the loop:
        # i corresponds to day T.
        # data_array[i-1] corresponds to day T-1.
        
        # The warmup processed `initial_data` which is `data_array[:initial_batch_size]`.
        # This means indices 0 to initial_batch_size-1 are already in the segmenter.
        # So when we start the loop at `i = initial_batch_size`, the segmenter has seen data up to `i-1`.
        # This matches the requirement perfectly. The segmenter state reflects knowledge up to T-1.
        
        for i in range(self.initial_batch_size, len(data_array) - 1):
            # 1. Process T-1
            point_t_minus_1 = data_array[i:i+1]
            self.segmenter.process_new_point(point_t_minus_1)
            
            segments = self.segmenter.get_segments()
            curr_num_segments = len(segments)
            
            # 2. Identify Day T info for trading
            idx_T = i + 1
            date_T = self.data.iloc[idx_T]['date']
            stock_price_T = self.data.iloc[idx_T]['close_stock']
            compound_val_T_minus_1 = point_t_minus_1[0, 1] # Value of T-1
            
            signal = 0
            
            # 3. Logic
            if curr_num_segments > prev_num_segments:
                # New segment created due to fluctuation
                # Clear all positions
                # "try to find new opportunities in future" -> implies wait?
                # For now, just close position (sell if long, cover if short)
                if self.position == 1:
                    signal = -1 # Sell to close
                elif self.position == -1:
                    signal = 1 # Buy to cover
                # If neutral, stay neutral
                
                # Note: If we just signal -1 when Long, _execute_trade handles it.
                # But _execute_trade logic for -1 is "Short".
                # We need a "Close" signal or adapt _execute_trade.
                # Let's adapt _execute_trade to handle "Close" or just use logic here.
                
                # Actually, if I signal -1 (Sell), and I am Long, it sells and goes Short.
                # The requirement says "clear all your positions".
                # So I should go to 0.
                
                # I will modify _execute_trade or handle it here.
                # Let's add a 'CLEAR' signal type or logic.
                self._clear_position(stock_price_T, date_T)
                
            elif curr_num_segments == prev_num_segments:
                # Compare current compound price (T-1? or T?)
                # "compare the current compound price with the mean of the last segment"
                # Usually "current" means the latest known, which is T-1 (since we just processed it).
                # Using T would be look-ahead if we haven't processed it?
                # But we are at Day T. We know Open of T. We might know Close of T if we trade at Close.
                # If we trade at Close of T, we technically know Close of T.
                # But the model hasn't seen T yet.
                # The prompt says: "run segmenter... with T-1... compare the current compound price..."
                # "Current" likely refers to the price we just processed (T-1) or the price of the day we are trading (T).
                # Given "avoid look ahead bias", using T-1 for signal generation is safer.
                # But if we trade at T Close, using T Close for signal is also common (Backtest on Close).
                # However, the prompt explicitly linked the process to T-1.
                # Let's use T-1 price for signal generation to be safe and consistent with "N_{T-1}".
                
                last_segment = segments[-1]
                segment_values = last_segment[:, 1]
                segment_mean = np.mean(segment_values)
                
                # Using T-1 value
                deviation = (compound_val_T_minus_1 - segment_mean) / segment_mean
                
                if deviation > self.threshold:
                    # Positive and over threshold -> Sell
                    signal = -1
                elif deviation < -self.threshold:
                    # Negative and larger (magnitude) -> Buy
                    signal = 1
                    
                self._execute_trade(signal, stock_price_T, date_T)
            
            # Update state
            prev_num_segments = curr_num_segments
            
            # Track Portfolio (Mark to Market at Close of T)
            total_val = self.cash + self.holdings * stock_price_T
            self.portfolio_value.append({'date': date_T, 'value': total_val})

    def _clear_position(self, price, date):
        if self.position == 1:
            # Sell all
            proceeds = self.holdings * price
            self.cash += proceeds
            self.holdings = 0
            self.position = 0
            self.signals.append({'date': date, 'type': 'CLOSE_LONG', 'price': price})
        elif self.position == -1:
            # Cover all
            cost = abs(self.holdings) * price
            self.cash -= cost
            self.holdings = 0
            self.position = 0
            self.signals.append({'date': date, 'type': 'CLOSE_SHORT', 'price': price})

    def _execute_trade(self, signal, price, date):
        # ... (existing logic, but need to ensure we don't flip if we just want to open/close?
        # The prompt says: "if it's positive ... consider selling it".
        # If I am neutral, selling means Shorting.
        # If I am Long, selling means Closing (and maybe Shorting?).
        # Let's stick to the previous logic: Flip position.
        # But if "Clear" happened, we are 0.
        
        if signal == 1: # Buy
            if self.position == 0:
                shares = self.cash / price
                self.holdings = shares
                self.cash = 0
                self.position = 1
                self.signals.append({'date': date, 'type': 'BUY', 'price': price})
            elif self.position == -1:
                # Reverse
                cost = abs(self.holdings) * price
                self.cash -= cost
                self.holdings = 0
                
                shares = self.cash / price
                self.holdings = shares
                self.cash = 0
                self.position = 1
                self.signals.append({'date': date, 'type': 'REVERSE_LONG', 'price': price})
                
        elif signal == -1: # Sell
            if self.position == 0:
                shares = self.cash / price
                self.holdings = -shares
                self.cash += shares * price
                self.position = -1
                self.signals.append({'date': date, 'type': 'SHORT', 'price': price})
            elif self.position == 1:
                # Reverse
                proceeds = self.holdings * price
                self.cash += proceeds
                self.holdings = 0
                
                shares = self.cash / price
                self.holdings = -shares
                self.cash += shares * price
                self.position = -1
                self.signals.append({'date': date, 'type': 'REVERSE_SHORT', 'price': price})

    def get_results(self):
        return pd.DataFrame(self.portfolio_value)
