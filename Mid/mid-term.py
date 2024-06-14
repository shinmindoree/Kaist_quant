
from datetime import timedelta
import pandas as pd
import statistics
import numpy as np


data = pd.read_excel("./processed_quant_mid.xlsx")
# Define the periods for lookback and holding
periods = [3, 6, 9, 12]

# Prepare a dictionary to hold the results
results = {}

# Calculate returns for each period
for J in periods:  # Lookback period
    for K in periods:  # Holding period
        # Prepare a list to store returns for all windows for this J-K combination
        all_returns = []

        # Iterate through the DataFrame by date for the rolling window
        for start_idx in range(len(data) - J - K + 1):
            # Lookback window
            start_date = data['Date'].iloc[start_idx]
            end_lookback_date = data['Date'].iloc[start_idx + J - 1]

            # Holding period window
            start_holding_date = data['Date'].iloc[start_idx + J]
            end_holding_date = data['Date'].iloc[start_idx + J + K - 1]

            # Calculate lookback returns
            lookback_prices_start = data.iloc[start_idx, 1:]
            lookback_prices_end = data.iloc[start_idx + J - 1, 1:]
            lookback_returns = (lookback_prices_end - lookback_prices_start) / lookback_prices_start

            # Identify top 30% and bottom 30% stocks
            top_30_threshold = lookback_returns.quantile(0.7)
            bottom_30_threshold = lookback_returns.quantile(0.3)

            # Calculate holding returns for top and bottom 30%
            holding_prices_start = data.iloc[start_idx + J, 1:]
            holding_prices_end = data.iloc[start_idx + J + K - 1, 1:]
            holding_returns = (holding_prices_end - holding_prices_start) / holding_prices_start

            # Filter returns
            top_30_returns = holding_returns[lookback_returns >= top_30_threshold]
            bottom_30_returns = holding_returns[lookback_returns <= bottom_30_threshold]

            # Combine and average the returns for this window
            window_returns = pd.concat([top_30_returns, -bottom_30_returns])
            all_returns.append(window_returns.mean())

        # Calculate the overall average return for this J-K combination
        results[f'{J}x{K}'] = sum(all_returns) / len(all_returns)

# Convert results to a DataFrame for display
results_df = pd.DataFrame(list(results.items()), columns=['Period', 'Average Return'])
results_df['Average Return'] = results_df['Average Return'].astype(float).map("{:.2%}".format)  # Format as percentage
results_df
