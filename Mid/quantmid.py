import pandas as pd

data = pd.read_csv("/Users/minsuk/Documents/PYTHON/Quant/processed_quant_mid.csv")
data.set_index('Date', inplace=True)


# Define the periods for lookback (J) and holding (K)
lookback_periods = [3, 6, 9, 12]  # in months
holding_periods = [3, 6, 9, 12]  # in months

lookback_returns = {}

for period in lookback_periods:
    lookback_returns[period] = data.pct_change(period)


# Initialize dictionaries to store future returns of Buy and Sell portfolios for each J-K combination

buy_portfolio_returns = {}
sell_portfolio_returns = {}

for j in lookback_periods:
    for k in holding_periods:
        buy_portfolio_returns[(j, k)] = []
        sell_portfolio_returns[(j, k)] = []



#####원래 코드######

# Function to calculate future returns for top and bottom 30% stocks
def calculate_future_returns(data, past_returns, holding_period):
    future_returns = data.pct_change(holding_period)
    top_30_returns = future_returns[past_returns >= past_returns.quantile(0.7)].mean(axis=1)
    bottom_30_returns = future_returns[past_returns <= past_returns.quantile(0.3)].mean(axis=1)
    return top_30_returns, bottom_30_returns


# Loop over each J-K combination to form portfolios and calculate future returns
for j in lookback_periods:
    for k in holding_periods:
        top_30_returns, bottom_30_returns = calculate_future_returns(data, lookback_returns[j], k)
        buy_portfolio_returns[(j, k)].append(top_30_returns)
        sell_portfolio_returns[(j, k)].append(bottom_30_returns)

# Example: Show average future returns for a single J-K combination to verify the calculation
# Convert the list of Series to a DataFrame for easier handling
example_j, example_k = 3, 3  # Example lookback and holding periods
buy_returns_example_data = pd.DataFrame(buy_portfolio_returns[(example_j, example_k)]).T.mean(axis=1)
sell_returns_example_data = pd.DataFrame(sell_portfolio_returns[(example_j, example_k)]).T.mean(axis=1)

# Display average returns for the example J-K combination
buy_avg_return = buy_returns_example_data.mean()
sell_avg_return = sell_returns_example_data.mean()


