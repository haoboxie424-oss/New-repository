import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Download 5 years of MSFT data
def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Calculate moving averages
def calculate_moving_averages(data, short_window=50, long_window=200):
    short_col = f'MA{short_window}'
    long_col = f'MA{long_window}'
    data[short_col] = data['Close'].rolling(window=short_window).mean()
    data[long_col] = data['Close'].rolling(window=long_window).mean()
    return data

# Identify golden cross (buy signals)
def identify_golden_cross(data, short_window=50, long_window=200):
    short_col = f'MA{short_window}'
    long_col = f'MA{long_window}'
    data['Signal'] = 0  # Initialize signal column with 0
    # Golden Cross occurs when the short MA crosses above the long MA
    data['GoldenCross'] = (data[short_col] > data[long_col]) & (data[short_col].shift(1) <= data[long_col].shift(1))
    return data

# Implement trading strategy
def implement_strategy(data, short_window=50, long_window=200):
    positions = []

    # Need at least the longest lookback period to calculate the moving averages
    min_period = max(short_window, long_window)
    data = data.iloc[min_period:].copy()

    buy_dates = data[data['GoldenCross'] == True].index.tolist()

    for buy_date in buy_dates:
        # Get buy price
        buy_price = data.loc[buy_date, 'Close']

        # Calculate target sell price (15% profit)
        target_price = buy_price * 1.15

        # Set maximum holding period
        max_sell_date = buy_date + pd.Timedelta(days=60)

        # Get data slice for potential sell period
        sell_period = data.loc[buy_date:max_sell_date].copy()

        # Check if target price is reached during the period
        target_reached = sell_period[sell_period['Close'] >= target_price]

        if not target_reached.empty:
            # Sell at first date target is reached
            sell_date = target_reached.index[0]
            sell_price = target_reached.loc[sell_date, 'Close']
            sell_reason = "Target reached"
        else:
            # Sell at end of maximum holding period
            sell_date_candidates = sell_period.index.tolist()
            if sell_date_candidates:
                sell_date = sell_date_candidates[-1]
                sell_price = data.loc[sell_date, 'Close']
                sell_reason = "Max holding period"
            else:
                # Skip if no valid sell date (should not happen in practice)
                continue

        # Calculate holding period in calendar days
        holding_days = (sell_date - buy_date).days

        # Calculate profit
        profit_pct = (sell_price / buy_price - 1) * 100

        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)

# Analyze the results
def analyze_results(positions):
    if positions.empty:
        return "No trading signals detected"

    # Summary statistics
    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    avg_profit = positions['ProfitPct'].mean()
    avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
    avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0

    avg_holding = positions['HoldingDays'].mean()

    target_reached = len(positions[positions['SellReason'] == 'Target reached'])
    max_period = len(positions[positions['SellReason'] == 'Max holding period'])

    print("\n===== Trading Strategy Results (Golden Cross)=====")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {win_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {loss_trades}")
    print(f"Average Profit: {avg_profit:.2f}%")

    return positions

# Main function
def main(short_window=50, long_window=200):
    # Get stock data
    data = get_stock_data("MSFT")

    # Calculate moving averages
    data = calculate_moving_averages(data, short_window, long_window)

    # Identify golden cross
    data = identify_golden_cross(data, short_window, long_window)

    # Implement strategy
    positions = implement_strategy(data, short_window, long_window)

    # Analyze results
    analyze_results(positions)

    # Return the detailed positions dataframe
    return positions


def run_strategy(short_window=50, long_window=200):
    if long_window <= short_window:
        long_window = short_window + 1

    data = get_stock_data("MSFT")
    data = calculate_moving_averages(data, short_window, long_window)
    data = identify_golden_cross(data, short_window, long_window)
    positions = implement_strategy(data, short_window, long_window)

    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0]) if not positions.empty else 0
    loss_trades = total_trades - win_trades
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit = positions['ProfitPct'].mean() if not positions.empty else 0
    avg_holding = positions['HoldingDays'].mean() if not positions.empty else 0

    summary = {
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate': round(win_rate, 2),
        'avg_profit_pct': round(float(avg_profit), 2),
        'avg_holding_days': round(float(avg_holding), 2),
        'total_return_pct': round(float((positions['ProfitPct'].sum()) if not positions.empty else 0), 2),
        'best_trade_pct': round(float(positions['ProfitPct'].max()) if not positions.empty else 0, 2),
        'worst_trade_pct': round(float(positions['ProfitPct'].min()) if not positions.empty else 0, 2),
    }

    return data, positions, summary


if __name__ == "__main__":
    positions = main()
    print("\nDetailed Trades:")
    print(positions.to_string())