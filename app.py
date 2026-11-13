import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go
import plotly.subplots as sp
import yfinance as yf

from trading_strategy import run_strategy

st.set_page_config(page_title='Trading Strategy Dashboard', layout='wide')

DEFAULT_PORTFOLIO_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'META', 'NFLX', 'TSLA', 'V', 'JPM',
    'XOM', 'WMT', 'AMD', 'INTC', 'COST', 'DIS', 'NKE', 'PEP', 'SHOP', 'CRM'
]


def format_trade_table(positions):
    if positions.empty:
        return pd.DataFrame(columns=[
            'BuyDate', 'BuyPrice', 'SellDate', 'SellPrice', 'HoldingDays', 'ProfitPct', 'SellReason'
        ])

    df = positions.copy()
    df['BuyDate'] = df['BuyDate'].dt.strftime('%Y-%m-%d')
    df['SellDate'] = df['SellDate'].dt.strftime('%Y-%m-%d')
    df['BuyPrice'] = df['BuyPrice'].round(2)
    df['SellPrice'] = df['SellPrice'].round(2)
    df['ProfitPct'] = df['ProfitPct'].round(2)
    return df[['BuyDate', 'BuyPrice', 'SellDate', 'SellPrice', 'HoldingDays', 'ProfitPct', 'SellReason']]  


def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def build_chart(data, positions, short_window=50, long_window=200):
    short_col = f'MA{short_window}'
    long_col = f'MA{long_window}'

    chart_df = data[['Open', 'High', 'Low', 'Close', short_col, long_col]].copy().reset_index()
    chart_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', short_col, long_col]
    chart_df['Date'] = pd.to_datetime(chart_df['Date'])

    buy_points = positions[['BuyDate', 'BuyPrice']].copy()
    buy_points.columns = ['Date', 'Price']
    buy_points['Date'] = pd.to_datetime(buy_points['Date'])

    sell_points = positions[['SellDate', 'SellPrice']].copy()
    sell_points.columns = ['Date', 'Price']
    sell_points['Date'] = pd.to_datetime(sell_points['Date'])

    macd_line, signal_line, hist = calculate_macd(data)
    macd_df = pd.DataFrame({
        'Date': data.index,
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': hist,
    }).dropna().reset_index(drop=True)

    fig = sp.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df['Date'],
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='Price',
            increasing_line_color='#16a34a',
            decreasing_line_color='#ef4444',
            increasing_fillcolor='#16a34a',
            decreasing_fillcolor='#ef4444',
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=chart_df['Date'], y=chart_df[short_col], mode='lines', name=f'MA{short_window}', line=dict(color='#f59e0b', width=2)),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=chart_df['Date'], y=chart_df[long_col], mode='lines', name=f'MA{long_window}', line=dict(color='#22c55e', width=2)),
        row=1,
        col=1,
    )

    for _, row in buy_points.iterrows():
        fig.add_vline(x=row['Date'], line_color='green', line_width=1, line_dash='dot')

    for _, row in sell_points.iterrows():
        fig.add_vline(x=row['Date'], line_color='red', line_width=1, line_dash='dot')

    fig.add_trace(
        go.Scatter(x=macd_df['Date'], y=macd_df['MACD'], mode='lines', name='MACD', line=dict(color='#6366f1', width=2)),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=macd_df['Date'], y=macd_df['Signal'], mode='lines', name='Signal', line=dict(color='#f97316', width=2)),
        row=2,
        col=1,
    )

    fig.add_bar(
        x=macd_df['Date'],
        y=macd_df['Histogram'],
        name='Histogram',
        marker_color=np.where(macd_df['Histogram'] >= 0, '#10b981', '#ef4444'),
        row=2,
        col=1,
    )

    fig.update_layout(
        title='MSFT Price Trend with MACD',
        template='plotly_white',
        height=850,
        legend=dict(
            orientation='v',
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            itemsizing='constant',
        ),
        margin=dict(l=20, r=180, t=40, b=20),
        hovermode='x unified',
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    fig.update_xaxes(rangeslider_visible=False, showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb', row=2, col=1)

    return fig


@st.cache_data
def get_portfolio_close_data(tickers):
    frames = []
    for ticker in tickers:
        history = yf.download(
            ticker,
            period='1y',
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=True,
        )
        if history.empty:
            continue
        close_series = history['Close'].copy()
        close_series.name = ticker
        close_series = close_series.dropna()
        frames.append(close_series)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).dropna(how='all')


def calculate_portfolio_metrics(tickers):
    close_df = get_portfolio_close_data(tickers)
    if close_df.empty:
        raise ValueError('No valid price data was returned for the selected tickers.')

    daily_returns = close_df.pct_change().dropna()
    if daily_returns.empty:
        raise ValueError('Not enough data to compute daily returns for the selected period.')

    portfolio_daily_return = daily_returns.mean(axis=1)
    individual_cumulative = (1 + daily_returns).cumprod() - 1
    portfolio_cumulative = (1 + portfolio_daily_return).cumprod() - 1
    allocation = pd.Series({ticker: 1 / len(tickers) for ticker in tickers})

    summary = {
        'portfolio_total_return_pct': round(float(portfolio_cumulative.iloc[-1] * 100), 2),
        'portfolio_volatility_pct': round(float(daily_returns.std().mean() * 100), 2),
        'selected_tickers': len(tickers),
    }

    return close_df, daily_returns, individual_cumulative, portfolio_cumulative, allocation, summary


def build_portfolio_allocation_chart(allocation):
    fig = go.Figure(data=[go.Pie(
        labels=allocation.index,
        values=allocation.values,
        hole=0.35,
        hovertemplate='%{label}: %{percent:.1%}<extra></extra>'
    )])
    fig.update_layout(
        title='Asset Allocation (Equal Weight)',
        margin=dict(l=20, r=180, t=60, b=20),
        height=500,
        legend=dict(
            orientation='v',
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            itemsizing='constant',
        ),
    )
    return fig


def build_portfolio_return_chart(individual_cumulative, portfolio_cumulative):
    fig = go.Figure()

    for ticker in individual_cumulative.columns:
        fig.add_trace(go.Scatter(
            x=individual_cumulative.index,
            y=individual_cumulative[ticker],
            mode='lines',
            name=ticker,
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>'
        ))

    fig.add_trace(go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='black', width=3),
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title='Cumulative Return Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis_tickformat='.0%',
        hovermode='x unified',
        legend=dict(
            orientation='v',
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            itemsizing='constant',
        ),
        margin=dict(l=20, r=200, t=60, b=40),
        height=560,
    )
    return fig


def main():
    short_ma = st.sidebar.slider('Short-term MA', min_value=10, max_value=100, value=50)
    long_ma = st.sidebar.slider('Long-term MA', min_value=100, max_value=300, value=200)

    if long_ma <= short_ma:
        long_ma = short_ma + 1
        st.sidebar.warning('The long-term MA must be greater than the short-term MA. It has been adjusted automatically.')

    data, positions, summary = run_strategy(short_window=short_ma, long_window=long_ma)

    st.title('Trading Strategy Dashboard')
    page = st.sidebar.radio('Navigation', ['Price Chart', 'Trade Statistics', 'Detailed Trades', 'Portfolio Analysis'])

    if page == 'Price Chart':
        st.subheader(f'MSFT Price Trend with MA{short_ma} / MA{long_ma} and MACD')
        st.plotly_chart(build_chart(data, positions, short_window=short_ma, long_window=long_ma), use_container_width=True)

    elif page == 'Trade Statistics':
        st.subheader('Overall Trade Statistics')
        col1, col2, col3 = st.columns(3)
        col1.metric('Total Trades', summary['total_trades'])
        col2.metric('Winning Trades', summary['win_trades'])
        col3.metric('Losing Trades', summary['loss_trades'])

        col4, col5, col6 = st.columns(3)
        col4.metric('Win Rate', f"{summary['win_rate']}%")
        col5.metric('Average Profit', f"{summary['avg_profit_pct']}%")
        col6.metric('Average Holding Days', summary['avg_holding_days'])

        col7, col8, col9 = st.columns(3)
        col7.metric('Total Return', f"{summary['total_return_pct']}%")
        col8.metric('Best Trade', f"{summary['best_trade_pct']}%")
        col9.metric('Worst Trade', f"{summary['worst_trade_pct']}%")

    elif page == 'Detailed Trades':
        st.subheader('Detailed Trade Record')
        trade_df = format_trade_table(positions)
        st.dataframe(trade_df, use_container_width=True)

    else:
        st.subheader('Portfolio Analysis')
        selected_tickers = st.multiselect(
            'Select up to 20 tickers',
            options=DEFAULT_PORTFOLIO_TICKERS,
            default=['AAPL', 'MSFT', 'AMZN', 'NVDA'],
            max_selections=20,
        )

        custom_tickers_input = st.text_input(
            'Or add custom tickers (comma separated)',
            value='',
            help='Examples: AAPL, MSFT, GOOG, TSLA'
        )

        custom_tickers = [
            ticker.strip().upper()
            for ticker in custom_tickers_input.split(',')
            if ticker.strip()
        ]
        combined_tickers = list(dict.fromkeys(selected_tickers + custom_tickers))

        if len(combined_tickers) > 20:
            st.warning('Please select no more than 20 total tickers.')
            combined_tickers = combined_tickers[:20]

        if not combined_tickers:
            st.info('Choose at least one stock ticker to analyse a portfolio.')
        else:
            with st.spinner('Downloading historical data and calculating portfolio metrics...'):
                try:
                    _, _, individual_cumulative, portfolio_cumulative, allocation, summary = calculate_portfolio_metrics(combined_tickers)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

            st.markdown('### Portfolio Summary')
            col1, col2, col3 = st.columns(3)
            col1.metric('Selected Stocks', len(combined_tickers))
            col2.metric('Portfolio Return', f"{summary['portfolio_total_return_pct']}%")
            col3.metric('Avg Daily Volatility', f"{summary['portfolio_volatility_pct']}%")

            st.plotly_chart(build_portfolio_allocation_chart(allocation), use_container_width=True)
            st.plotly_chart(build_portfolio_return_chart(individual_cumulative, portfolio_cumulative), use_container_width=True)


if __name__ == '__main__':
    main()
