import pandas as pd
import streamlit as st
import altair as alt

from trading_strategy import run_strategy

st.set_page_config(page_title='Trading Strategy Dashboard', layout='wide')


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


def build_chart(data, positions):
    chart_df = data[['Close', 'MA50', 'MA200']].copy()
    chart_df.index = chart_df.index.strftime('%Y-%m-%d')
    chart_df = chart_df.reset_index().rename(columns={'index': 'Date'})

    buy_points = positions[['BuyDate', 'BuyPrice']].copy()
    buy_points.columns = ['Date', 'Price']
    buy_points['Date'] = pd.to_datetime(buy_points['Date']).dt.strftime('%Y-%m-%d')

    sell_points = positions[['SellDate', 'SellPrice']].copy()
    sell_points.columns = ['Date', 'Price']
    sell_points['Date'] = pd.to_datetime(sell_points['Date']).dt.strftime('%Y-%m-%d')

    line_chart = alt.Chart(chart_df).mark_line().encode(
        x='Date:T',
        y='Close:Q',
        color=alt.value('#1f77b4')
    ).properties(width=1200, height=650)

    ma50 = alt.Chart(chart_df).mark_line(stroke='#ff7f0e', strokeWidth=2).encode(
        x='Date:T',
        y='MA50:Q'
    )

    ma200 = alt.Chart(chart_df).mark_line(stroke='#2ca02c', strokeWidth=2).encode(
        x='Date:T',
        y='MA200:Q'
    )

    buy = alt.Chart(buy_points).mark_circle(size=80, color='green').encode(
        x='Date:T',
        y='Price:Q'
    )

    sell = alt.Chart(sell_points).mark_circle(size=80, color='red').encode(
        x='Date:T',
        y='Price:Q'
    )

    return (line_chart + ma50 + ma200 + buy + sell).interactive()


def main():
    data, positions, summary = run_strategy()

    st.title('Trading Strategy Dashboard')
    page = st.sidebar.radio('Navigation', ['Price Chart', 'Trade Statistics', 'Detailed Trades'])

    if page == 'Price Chart':
        st.subheader('MSFT Price Trend with Moving Averages and Trade Points')
        st.altair_chart(build_chart(data, positions), use_container_width=True)

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

    else:
        st.subheader('Detailed Trade Record')
        trade_df = format_trade_table(positions)
        st.dataframe(trade_df, use_container_width=True)


if __name__ == '__main__':
    main()
