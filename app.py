"""
Stock Screener Main Class
"""

## IMPORT UTILS FUNCTIONS ##
from screener import StockScreener, get_tickers, get_stocks
import streamlit as st



if __name__=='__main__':
    
    # Streamlit Config 
    st.set_page_config(page_title="Stock Screener", page_icon=":chart_with_upwards_trend:")

    filters = []
    
    # Get sp500 tickers and sectors
    nasdaq = get_tickers()
    # Create Stock objects
    # Check the session state before loading data
    if 'stocks' not in st.session_state:
        st.session_state.nasdaq_stocks = get_stocks(nasdaq)

    # Use session state within the app
    screener = StockScreener(st.session_state.nasdaq_stocks, filters)

    # Create streamlit app
    screener.create_app()
