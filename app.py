"""
Stock Screener Main Class
"""

## IMPORT UTILS FUNCTIONS ##
from screener import StockScreener, get_sp_tickers, get_sp500_stocks
import streamlit as st
    


   
# # Run example with 2 stocks
# filters = [lambda stock: filter_sector(stock, 'Interactive Media & Services'),
#           lambda stock: filter_price(stock, 60, 200),
#           lambda stock: filter_metric(stock, 'profit_margin', '>', 15),
#           lambda stock: filter_technical_indicator(stock, 'UpperBand', '>', 'price'),
#           lambda stock: filter_technical_indicator(stock, 'LowerBand', '<', 'price'),
# ]

# price1 = get_stock_price('GOOGL')
# data1 = get_historical('GOOGL')
# price2 = get_stock_price('GOOG')
# data2 = get_historical('GOOG')

# sp500_stocks = [Stock('GOOGL', 'Interactive Media & Services', price1, data1), 
#                 Stock('GOOG', 'Interactive Media & Services', price2, data2)
# ]

# # Screener
# screener = StockScreener(sp500_stocks, filters)

# # Create streamlit app
# screener.create_app()





if __name__=='__main__':
    
    # Streamlit Config 
    st.set_page_config(page_title="Stock Screener", page_icon=":chart_with_upwards_trend:")

    filters = []
    
    sp500 = get_sp_tickers()
    # Get sp500 tickers and sectors
    sp500_stocks = get_sp500_stocks(sp500)
    # Screener
    screener = StockScreener(sp500_stocks, filters)

    # Create streamlit app
    screener.create_app()
