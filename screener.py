# -*- coding: utf-8 -*-
"""
StockScreener clas and utils functions
"""
## IMPORT STOCK DATA ##
from stock import (Stock, filter_sector, filter_price, filter_metric, filter_technical_indicator,
                        get_stock_price, get_historical)
import requests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler

## StockScreener Class ##

class StockScreener:
    def __init__(self, stocks, filters):
        self.stocks = stocks
        self.filters = filters
        self.models = {}
        self.scaler = StandardScaler()


    # Select stocks that pass all filters
    def apply_filters(self):
        filtered_stocks = []
        for stock in self.stocks:
            passed_all_filters = True
            for filter_func in self.filters:
                if not filter_func(stock):
                    passed_all_filters = False
                    break
            if passed_all_filters:
                filtered_stocks.append(stock)
        return filtered_stocks
    
    # Train deep learning models on selected stocks
    def train_models(self):
        
        # Get data for training and testing
        filtered_stocks = self.apply_filters()

        # Create a progress bar and a text placeholder
        progress = 0
        progress_bar = st.progress(0)
        training_models = st.empty()

        for i, stock in enumerate(filtered_stocks):
            training_models.markdown(f"**Training Models... {progress}%**<br>"
                            f"**Processing Ticker {i+1}/{len(filtered_stocks)}: {stock.ticker}**",
                            unsafe_allow_html=True)
            
            train_data = stock.technical_indicators
            train_labels = stock.label
            if len(train_data) == 0:
                continue
            
            # Normalize the data
            train_data = self.scaler.fit_transform(train_data)
            train_labels = np.array(train_labels)

            # Create and train model
            model = create_model(train_data)
            history = model.fit(train_data, train_labels, epochs=35)
            self.models[stock.ticker] = model, history
        
            # Update the progress bar after training
            progress= int((i + 1)/len(filtered_stocks) * 100)
            progress_bar.progress(progress)
    
        progress_bar.empty()
        training_models.empty()
        
        return filtered_stocks
     
    # Predict whether new stocks will pass filters
    def predict_stocks(self, new_stocks):
        # Make predictions for each stock using its corresponding model
        predicted_stocks = []

        # Create a progress bar and a text placeholder
        progress = 0
        progress_bar = st.progress(0)
        predicting_models = st.empty()

        for i, stock in enumerate(new_stocks):
            if stock.ticker in self.models:
                # Show that it is processing the ticker
                predicting_models.markdown(f"**Predicting Stocks... {progress}%**<br>"
                                        f"**Processing Ticker {i+1}/{len(new_stocks)}: {stock.ticker}**",
                                        unsafe_allow_html=True)

                model, _ = self.models[stock.ticker]
                new_features_aux = np.array(stock.today_technical_indicators).reshape(1, -1)
                new_stock_data = self.scaler.fit_transform(new_features_aux)
                prediction = model.predict(new_stock_data)
                stock.prediction = prediction
                if prediction > 0.5:
                    predicted_stocks.append(stock)

                # Update the progress bar after prediction
                progress = int((i + 1) / len(new_stocks) * 100)
                progress_bar.progress(progress)

        progress_bar.empty()
        predicting_models.empty()

        return predicted_stocks
    
    def reset_training(self):
        st.session_state.trained = False

    # Create a web app for the stock screener
    def create_app(self):
        # Initialize session state
        if "trained" not in st.session_state:
            st.session_state.trained = False

        st.title(":grey[🚀NASDAQ 100 STOCK SCREENER 📈]")

        # Create sidebar for filtering options
        sector_list = sorted(list(set(stock.sector for stock in self.stocks)))
        selected_sector = st.sidebar.selectbox("Sector", ["All"] + sector_list, on_change=self.reset_training)

        min_price = st.sidebar.number_input("Min Price", value=0.0, step=0.01, on_change=self.reset_training)
        max_price = st.sidebar.number_input("Max Price", value=1000000.0, step=0.01, on_change=self.reset_training)

        
        metric_list = sorted(list(set(metric for stock in self.stocks for metric in stock.metrics)))
        selected_metric = st.sidebar.selectbox("Metric", ["All"] + metric_list, on_change=self.reset_training)

        metric_operator_list = [">", ">=", "<", "<=", "=="]
        selected_metric_operator = st.sidebar.selectbox("Metric Operator", metric_operator_list, on_change=self.reset_training)

        metric_value = st.sidebar.text_input("Metric Value", "Enter value or the word price", on_change=self.reset_training)
        try:
            metric_value = float(metric_value)
            print(metric_value)
        except:
            pass
        indicator_list = sorted(list(set(indicator for stock in self.stocks for indicator in stock.today_technical_indicators.keys())))
        selected_indicator = st.sidebar.selectbox("Indicator", ["All"] + indicator_list, on_change=self.reset_training)

        indicator_operator_list = [">", ">=", "<", "<=", "=="]
        selected_indicator_operator = st.sidebar.selectbox("Indicator Operator", indicator_operator_list, on_change=self.reset_training)

        indicator_value = st.sidebar.text_input("Indicator Value", "Enter value or the word price", on_change=self.reset_training)
        try:
            indicator_value = float(indicator_value)
            print(indicator_value)
        except:
            pass
        
        # Update filters list with user inputs
        new_filters = []
        if selected_sector != "All":
            new_filters.append(lambda stock: filter_sector(stock, selected_sector))
        if selected_metric != "All":
            new_filters.append(lambda stock: filter_metric(stock, selected_metric, selected_metric_operator, metric_value))
        if selected_indicator != "All":
            new_filters.append(lambda stock: filter_technical_indicator(stock, selected_indicator, selected_indicator_operator, indicator_value))
        new_filters.append(lambda stock: filter_price(stock, min_price, max_price))
        self.filters = new_filters
        
        # Provide feedback to the user instead of writing to the empty container
        if st.sidebar.button("Apply Filters"):
            # with st.spinner(text='Applying filters...'):
            filtered_stocks = self.apply_filters()
            st.success('Filters applied!')
            display_filtered_stocks(filtered_stocks, selected_metric, selected_indicator)

        # Update model training to check if models should be retrained
        if st.sidebar.button("Train and Predict"):
            filtered_stocks = self.train_models()
            predicted_stocks = self.predict_stocks(filtered_stocks)
            display_filtered_stocks(predicted_stocks, selected_metric, selected_indicator, self.models)
            st.success('Training and prediction completed!')


# Simple Dense model 
def create_model(train_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_progress_bar():
    progress = 0
    progress_bar = st.progress(0)
    displaying_stocks = st.empty()
    return progress_bar, displaying_stocks

def display_stock_metrics(stock, tab):
    # Divide the metrics into three lists for the three columns
    metrics = list(stock.metrics.items())
    num_metrics = len(metrics)
    col1_metrics = metrics[:num_metrics//3]
    col2_metrics = metrics[num_metrics//3:(2*num_metrics)//3]
    col3_metrics = metrics[(2*num_metrics)//3:]
    # Create three columns inside each tab
    col1, col2, col3 = tab.columns(3)

    # Display the metrics in the three columns
    for metric, value in col1_metrics:
        col1.metric(metric, value)
    for metric, value in col2_metrics:
        col2.metric(metric, value)
    for metric, value in col3_metrics:
        col3.metric(metric, value)

def plot_stock_data(stock, tab):
    fig, ax = plt.subplots()
    ax.plot(stock.data.index, stock.data["Close"])
    ax.set_title(f"{stock.ticker} Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    tab.pyplot(fig)
    plt.close(fig)

def display_filtered_stocks(filtered_stocks, selected_metric, selected_indicator, models=None):
    progress_bar, displaying_stocks = create_progress_bar()

    num_stocks = len(filtered_stocks)
    if num_stocks == 0:
        st.write("No stocks match the specified criteria.")
    else:
        filtered_tickers = [stock.ticker for stock in filtered_stocks]
        tabs = st.tabs(filtered_tickers)
        for n, stock in enumerate(filtered_stocks):
            display_stock_metrics(stock, tabs[n])
            plot_stock_data(stock, tabs[n])

            # Your existing code to handle models...
            try:
                model, history = models[filtered_stocks[n].ticker]
                
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                # Plot training loss
                ax[0].plot(history.history['loss'])
                ax[0].set_title(f"{filtered_stocks[n].ticker} Training Loss")
                ax[0].set_ylabel("Loss")
                
                # Plot training accuracy
                ax[1].plot(history.history['accuracy'])
                ax[1].set_title(f"{filtered_stocks[n].ticker} Training Accuracy")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("Accuracy")
                
                # Show the plot in the streamlit app
                tabs[n].pyplot(fig) 
                plt.close(fig)          
         
            except:
                tabs[n].write("")

            # Update the progress bar after each stock
            progress = int((n + 1) / num_stocks * 100)
            progress_bar.progress(progress)
            displaying_stocks.markdown(f"**Creating DataFrame... {progress}%**<br>"
                                       f"**Processing Stock {n+1}/{num_stocks}: {filtered_tickers[n]}**",
                                       unsafe_allow_html=True)

    # Display table of filtered stocks info
    table_data = [[s.ticker, s.sector, s.price, s.metrics.get(selected_metric, "N/A"), s.today_technical_indicators.get(selected_indicator, "N/A"), float(s.prediction) if s.prediction != 0 else "N/A"] for s in filtered_stocks]
    table_columns = ["Ticker", "Sector", "Price", f"Metric: {selected_metric}", f"Indicator: {selected_indicator}", "Prediction" if any(s.prediction != 0 for s in filtered_stocks) else ""]
    st.write(pd.DataFrame(table_data, columns=table_columns))

    # Clear the progress elements after loading
    progress_bar.empty()
    displaying_stocks.empty()


## GET STOCK DATA ##
def get_tickers():
    # Get sp500 ticker and sector
    # url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table', {'id': 'constituents'})
    rows = table.find_all('tr')[1:]  # skip the header row
    
    tickers_data = []
    
    for row in rows:
        cells = row.find_all('td')
        ticker = cells[1].text.strip()
        # ticker = ticker.replace('.', '-')
        company = cells[0].text.strip()
        sector = cells[2].text.strip()
        tickers_data.append({'ticker': ticker, 'company': company, 'sector': sector})

    # tickers_data = tickers_data[:10] ## For testing purposes

    return tickers_data


def load_stocks(nasdaq, progress_bar, status_text):
    stocks = []
    for i, stock in enumerate(nasdaq):
        try:
            price = get_stock_price(stock['ticker'])
            data = get_historical(stock['ticker'])
            
            if data is not None and len(data) > 0:
                stocks.append(Stock(stock['ticker'], stock['sector'], price, data))

            # Update the UI with the progress
            progress = int((i + 1) / len(nasdaq) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Loading stock data... {progress}%. Stock no. {i+1}/{len(nasdaq)}: {stock['ticker']}")
        
        except Exception as e:
            st.error(f"There was an issue with {stock['ticker']}: No data found, symbol may be delisted")
    
    # Return the loaded data
    return stocks

# Run screener for all sp500 tickers
@st.cache_resource(ttl=23*3600, show_spinner=False)
def get_stocks(tickers):
    # Create empty placeholder elements for progress and status text
    progress_bar = st.empty()
    status_text = st.empty()

    # Initialize the progress bar to zero
    progress_bar.progress(0)

    # Load the stocks and update the UI progress
    stocks = load_stocks(tickers, progress_bar, status_text)

    # Clear the progress elements after loading
    progress_bar.empty()
    status_text.empty()
    
    return stocks