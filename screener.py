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
    
    # @st.cache_data
    # Train deep learning models on selected stocks
    def train_models(self):
        training_models = st.empty()
        training_models.write("Training Model for each Ticker...")
        # Get data for training and testing
        filtered_stocks = self.apply_filters()

        for stock in filtered_stocks:
            train_data = stock.technical_indicators
            train_labels = stock.label
            if len(train_data) == 0:
                continue
            
            # Normalize the data
            train_data = self.scaler.fit_transform(train_data)
            train_labels = np.array(train_labels)

            # Create and train model
            model = create_model(train_data)
            history = model.fit(train_data, train_labels, epochs=100)
            self.models[stock.ticker] = model, history
        
        training_models.empty()
        
        return filtered_stocks
     
    # @st.cache_data
    # Predict whether new stocks will pass filters
    def predict_stocks(self, new_stocks):

        # Make predictions for each stock using its corresponding model
        predicted_stocks = []
        for stock in new_stocks:
            if stock.ticker in self.models:
                model, _ = self.models[stock.ticker]
                new_features_aux = np.array(stock.today_technical_indicators).reshape(1, -1)
                new_stock_data = self.scaler.fit_transform(new_features_aux)
                prediction = model.predict(new_stock_data)
                stock.prediction = prediction
                if prediction > 0.5:
                    predicted_stocks.append(stock)

        return predicted_stocks
    

    # Create a web app for the stock screener
    def create_app(self):

        st.title(":grey[STOCK SCREENER]")


        # Create sidebar for filtering options
        sector_list = sorted(list(set(stock.sector for stock in self.stocks)))
        selected_sector = st.sidebar.selectbox("Sector", ["All"] + sector_list)

        min_price = st.sidebar.number_input("Min Price", value=0.0, step=0.01)
        max_price = st.sidebar.number_input("Max Price", value=1000000.0, step=0.01)

        
        metric_list = sorted(list(set(metric for stock in self.stocks for metric in stock.metrics)))
        selected_metric = st.sidebar.selectbox("Metric", ["All"] + metric_list)

        metric_operator_list = [">", ">=", "<", "<=", "=="]
        selected_metric_operator = st.sidebar.selectbox("Metric Operator", metric_operator_list)

        metric_value = st.sidebar.text_input("Metric Value", "Enter value or the word price")
        try:
            metric_value = float(metric_value)
            print(metric_value)
        except:
            pass
        indicator_list = sorted(list(set(indicator for stock in self.stocks for indicator in stock.today_technical_indicators.keys())))
        selected_indicator = st.sidebar.selectbox("Indicator", ["All"] + indicator_list)

        indicator_operator_list = [">", ">=", "<", "<=", "=="]
        selected_indicator_operator = st.sidebar.selectbox("Indicator Operator", indicator_operator_list)

        indicator_value = st.sidebar.text_input("Indicator Value", "Enter value or the word price")
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
        
        # Create "Apply Filters" button
        if st.sidebar.button("Apply Filters"):

            # Apply filters
            filtered_stocks = self.apply_filters()
            
            # Display visualizations for filtered stocks
            display_filtered_stocks(filtered_stocks, selected_metric, selected_indicator)
        
        # Create "Train and Predict Models" button
        if st.sidebar.button("Train and Predict Model"):
            # Train models for each filtered stock
            filtered_stocks = self.train_models()
            # Predict models
            predicted_stocks = self.predict_stocks(filtered_stocks)
            
            # Display visualizations for filtered stocks
            display_filtered_stocks(predicted_stocks, selected_metric, selected_indicator, self.models)




# Simple Dense model 
def create_model(train_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def display_filtered_stocks(filtered_stocks, selected_metric, selected_indicator, models=None):
    # Display filtered stocks
    if len(filtered_stocks) == 0:
        st.write("No stocks match the specified criteria after Predicting.")
    else:
        filtered_tickers = [stock.ticker for stock in filtered_stocks]
        tabs = st.tabs(filtered_tickers)
        for n in range(len(tabs)):
            # Divide the metrics into three lists for the three columns
            metrics = list(filtered_stocks[n].metrics.items())
            num_metrics = len(metrics)
            col1_metrics = metrics[:num_metrics//3]
            col2_metrics = metrics[num_metrics//3:(2*num_metrics)//3]
            col3_metrics = metrics[(2*num_metrics)//3:]
            # Create three columns inside each tab
            col1, col2, col3 = tabs[n].columns(3)
         
            # Display the metrics in the three columns
            for metric, value in col1_metrics:
                col1.metric(metric, value)
            for metric, value in col2_metrics:
                col2.metric(metric, value)
            for metric, value in col3_metrics:
                col3.metric(metric, value)
         
            fig, ax = plt.subplots()
            ax.plot(filtered_stocks[n].data.index, filtered_stocks[n].data["Close"])
            ax.set_title(f"{filtered_stocks[n].ticker} Close Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            tabs[n].pyplot(fig)                
    
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
         
            except:
                tabs[n].write("")
                
    # Display table of filtered stocks info
    table_data = [[s.ticker, s.sector, s.price, s.metrics.get(selected_metric, "N/A"), s.today_technical_indicators.get(selected_indicator, "N/A"), float(s.prediction) if s.prediction != 0 else "N/A"] for s in filtered_stocks]
    table_columns = ["Ticker", "Sector", "Price", f"Metric: {selected_metric}", f"Indicator: {selected_indicator}", "Prediction" if any(s.prediction != 0 for s in filtered_stocks) else ""]
    st.write(pd.DataFrame(table_data, columns=table_columns))


## GET SP 500 STOCK DATA ##

def get_sp_tickers():
    # Get sp500 ticker and sector
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table', {'class': 'wikitable sortable'})
    rows = table.find_all('tr')[1:]  # skip the header row
    
    sp500 = []
    
    for row in rows:
        cells = row.find_all('td')
        ticker = cells[0].text.strip()
        company = cells[1].text.strip()
        sector = cells[3].text.strip()
        sp500.append({'ticker': ticker, 'company': company, 'sector': sector})

    return sp500

@st.cache_data
# Run screener for all sp500 tickers
def get_sp500_stocks(sp500):
    
    sp500_stocks = []
    # Streamlit Text
    stock_download = st.empty()
    stock_issues = st.empty()
    # Create Stock object for every stock with data
    for stock in sp500:
        stock_download.write(f"Downloading {stock['ticker']} Data")
        try:
            price = get_stock_price(stock['ticker'])
            data = get_historical(stock['ticker'])
            sp500_stocks.append(Stock(stock['ticker'], stock['sector'], price, data))
            stock_download.empty()
        except:
            stock_issues.write(f"There was an issue with {stock['ticker']}. ")
    
    stock_issues.empty()
    
    return sp500_stocks