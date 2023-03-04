# Stock Screener
This is a stock screener that allows users to filter through a list of stocks based on various criteria such as sector, price, metric, and technical indicators.

## Features
1. Users can filter by sector, price range, metric, technical indicators and predictions
2. The app displays a list of stocks that match the user's criteria
3. Users can view additional information on a stock, such as its price and sector, by clicking on its ticker symbol
4. Users can view a chart of a stock's price history by clicking on its ticker symbol
5. Users can view charts of training metrics for each ticker symbol

Deep Learning Objective:
- Predict if the price will go up in the next 10 days

## Requirements
Python 3.7 or higher
Required packages are listed in the **requirements.txt** file

## Installation
1. Clone the repository
2. Navigate to the project directory
3. Install the required packages by running ```pip install -r requirements.txt```
4. Run the app by running ```streamlit run app.py```

## Usage
1. Open the app by running ```streamlit run app.py```
2. Use the sidebar on the left to select the filters you wish to apply
3. Click the "Apply Filters" button to apply the filters and display the list of filtered stocks
4. Use the "Train and Predict" button to train the model and apply the filters including prediction filter
5. Click on a stock's ticker symbol to view additional information or a chart of its price history


## Contributions
Contributions are welcome! Please create a pull request for any changes you would like to make.
