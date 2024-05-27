from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import os
from keras.models import load_model
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import datetime
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(layout="wide")
st.header('Stock price prediction', divider='rainbow')
ticker=st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "SBIN.NS")
Stock = yf.Ticker(ticker)
PredData= Stock.history(interval="1mo",start = '2010-01-03',end='2024-01-03')


dfSet=PredData.copy()
dfSet1=dfSet.copy()
dfSet1['Difference'] = dfSet1['Close'].shift(-1) - dfSet1['Close']

dfSet1 = dfSet1.drop(dfSet1.index[-1])
dfSet1 = dfSet1[['Difference']]

dfset_train = dfSet1.iloc[0:int(0.8 * len(dfSet1)), :]
dfset_test = dfSet1.iloc[int(0.8 * len(dfSet1)):, :]

# Feature Scaling


training_set = dfSet1.iloc[:, 0:1].values  # Extracting the "Difference" column as numpy array

# Scaling the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating the data structure with 7 timesteps and 1 output
X_train = []  # Memory with 7 days from day i
y_train = []  # Day i
for i in range(30, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - 30:i, 0])
    y_train.append(training_set_scaled[i, 0])

# Convert lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping: Adding 3rd dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Now, to integrate the forecasting part
X_forecast = np.array(X_train[-1, 1:])  # Selecting the last 6 values from X_train
X_forecast = np.append(X_forecast, y_train[-1])  # Appending the last actual y_train value
X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))  # Reshaping for LSTM input

# Now, X_train and y_train are prepared for training, and X_forecast is prepared for forecasting.


load_existing_model = True  # Define the variable load_existing_model
save_model = True  # Define the variable save_model

if load_existing_model and os.path.exists(Stock.ticker + "_lstm_model.h5"):
    regressor = load_model(Stock.ticker + "_lstm_model.h5")
else:
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=200, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=200, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=50, batch_size=8)

    if save_model:
        regressor.save(Stock.ticker + "_lstm_model.h5")




# Testing

real_stock_price = dfset_test.iloc[:, 0:1].values

# To predict, we need stock prices of 7 days before the test set
# So combine train and test set to get the entire df set
dfset_total = pd.concat((dfset_train, dfset_test), axis=0)
testing_set = dfset_total[len(dfset_total) - len(dfset_test) - 30:].values
testing_set = testing_set.reshape(-1, 1)
# -1=till last row, (-1,1)=>(80,1). otherwise only (80,0)

# Feature scaling
testing_set = sc.transform(testing_set)

# Create df structure
X_test = []
for i in range(30, len(testing_set)):
    X_test.append(testing_set[i - 30:i, 0])
# Convert list to numpy arrays
X_test = np.array(X_test)

# Reshaping: Adding 3rd dimension
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Testing Prediction
predicted_stock_price = regressor.predict(X_test)

# Getting original prices back from scaled values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Forecasting Prediction
forecasted_stock_price = regressor.predict(X_forecast)

# Getting original prices back from scaled values
forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

lstm_pred = forecasted_stock_price[0, 0]

latest_closed_price = dfSet.iloc[-1]['Close']
print("Latest closed price:", latest_closed_price)

final_forecast= latest_closed_price + lstm_pred


error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
final_forecast= round(final_forecast, 3)
error_lstm = round(error_lstm, 2)

print("##############################################################################")


# print("Next weeks final prediction is: ")
# print(final_forecast)
# print("RSME : " + str(error_lstm))
# PredData= Stock.history(interval="1mo",start = '2010-01-03',end='2024-02-03')
# PredData
st.subheader('Next months final prediction is: ' + '   '+'"'+str(final_forecast)+'"')





########################################################################################################################3

def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

@st.cache_data
def load_dataset(ticker, start_date, end_date):
    stock_df = fetch_data(ticker, start_date, end_date)
    stock_df["BarColor"] = stock_df.apply(lambda row: "red" if row["Open"] > row["Close"] else "green", axis=1)
    stock_df["Date_str"] = stock_df.index.strftime("%Y-%m-%d")
    return stock_df

# Function to handle period logic, including "Custom"
def apply_period(selected_period, start_date, end_date):
    if selected_period != "Custom":
        timedelta_value = get_timedelta(selected_period)
        if timedelta_value is not None:
            start_date = datetime.now() - timedelta_value
        else:  # "Max" selected
            start_date = None
    return start_date

# Function to convert period to timedelta
def get_timedelta(period):
    if period == "1 Month":
        return timedelta(days=30)
    elif period == "3 Months":
        return timedelta(days=90)
    elif period == "1 Year":
        return timedelta(days=365)
    elif period == "5 Years":
        return timedelta(days=365*5)
    else:  # "Max" or "Custom"
        return None

# # Dashboard title and user input for stock ticker, start date, and end date
# st.title("Line Graph App with Date Range")

#ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")

# Option to select start date and end date
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Default period set to "1 Year"
selected_period = st.sidebar.radio(
    "Select Period of Visualization",
    ["1 Month", "3 Months", "1 Year", "5 Years", "Max", "Custom"],
    index=2  # Set default index to "1 Year"
)

# Apply period logic
start_date = apply_period(selected_period, start_date, end_date)

# Load dataset based on user input and selected period
end_date = end_date if selected_period != "Custom" else datetime.now()
apple_df = load_dataset(ticker, start_date, end_date)

# Chart creation function (modified for line graph)
def create_chart(df):
    source = ColumnDataSource(data=df)  # Create a ColumnDataSource

    line_graph = figure(x_axis_type="datetime", height=500, x_range=(min(df.index), max(df.index)))
    line_graph.line("Date", "Close", source=source, line_width=2, color="blue", legend_label="Close Price")

    hover_tool = HoverTool(tooltips=[("Date", "@Date_str"), ("Close", "@Close{0.00}")])  # Adjust tooltip formatting
    line_graph.add_tools(hover_tool)

    return line_graph

# Display the chart
chart = create_chart(apple_df)
st.bokeh_chart(chart, use_container_width=True)

#######################################################################################################################

fundamental_data , pricing_data, news   = st.tabs(["fundamental data" , "pricing data","Top 10 News" ])

with pricing_data: 
    st.write("price")
    # Display data description and last eight rows side by side
    st.write("### Data Description and Last Eight Rows")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Data Description")
        st.write(apple_df.describe())
    with col2:
        st.write("### Last Eight Rows")
        st.write(apple_df.tail(8))

with fundamental_data:
    st.header("fundamental data") 
    # Fetch information for the specified ticker
    stock = yf.Ticker(ticker)

    # Display general information about the ticker
    st.subheader("financials")
    st.write(stock.financials)
    st.subheader("balancesheet")
    st.write(stock.balancesheet)
    st.subheader("cashflow")
    st.write(stock.cashflow )
    # st.write(actions)

from stocknews import StockNews
with news:
    st.write("news") 
    st.header(f'News of {ticker}') 
    sn=StockNews(ticker,save_news=False)
    df_news = sn.read_rss()
    ticker2 = ticker.split('.')[0].upper()
    exchange = 'NSE'
    url = f'https://www.google.com/finance/quote/{ticker2}:{exchange}'
    try:
        st.subheader('news')
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        news1 = soup.find(class_='Yfwt5').text
        st.write(news1)
    except:
        st.write('')


    for i in range(10):
        #st.subheader(f'News {i+1}')
        st.subheader('news')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
