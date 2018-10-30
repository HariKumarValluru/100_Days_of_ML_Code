# Data Analysis for stock prices

# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from pandas_datareader import data, wb

# setting start date and end date
start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)

# getting the banks dataset from quandl

# Bank of America
BAC = data.DataReader("BAC", 'quandl', start, end).sort_index()

# CitiGroup
C = data.DataReader("C", 'quandl', start, end).sort_index()

# Goldman Sachs
GS = data.DataReader("GS", 'quandl', start, end).sort_index()

# JPMorgan Chase
JPM = data.DataReader("JPM", 'quandl', start, end).sort_index()

# Morgan Stanley
MS = data.DataReader("MS", 'quandl', start, end).sort_index()

# Wells Fargo
WFC = data.DataReader("WFC", 'quandl', start, end).sort_index()

# creating list of tickers
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

# concatenating all the banks data into single dataset
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)

# setting the column name levels
bank_stocks.columns.names = ['Banks Ticker', 'Stock Info']

# max close price for each bank's stock
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()

# creating an empty dataframe for returns
returns = pd.DataFrame()

for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].astype(float).pct_change()

# creating a pairplot
sns.pairplot(returns[1:])