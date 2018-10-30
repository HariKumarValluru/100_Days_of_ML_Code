# Data Analysis for stock prices

# Importing the libraries
import pandas as pd
import numpy as np
import datetime
from pandas_datareader import data, wb

# setting start date and end date
start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)

# getting the banks dataset from robinhood

# Bank of America
BAC = data.DataReader("BAC", 'robinhood', start, end)

# CitiGroup
C = data.DataReader("C", 'robinhood', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'robinhood', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'robinhood', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'robinhood', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'robinhood', start, end)