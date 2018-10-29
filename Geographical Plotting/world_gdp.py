# 2014 World GDP

# Importing the libraries
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, iplot, plot, init_notebook_mode

init_notebook_mode(connected=True)

# loading the dataset
dataset = pd.read_csv("2014_World_GDP.csv")

data = {'type': 'choropleth',
        'locations': dataset['CODE'],
        'z': dataset['GDP (BILLIONS)'],
        'text': dataset['COUNTRY'],
        'colorbar': {'title': 'GDP in Billions USD'}
        }

layout = {'title': '2014 Global GDP', 
          'geo': {'showframe': False,
                  'projection': {'type': 'natural earth'}
                  }
        }

choroMap = go.Figure(data = [data], layout = layout)

plot(choroMap)