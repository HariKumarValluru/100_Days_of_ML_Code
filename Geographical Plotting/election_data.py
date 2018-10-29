# 2012 Election Data

# impoting the libraries
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, iplot, plot, init_notebook_mode

# loading the dataset
dataset = pd.read_csv("2012_Election_Data.csv")

data = {'type': 'choropleth',
        'colorscale': 'Viridis',
        'reversescale': True,
        'locations': dataset['State Abv'],
        'z': dataset['Voting-Age Population (VAP)'],
        'locationmode': 'USA-states',
        'text': dataset['State'],
        'colorbar': {'title': 'Voting-Age Population'}
        }

layout = {'title': '2012 Election Data',
          'geo': {'scope':'usa',
                  'showlakes': True,
                  'lakecolor': 'rgb(85,173,240)'}
          }
          
choroMap = go.Figure(data = [data], layout = layout)

plot(choroMap)