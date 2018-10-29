# 2014 World Power Consumption

# impoting the libraries
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, iplot, plot, init_notebook_mode

# loading the dataset
dataset = pd.read_csv('2014_World_Power_Consumption.csv')

data = {'type': 'choropleth',
        'locations': dataset['Country'],
        'locationmode': 'country names',
        'colorscale': 'Viridis',
        'reversescale': True,
        'z': dataset['Power Consumption KWH'],
        'text': dataset['Country'],
        'colorbar': {'title': 'Power Consumption KWH'}
        }

layout = {'title': 'Power Consumption KWH',
          'geo': {'showframe': False, 
                  'projection': {'type':'mercator'}
                  }
          }

choromap = go.Figure(data = [data], layout = layout)

plot(choromap)