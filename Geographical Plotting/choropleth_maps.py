# Choropleth Maps with Plotly

# Importing the libraries
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, iplot, plot, init_notebook_mode

init_notebook_mode(connected=True)

data = dict(type='choropleth',
            locations=['AZ','CA','NY'], 
            locationmode='USA-states',
            colorscale='Portland',
            text=['text 1','text 2', 'text 3'],
            z=[1.0,2.0,3.0],
            colorbar={'title':'Colorbar'})

layout = dict(geo={'scope':'usa'})

choroMap = go.Figure(data = [data], layout=layout)

plot(choroMap)

# Importing pandas
import pandas as pd

# Loading dataset
dataset = pd.read_csv("2011_US_AGRI_EXPORTS.csv")
data = dict(type = 'choropleth',
        colorscale = 'YlOrRd',
        locations = dataset['code'],
        locationmode = 'USA-states',
        z = dataset['total exports'],
        text = dataset['text'],
        marker = dict(
                line = {
                        'color' : 'rgb(255,255,255)',
                        'width': 2
                        }
                ),
        colorbar = {'title' : 'Millions USD'}
        )
layout = dict(title = '2011 US Agriculture Exports by State',
              geo={'scope':'usa','showlakes':True, 
                   'lakecolor': 'rgb(85,173,214)'})
choroMap2 = go.Figure(data = [data], layout=layout)
plot(choroMap2)
