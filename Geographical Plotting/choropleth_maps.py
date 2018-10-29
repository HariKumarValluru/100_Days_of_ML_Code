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

