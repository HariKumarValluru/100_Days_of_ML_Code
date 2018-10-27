# Basic visualization using matplotlib

# Importing the numpy library
import numpy as np
x = np.arange(0, 100)
y = x*2
z = x**2

# Importing the matplotlib library for visualising the data
import matplotlib.pyplot as plt

# Create plot in a functional way
plt.plot(x,y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Basic Visualization')

# create plot in an object oriented way
# creating the figure object
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Basic Visualization')