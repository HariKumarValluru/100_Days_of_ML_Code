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

# create fig object with two axes on it
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([.2,.5,.2,.2])

# ploting x,y on both axes
ax1.plot(x,y, color = 'red')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax2.plot(x,y, color = 'green')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([.2,.5,.4,.4])

# Large Plot
ax1.plot(x,z)
ax1.set_xlabel('X')
ax1.set_ylabel('Z')

# Insert
ax2.plot(x,y)
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.set_title('Zoom')
ax2.set_xlim([20,22])
ax2.set_ylim([30,50])

# create subplots
fig, axes = plt.subplots(1,2, figsize=(12,2))
axes[0].plot(x,y, color='blue', linestyle='--', linewidth=3)
axes[1].plot(x,z, color='red', linewidth=3)
