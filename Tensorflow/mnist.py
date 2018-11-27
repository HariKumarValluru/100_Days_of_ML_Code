# classify what number is written down with tensorflow
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# reading data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape)
print(mnist.test.images.shape)

import matplotlib.pyplot as plt
print(mnist.train.images)

print(mnist.train.images[1].shape)

# reshaping the image and plotting
plt.imshow(mnist.train.images[1].reshape(28, 28))

print(mnist.train.images[1].reshape(28, 28))

# gray image
plt.imshow(mnist.train.images[1].reshape(28, 28), cmap="gist_gray")