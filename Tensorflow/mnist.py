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

x = tf.placeholder(tf.float32, shape=[None, 784])

# weights
W = tf.Variable(tf.zeros([784, 10]))

# bias
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

y_true = tf.placeholder(tf.float32, shape=[None, 10])

# minimizing the error
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_true, logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)

train = optimizer.minimize(cross_entropy)

# initializing the global variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(5000):
        batch_x, batch_y =mnist.train.next_batch(100)
        
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
        
    matches = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    
    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))