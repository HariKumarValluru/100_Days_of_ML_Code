# Tensorflow basics

# importing tensorflow
import tensorflow as tf

# creating a constant
hello = tf.constant("Hello World")
type(hello)

x = tf.constant(100)
type(x)

# creating a tensorflow session
sess = tf.Session()

sess.run(hello)