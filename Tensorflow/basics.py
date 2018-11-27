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
sess.run(x)

type(sess.run(x))

# Operations
x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print("Operations with constants")
    print("Addition: ", sess.run(x+y))
    print("Substraction: ", sess.run(x-y))
    print("Multiplication: ", sess.run(x*y))
    print("Division: ", sess.run(x/y))
    
# placeholders
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

print(x)

add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)

d = {x: 20, y:30}

with tf.Session() as sess:
    print("Operations with placeholders")
    print("Addition: ", sess.run(add, feed_dict=d))
    print("Substraction: ", sess.run(sub, feed_dict=d))
    print("Multiplication: ", sess.run(mul, feed_dict=d))

import numpy as np
a = np.array([[5., 5.]])
b = np.array([[2.],[2.]])

print(a.shape, b.shape)

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)