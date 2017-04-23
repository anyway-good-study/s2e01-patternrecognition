import tensorflow as tf
import numpy as np


xy = np.loadtxt("input/logistic-reg-train.txt", unpack=True, dtype="float32")
x_data = xy[0:-1]
y_data = xy[-1]

print(x_data)
#print(y_data)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# Cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
opt = tf.train.GradientDescentOptimizer(a)
train = opt.minimize(cost)

# Before, starting, initialize the variables. We will 'run' this first.
init = tf.global_variables_initializer()
#init = tf.initialize_all_variables() # It will be deprecated.

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 50 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print("=== Prediction ===")
print(sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}))
print(sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}))
print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}))