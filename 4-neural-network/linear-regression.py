import tensorflow as tf

"""
x_data = [[1, 1, 1, 1, 1],
          [0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]
y_data = [1, 2, 3, 4, 5]"""
import numpy as np
xy = np.loadtxt("input/linear-reg-train.txt", unpack=True, dtype="float32")
x_data = xy[0:-1]
y_data = xy[-1]

print(x_data)
print(y_data)

W = tf.Variable(tf.random_uniform([1, 3], -1, 1))
#W1 = tf.Variable(tf.random_uniform([1], -1, 1))
#W2 = tf.Variable(tf.random_uniform([1], -1, 1))
#b = tf.Variable(tf.random_uniform([1], -1, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
#hypothesis = W1 * x1_data + W2 * x2_data + b
#hypothesis = tf.matmul(W, x_data) + b
hypothesis = tf.matmul(W, X)

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
opt = tf.train.GradientDescentOptimizer(a)
train = opt.minimize(cost)

# Before, starting, initialize the variables. We will 'run' this first.
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    #sess.run(train, feed_dict={X1: x1_data, X2: x2_data, Y: y_data})
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 50 == 0:
        #print(step, sess.run(cost, feed_dict={X1: x1_data, X2: x2_data, Y: y_data}), sess.run(W1), sess.run(W2), sess.run(b))
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

#print(sess.run(hypothesis, feed_dict={X: 5}))
#print(sess.run(hypothesis, feed_dict={X: 2.5}))