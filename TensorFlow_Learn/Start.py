# below 2 lines remove warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# initializing 0-d tensors w/types, printing type
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) #tf.float32 implicitly
print(node1, node2)

# runing session. must be init beforehand
print(sess.run([node1, node2]))

node3 = tf.add(node1,node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

# placeholders, promising values for later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # same as tf.add(a,b)

print(sess.run(adder_node, {a:3, b:4.5}))
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a:3, b:4.5}))

# variables. Unlike constants, not initialized at the moment
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# variables are only initialized when:
init = tf.global_variables_initializer()
sess.run(init) # this init variables

# now feed linear model w/ placeholder x and run sess
print(sess.run(linear_model, {x:[1,2,3,4]}))

# loss function, sum of square differences
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas) # fxn for sum
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# guess & change variable values for better results
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb]) # sess must be run to actually change the vars
print(sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]}))

# tf.train API
# now actual ML using gradient descent, and training
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset variable to orig values
for i in range(1000):
    sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})

curr_W, curr_b, curr_loss = sess.run([W,b,loss],{x:[1,2,3,4], y:[0,-1,-2,-3]})
print("W: %s b: %s loss: %s"%(curr_W,curr_b,curr_loss))
