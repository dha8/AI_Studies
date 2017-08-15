import tensorflow as tf
# numpy used to load, manip, preprocess data
import numpy as np

# Declare list of features. one real-valued feature
features = [tf.contrib.layers.real_valued_column("x",dimension=1)]

# An estimator : invokes training & evaluation
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# set up data sets. one for training, one for evaluation
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=4,
                                              num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# train 1000 steps
estimator.fit(input_fn=input_fn, steps=1000)

# evaluate performance of model
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
