import tensorflow as tf

# Initialize weight variable as random tensor with stddev 0.1
def weight_variable(shape, std_dev=0.1):
  initial = tf.truncated_normal(shape, stddev=std_dev)
  return tf.Variable(initial)

# Initialize bias as constant 0.01
def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)