from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

FLAGS = None

quantization_bits = 3

quantization_range = 1

def run():
  sess = tf.InteractiveSession()
  G = tf.get_default_graph()

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def quantize(x, k):
    n = float(2 ** k - 1)
    with G.gradient_override_map({"Round": "Identity"}):
      return tf.round(x * n) / n

  # Create a linear vector tensor with a given size
  def create_linear_vector_tensor(min, max, step_size):
    size = (max - min) / step_size
    a = np.zeros(shape=int(size)+1)
    i = min
    j=0
    while j <= size:
      a[j] = float("{0:.6f}".format(i))
      j += 1
      i += step_size

    with tf.name_scope('linear_tensor'):
      linear_tensor = tf.constant(a, tf.float32)
      variable_summaries(linear_tensor)
    return linear_tensor

  def create_random_weights_vector(size, stddev):
    with tf.name_scope('random_tensor'):
      random_tensor = tf.Variable(tf.random_normal([size], stddev=stddev))
      variable_summaries(random_tensor)
    return random_tensor

  def quantize_test(input_tensor):
    with tf.name_scope('quantized_tensor'):
      # Get the max value in the input tensor
      max_val_index = tf.argmax(input_tensor, output_type=tf.int32)
      max_val = sess.run(input_tensor[max_val_index])
      # Get the min value in the input tensor
      min_val_index = tf.argmin(input_tensor, output_type=tf.int32)
      min_val = sess.run(input_tensor[min_val_index])

      # Quantization
      quantized_tensor = tf.fake_quant_with_min_max_args(input_tensor, min_val, max_val, quantization_bits, False, 'quantized_tensor')
      variable_summaries(quantized_tensor)
    return quantized_tensor

  input_array = create_linear_vector_tensor(-.25, .25, 0.001)
  quantized = quantize_test(input_array)

  input_array2 = create_linear_vector_tensor(-0.3, 0.2, 0.001)
  quantized2 = quantize_test(input_array2)

  weights = create_random_weights_vector(50, 0.01)
  input = create_random_weights_vector(50, 0.01)

  max_val_index = tf.argmax(weights, output_type=tf.int32)
  # max_val = sess.run(weights[max_val_index])

  # Merge all the summaries and write them out to
  merged = tf.summary.merge_all()
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  quantized_weights = quantize_test(weights) # This has to be after the variable initializer

  sess.run(input_array)

  summary = sess.run(merged)
  test_writer.add_summary(summary)

  test_writer.close()


  print('Done!')

  plt.plot(sess.run(input_array))
  plt.plot(sess.run(quantized))
  plt.legend(['Input Tensor', 'Quantized Tensor'])
  plt.show()

  plt.plot(sess.run(input_array2))
  plt.plot(sess.run(quantized2))
  plt.legend(['Input Tensor', 'Quantized Tensor'])
  plt.show()

  plt.plot(sess.run(weights))
  plt.plot(sess.run(quantized_weights))
  plt.legend(['Input Weights', 'Quantized Weights'])
  plt.show()


def clean_log_dir(dir):
  if os.path.isdir(dir):
    print("Removing existing log directory ", dir)
    shutil.rmtree(dir)

def main(_):
  clean_log_dir(FLAGS.log_dir)
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  run()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/quant_test/mnist/input_data'),
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/quant_test/test_2_summaries'),
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
