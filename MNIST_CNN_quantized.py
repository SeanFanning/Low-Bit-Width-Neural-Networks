# Based on tensorflows mnist tutorial for TensorBoard
# Modified to add 2 conv layers and quantization

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from src.custom_convolution_layers import conv_layer_quantized_add_noise, conv_layer_quantized, conv_layer
from src.custom_fully_connected_layers import fc_layer_quantized_add_noise, fc_layer_quantized, fc_layer

FLAGS = None

num_layers = 2  # Set the number of Fully Connected Layers
quantization_enabled = False

noise_stddev = 0.05
noise_enabled_fc = False
noise_enabled_conv = False

# Conv 1
conv1_w_bits = 3
conv1_w_min = -0.3
conv1_w_max = 0.3
conv1_b_bits = 3
conv1_b_min = -0.3
conv1_b_max = 0.3
conv1_a_bits = 3
conv1_a_min = -8
conv1_a_max = 8

# Conv 2
conv2_w_bits = 3
conv2_w_min = -0.3
conv2_w_max = 0.3
conv2_b_bits = 3
conv2_b_min = -0.3
conv2_b_max = 0.3
conv2_a_bits = 3
conv2_a_min = -8
conv2_a_max = 8

# Fully Connected 1
fc1_depth = 250
fc1_w_bits = 3
fc1_w_min = -0.3
fc1_w_max = 0.3
fc1_b_bits = 3
fc1_b_min = -0.3
fc1_b_max = 0.3
fc1_a_bits = 3
fc1_a_min = -8
fc1_a_max = 8

# Fully Connected 2 (OUTPUT)
fc2_depth = 10
fc2_w_bits = 3
fc2_w_min = -0.3
fc2_w_max = 0.3
fc2_b_bits = 3
fc2_b_min = -0.3
fc2_b_max = 0.3
fc2_a_bits = 3
fc2_a_min = -8
fc2_a_max = 8

# Fully Connected 3 (MIDDLE)
fc3_depth = 250
fc3_w_bits = 3
fc3_w_min = -0.3
fc3_w_max = 0.3
fc3_b_bits = 3
fc3_b_min = -0.3
fc3_b_max = 0.3
fc3_a_bits = 3
fc3_a_min = -8
fc3_a_max = 8


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  def create_fc_layer(input_tensor, input_dim, output_dim, bits_w, max_w, bits_b, max_b, bits_a, max_a, noise_stddev, layer_name, act=tf.nn.relu):
    if(quantization_enabled == False):
      return fc_layer(input_tensor, input_dim, output_dim, layer_name, act)
    elif(noise_enabled_fc == False):
      return fc_layer_quantized(input_tensor, input_dim, output_dim, bits_w, max_w, bits_b, max_b, bits_a, max_a, layer_name, act)
    else:
      return fc_layer_quantized_add_noise(input_tensor, input_dim, output_dim, bits_w, max_w, bits_b, max_b, bits_a, max_a, noise_stddev, layer_name, act)

  def create_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, bits_w, max_w, bits_b, max_b, bits_a, max_a, noise_stddev, layer_name):
    if(quantization_enabled == False):
      return conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, layer_name)
    elif(noise_enabled_conv == False):
      return conv_layer_quantized(input_data, num_input_channels, num_filters, filter_shape, pool_shape, bits_w, max_w, bits_b, max_b, bits_a, max_a, layer_name)
    else:
      return conv_layer_quantized_add_noise(input_data, num_input_channels, num_filters, filter_shape, pool_shape, bits_w, max_w, bits_b, max_b, bits_a, max_a, noise_stddev, layer_name)





  layer1 = create_conv_layer(image_shaped_input, 1, 32, [5, 5], [2, 2], conv1_w_bits, conv1_w_max, conv1_b_bits, conv1_b_max, conv1_a_bits, conv1_a_max, layer_name='conv1')
  layer2 = create_conv_layer(layer1, 32, 64, [5, 5], [2, 2], conv2_w_bits, conv2_w_max, conv2_b_bits, conv2_b_max, conv2_a_bits, conv2_a_max, layer_name='conv2')


  with tf.name_scope('flatten'):
    x_flattened = tf.reshape(layer2, [-1, 7*7*64])

  if(num_layers == 3):
    hidden1 = create_fc_layer(x_flattened, 7*7*64, fc1_depth, fc1_w_bits, fc1_w_max, fc1_b_bits, fc1_b_max, fc1_a_bits, fc1_a_max, noise_stddev, 'fully_connected1')

    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.summary.scalar('dropout_keep_probability', keep_prob)
      dropped = tf.nn.dropout(hidden1, keep_prob)

    # Middle Layer (using settings 3)
    hidden2 = create_fc_layer(dropped, fc1_depth, fc3_depth, fc3_w_bits, fc3_w_max, fc3_b_bits, fc3_b_max, fc3_a_bits, fc3_a_max, noise_stddev, 'fully_connected_middle')

    with tf.name_scope('dropout2'):
      keep_prob2 = tf.placeholder(tf.float32)
      tf.summary.scalar('dropout2_keep_probability', keep_prob2)
      dropped2 = tf.nn.dropout(hidden2, keep_prob2)

    # Do not apply softmax activation yet, see below.
    y = create_fc_layer(dropped2, fc3_depth, fc2_depth, fc2_w_bits, fc2_w_max, fc2_b_bits, fc2_b_max, fc2_a_bits, fc2_a_max, noise_stddev, 'fully_connected2', act=tf.identity)

  elif(num_layers == 1):
    keep_prob = tf.placeholder(tf.float32)  # Need to create placeholders
    keep_prob2 = tf.placeholder(tf.float32) # Even though they wont be used
    y = create_fc_layer(x_flattened, 7*7*64, 10, fc2_w_bits, fc2_w_max, fc2_b_bits, fc2_b_max, fc2_a_bits, fc2_a_max, noise_stddev, layer_name='fully_connected', act=tf.identity)

  else: # Otherwise create 2 layers
    hidden1 = create_fc_layer(x_flattened, 7*7*64, fc1_depth, fc1_w_bits, fc1_w_max, fc1_b_bits, fc1_b_max, fc1_a_bits, fc1_a_max, noise_stddev, 'fully_connected1')

    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.summary.scalar('dropout_keep_probability', keep_prob)
      dropped = tf.nn.dropout(hidden1, keep_prob)

    keep_prob2 = tf.placeholder(tf.float32)  # Need to create a placholder

    # Do not apply softmax activation yet, see below.
    y = create_fc_layer(dropped, fc1_depth, fc2_depth, fc2_w_bits, fc2_w_max, fc2_b_bits, fc2_b_max, fc2_a_bits, fc2_a_max, noise_stddev, 'fully_connected2', act=tf.identity)




  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          labels=y_, logits=y)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(64, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k, keep_prob2: k}


  for i in range(FLAGS.max_steps):
    if FLAGS.load == True:
      break
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

  # # Add ops to save and restore all the variables.
  # saver = tf.train.Saver()
  # if FLAGS.load == False:
  #   # Save the variables to disk.
  #   save_path = saver.save(sess, "/tmp/test_model.ckpt")
  #   print("Model saved in file: %s" % save_path)
  # else:
  #   saver.restore(sess, "/tmp/test_model.ckpt")
  #   print("Model restored.")
  train_writer.close()


  summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
  test_writer.add_summary(summary, 1999)
  test_writer.close()
  print('Accuracy at Completion: %s' % (acc))
  # sess.run(tf.contrib.memory_stats.BytesInUse())



def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    print("Deleting existing dir: ", FLAGS.log_dir)
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)


  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=5000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--load', nargs='?', const=True, type=bool, default=False,
                      help='If true, load the model from /tmp/test_model.ckpt')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      # default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist_with_summaries/logs/mnist_with_summaries'),
      default = 'logs/mnist_with_summaries',
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
