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

FLAGS = None

quantization_bits = 4
quantization_range = 1



def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)

  variables_initialized = False

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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

  # TODO: Quantize tensor
  def fake_quantize_tensor(input_tensor, min_val, max_val, name):
    # Hardcode the min and max for now
    # I think these cant be dynamic
    with tf.name_scope(name):
      # Get the max value in the input tensor
      # max_val = sess.run(tf.reduce_max(input_tensor))
      # Get the min value in the input tensor
      # min_val = sess.run(tf.reduce_min(input_tensor))
      # if(max_val == min_val):
      #   min_val = -max_val



      # Quantization
      quantized_tensor = tf.fake_quant_with_min_max_args(input_tensor, min_val, max_val, quantization_bits, False, name)
      variable_summaries(quantized_tensor)
      return quantized_tensor

  # TODO: Create a normal fully connected layer
  def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)

      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations


  # TODO: Create a fully connected layer with quantized variables
  def fc_layer_quantized(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        # biases = bias_variable([output_dim])
        biases = weight_variable([output_dim])
        variable_summaries(biases)

      # if(variables_initialized == False):
      #   tf.global_variables_initializer().run()
      #   print("Initializing Variables")

      with tf.name_scope('quantized_weights'):
        #quantized_weights = tf.fake_quant_with_min_max_args(weights, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
        #variable_summaries(quantized_weights)
        quantized_weights = fake_quantize_tensor(weights, -0.25, 0.25, name="quantized_weights")
      with tf.name_scope('quantized_biases'):
        #quantized_biases = tf.fake_quant_with_min_max_args(biases, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
        #variable_summaries(quantized_biases)
        quantized_biases = fake_quantize_tensor(biases, -0.25, 0.25, name="quantized_biases") # TODO: Biases seem to need higher bitwidths, also they train weirdly
        # quantized_biases = biases
      with tf.name_scope('quantized_Wx_plus_b'):
        preactivate_q = tf.matmul(input_tensor, quantized_weights) + quantized_biases
        #quantized_preactivate = tf.fake_quant_with_min_max_args(preactivate_q, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
        quantized_preactivate = fake_quantize_tensor(preactivate_q, -8, 8, name="quantized_preactivate")
        variable_summaries(quantized_preactivate)
        tf.summary.histogram('quantized_pre_activations', quantized_preactivate)

      quantized_activations = act(quantized_preactivate, name='quantized_activation')
      tf.summary.histogram('quantized activations', quantized_activations)
      return quantized_activations



  # TODO: Create a normal Conv layer
  def conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, layer_name):
    with tf.name_scope(layer_name):
      # setup the filter input shape for tf.nn.conv_2d
      conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

      # initialise weights and bias for the filter
      with tf.name_scope('weights'):
        # weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name + '_W')
        weights = weight_variable(conv_filt_shape)
        variable_summaries(weights)
      with tf.name_scope('biases'):
        # bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')
        # biases = bias_variable([num_filters])
        biases = weight_variable([num_filters])
        variable_summaries(weights)

      with tf.name_scope('out_layer'):
        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
        out_layer += biases
        variable_summaries(out_layer)
        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)

      # now perform max pooling
      # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
      # calculated).  It must be 4D to toe of frogmatch the convolution - in this case, for each image we want to use a 2 x 2 area
      # applied to each channel
      ksize = [1, pool_shape[0], pool_shape[1], 1]
      # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
      # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
      # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
      # to do strides of 2 in the x and y directions.
      strides = [1, 2, 2, 1]
      out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

      return out_layer


  # TODO: Create a Conv layer with Quantized variables
  def conv_layer_quantized(input_data, num_input_channels, num_filters, filter_shape, pool_shape, layer_name):
    with tf.name_scope(layer_name):
      # setup the filter input shape for tf.nn.conv_2d
      conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

      # initialise weights and bias for the filter
      with tf.name_scope('weights'):
        weights = weight_variable(conv_filt_shape)
        variable_summaries(weights)
      with tf.name_scope('biases'):
        # biases = bias_variable([num_filters])
        biases = weight_variable([num_filters])
        variable_summaries(biases)

      # Quantization
      with tf.name_scope('quantized_weights'):
        quantized_weights = fake_quantize_tensor(weights, -0.25, 0.25, name="quantized_weights")
      with tf.name_scope('quantized_biases'):
        quantized_biases = fake_quantize_tensor(biases, -0.25, 0.25, name="quantized_biases")

      with tf.name_scope('out_layer'):
        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, quantized_weights, [1, 1, 1, 1], padding='SAME')
        out_layer += quantized_biases
        quantized_out_layer = fake_quantize_tensor(out_layer, -6, 6, name="quantized_out_layer")
        # apply a ReLU non-linear activation
        quantized_out_layer = tf.nn.relu(quantized_out_layer)

      # now perform max pooling
      # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
      # calculated).  It must be 4D to toe of frogmatch the convolution - in this case, for each image we want to use a 2 x 2 area
      # applied to each channel
      ksize = [1, pool_shape[0], pool_shape[1], 1]
      # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
      # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
      # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
      # to do strides of 2 in the x and y directions.
      strides = [1, 2, 2, 1]
      quantized_out_layer = tf.nn.max_pool(quantized_out_layer, ksize=ksize, strides=strides, padding='SAME')

      return quantized_out_layer

  # with tf.name_scope('ConvolutionLayers'):
  layer1 = conv_layer_quantized(image_shaped_input, 1, 32, [5, 5], [2, 2], layer_name='conv1')
  layer2 = conv_layer_quantized(layer1, 32, 64, [5, 5], [2, 2], layer_name='conv2')

  with tf.name_scope('flatten'):
    x_flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

  # with tf.name_scope('FullyConnectedLayers'):
  # Layer 1 784x250
  hidden1 = fc_layer_quantized(x_flattened, 7 * 7 * 64, 250, 'fully_connected1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Layer 2 250x10
  # Do not apply softmax activation yet, see below.
  y = fc_layer_quantized(dropped, 250, 10, 'fully_connected2', act=tf.identity)

  # test_layer = nn_layer(y, 10, 2, 'test_layer')

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
    return {x: xs, y_: ys, keep_prob: k}

  variables_initialized = True

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
  test_writer.add_summary(summary, 999)
  test_writer.close()
  print('Accuracy at Completion: %s' % (acc))
  # sess.run(tf.contrib.memory_stats.BytesInUse())



def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
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
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist_with_summaries/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
