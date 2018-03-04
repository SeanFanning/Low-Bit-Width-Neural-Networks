import tensorflow as tf

from src.variable_summaries import variable_summaries
from src.quantize_tensor import fake_quantize_tensor
from src.variable_initialization import weight_variable


# Create a normal Conv layer
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


# Create a Conv layer with Quantized variables
def conv_layer_quantized(input_data, num_input_channels, num_filters, filter_shape, pool_shape, bits_w, max_w, bits_b, max_b, bits_a, max_a, layer_name):
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
      quantized_weights = fake_quantize_tensor(weights, bits_w, -max_w, max_w, name="quantized_weights")
    with tf.name_scope('quantized_biases'):
      quantized_biases = fake_quantize_tensor(biases, bits_b, -max_b, max_b, name="quantized_biases")

    with tf.name_scope('out_layer'):
      # setup the convolutional layer operation
      out_layer = tf.nn.conv2d(input_data, quantized_weights, [1, 1, 1, 1], padding='SAME')
      out_layer += quantized_biases
      quantized_out_layer = fake_quantize_tensor(out_layer, bits_a, -max_a, max_a, name="quantized_out_layer")
      # apply a ReLU non-linear activation
      quantized_out_layer = tf.nn.relu(quantized_out_layer)

    ksize = [1, pool_shape[0], pool_shape[1], 1]

    strides = [1, 2, 2, 1]
    quantized_out_layer = tf.nn.max_pool(quantized_out_layer, ksize=ksize, strides=strides, padding='SAME')

    return quantized_out_layer


# Create a Conv layer with Quantized variables, adding random noise
def conv_layer_quantized_add_noise(input_data, num_input_channels, num_filters, filter_shape, pool_shape, bits_w, max_w, bits_b, max_b, bits_a, max_a, noise_stddev, layer_name):
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
    with tf.name_scope('additive_noise'):  # Add some noise to move some nodes into activation if they are close
      noise = tf.random_normal([num_filters], mean=0.0, stddev=noise_stddev)
      variable_summaries(noise)

    # Quantization
    with tf.name_scope('quantized_weights'):
      quantized_weights = fake_quantize_tensor(weights, bits_w, -max_w, max_w, name="quantized_weights")
    with tf.name_scope('quantized_biases'):
      quantized_biases = fake_quantize_tensor(biases, bits_b, -max_b, max_b, name="quantized_biases")
    with tf.name_scope('quantized_noise'):
      quantized_noise = fake_quantize_tensor(noise, bits_b, -noise_stddev*2, noise_stddev*2, name="quantized_noise")

    with tf.name_scope('out_layer'):
      # setup the convolutional layer operation
      out_layer = tf.nn.conv2d(input_data, quantized_weights, [1, 1, 1, 1], padding='SAME')
      out_layer = out_layer + quantized_biases + quantized_noise
      quantized_out_layer = fake_quantize_tensor(out_layer, bits_a, -max_a, max_a, name="quantized_out_layer")
      # apply a ReLU non-linear activation
      quantized_out_layer = tf.nn.relu(quantized_out_layer)

    ksize = [1, pool_shape[0], pool_shape[1], 1]

    strides = [1, 2, 2, 1]
    quantized_out_layer = tf.nn.max_pool(quantized_out_layer, ksize=ksize, strides=strides, padding='SAME')

    return quantized_out_layer