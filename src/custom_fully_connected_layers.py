import tensorflow as tf

from src.variable_summaries import variable_summaries
from src.quantize_tensor import fake_quantize_tensor
from src.variable_initialization import weight_variable, bias_variable


# Create a normal fully connected layer without quantization
def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      # tf.summary.histogram('pre_activations', preactivate)

    activations = act(preactivate, name='activation')
    # tf.summary.histogram('activations', activations)
    variable_summaries(activations)
    return activations


# Create a fully connected layer with quantized variables
def fc_layer_quantized(input_tensor, input_dim, output_dim, bits_w, max_w, bits_b, max_b, bits_a, max_a, layer_name, act=tf.nn.relu):
  with tf.name_scope(layer_name):
    with tf.name_scope('input'):
      variable_summaries(input_tensor)
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      # biases = weight_variable([output_dim]) # Biases work better when initialised randomly
      variable_summaries(biases)

    with tf.name_scope('quantized_weights'):
      # quantized_weights = tf.Variable([input_dim, output_dim], collections=[tf.GraphKeys.GLOBAL_VARIABLES]) # Initialise quantized values as global vars
      quantized_weights = fake_quantize_tensor(weights, bits_w, -max_w, max_w, name="quantized_weights")
    with tf.name_scope('quantized_biases'):
      # quantized_biases = tf.Variable(output_dim, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
      quantized_biases = fake_quantize_tensor(biases, bits_b, -max_b, max_b, name="quantized_biases")
    with tf.name_scope('quantized_Wx_plus_b'):
      preactivate_q = tf.matmul(input_tensor, quantized_weights) + quantized_biases
      quantized_preactivate = fake_quantize_tensor(preactivate_q, bits_a, -max_a, max_a, name="quantized_preactivate")
      # variable_summaries(quantized_preactivate)

    # quantized_activations = tf.Variable(output_dim, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    quantized_activations = act(quantized_preactivate, name='quantized_activation') # Relu by default
    variable_summaries(quantized_activations)
    return quantized_activations


# Create a fully connected layer with quantized variables, adding random noise
def fc_layer_quantized_add_noise(input_tensor, input_dim, output_dim, bits_w, max_w, bits_b, max_b, bits_a, max_a, noise_stddev, layer_name, act=tf.nn.relu):
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      # biases = bias_variable([output_dim])
      biases = weight_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('additive_noise'): # Add some noise to move some nodes into activation if they are close
      noise = tf.random_normal([output_dim], mean=0.0, stddev=noise_stddev)
      variable_summaries(noise)

    with tf.name_scope('quantized_weights'):
      # quantized_weights = tf.Variable([input_dim, output_dim], collections=[tf.GraphKeys.GLOBAL_VARIABLES])
      quantized_weights = fake_quantize_tensor(weights, bits_w, -max_w, max_w, name="quantized_weights")
    with tf.name_scope('quantized_biases'):
      # quantized_biases = tf.Variable(output_dim, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
      quantized_biases = fake_quantize_tensor(biases, bits_b, -max_b, max_b, name="quantized_biases")
    with tf.name_scope('quantized_noise'):
      quantized_noise = fake_quantize_tensor(noise, bits_b, -noise_stddev*2, noise_stddev*2, name="quantized_noise")

    with tf.name_scope('quantized_Wx_plus_b'):
      preactivate_q = tf.matmul(input_tensor, quantized_weights) + quantized_biases + quantized_noise
      quantized_preactivate = fake_quantize_tensor(preactivate_q, bits_a, -max_a, max_a, name="quantized_preactivate")
      # variable_summaries(quantized_preactivate)

    # quantized_activations = tf.Variable(output_dim, collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    quantized_activations = act(quantized_preactivate, name='quantized_activation') # Relu by default
    variable_summaries(quantized_activations)
    return quantized_activations
