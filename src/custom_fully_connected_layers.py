import tensorflow as tf

from src.variable_summaries import variable_summaries
from src.quantize_tensor import fake_quantize_tensor
from src.variable_initialization import weight_variable, bias_variable


# Create a normal fully connected layer
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


# Create a fully connected layer with quantized variables
def fc_layer_quantized(input_tensor, input_dim, output_dim, bits_w, max_w, bits_b, max_b, bits_a, max_a, layer_name, act=tf.nn.relu):
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
      quantized_weights = fake_quantize_tensor(weights, bits_w, -max_w, max_w, name="quantized_weights")
    with tf.name_scope('quantized_biases'):
      #quantized_biases = tf.fake_quant_with_min_max_args(biases, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
      #variable_summaries(quantized_biases)
      quantized_biases = fake_quantize_tensor(biases, bits_b, -max_b, max_b, name="quantized_biases") # TODO: Biases seem to need higher bitwidths, also they train weirdly
      # quantized_biases = biases
    with tf.name_scope('quantized_Wx_plus_b'):
      preactivate_q = tf.matmul(input_tensor, quantized_weights) + quantized_biases
      #quantized_preactivate = tf.fake_quant_with_min_max_args(preactivate_q, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
      quantized_preactivate = fake_quantize_tensor(preactivate_q, bits_a, -max_a, max_a, name="quantized_preactivate")
      variable_summaries(quantized_preactivate)
      #tf.summary.histogram('quantized_pre_activations', quantized_preactivate)

    quantized_activations = act(quantized_preactivate, name='quantized_activation') # Relu by default
    # tf.summary.histogram('quantized activations', quantized_activations)
    variable_summaries(quantized_activations)
    return quantized_activations


# Create a fully connected layer with quantized variables, adding random noise
def fc_layer_quantized_add_noise(input_tensor, input_dim, output_dim, bits_w, max_w, bits_b, max_b, bits_a, max_a, noise_stddev, layer_name, act=tf.nn.relu):
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
    with tf.name_scope('additive_noise'): # Add some noise to move some nodes into activation if they are close
      noise = tf.random_normal([output_dim], mean=0.0, stddev=noise_stddev)
      variable_summaries(noise)

    with tf.name_scope('quantized_weights'):
      #quantized_weights = tf.fake_quant_with_min_max_args(weights, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
      #variable_summaries(quantized_weights)
      quantized_weights = fake_quantize_tensor(weights, bits_w, -max_w, max_w, name="quantized_weights")
    with tf.name_scope('quantized_biases'):
      #quantized_biases = tf.fake_quant_with_min_max_args(biases, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
      #variable_summaries(quantized_biases)
      quantized_biases = fake_quantize_tensor(biases, bits_b, -max_b, max_b, name="quantized_biases")
      # quantized_biases = biases

    with tf.name_scope('quantized_Wx_plus_b'):
      preactivate_q = tf.matmul(input_tensor, quantized_weights) + quantized_biases + noise
      #quantized_preactivate = tf.fake_quant_with_min_max_args(preactivate_q, -quantization_range/2, quantization_range/2, quantization_bits, narrow_range=False, name='quantized_weights')
      quantized_preactivate = fake_quantize_tensor(preactivate_q, bits_a, -max_a, max_a, name="quantized_preactivate")
      variable_summaries(quantized_preactivate)
      #tf.summary.histogram('quantized_pre_activations', quantized_preactivate)

    quantized_activations = act(quantized_preactivate, name='quantized_activation') # Relu by default
    # tf.summary.histogram('quantized activations', quantized_activations)
    variable_summaries(quantized_activations)
    return quantized_activations
