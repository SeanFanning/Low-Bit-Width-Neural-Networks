import tensorflow as tf

from src.variable_summaries import variable_summaries

# Creates a fake quantized tensor with 32 bit floating point values rounded to low bitwidth equivalent

def fake_quantize_tensor(input_tensor, quantization_bits, min_val, max_val, name):
  with tf.name_scope(name):

    # TODO: Min and Max values need to be given manually right now
    # Get the max value in the input tensor
    # max_val = sess.run(tf.reduce_max(input_tensor))
    # Get the min value in the input tensor
    # min_val = sess.run(tf.reduce_min(input_tensor))
    # if(max_val == min_val):
    #   min_val = -max_val # If biases are initialized as a constant

    # Quantization
    quantized_tensor = tf.fake_quant_with_min_max_args(input_tensor, min_val, max_val, quantization_bits, False, name)
    variable_summaries(quantized_tensor)
    return quantized_tensor