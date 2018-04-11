# Hard code quantization to 2,2,4 for now
l1_b_base = -0.4
l1_b_steps = 2 ** 2 # Hardcode to 2 bits
l1_b_step_size = 0.2  # (0.2-l1_b_base)/l1_b_steps (No idea what I was doing here) hard code it for now
l1_w_base = -0.4
l1_w_steps = 2 ** 2
l1_w_step_size = 0.2  # (0.2 - l1_w_base) / l1_w_steps

l2_b_base = -0.4
l2_b_steps = 2 ** 2
l2_b_step_size = 0.2  # (0.2 - l2_b_base) / l2_b_steps
l2_w_base = -0.4
l2_w_steps = 2 ** 2
l2_w_step_size = 0.2  # (0.2 - l2_w_base) / l2_w_steps




# Returns the quantized versions of the values

def get_biases(biases):
  quantized_values = biases
  i = 0
  for x in biases:  # For each bias value
    for step in range(0, l2_b_steps):  # For each step in range
      step_val = round(l2_b_base + step * l2_b_step_size, 4)  # Value of current step
      if (x > (step_val - l2_b_step_size / 2) and x < (step_val + l2_b_step_size / 2)):  # If bias value is within the range of this step
        # print(x, " -> ", step_val)
        quantized_values[i] = step_val
    i += 1
  return quantized_values


def get_weights(weights):
  quantized_values = weights
  j = 0
  for y in weights:  # For each weight array
    i = 0
    for x in y:  # For each weight value in array
      for step in range(0, l2_w_steps):  # For each step in range
        step_val = round(l2_w_base + step * l2_w_step_size, 4)  # Value of current step
        if (x > (step_val - l2_w_step_size / 2) and x < (step_val + l2_w_step_size / 2)):  # If bias value is within the range of this step
          # print(x, " -> ", step_val)
          quantized_values[j][i] = step_val
      i += 1
    j += 1
  return quantized_values