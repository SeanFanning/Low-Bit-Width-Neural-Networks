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
    quantized_values[i] = fixed_point_quantize(x)
    print(x, " -> ", quantized_values[i])
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


def calculate_Q(q_s, weights):
  Q = weights
  j = 0
  for y in weights:  # For each weight array
    i = 0
    for w in y:  # For each weight value in array
      x = w * q_s
      Q[j][i] = fixed_point_quantize(x)
      i += 1
    j += 1

  return Q

# Returns the signed fixed point value using 2s compliment
# Hardcoded for 4 bit signed Q to 2^-7
def fixed_point_quantize(x):
  # Values are quantized to 4 bits signed
  a = 2 ** -5 # Define the weight of each bit
  b = 2 ** -6
  c = 2 ** -7
  varience = c/2
  if(x > a + b + c - varience): # abc
    return 7
  elif(x > a + b - varience): # ab
    return 6
  elif(x > a + c - varience): # a c
    return 5
  elif(x > a - varience): # a
    return 4
  elif(x > b + c - varience): #  bc
    return 3
  elif(x > b - varience): #  b
    return 2
  elif(x > c - varience): #   c
    return 1
  elif(x > - varience): #
    return 0
  elif(x > - b + c - varience): # -  c
    return -1
  elif(x > - b - varience): # - b
    return -2
  elif(x > - a + c - varience): # - bc
    return -3
  elif(x > - a - varience): # -a
    return 4
  elif(x > - a - c - varience): # -a c
    return -5
  elif(x > - a - b - varience): # -ab
    return -6
  elif(x > - a - b - c - varience): # -abc
    return -7
  else:
    return -8