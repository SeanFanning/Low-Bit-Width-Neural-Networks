# Low-Bit-Width-Neural-Networks

Fixed point multiplication free only works on single layer quantized network
1.  Train network on MNIST using 4-bit quantization
      $ python MNIST_CNN_quantized.py
2.  Model parameters are exported to csv files in /Parameters
3.  Run fixed point inference on the sample input data
      $ ./inference_fixedpoint
4.  Classification results should be the same
