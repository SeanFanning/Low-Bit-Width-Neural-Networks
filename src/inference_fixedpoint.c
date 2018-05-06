#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fixed_point_calculations.h"


// Using Identity
float * calc_activations(float *input, float *biases, float **weights, int length){

    float *act = malloc (sizeof (float) * length); // Activations should be the same size as the biases

    // y = x1w1 + x2w2 + x3w3 + b
    for(int i=0; i<length; i++){    // Per node
        float e = 0;
        for(int j=0; j<784; j++){   // Per input
            e += input[j] * weights[j][i];
        }
        e += biases[i];
        // printf("e = %f\n", e);
        act[i] = e;
    }
    return(act);
}

int * calc_activations_fixed_point(int * q_i, int ** Q, int * biases, int length){
    int *act = malloc (sizeof (int) * length);

    // y = x1w1 + x2w2 + x3w3 + b
    // y = qi[step_size * w] + b
    for(int i=0; i<length; i++){    // Per node
        int e = 0;
        for(int j=0; j<784; j++){   // Per input
            //e += input[j] * weights[j][i];
            e += calc_multiplication(q_i[j], Q[j][i]);
            //e += q_i[j] * Q[j][i];
        }

        //int a = e >> 7;
        e += biases[i];
        int activation = e >> 7;
        printf("e= %d\ta= %d\tb= %d\n", e, activation, biases[i]);
        //activation /= 128; // Shift right to dequantize
        //activation += biases[i]; // Could reduce this by increasing bit size of activations
        act[i] = activation;

        //act[i] = e + (biases[i]);
    }
    return(act);
}

int main() {
    int layer1_length = 10;

    float * input = get_biases("../input_reshaped_quantized.csv", 784);
    float * biases = get_biases("../Parameters/floating_biases.csv", 10);
    int * biases_fixed = get_fixed_biases("../Parameters/biases.csv", 10);
    float ** weights = get_weights("../Parameters/weights.csv", 784, 10);
    int ** Q = get_Q("../Parameters/Q.csv", 784, 10);

    // First calculate the activations the normal way
    float * activations = calc_activations(input, biases, weights, layer1_length);

    printf("Done Normal\n");

    // Get the quantization step of the input values
    int *q_i = malloc (sizeof (int) * 784);
    for(int i=0; i<784; i++){
        q_i[i] = get_quantize_step(input[i], 0, 0.066667, 16);
    }
    int * q_activations = calc_activations_fixed_point(q_i, Q, biases_fixed, layer1_length);

    printf("Normal Activation:\t\tFixed Point Multiplication Free Activation: \n");
    for(int i=0; i<10; i++){
        printf("%f\t\t\t%d\n", activations[i], q_activations[i]);
    }

    int maxout_normal = get_maxout_float(activations);
    int maxout_fixedpoint = get_maxout(q_activations);

    printf("\n\nNormal Classification Result:\t\t\t\t%d\nFixed Point Multiplication Free Classification Result:\t%d\n", maxout_normal, maxout_fixedpoint);

    return 0;
}
