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

float * calc_activations_fixed_point(int * q_i, int ** Q, float * biases, int length){
    float *act = malloc (sizeof (float) * length);

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
        float activation = e;
        activation /= 128; // Single floating point division per node (Compared to 784 mults per node)
        activation += biases[i]; // Could reduce this by increasing bit size of activations
        act[i] = activation;
    }
    return(act);
}

int main() {
    int layer1_length = 10;

    float * input = get_biases("../input_reshaped_quantized.csv", 784);
    float * biases = get_biases("../Parameters/biases.csv", 10);
    float ** weights = get_weights("../Parameters/weights.csv", 784, 10);
    int ** Q = get_Q("../Parameters/Q.csv", 784, 10);

    // First calculate the activations the normal way
    float * activations = calc_activations(input, biases, weights, layer1_length);

    // Get the quantization step of the input values
    int *q_i = malloc (sizeof (int) * 784);
    for(int i=0; i<784; i++){
        q_i[i] = get_quantize_step(input[i], 0, 0.066667, 16);
    }
    float * q_activations = calc_activations_fixed_point(q_i, Q, biases, layer1_length);

    printf("Normal Activation:\t\tFixed Point Multiplication Free Activation: \n");
    for(int i=0; i<10; i++){
        printf("%f\t\t\t%f\n", activations[i], q_activations[i]);
    }

    return 0;
}
