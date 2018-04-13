#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Struct to store 4 bit quantized value
typedef struct fixedPoint{ // C will automatically pad this to 8 bits though
    int value : 4; // 4 bits signed value
} fixedPoint;

float ** get_weights(char *filename, int size_x, int size_y){
    char buffer[1024];
    char *record, *line;
    int i=0, j=0;
    FILE *fstream = fopen(filename, "r");

    float ** mat = (float **) malloc(sizeof(float) * size_x * size_y);
    for(int k=0; k<1000; k++){
        mat[k]=(float *) malloc(size_y * sizeof(float));
    }


    if(fstream == NULL){
        printf("\n file opening failed ");
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL){
        record = strtok(line, ",");
        while(record != NULL) {
            mat[i][j++] = atof(record);
            // printf("%d, %d %f\n", i, j, mat[i][j]);
            record = strtok(NULL,",");
        }
        ++i;
        j=0;
    }
    return(mat);
}

float * get_biases(char *filename, int size){
    char buffer[1024];
    char *record, *line;
    int j=0;
    float *mat = malloc (sizeof (float) * size);
    FILE *fstream = fopen(filename, "r");

    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL){
        record = strtok(line, ",");
        while(record != NULL) {
            mat[j++] = atof(record);
            //printf("%d, %f\n", j, mat[j]);
            record = strtok(NULL,";");
        }
    }
    return(mat);
}


float quantize_value(float x, float base, float step_size, int steps){
    if(x < base){   // Limit x to the min and max values
        return(base);
    }
    if(x > base+(steps)*step_size){
        return(base+(steps)*step_size);
    }
    for(int i=0; i<steps; i++){ // For each step in range
        float step_val = base + i * step_size;
        if((x > (step_val - step_size / 2)) && (x < (step_val + step_size / 2))){
            // printf("%f -> %f\n", x, step_val);
            float v = round(step_val * 100000) / 100000;    // Round value to ensure zero value
            return(v);
        }
    }
    printf("Oh no x = %f\n", x);
}

//  Returns the quantization step (q_i)
int get_quantize_step(float x, float base, float step_size, int steps){
    if(x < base){   // Limit x to the min and max values
        return(0);
    }
    if(x > base+(steps)*step_size){
        return(steps);
    }
    for(int i=0; i<steps; i++){ // For each step in range
        float step_val = base + i * step_size;
        if((x > (step_val - step_size / 2)) && (x < (step_val + step_size / 2))){
            //printf("%f -> %d\n", x, i);
            return(i);
        }
    }
    printf("Oh no x = %f\n", x);
}

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
        //e = quantize_value(e, -1.866667, 0.266667, 15);    // Hardcode quantization of activations to 4 bits
        // printf("e = %f\n", e);
        act[i] = e;
    }
    return(act);
}

float calc_multiplication(int q_i, int Q){
    int x = 0;

    // TODO: Convert this into a binary search
    // Maybe hardcode each value have an array 4x16
    switch(q_i){
        case 0:     // li 0
            return(0);
        case 1:     // move
            return(Q);
        case 2:     // sll 1
            x = Q << 1;
            return(x);
        case 3:     // sll 1; add
            x = Q << 1;
            x += Q;
            return(x);
        case 4:     // sll 2
            x = Q << 2;
            return(x);
        case 5:     // sll 2; add
            x =
            Q << 2;
            x += Q;
            return(x);
        case 6:     // sll 1; add; sll 1
            x = Q << 1;
            x += Q;
            x = x << 1;
            return(x);
        case 7:     // sll 3; sub
            x = Q << 3;
            x -= Q;
            return(x);
        case 8:     // sll 3
            x = Q << 3;
            return(x);
        case 9:     // sll 3; add
            x = Q << 3;
            x += Q;
            return(x);
        case 10:    // sll 2; add; sll 1
            x = Q << 2;
            x += Q;
            x = x << 1;
            return(x);
        case 11:    // sll 3; add; add; add
            x = Q << 3;
            x += Q;
            x += Q;
            x += Q;
            return(x);
        case 12:    // sll 1; add; sll 2  (* 3 * 4)
            x = Q << 1;
            x += Q;
            x = x << 2;
            return(x);
        case 13:    // sll 4; sub; sub; sub
            x = Q << 4;
            x -= Q;
            x -= Q;
            x -= Q;
            return(x);
        case 14:    // sll 4; sub; sub
            x = Q << 4;
            x -= Q;
            x -= Q;
            return(x);
        case 15:    // sll 4; sub
            x = Q << 4;
            x -= Q;
            return(x);
    }
}

// Returns the signed fixed point value using 2s compliment
// Hardcoded for 4 bit Q to 2^-6
int fixed_point_quantize(float x){

    if(x < -0.026){
        return(0b1110);
    }
    else if(x < -0.01){
        return(-1);
    }
    else if(x > 0.01){
        return(0b0001);
    }
    else{
        return(0b0000);
    }
}

int ** calc_Q(float q_s, float **weights, int length){
    int ** Q = (int **) malloc (sizeof (int) * 784 * length);

    for(int i=0; i<784; i++){
        Q[i] = (int *) malloc(sizeof(int) * length);
        float y=0;
        for(int j=0; j<length; j++){
            float x = q_s * weights[i][j];
            fixedPoint v;
            v.value = fixed_point_quantize(x);
            printf("w = %f\tx = %f\tFixed Point Quantized = %d\n", weights[i][j], x, v.value);
            Q[i][j] = v.value;
        }
    }

    return(Q);
}

float * calc_activations_fixed_point(int * q_i, int ** Q, float * biases, int length){
    float *act = malloc (sizeof (float) * length);

    // y = x1w1 + x2w2 + x3w3 + b
    // y = qi[step_size * w] + b
    for(int i=0; i<length; i++){    // Per node
        int e = 0;
        for(int j=0; j<784; j++){   // Per input
            //e += input[j] * weights[j][i];
            //printf("e = %f\n", e);
            // TODO: This
            e += calc_multiplication(q_i[j], Q[j][i]);
            //e += q_i[j] * Q[j][i];
        }

        float activation = e;
        activation /= 64;
        activation += biases[i];
        // e = quantize_value(e, -1.866667, 0.266667, 15);    // Hardcode quantization of activations to 4 bits
        // printf("e = %f\n", e);
        act[i] = activation;
    }
    return(act);
}

int main() {
    int layer1_length = 10;

    float * input = get_biases("../input_reshaped_quantized.csv", 784);
    float * biases = get_biases("../biases.csv", 10);
    float ** weights = get_weights("../weights.csv", 784, 10);

    // First calculate the activations the normal way
    float * activations = calc_activations(input, biases, weights, layer1_length);

    // Then do the multiplication free fixed point method
    // Get the quantization step of the input values
    int *q_i = malloc (sizeof (int) * 784);
    for(int i=0; i<784; i++){
        q_i[i] = get_quantize_step(input[i], 0, 0.066667, 16);
    }

    int ** Q = calc_Q(0.066667, weights, 10);

    float * q_activations = calc_activations_fixed_point(q_i, Q, biases, layer1_length);

    printf("Normal Activation:\t\tFixed Point Multiplication Free Activation: \n");
    for(int i=0; i<10; i++){
        printf("%f\t\t\t%f\n", activations[i], q_activations[i]);
    }


    return 0 ;
}
