#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


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

float * get_ReLU(float *input, int length){
    for(int i=0; i<length; i++){
        if(input[i] < 0){
            input[i] = 0;
        }
    }
    return(input);
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
        e = quantize_value(e, -1.866667, 0.266667, 15);    // Hardcode quantization of activations to 4 bits
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

// Using Q function to efficiently calculate activations
float * calc_activations_optimised(int *input, float *biases, int **weights, int length){

    float *act = malloc (sizeof (float) * length); // Activations should be the same size as the biases
    float step_size = 0.066667; // q_s
    float w[4] = {-0.4, -0.2, 0, 0.2};
    float *Q = malloc (sizeof (float) * 4);

    // Calculate the possible values of Q
    for(int i=0; i<4; i++){
        Q[i] = step_size * w[i];
    }

    // y = x1w1 + x2w2 + x3w3 + b
    // y = qi[step_size * w] + b
    for(int i=0; i<length; i++){    // Per node
        float e = 0;
        for(int j=0; j<784; j++){   // Per input
            //e += input[j] * weights[j][i];
            //printf("e = %f\n", e);
            // TODO: This
            e += calc_multiplication(input[j], weights[j][i]);
        }
        e += biases[i];
        // e = quantize_value(e, -1.866667, 0.266667, 15);    // Hardcode quantization of activations to 4 bits
        // printf("e = %f\n", e);
        act[i] = e;
    }
    return(act);
}

int main() {
    int layer1_length = 10;

    float * input = get_biases("../input_reshaped_quantized.csv", 784);

    float * biases = get_biases("../biases.csv", 10);

    float ** weights = get_weights("../weights.csv", 784, 10);

//    for(int i=0; i<784; i++){
//        for(int j=0; j<10; j++){
//            //printf("%f ", weights[i][j]);
//        }
//        printf("\n");
//    }

    //float * activations = malloc (sizeof (float) * 10);
    float * activations = calc_activations(input, biases, weights, layer1_length);

    printf("Normal Activation:\n");
    for(int i=0; i<10; i++){
        printf("%f\n", activations[i]);
    }

    // Get the quantization step of the input values
    int *q_i = malloc (sizeof (int) * 784);
    for(int i=0; i<784; i++){
        q_i[i] = get_quantize_step(input[i], 0, 0.066667, 16);
    }

    // Get the quantization step of the weights
    int ** w_i = (int **) malloc (sizeof (int) * 784 * 100);
    for(int i=0; i<784; i++){
        w_i[i] = (int *) malloc(sizeof(int) * 10);
        for(int j=0; j<10; j++){
            int v = get_quantize_step(weights[i][j], -0.4, 0.2, 4);
            w_i[i][j] = v;
            //printf("w = %d\n", v);
        }
    }

    float * q_activations = calc_activations_optimised(q_i, biases, w_i, layer1_length);

    printf("Q Activation: \n");
    for(int i=0; i<10; i++){
        printf("%f\n", q_activations[i]);
    }


    return 0 ;
}