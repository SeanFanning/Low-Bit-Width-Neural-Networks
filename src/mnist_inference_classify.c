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

// Using Q function to efficiently calculate activations
float * calc_activations_optimised(float *input, float *biases, float **weights, int length){

    float *act = malloc (sizeof (float) * length); // Activations should be the same size as the biases

    float step_size = 0.066667;
    float w[4] = [-0.4, -0.2, 0, 0.2];

    // y = x1w1 + x2w2 + x3w3 + b
    // y = qi[step_size * w] + b
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

int main() {
    int layer1_length = 10;

    float * input = get_biases("../input_reshaped_quantized.csv", 784);

    float * biases = get_biases("../biases.csv", 10);

//    for(int i=0; i<784; i++){
//        printf("%f\n", input[i]);
//    }

    float ** weights = get_weights("../weights.csv", 784, 10);

//    for(int i=0; i<784; i++){
//        for(int j=0; j<10; j++){
//            //printf("%f ", weights[i][j]);
//        }
//        printf("\n");
//    }

    //float * activations = malloc (sizeof (float) * 10);
    float * activations = calc_activations(input, biases, weights, layer1_length);

    for(int i=0; i<10; i++){
        printf("%f\n", activations[i]);
    }

    return 0 ;
}