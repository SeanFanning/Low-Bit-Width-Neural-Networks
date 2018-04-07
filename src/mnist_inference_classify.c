#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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

// Using ReLU
float * calc_activations(float *input, float *biases, float **weights, int length){

    float *act = malloc (sizeof (float) * length); // Activations should be the same size as the biases

    for(int i=0; i<length; i++){
        
    }

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

    float *activations = malloc (sizeof (float) * 10);

    return 0 ;
}