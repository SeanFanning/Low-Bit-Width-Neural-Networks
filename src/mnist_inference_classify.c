#include <stdio.h>
#include <stdlib.h>
#include <string.h>


float ** get_weights(char *filename, int size_x, int size_y){
    char buffer[1024];
    char *record, *line;
    int i=0, j=0;

    float ** mat = (float **) malloc(sizeof(float) * size_x * size_y);
    for(int k=0; k<1000; k++){
        mat[k]=(float *) malloc(1000*sizeof(float));
    }
    FILE *fstream = fopen(filename, "r");
    if(fstream == NULL){
        printf("\n file opening failed ");
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL){
        record = strtok(line, ",");
        while(record != NULL) {
            //printf("record : %s", record); //here you can put the record into the array as per your requirement.
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
    if(fstream == NULL){
        printf("\n file opening failed ");
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL){
        record = strtok(line, ",");
        while(record != NULL) {
            mat[j++] = atof(record);
            record = strtok(NULL,";");
        }
    }

    return(mat);
}

int main() {
    float *biases = get_biases("../biases.csv", 10);

//    for(int i=0; i<10; i++){
//        printf("%f\n", biases[i]);
//    }

    float** weights = get_weights("../weights.csv", 784, 10);

    for(int i=0; i<784; i++){
        for(int j=0; j<10; j++){
            //printf("%f ", weights[i][j]);
        }
        printf("\n");
    }

    return 0 ;
}