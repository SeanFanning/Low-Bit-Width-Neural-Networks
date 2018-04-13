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
            record = strtok(NULL,",");
        }
        ++i;
        j=0;
    }
    return(mat);
}

int ** get_Q(char *filename, int size_x, int size_y){
    char buffer[1024];
    char *record, *line;
    int i=0, j=0;
    FILE *fstream = fopen(filename, "r");

    int ** mat = (int **) malloc(sizeof(int) * size_x * size_y);
    for(int k=0; k<1000; k++){
        mat[k]=(int *) malloc(size_y * sizeof(int));
    }

    if(fstream == NULL){
        printf("\n file opening failed ");
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL){
        record = strtok(line, ",");
        while(record != NULL) {
            mat[i][j++] = atof(record);
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
            record = strtok(NULL,";");
        }
    }
    return(mat);
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
