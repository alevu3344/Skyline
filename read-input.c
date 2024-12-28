#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int read_input(){
    int rows, cols;
    char buffer[256];

    // Read and parse the number of rows
    if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
        fprintf(stderr, "Error reading rows line.\n");
        return 1;
    }
    if (sscanf(buffer, "%d", &cols) != 1) {
        fprintf(stderr, "Invalid input for number of rows.\n");
        return 1;
    }

    // Read and parse the number of columns
    if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
        fprintf(stderr, "Error reading columns line.\n");
        return 1;
    }
    if (sscanf(buffer, "%d", &rows) != 1) {
        fprintf(stderr, "Invalid input for number of columns.\n");
        return 1;
    }

    // Allocate memory for the matrix
    float **matrix = malloc(rows * sizeof(float *));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        return 1;
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(float));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Memory allocation error.\n");
            return 1;
        }
    }

    // Read the matrix values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (scanf("%f", &matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading matrix values.\n");
                return 1;
            }
        }
    }

    // Print the matrix to verify
    printf("%d\n", cols);
    printf("%d\n", rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);

    return 0;
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
    
    read_input();
    }


    
    MPI_Finalize();
    return 0;
}