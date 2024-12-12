
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>

#include "hpc.h"

typedef struct
{
    float *P; /* coordinates P[i][j] of point i               */
    int N;    /* Number of points (rows of matrix P)          */
    int D;    /* Number of dimensions (columns of matrix P)   */
} points_t;

void read_input(points_t *points)
{
    char buf[2048];
    int N, D;
    float *P;

    // Open the input file
    const char *filename = "datasets/test3.in"; // Adjust filename if needed
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "FATAL: Unable to open file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read the dimension (D)
    if (1 != fscanf(file, "%d", &D))
    {
        fprintf(stderr, "FATAL: Cannot read the dimension from file\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);

    // Ignore the rest of the first line
    if (NULL == fgets(buf, sizeof(buf), file))
    {
        fprintf(stderr, "FATAL: Cannot read the first line\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read the number of points (N)
    if (1 != fscanf(file, "%d", &N))
    {
        fprintf(stderr, "FATAL: Cannot read the number of points\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    // Allocate memory for points
    P = (float *)malloc(D * N * sizeof(*P));
    assert(P);

    // Read the points
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < D; k++)
        {
            if (1 != fscanf(file, "%f", &(P[i * D + k])))
            {
                fprintf(stderr, "FATAL: Failed to get coordinate %d of point %d\n", k, i);
                fclose(file);
                free(P);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);

    points->P = P;
    points->N = N;
    points->D = D;
}

void free_points(points_t *points)
{
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

/* Returns 1 iff |p| dominates |q| */
int dominates(const float *p, const float *q, int D)
{
    /* The following loops could be merged, but the keep them separated
       for the sake of readability */
    for (int k = 0; k < D; k++)
    {
        if (p[k] < q[k])
        {
            return 0;
        }
    }
    for (int k = 0; k < D; k++)
    {
        if (p[k] > q[k])
        {
            return 1;
        }
    }
    return 0;
}

/**
 * Compute the skyline of `points`. At the end, `s[i] == 1` iff point
 * `i` belongs to the skyline. The function returns the number `r` of
 * points that belongs to the skyline. The caller is responsible for
 * allocating the array `s` of length at least `points->N`.
 */
int skyline(const points_t *points, int *s)
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    int r = N;

    for (int i = 0; i < N; i++)
    {
        s[i] = 1;
    }

    for (int i = 0; i < N; i++)
    {
        if (s[i])
        {
            for (int j = 0; j < N; j++)
            {
                if (s[j] && dominates(&(P[i * D]), &(P[j * D]), D))
                {
                    s[j] = 0;
                    r--;
                }
            }
        }
    }
    return r;
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard ouptut. `s[i] == 1` iff point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
void print_skyline(const points_t *points, const int *s, int r)
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    printf("%d\n", D);
    printf("%d\n", r);
    for (int i = 0; i < N; i++)
    {
        if (s[i])
        {
            for (int k = 0; k < D; k++)
            {
                printf("%f ", P[i * D + k]);
            }
            printf("\n");
        }
    }
}

void print_skyline_to_file(char * filename, const points_t *points, const int *s, int r)
{
    FILE *file = fopen(filename, "w");
    fopen(filename, "w");
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    fprintf(file, "%d\n", D);
    fprintf(file, "%d\n", r);
    for (int i = 0; i < N; i++)
    {
        if (s[i])
        {
            for (int k = 0; k < D; k++)
            {
                fprintf(file, "%f ", P[i * D + k]);
            }
            fprintf(file, "\n");
        }
    }
    fclose(file);
}

void scatter_points(points_t *points, points_t *local_points, int rank, int size)
{

    int D, N;
    if (rank == 0)
    {
        D = points->D;
        N = points->N;
    }

    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int WORKER_NPOINTS = N / size;
    int MASTER_NPOINTS = WORKER_NPOINTS + (N % size);

    int local_num_points = rank == 0 ? MASTER_NPOINTS : WORKER_NPOINTS;

    local_points->D = D;
    local_points->N = local_num_points;
    local_points->P = (float *)malloc(D * local_num_points * sizeof(float));

    float *P = points->P;

    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = i == 0 ? (MASTER_NPOINTS * D) : WORKER_NPOINTS * D;
            displs[i] = i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1];
        }
    }

    MPI_Scatterv(P, sendcounts, displs, MPI_FLOAT, local_points->P, D * local_num_points, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(sendcounts);
        free(displs);
    }
}

void gather_points(points_t *gathered, points_t *local_points, int *num_skyline_points_perp, int rank, int size)
{
    int D = local_points->D;
    int N = local_points->N;

    int *num_points_per_process = num_skyline_points_perp;

    int total_points = 0;
    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            total_points += num_points_per_process[i];
        }
    }

    if (rank == 0)
    {
        gathered->D = D;
        gathered->N = total_points;
        gathered->P = (float *)malloc(D * total_points * sizeof(float));
    }

    int *displs = (int *)malloc(size * sizeof(int));

    if (rank == 0)
    {

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            displs[i] = offset;
            offset += num_points_per_process[i] * D;
        }
    }

    int *recvcounts = (int *)malloc(size * sizeof(int));
    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = num_points_per_process[i] * D;
        }
    }

    MPI_Gatherv(
        local_points->P,
        D * N,
        MPI_FLOAT,
        gathered->P,
        recvcounts,
        displs,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    points_t points;
    int rank, size;
    int WORKER_NPOINTS;
    int MASTER_NPOINTS;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        read_input(&points);
        WORKER_NPOINTS = points.N / size;
        MASTER_NPOINTS = WORKER_NPOINTS + (points.N % size);
    }

    MPI_Bcast(&WORKER_NPOINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MASTER_NPOINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double tstart = 0;

    if (rank == 0)
    {
        tstart = hpc_gettime();
    }

    points_t local_points_struct;

    scatter_points(&points, &local_points_struct, rank, size);

    

    int D = local_points_struct.D;

    float *local_skyline_points = (float *)malloc(D * local_points_struct.N * sizeof(float));

    int *local_skyline = (int *)malloc(local_points_struct.N * sizeof(int));

    int local_num_skyline = skyline(&local_points_struct, local_skyline);

    float *local_points = local_points_struct.P;

    int skyline_index = 0; // Tracks where to write in `local_skyline_points`
    for (int i = 0; i < local_points_struct.N; i++)
    {
        if (local_skyline[i] == 1)
        {
            for (int k = 0; k < D; k++)
            {
                local_skyline_points[skyline_index * D + k] = local_points[i * D + k];
            }
            skyline_index++;
        }
    }
    points_t local_sk_points_struct;
    local_sk_points_struct.D = D;
    local_sk_points_struct.N = local_num_skyline;
    local_sk_points_struct.P = local_skyline_points;

    MPI_Barrier(MPI_COMM_WORLD);

    int *num_skyline_points_per_process = (int *)malloc(size * sizeof(int));

    MPI_Gather(&local_num_skyline, 1, MPI_INT, num_skyline_points_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // printo the num_skyline_points_per_process

    printf("Rank: %d, Num Skyline Points: %d\n", rank, local_num_skyline);

    // calculate total number of skyline points
    int total_skyline_points = 0;
    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            total_skyline_points += num_skyline_points_per_process[i];
        }
    }

    points_t gathered;

    gather_points(&gathered, &local_sk_points_struct, num_skyline_points_per_process, rank, size);

    double elapsed;

    MPI_Barrier(MPI_COMM_WORLD);

    /**
     * Snodo chiave
     */

    int global_num_skyline = 0;

    int N;

    if (rank == 0)
    {
        N = gathered.N;
        D = gathered.D;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float *S_first = (float *)malloc(D * N * sizeof(float));

    int * visited_S_first = (int *)malloc(D * N * sizeof(int));

    for(int i = 0; i < N; i++){
        visited_S_first[i] = 0;
    }

    if (rank == 0)
    {

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < D; j++)
            {
                S_first[i * D + j] = gathered.P[i * D + j];
            }
        }
    }

    // I send S' to every process using a broadcast
    MPI_Bcast(S_first, N * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /**
     * Each processor P_i then determines for a
     * subset S_i' ⊆ S' of points which points in S_i' are dominated
     * by points in S' and removes these points from S_i. At the
     * end of this round, we perform another all-to-all communi-
     * cation to collect the points in sets S_i' that were not deleted in
     * processor P_0 . These points form sky(S), and processor P0
     * returns this set to the user.
     */

    WORKER_NPOINTS = N / size;
    MASTER_NPOINTS = WORKER_NPOINTS + (N % size);

    points_t partition;
    partition.D = D;
    partition.N = rank == 0 ? MASTER_NPOINTS : WORKER_NPOINTS;
    int offset = rank == 0 ? 0 : MASTER_NPOINTS + (rank - 1) * WORKER_NPOINTS;
    partition.P = S_first + offset * D;

    // now each processor checks his partition of S' against the other point of S' and removes dominated points from its own partition

    // to do so more efficiently, each process skips the checking against the points in its own partition, defining a start and end index between which not to check
    int start_index = offset;
    int end_index = offset + partition.N;

    int *s = (int *)malloc(partition.N * sizeof(int));
    int r = partition.N;

    // Debug: print partition details
    printf("Rank: %d, Partition N: %d, Offset: %d, Start Index: %d, End Index: %d\n",
           rank, partition.N, offset, start_index, end_index);

    for (int i = 0; i < partition.N; i++)
    {
        s[i] = 1;
    }

    

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < partition.N; i++)
    {
        if (s[i])
        {
            int j = 0;
            while (j < N)
            {   
                if (j == start_index)
                {
    
                    j = end_index;
                }
    
                if (s[i] && dominates(&(S_first[j * D]), &(partition.P[i * D]), D))
                {

                    s[i] = 0;
                    r--;
                }
                
                
                j++;
            }
        }
    }

    /*

    char filename_visits[100];
    sprintf(filename_visits, "visited%d.txt", rank);
    FILE *file_visits = fopen(filename_visits, "w");
    for (int i = 0; i < N; i++)
    {
        fprintf(file_visits, "S_first[%d] : %d\n", i, visited_S_first[i]);
    }

    */

    
    // DA QUI IN GIU FUNZIONA
    char filename[100];
    sprintf(filename, "skyline%d.txt", rank);

    print_skyline_to_file(filename, &partition, s, r);

    /*
    FILE *file = fopen(filename, "w");
    for (int i = 0; i < partition.N; i++)
    {
        if (s[i])
        {
            for (int k = 0; k < D; k++)
            {
                fprintf(file, "%f ", partition.P[i * D + k]);
            }
            fprintf(file, "\n");
        }
    }
    fclose(file);
    */

    MPI_Barrier(MPI_COMM_WORLD);

    // gather the total r from all processes using mpi_reduce

    MPI_Reduce(&r, &global_num_skyline, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // now we need to gather the points that were not deleted in the partition, to do so
    // we just need to concatenate the various "s" arrays, then the print skyline will handle the rest

    // int *global_skyline = (int *)malloc(N * sizeof(int));

    // MPI_Gather(s, partition.N, MPI_INT, global_skyline, partition.N, MPI_INT, 0, MPI_COMM_WORLD);

    int *global_skyline = NULL;
    int *sizes = NULL, *displs = NULL;

    // Each process sends its local size (partition.N) to the master process
    int local_size = partition.N;
    if (rank == 0)
    {
        sizes = (int *)malloc(size * sizeof(int)); // Size of data from each process
    }
    MPI_Gather(&local_size, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Master process calculates total size and displacements
    int total_size = 0;
    if (rank == 0)
    {
        displs = (int *)malloc(size * sizeof(int)); // Displacements in the receive buffer
        displs[0] = 0;                              // First displacement is 0
        for (int i = 1; i < size; i++)
        {
            displs[i] = displs[i - 1] + sizes[i - 1];
        }
        total_size = displs[size - 1] + sizes[size - 1];

        global_skyline = (int *)malloc(total_size * sizeof(int)); // Allocate the global buffer
    }

    // Use MPI_Gatherv to gather arrays of varying sizes
    MPI_Gatherv(s, local_size, MPI_INT,
                global_skyline, sizes, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Master process can now use `global_skyline`

    if (rank == 0)
    {
        elapsed = hpc_gettime() - tstart;

        print_skyline(&gathered, global_skyline, global_num_skyline);

        fprintf(stderr, "\n\t%d points\n", points.N);
        fprintf(stderr, "\t%d dimensions\n", points.D);
        fprintf(stderr, "\t%d points in skyline\n\n", global_num_skyline);
        fprintf(stderr, "Execution time (s) %f\n", elapsed);

        free_points(&points);

        free_points(&gathered);

        free(global_skyline); 
        free(sizes);
        free(displs);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    free(local_points);
    partition.P = NULL;
    free(local_skyline);
    free(local_skyline_points);
    free(num_skyline_points_per_process);
    free(S_first);
    MPI_Finalize();
    return 0;
}