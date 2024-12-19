/****************************************************************************
 * Alessandro Valmori
 * 0001089308
 * 
 * Per compilare:
 *
 *       mpicc -std=c99 -Wall -Wpedantic -O2 mpi-skyline.c -o mpi-skyline
 *
 * Per eseguire il programma:
 *
 *       mpirun -np <num_processes> mpi-skyline < input > output  
 *
 ****************************************************************************/

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

void read_input( points_t *points )
{
    char buf[1024];
    int N, D;
    float *P;

    if (1 != scanf("%d", &D)) {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != scanf("%d", &N)) {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float*)malloc( D * N * sizeof(*P) );
    assert(P);
    for (int i=0; i<N; i++) {
        for (int k=0; k<D; k++) {
            if (1 != scanf("%f", &(P[i*D + k]))) {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
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
/**
 * @brief Scatters points data from the master process (rank 0) to all processes in an MPI environment.
 *
 * This function is designed to distribute a set of points stored in the `points_t` structure
 * from the master process to all other processes in an MPI communicator. Each process, including
 * the master, will receive a subset of the points, ensuring a balanced distribution (or close to it).
 * 
 * - The master process gets an additional portion of points if the total number of points
 *   (`N`) is not evenly divisible by the number of processes (`size`).
 * - Uses MPI's `MPI_Scatterv` to perform the distribution efficiently.
 *
 * @param points Pointer to the `points_t` structure holding the complete dataset (only used by the master process).
 *               - `points->P`: Flat array of size `N * D` (N points, each with D dimensions).
 *               - `points->D`: Number of dimensions per point.
 *               - `points->N`: Total number of points in the dataset.
 *               This parameter is ignored for non-master processes.
 *
 * @param local_points Pointer to the `points_t` structure where the received subset of points will be stored.
 *                     This structure will be allocated and filled by the function for all processes.
 *                     - `local_points->P`: Will contain the subset of points assigned to this process.
 *                     - `local_points->D`: Number of dimensions per point (same as `points->D`).
 *                     - `local_points->N`: Number of points received by this process.
 *
 * @param rank Rank of the current process in the MPI communicator (from 0 to `size - 1`).
 * @param size Total number of processes in the MPI communicator.
 *
 * @note The function dynamically allocates memory for `local_points->P` on all processes.
 *       The caller is responsible for freeing this memory after use.
 *       For the master process, the `points` parameter should be initialized with valid data
 *       before calling this function.
 */
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

/**
 * @brief Gathers points data from all processes in an MPI environment to the master process (rank 0).
 *
 * This function collects subsets of points stored in the `local_points` structures of each process
 * and gathers them into a single `gathered` structure on the master process (rank 0).
 * The number of points contributed by each process is specified by the `num_skyline_points_perp` array.
 * 
 * - Uses `MPI_Gatherv` for efficient gathering of data.
 * - The data is concatenated in `gathered->P` on the master process, maintaining the order of ranks.
 *
 * @param gathered Pointer to the `points_t` structure where the gathered points will be stored (only used by the master process).
 *                 This parameter is ignored for non-master processes.
 *                 - `gathered->P`: Will contain all points from all processes.
 *                 - `gathered->D`: Number of dimensions per point (same as `local_points->D`).
 *                 - `gathered->N`: Total number of points gathered from all processes.
 *                 Memory for `gathered->P` is dynamically allocated by the function and must be freed by the caller on rank 0.
 *
 * @param local_points Pointer to the `points_t` structure holding the subset of points for this process.
 *                     - `local_points->P`: Flat array of size `N * D` (N points, each with D dimensions).
 *                     - `local_points->D`: Number of dimensions per point.
 *                     - `local_points->N`: Number of points in the local subset.
 *
 * @param num_skyline_points_perp Array of size `size` (number of processes) where each entry specifies
 *                                the number of points contributed by the corresponding process.
 *                                Must be valid on all processes.
 *
 * @param rank Rank of the current process in the MPI communicator (from 0 to `size - 1`).
 * @param size Total number of processes in the MPI communicator.
 *
 * @note The function dynamically allocates memory for `gathered->P` on rank 0.
 *       The caller must free this memory after use.
 * @note The `num_skyline_points_perp` array must be consistent across all processes.
 */
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
    int D;
    int N;
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

    D = local_points_struct.D;

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

    int *num_skyline_points_per_process = (int *)malloc(size * sizeof(int));

    MPI_Gather(&local_num_skyline, 1, MPI_INT, num_skyline_points_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    int global_num_skyline = 0;

    if (rank == 0)
    {
        N = gathered.N;
        D = gathered.D;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float *S_first = (float *)malloc(D * N * sizeof(float));

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

    MPI_Bcast(S_first, N * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    points_t partition;
    partition.D = D;
    partition.N = local_sk_points_struct.N;

    int *offsets = (int *)malloc(size * sizeof(int));

    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            offsets[i] = i == 0 ? 0 : (offsets[i - 1] + num_skyline_points_per_process[i - 1]);
        }
    }

   
    MPI_Bcast(offsets, size, MPI_INT, 0, MPI_COMM_WORLD);

    partition.P = S_first + offsets[rank] * D;

    
    int start_index = offsets[rank];
    int end_index = offsets[rank] + partition.N;

    int *s = (int *)malloc(partition.N * sizeof(int));
    int r = partition.N;

    for (int i = 0; i < partition.N; i++)
    {
        s[i] = 1;
    }


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

    MPI_Reduce(&r, &global_num_skyline, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

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


    MPI_Gatherv(s, local_size, MPI_INT,
                global_skyline, sizes, displs, MPI_INT,
                0, MPI_COMM_WORLD);



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

    free(local_points);
    partition.P = NULL;
    free(local_skyline);
    free(local_skyline_points);
    free(num_skyline_points_per_process);
    free(S_first);
    MPI_Finalize();
    return 0;
}