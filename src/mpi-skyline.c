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

void read_input(const char *filename, points_t *points)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "FATAL: cannot open file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    char buf[1024];
    int N, D;
    float *P;

    if (1 != fscanf(file, "%d", &D)) {
        fprintf(stderr, "FATAL: cannot read the dimension\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);

    if (NULL == fgets(buf, sizeof(buf), file)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: cannot read the first line\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    if (1 != fscanf(file, "%d", &N)) {
        fprintf(stderr, "FATAL: cannot read the number of points\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    P = (float *)malloc(D * N * sizeof(*P));
    if (!P) {
        fprintf(stderr, "FATAL: memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < D; k++) {
            if (1 != fscanf(file, "%f", &(P[i * D + k]))) {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                free(P);
                fclose(file);
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
    // Ensure proper usage by checking command-line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    points_t points;
    int rank, size;
    int WORKER_NPOINTS;
    int MASTER_NPOINTS;
    int D;
    int N;

    // Determine the rank of the process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        const char *filename = argv[1]; 
        // Read input points from the specified file
        read_input(filename, &points);

        // Calculate the number of points each worker will process
        WORKER_NPOINTS = points.N / size;
        MASTER_NPOINTS = WORKER_NPOINTS + (points.N % size);
    }

    // Broadcast the number of points each process handles to all processes
    MPI_Bcast(&WORKER_NPOINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MASTER_NPOINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double tstart = 0;

    if (rank == 0)
    {
        // Start the timer on the master process
        tstart = hpc_gettime();
    }

    points_t local_points_struct;

    // Scatter the points from the master process to all processes
    scatter_points(&points, &local_points_struct, rank, size);

    // Extract the dimensionality of the points
    D = local_points_struct.D;

    // Allocate memory for local skyline points and flags
    float *local_skyline_points = (float *)malloc(D * local_points_struct.N * sizeof(float));
    int *local_skyline = (int *)malloc(local_points_struct.N * sizeof(int));

    // Compute the local skyline points
    int local_num_skyline = skyline(&local_points_struct, local_skyline);

    // Gather the actual skyline points based on the flags
    int skyline_index = 0;
    for (int i = 0; i < local_points_struct.N; i++)
    {
        if (local_skyline[i] == 1)
        {
            for (int k = 0; k < D; k++)
            {
                local_skyline_points[skyline_index * D + k] = local_points_struct.P[i * D + k];
            }
            skyline_index++;
        }
    }

    points_t local_sk_points_struct;
    local_sk_points_struct.D = D;
    local_sk_points_struct.N = local_num_skyline;
    local_sk_points_struct.P = local_skyline_points;

    // Gather the number of skyline points from each process to the master
    int *num_skyline_points_per_process = (int *)malloc(size * sizeof(int));
    MPI_Gather(&local_num_skyline, 1, MPI_INT, num_skyline_points_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_skyline_points = 0;
    if (rank == 0)
    {
        // Calculate the total number of skyline points across all processes
        for (int i = 0; i < size; i++)
        {
            total_skyline_points += num_skyline_points_per_process[i];
        }
    }

    points_t gathered;

    // Gather all local skyline points into a single structure at the master process
    gather_points(&gathered, &local_sk_points_struct, num_skyline_points_per_process, rank, size);

    double elapsed;
    int global_num_skyline = 0;

    if (rank == 0)
    {
        // Share the total number and dimensionality of gathered points
        N = gathered.N;
        D = gathered.D;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize the global skyline points array
    float *S_first = (float *)malloc(D * N * sizeof(float));

    if (rank == 0)
    {
        // Copy gathered points into the global skyline array
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < D; j++)
            {
                S_first[i * D + j] = gathered.P[i * D + j];
            }
        }
    }

    MPI_Bcast(S_first, N * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Partition the skyline points for each process based on offsets
    points_t partition;
    partition.D = D;
    partition.N = local_sk_points_struct.N;

    int *offsets = (int *)malloc(size * sizeof(int));
    if (rank == 0)
    {
        // Calculate offsets for each process's portion of the global skyline
        for (int i = 0; i < size; i++)
        {
            offsets[i] = i == 0 ? 0 : (offsets[i - 1] + num_skyline_points_per_process[i - 1]);
        }
    }

    MPI_Bcast(offsets, size, MPI_INT, 0, MPI_COMM_WORLD);
    partition.P = S_first + offsets[rank] * D;

    // Perform a second pass to remove dominated points
    int start_index = offsets[rank];
    int end_index = offsets[rank] + partition.N;
    int *s = (int *)malloc(partition.N * sizeof(int));
    int r = partition.N;

    // Initialize all points as part of the skyline
    for (int i = 0; i < partition.N; i++)
    {
        s[i] = 1;
    }

    // Check if any point is dominated by points in the global skyline
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

    // Reduce the total number of skyline points globally
    MPI_Reduce(&r, &global_num_skyline, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather all skyline flags to the master process
    int *global_skyline = NULL;
    int *sizes = NULL, *displs = NULL;
    int local_size = partition.N;

    if (rank == 0)
    {
        sizes = (int *)malloc(size * sizeof(int));
    }

    MPI_Gather(&local_size, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        displs = (int *)malloc(size * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            displs[i] = displs[i - 1] + sizes[i - 1];
        }

        global_skyline = (int *)malloc((displs[size - 1] + sizes[size - 1]) * sizeof(int));
    }

    MPI_Gatherv(s, local_size, MPI_INT, global_skyline, sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Print results and execution time
        elapsed = hpc_gettime() - tstart;

        print_skyline(&gathered, global_skyline, global_num_skyline);

        fprintf(stderr, "\n\t%d points\n", points.N);
        fprintf(stderr, "\t%d dimensions\n", points.D);
        fprintf(stderr, "\t%d points in skyline\n\n", global_num_skyline);
        fprintf(stderr, "Execution time (s) %f\n", elapsed);

        // Free dynamically allocated memory for gathered data
        free_points(&points);
        free_points(&gathered);
        free(global_skyline);
        free(sizes);
        free(displs);
    }

    // Clean up dynamically allocated memory
    partition.P = NULL;
    free(local_skyline);
    free(local_skyline_points);
    free(num_skyline_points_per_process);
    free(S_first);

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
