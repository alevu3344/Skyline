
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "hpc.h"

typedef struct
{
    float *P; /* coordinates P[i][j] of point i               */
    int N;    /* Number of points (rows of matrix P)          */
    int D;    /* Number of dimensions (columns of matrix P)   */
} points_t;

void read_input(points_t *points)
{
    char buf[1024];
    int N, D;
    float *P;

    // Open the input file
    const char *filename = "datasets/test1.in"; // Adjust filename if needed
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

    printf("D = %d\n", D);
    printf("N = %d\n", N);

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

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    points_t points;
    int rank, size;
    int D, N;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        read_input(&points);
        D = points.D;
        N = points.N;
    }


    double tstart = 0;

    if (rank == 0)
    {
        tstart = hpc_gettime();
    }

    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int WORKER_NPOINTS = N / size;
    int MASTER_NPOINTS;

    if (rank == 0)
    {
        MASTER_NPOINTS = WORKER_NPOINTS + (N % size);
    }

    int local_num_points = rank == 0 ? MASTER_NPOINTS : WORKER_NPOINTS;


    MPI_Bcast(&MASTER_NPOINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&WORKER_NPOINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);



    int *local_skyline;
    float *local_points;

    local_skyline = (int *)malloc(local_num_points * sizeof(int));
    local_points = (float *)malloc(D * local_num_points * sizeof(float));

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

    MPI_Scatterv(points.P, sendcounts, displs, MPI_FLOAT, local_points, D * local_num_points, MPI_FLOAT, 0, MPI_COMM_WORLD);


    float *local_skyline_points = (float *)malloc(D * local_num_points * sizeof(float));
    points_t local_points_struct;
    local_points_struct.D = D;
    local_points_struct.N = local_num_points;
    local_points_struct.P = local_points;

    int local_num_skyline = skyline(&local_points_struct, local_skyline);

    int skyline_index = 0; // Tracks where to write in `local_skyline_points`
    for (int i = 0; i < local_num_points; i++)
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

    MPI_Barrier(MPI_COMM_WORLD);

    int *num_skyline_points_per_process = (int *)malloc(size * sizeof(int));

    MPI_Gather(&local_num_skyline, 1, MPI_INT, num_skyline_points_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
    if (rank == 0)
    {
        gathered.D = D;
        gathered.N = total_skyline_points;
        gathered.P = (float *)malloc(D * total_skyline_points * sizeof(float));
    }

    if (rank == 0)
    {

        int offset = 0;
        for (int i = 0; i < size; i++)
        {

            displs[i] = offset;
            offset += num_skyline_points_per_process[i] * D;
        }
    }

    int *recvcounts = NULL;
    if (rank == 0)
    {
        recvcounts = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = num_skyline_points_per_process[i] * D;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gatherv(
        local_skyline_points,
        D * local_num_skyline,
        MPI_FLOAT,
        gathered.P,
        recvcounts,
        displs,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD);

    double elapsed;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {

        int *global_skyline = (int *)malloc(total_skyline_points * sizeof(int));

        int global_num_skyline = skyline(&gathered, global_skyline);

        elapsed = hpc_gettime() - tstart;

        print_skyline(&gathered, global_skyline, global_num_skyline);

        fprintf(stderr, "\n\t%d points\n", points.N);
        fprintf(stderr, "\t%d dimensions\n", points.D);
        fprintf(stderr, "\t%d points in skyline\n\n", global_num_skyline);
        fprintf(stderr, "Execution time (s) %f\n", elapsed);

        free(displs);
        free(recvcounts);
        free(sendcounts);
        free_points(&points);
        free_points(&gathered);
        free(global_skyline);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    free(local_points);
    free(local_skyline);
    free(local_skyline_points);
    free(num_skyline_points_per_process);
    MPI_Finalize();
    return 0;
}