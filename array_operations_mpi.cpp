#define _CRT_SECURE_NO_WARNINGS  

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Constants
#define DEFAULT_ARRAY_SIZE 1000
#define MS_PER_SEC 1000.0

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get array size from command line or environment variable
    int ARRAY_SIZE = DEFAULT_ARRAY_SIZE;
    if (rank == 0) {
        if (argc > 1) ARRAY_SIZE = atoi(argv[1]);

        char* array_size_str = NULL;
        size_t len;
        if (_dupenv_s(&array_size_str, &len, "ARRAY_SIZE") == 0 && array_size_str != NULL) {
            ARRAY_SIZE = atoi(array_size_str);
            free(array_size_str);
        }
        printf("Using array size: %d\n", ARRAY_SIZE);
    }

    // Broadcast array size to all processes
    MPI_Bcast(&ARRAY_SIZE, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Validate array size
    if (ARRAY_SIZE <= 0) {
        if (rank == 0) fprintf(stderr, "Error: Array size must be positive\n");
        MPI_Finalize();
        return 1;
    }
    if (ARRAY_SIZE < size) {
        if (rank == 0) fprintf(stderr, "Error: Array size must be >= number of processes\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate local array size with balanced distribution
    // Extra elements are distributed to lower-rank processes
    int local_size = ARRAY_SIZE / size + ((rank < ARRAY_SIZE % size) ? 1 : 0);

    double* arr1 = (double*)malloc(local_size * ARRAY_SIZE * sizeof(double));
    double* arr2 = (double*)malloc(local_size * ARRAY_SIZE * sizeof(double));
    double* results = (double*)malloc(local_size * ARRAY_SIZE * sizeof(double));

    if (!arr1 || !arr2 || !results) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    srand((unsigned int)(time(NULL) + rank));

    for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
        arr1[i] = (double)rand() / RAND_MAX * 10.0 + 1.0;
        arr2[i] = (double)rand() / RAND_MAX * 10.0 + 1.0;
    }

    // ===== SEQUENTIAL OPERATIONS (Root process only) =====
    if (rank == 0) {
        printf("\nSequential operations:\n");

        double* seq_arr1 = (double*)malloc(local_size * ARRAY_SIZE * sizeof(double));
        double* seq_arr2 = (double*)malloc(local_size * ARRAY_SIZE * sizeof(double));
        double* seq_res = (double*)malloc(local_size * ARRAY_SIZE * sizeof(double));

        if (!seq_arr1 || !seq_arr2 || !seq_res) {
            fprintf(stderr, "Error: Sequential memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
            seq_arr1[i] = arr1[i];
            seq_arr2[i] = arr2[i];
        }

        double start, end;

        // ---- Addition ----
        start = MPI_Wtime();
        for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
            seq_res[i] = seq_arr1[i] + seq_arr2[i];
        }
        end = MPI_Wtime();
        printf("Addition: %.3f ms\n", (end - start) * MS_PER_SEC);

        // ---- Subtraction ----
        start = MPI_Wtime();
        for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
            seq_res[i] = seq_arr1[i] - seq_arr2[i];
        }
        end = MPI_Wtime();
        printf("Subtraction: %.3f ms\n", (end - start) * MS_PER_SEC);

        // ---- Multiplication ----
        start = MPI_Wtime();
        for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
            seq_res[i] = seq_arr1[i] * seq_arr2[i];
        }
        end = MPI_Wtime();
        printf("Multiplication: %.3f ms\n", (end - start) * MS_PER_SEC);

        // ---- Division (with zero check) ----
        start = MPI_Wtime();
        for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
            seq_res[i] = (seq_arr2[i] != 0.0) ? seq_arr1[i] / seq_arr2[i] : 0.0;
        }
        end = MPI_Wtime();
        printf("Division: %.3f ms\n", (end - start) * MS_PER_SEC);

        free(seq_arr1);
        free(seq_arr2);
        free(seq_res);
    }

    // Synchronize all processes before parallel operations
    MPI_Barrier(MPI_COMM_WORLD);

    // ===== PARALLEL OPERATIONS =====
    if (rank == 0) printf("\nParallel operations (MPI size = %d):\n", size);

    double local_start, local_end, global_start, global_end;

    // ---- Parallel Addition ----
    local_start = MPI_Wtime();
    for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
        results[i] = arr1[i] + arr2[i];
    }
    local_end = MPI_Wtime();
    // Find earliest start and latest end across all processes
    MPI_Reduce(&local_start, &global_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_end, &global_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Addition: %.3f ms\n", (global_end - global_start) * MS_PER_SEC);
    }

    // ---- Parallel Subtraction ----
    local_start = MPI_Wtime();
    for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
        results[i] = arr1[i] - arr2[i];
    }
    local_end = MPI_Wtime();
    MPI_Reduce(&local_start, &global_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_end, &global_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Subtraction: %.3f ms\n", (global_end - global_start) * MS_PER_SEC);
    }

    // ---- Parallel Multiplication ----
    local_start = MPI_Wtime();
    for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
        results[i] = arr1[i] * arr2[i];
    }
    local_end = MPI_Wtime();
    MPI_Reduce(&local_start, &global_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_end, &global_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Multiplication: %.3f ms\n", (global_end - global_start) * MS_PER_SEC);
    }

    // ---- Parallel Division (with zero check) ----
    local_start = MPI_Wtime();
    for (int i = 0; i < local_size * ARRAY_SIZE; i++) {
        results[i] = (arr2[i] != 0.0) ? arr1[i] / arr2[i] : 0.0;
    }
    local_end = MPI_Wtime();
    MPI_Reduce(&local_start, &global_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_end, &global_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Division: %.3f ms\n", (global_end - global_start) * MS_PER_SEC);
    }

    // Free allocated memory
    free(arr1);
    free(arr2);
    free(results);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}