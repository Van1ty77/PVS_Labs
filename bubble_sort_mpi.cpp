#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

// Defaults
const int DEFAULT_ARRAY_SIZE = 200000;

// Sequential Bubble Sort
void sequential_bubble_sort(int* arr, int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Parallel Bubble Sort (Odd-Even Sort)
void parallel_bubble_sort(int* arr, int local_size, int rank, int size, MPI_Comm comm) {
    int n = local_size;
    for (int phase = 0; phase < n; ++phase) {
        if (phase % 2 == 0) {
            for (int i = 0; i < local_size - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    int temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
        }
        else {
            for (int i = 1; i < local_size - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    int temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
        }

        // Communicate and merge sorted sub-arrays
        if (rank > 0 && (phase % 2 == 0)) {
            MPI_Send(&arr[0], local_size, MPI_INT, rank - 1, 0, comm);
        }

        if (rank < size - 1 && (phase % 2 == 0)) {
            MPI_Recv(&arr[0], local_size, MPI_INT, rank + 1, 0, comm, MPI_STATUS_IGNORE);
            if (arr[local_size - 1] > arr[0]) {
                int temp = arr[local_size - 1];
                arr[local_size - 1] = arr[0];
                arr[0] = temp;
            }
        }
        if (rank < size - 1 && (phase % 2 != 0)) {
            MPI_Send(&arr[0], local_size, MPI_INT, rank + 1, 0, comm);
        }

        if (rank > 0 && (phase % 2 != 0)) {
            MPI_Recv(&arr[0], local_size, MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            if (arr[local_size - 1] > arr[0]) {
                int temp = arr[local_size - 1];
                arr[local_size - 1] = arr[0];
                arr[0] = temp;
            }
        }
        MPI_Barrier(comm);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("\nRunning with MPI size: %d\n", size);
    }

    // Get ARRAY_SIZE from environment variable
    int ARRAY_SIZE = DEFAULT_ARRAY_SIZE;
    if (rank == 0) {
        char* array_size_str;
        size_t len;
        errno_t err = _dupenv_s(&array_size_str, &len, "ARRAY_SIZE");
        if (err == 0 && array_size_str != NULL) {
            ARRAY_SIZE = atoi(array_size_str);
            free(array_size_str);
        }
        printf("Array Size (read by process 0): %d\n", ARRAY_SIZE);
    }

    // Broadcast ARRAY_SIZE to all processes
    MPI_Bcast(&ARRAY_SIZE, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (ARRAY_SIZE <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: ARRAY_SIZE must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (ARRAY_SIZE < size) {
        if (rank == 0) {
            fprintf(stderr, "Error: ARRAY_SIZE must be greater than or equal to number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate local array size
    int local_size = ARRAY_SIZE / size;
    int remainder = ARRAY_SIZE % size;
    if (rank < remainder) {
        local_size++;
    }

    // Initialize local array
    int* arr = (int*)malloc(local_size * sizeof(int));
    if (arr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Seed the random number generator with the rank to ensure different processes get different numbers
    time_t t;
    srand((unsigned)time(&t) + rank);

    // Fill local arrays with random values
    for (int i = 0; i < local_size; ++i) {
        arr[i] = rand() % 1000 + 1;
    }

    // --- Sequential Operations (on rank 0 only) ---
    if (rank == 0) {
        printf("\nSequential Operations (Rank 0):\n");
        int* arr_seq = (int*)malloc(local_size * sizeof(int));
        if (arr_seq == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for arr_seq.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < local_size; i++) {
            arr_seq[i] = arr[i];
        }

        // Sequential Bubble Sort
        clock_t start_seq = clock();
        sequential_bubble_sort(arr_seq, local_size);
        clock_t stop_seq = clock();
        double duration_seq = (double)(stop_seq - start_seq) * 1000000.0 / CLOCKS_PER_SEC;
        printf("Sequential Bubble Sort time: %f microseconds\n", duration_seq);

        free(arr_seq);

    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Parallel Operations (MPI only) ---

    clock_t start_par = clock();
    parallel_bubble_sort(arr, local_size, rank, size, MPI_COMM_WORLD);
    clock_t stop_par = clock();

    MPI_Barrier(MPI_COMM_WORLD);
    double duration_par = (double)(stop_par - start_par) * 1000000.0 / CLOCKS_PER_SEC;


    double max_duration_par;
    MPI_Reduce(&duration_par, &max_duration_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nParallel Operations (MPI size = %d) :\n", size);
        printf("Parallel Bubble Sort Time:  %f microseconds\n", max_duration_par);
    }
    free(arr);

    MPI_Finalize();
    return 0;
}