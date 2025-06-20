#include <stdio.h>
#include <stdlib.h>
#include <chrono>

double get_time() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    return duration_cast<duration<double>>(now.time_since_epoch()).count();
}

void matrix_operations(float* A, float* B, float* C, int width, int height, int op) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            switch (op) {
            case 0: C[idx] = A[idx] + B[idx]; break;
            case 1: C[idx] = A[idx] - B[idx]; break;
            case 2: C[idx] = A[idx] * B[idx]; break;
            case 3: C[idx] = A[idx] / (B[idx] + 0.0001f); break;
            }
        }
    }
}

int main() {
    const int N = 10000;
    const char* ops[] = { "Addition", "Subtraction", "Multiplication", "Division" };

    int total = N * N;
    size_t mem_size = total * sizeof(float);

    printf("Matrix size: %d x %d (%d elements, %.2f MB)\n",
        N, N, total, mem_size / (1024.0 * 1024.0));

    float* A = (float*)malloc(mem_size);
    float* B = (float*)malloc(mem_size);
    float* C = (float*)malloc(mem_size);

    // Инициализация
    for (int i = 0; i < total; i++) {
        A[i] = (float)rand() / RAND_MAX * 100.0f;
        B[i] = (float)rand() / RAND_MAX * 100.0f + 0.1f;
    }

    for (int op = 0; op < 4; op++) {
        double start = get_time();
        matrix_operations(A, B, C, N, N, op);
        double end = get_time();

        printf("%-12s: %.5f sec\n", ops[op], end - start);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}