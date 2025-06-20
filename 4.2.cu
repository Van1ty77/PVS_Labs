#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH  10000
#define HEIGHT 10000

enum Operation { ADD, SUB, MUL, DIV };

__global__ void matrixOpsKernel(float* A, float* B, float* C,
	int width, int height, Operation op) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		switch (op) {
		case ADD: C[idx] = A[idx] + B[idx]; break;
		case SUB: C[idx] = A[idx] - B[idx]; break;
		case MUL: C[idx] = A[idx] * B[idx]; break;
		case DIV: C[idx] = A[idx] / (B[idx] + 0.0001f); break;
		}
	}
}

void matrixOps(float* A, float* B, float* C, int width, int height,
	Operation op, dim3 blockSize) {
	size_t size = width * height * sizeof(float);
	float* d_A, * d_B, * d_C;

	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	matrixOpsKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, width, height, op);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main() {
	const int blockSizes[] = { 64, 128, 256, 512, 1024 };
	const char* opNames[] = { "Addition", "Subtraction", "Multiplication", "Division" };

	int width = WIDTH, height = HEIGHT;
	int n = width * height;
	size_t size = n * sizeof(float);

	float* A = (float*)malloc(size);
	float* B = (float*)malloc(size);
	float* C = (float*)malloc(size);

	// Инициализация
	for (int i = 0; i < n; i++) {
		A[i] = (float)rand() / RAND_MAX * 100.0f;
		B[i] = (float)rand() / RAND_MAX * 100.0f + 0.1f;
	}

	printf("Matrix operations (%dx%d = %d elements):\n", width, height, n);

	for (int i = 0; i < sizeof(blockSizes) / sizeof(blockSizes[0]); i++) {
		int blockSize = blockSizes[i];
		dim3 block(blockSize, 1);
		dim3 grid((width * height + block.x - 1) / block.x);

		printf("\nBlock size %4d:\n", blockSize);

		for (int op = ADD; op <= DIV; op++) {
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);
			matrixOps(A, B, C, width, height, (Operation)op, block);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);

			float ms;
			cudaEventElapsedTime(&ms, start, stop);

			printf("  %-14s: %7.3f ms (%.3f s)\n", opNames[op], ms, ms / 1000.0);
		}
	}

	free(A);
	free(B);
	free(C);

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
                return 1;
        }

	return 0;
}

