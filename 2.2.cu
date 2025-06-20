#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ void swap(float& a, float& b) {
	float t = a;
	a = b;
	b = t;
}

__global__ void bitonicSortStep(float* dev_values, int j, int k) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int ixj = i ^ j;

	if (ixj > i) {
		if ((i & k) == 0) {
			if (dev_values[i] > dev_values[ixj]) {
				swap(dev_values[i], dev_values[ixj]);
			}
		}
		else {
			if (dev_values[i] < dev_values[ixj]) {
				swap(dev_values[i], dev_values[ixj]);
			}
		}
	}
}

void bitonicSort(float* values, int numElements, int block_size) {
	float* dev_values;
	cudaMalloc((void**)&dev_values, numElements * sizeof(float));
	cudaMemcpy(dev_values, values, numElements * sizeof(float), cudaMemcpyHostToDevice);

	int threads = block_size;
	int blocks = (numElements + threads - 1) / threads;

	for (int k = 2; k <= numElements; k *= 2) {
		for (int j = k >> 1; j > 0; j = j >> 1) {
			bitonicSortStep << <blocks, threads >> > (dev_values, j, k);
			cudaDeviceSynchronize();
		}
	}

	cudaMemcpy(values, dev_values, numElements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
}

int main() {
	const int n = 1000000;
	const int block_sizes[] = { 64, 128, 256, 512, 1024 };

	for (int i = 0; i < sizeof(block_sizes) / sizeof(block_sizes[0]); i++) {
		int block_size = block_sizes[i];
		float* arr = (float*)malloc(n * sizeof(float));

		for (int j = 0; j < n; j++) {
			arr[j] = (float)rand() / RAND_MAX * 1000.0f;
		}

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		bitonicSort(arr, n, block_size);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		printf("Block size %4d: %.3f s\n", block_size, milliseconds / 1000.0);

		free(arr);
	}

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
                return 1;
        }

	return 0;
}
