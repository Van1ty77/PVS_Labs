#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void merge(float *arr, int l, int m, int r) {
	int n1 = m - l + 1;
	int n2 = r - m;

	float *L = (float *)malloc(n1 * sizeof(float));
	float *R = (float *)malloc(n2 * sizeof(float));

	for (int i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for (int j = 0; j < n2; j++)
		R[j] = arr[m + 1 + j];

	int i = 0, j = 0, k = l;
	while (i < n1 && j < n2) {
		if (L[i] <= R[j]) {
			arr[k++] = L[i++];
		} else {
			arr[k++] = R[j++];
		}
	}

	while (i < n1) {
		arr[k++] = L[i++];
	}
	while (j < n2) {
		arr[k++] = R[j++];
	}

	free(L);
	free(R);
}

void mergeSort(float *arr, int l, int r) {
	if (l < r) {
		int m = l + (r - l) / 2;
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);
		merge(arr, l, m, r);
	}
}

int main() {
	const int n = 10000000;
	float *arr = (float *)malloc(n * sizeof(float));

	// Инициализация массива случайными числами
	for (int i = 0; i < n; i++) {
		arr[i] = (float)rand() / RAND_MAX * 1000.0f;
	}

	clock_t start = clock();
	mergeSort(arr, 0, n - 1);
	clock_t end = clock();

	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Merge Sort for %d elements: %.3f seconds\n", n, time_spent);

	free(arr);
	return 0;
}
