#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <random>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

// Defaults
const int DEFAULT_ARRAY_SIZE = 1000000;
const int DEFAULT_SEQUENTIAL_THRESHOLD = 1000;

// Sequential QuickSort
void sequential_quicksort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        int pi = i + 1;

        sequential_quicksort(arr, low, pi - 1);
        sequential_quicksort(arr, pi + 1, high);
    }
}

// Parallel QuickSort with OpenMP
void parallel_quicksort(vector<int>& arr, int low, int high) {
    if (low < high) {
        if (high - low < DEFAULT_SEQUENTIAL_THRESHOLD) {
            sequential_quicksort(arr, low, high);
            return;
        }

        int pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        int pi = i + 1;

        #pragma omp task if(high - low > DEFAULT_SEQUENTIAL_THRESHOLD)  // Create a task only if the subarray is large enough
        parallel_quicksort(arr, low, pi - 1);

        #pragma omp task if(high - low > DEFAULT_SEQUENTIAL_THRESHOLD)
        parallel_quicksort(arr, pi + 1, high);

        #pragma omp taskwait  // Wait for all generated tasks to complete
    }
}

int main() {
    // Get ARRAY_SIZE from environment variable
    const char* array_size_str = getenv("ARRAY_SIZE");
    int ARRAY_SIZE = DEFAULT_ARRAY_SIZE;
    if (array_size_str != nullptr) {
        ARRAY_SIZE = atoi(array_size_str);
    }

    // Get SEQUENTIAL_THRESHOLD from environment variable
    const char* sequential_threshold_str = getenv("SEQUENTIAL_THRESHOLD");
    int SEQUENTIAL_THRESHOLD = DEFAULT_SEQUENTIAL_THRESHOLD;
    if (sequential_threshold_str != nullptr) {
        SEQUENTIAL_THRESHOLD = atoi(sequential_threshold_str);
    }

    vector<int> arr(ARRAY_SIZE);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 1000);

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        arr[i] = distrib(gen);
    }

    vector<int> arr_seq = arr;
    vector<int> arr_par = arr;

    cout << "Array size: " << ARRAY_SIZE << endl;
    cout << "Sequential threshold: " << SEQUENTIAL_THRESHOLD << endl;

    // Sequential QuickSort
    auto start_seq = high_resolution_clock::now();
    sequential_quicksort(arr_seq, 0, ARRAY_SIZE - 1);
    auto stop_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<microseconds>(stop_seq - start_seq);
    cout << "Sequential QuickSort time: " << duration_seq.count() << " microseconds" << endl;

    // Parallel QuickSort
    int num_threads = omp_get_max_threads();
    cout << "Using " << num_threads << " threads" << endl;

    auto start_par = high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp single  // Only one thread does the initial call
        parallel_quicksort(arr_par, 0, ARRAY_SIZE - 1);

        #pragma omp taskwait // Ensure all tasks are completed
    }
    auto stop_par = high_resolution_clock::now();
    auto duration_par = duration_cast<microseconds>(stop_par - start_par);
    cout << "Parallel QuickSort time: " << duration_par.count() << " microseconds" << endl;

    // Verify the sort
    bool sorted_correctly = is_sorted(arr_par.begin(), arr_par.end());
    cout << "Is sorted correctly: " << (sorted_correctly ? "Yes" : "No") << endl;

    return 0;
}
