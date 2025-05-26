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
const int DEFAULT_ARRAY_SIZE = 1000;
const int DEFAULT_SEQUENTIAL_THRESHOLD = 100;

int main() {
    // Get parameters from environment variables
    const char* array_size_str = getenv("ARRAY_SIZE");
    int ARRAY_SIZE = DEFAULT_ARRAY_SIZE;
    if (array_size_str != nullptr) {
        ARRAY_SIZE = atoi(array_size_str);
    }

    const char* sequential_threshold_str = getenv("SEQUENTIAL_THRESHOLD");
    int SEQUENTIAL_THRESHOLD = DEFAULT_SEQUENTIAL_THRESHOLD;
    if (sequential_threshold_str != nullptr) {
        SEQUENTIAL_THRESHOLD = atoi(sequential_threshold_str);
    }

    // Output parameters
    cout << "Array size: " << ARRAY_SIZE << endl;
    cout << "Sequential threshold: " << SEQUENTIAL_THRESHOLD << endl;

    // Initialize arrays
    vector<vector<double>> arr1(ARRAY_SIZE, vector<double>(ARRAY_SIZE));
    vector<vector<double>> arr2(ARRAY_SIZE, vector<double>(ARRAY_SIZE));
    vector<vector<double>> arr_sum(ARRAY_SIZE, vector<double>(ARRAY_SIZE));
    vector<vector<double>> arr_diff(ARRAY_SIZE, vector<double>(ARRAY_SIZE));
    vector<vector<double>> arr_prod(ARRAY_SIZE, vector<double>(ARRAY_SIZE));
    vector<vector<double>> arr_div(ARRAY_SIZE, vector<double>(ARRAY_SIZE));

    // Fill arrays with random values
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        for (int j = 0; j < ARRAY_SIZE; ++j) {
            arr1[i][j] = dis(gen);
            arr2[i][j] = dis(gen);
        }
    }

    // Sequential operations
    cout << "Sequential operations:" << endl;

    auto start_sum_seq = high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        for (int j = 0; j < ARRAY_SIZE; ++j) {
            arr_sum[i][j] = arr1[i][j] + arr2[i][j];
        }
    }
    auto stop_sum_seq = high_resolution_clock::now();
    auto duration_sum_seq = duration_cast<microseconds>(stop_sum_seq - start_sum_seq);
    cout << "Sequential Sum time: " << duration_sum_seq.count() << " microseconds" << endl;

    auto start_diff_seq = high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        for (int j = 0; j < ARRAY_SIZE; ++j) {
            arr_diff[i][j] = arr1[i][j] - arr2[i][j];
        }
    }
    auto stop_diff_seq = high_resolution_clock::now();
    auto duration_diff_seq = duration_cast<microseconds>(stop_diff_seq - start_diff_seq);
    cout << "Sequential Difference time: " << duration_diff_seq.count() << " microseconds" << endl;

    auto start_prod_seq = high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        for (int j = 0; j < ARRAY_SIZE; ++j) {
            arr_prod[i][j] = arr1[i][j] * arr2[i][j];
        }
    }
    auto stop_prod_seq = high_resolution_clock::now();
    auto duration_prod_seq = duration_cast<microseconds>(stop_prod_seq - start_prod_seq);
    cout << "Sequential Product time: " << duration_prod_seq.count() << " microseconds" << endl;

    auto start_div_seq = high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        for (int j = 0; j < ARRAY_SIZE; ++j) {
            if (arr2[i][j] != 0.0) {
                arr_div[i][j] = arr1[i][j] / arr2[i][j];
            } else {
                arr_div[i][j] = 0.0;
            }
        }
    }
    auto stop_div_seq = high_resolution_clock::now();
    auto duration_div_seq = duration_cast<microseconds>(stop_div_seq - start_div_seq);
    cout << "Sequential Division time: " << duration_div_seq.count() << " microseconds" << endl;


    // Parallel section
    vector<int> num_threads_list = {1, 2, 4, 8, 16};

    cout << "\nParallel operations:" << endl;
    for (int num_threads : num_threads_list) {
        cout << "\nNumber of threads: " << num_threads << endl;
        omp_set_num_threads(num_threads);

        //Sum
        auto start_sum_par = high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            for (int j = 0; j < ARRAY_SIZE; ++j) {
                arr_sum[i][j] = arr1[i][j] + arr2[i][j];
            }
        }
        auto stop_sum_par = high_resolution_clock::now();
        auto duration_sum_par = duration_cast<microseconds>(stop_sum_par - start_sum_par);
        cout << "Parallel Sum time: " << duration_sum_par.count() << " microseconds" << endl;

        //Difference
        auto start_diff_par = high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            for (int j = 0; j < ARRAY_SIZE; ++j) {
                arr_diff[i][j] = arr1[i][j] - arr2[i][j];
            }
        }
        auto stop_diff_par = high_resolution_clock::now();
        auto duration_diff_par = duration_cast<microseconds>(stop_diff_par - start_diff_par);
        cout << "Parallel Difference time: " << duration_diff_par.count() << " microseconds" << endl;

        //Product
        auto start_prod_par = high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            for (int j = 0; j < ARRAY_SIZE; ++j) {
                arr_prod[i][j] = arr1[i][j] * arr2[i][j];
            }
        }
        auto stop_prod_par = high_resolution_clock::now();
        auto duration_prod_par = duration_cast<microseconds>(stop_prod_par - start_prod_par);
        cout << "Parallel Product time: " << duration_prod_par.count() << " microseconds" << endl;

        //Division
        auto start_div_par = high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            for (int j = 0; j < ARRAY_SIZE; ++j) {
                if (arr2[i][j] != 0.0) {
                    arr_div[i][j] = arr1[i][j] / arr2[i][j];
                } else {
                    arr_div[i][j] = 0.0;
                }
            }
        }
        auto stop_div_par = high_resolution_clock::now();
        auto duration_div_par = duration_cast<microseconds>(stop_div_par - start_div_par);
        cout << "Parallel Division time: " << duration_div_par.count() << " microseconds" << endl;
    }

    return 0;
}