#include <cmath>
#include <omp.h>

extern "C"
{
void feed_forward_simd(int N, const float* input, const float* weights, float* output) {
    #pragma omp simd
    for (int idx = 0; idx < N; ++idx) {
        output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
    }
}

void feed_forward_mult(int N, const float* input, const float* weights, float* output, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int idx = 0; idx < N; ++idx) {
        output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
    }
}


void feed_forward_MultSIMD(int N, const float* input, const float* weights, float* output, int num_threads) {
        // omp_set_num_threads(num_threads);
        #pragma omp parallel for 
        for (int idx = 0; idx < N; ++idx) {
            #pragma omp simd
            output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
        }
    }


}