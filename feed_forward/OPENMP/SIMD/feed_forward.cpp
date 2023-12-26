#include <cmath>
#include <omp.h>

#define PD 64
#define WILL_READ_AND_WRITE 1

#define LOCALITY_LOW 1

extern "C"
{
void feed_forward_simd(int N, const float* input, const float* weights, float* output) {
    #pragma omp simd
    for (int idx = 0; idx < N; ++idx) {
        output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
    }
}

void feed_forward_mult(int N, const float* input, const float* weights, float* output, int num_threads) {
    // omp_set_num_threads(num_threads);
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

// PREFETECH DATA
void feed_forward_simd_prefetch(int N, const float* input, const float* weights, float* output) {
    #pragma omp simd
    for (int idx = 0; idx < N; ++idx) {
        if ((idx%32) == 0)
        {
            __builtin_prefetch(&input[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
            __builtin_prefetch(&weights[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
        }
        output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
    }
}

void feed_forward_MultSIMD_prefetch(int N, const float* input, const float* weights, float* output, int num_threads) {
        // omp_set_num_threads(num_threads);
        #pragma omp parallel for 
        for (int idx = 0; idx < N; ++idx) {
            
            if ((idx%32) == 0)
            {
                __builtin_prefetch(&input[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
                __builtin_prefetch(&weights[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
            }
            #pragma omp simd
            output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
        }
    }

}