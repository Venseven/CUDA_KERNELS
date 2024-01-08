#include <iostream>
#include <cmath>
#include <omp.h>


#define PD 64
#define WILL_READ_AND_WRITE 1
#define WILL_READ 1
#define LOCALITY_LOW 1

// PREFETCH DATA
void feed_forward_simd_prefetch(int N, const float* input, const float* weights, float* output) {
        omp_set_num_threads(4);
        #pragma omp parallel for 
        for (int idx = 0; idx < N; ++idx) {
            
            if ((idx%32) == 0)
            {
                __builtin_prefetch(&input[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
                __builtin_prefetch(&weights[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
            }
            output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
        }
}

void feed_forward_simd_(int N, const float* input, const float* weights, float* output) {
        omp_set_num_threads(4);
        #pragma omp parallel for 
        for (int idx = 0; idx < N; ++idx) {
            
            if ((idx%32) == 0)
            {
                __builtin_prefetch(&input[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
                __builtin_prefetch(&weights[idx+PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
            }
            output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
        }
} 
int main() {
    const int N = 100;  // You can adjust the size as needed
    float input[N];
    float weights[N];
    float output[N];

    // Initialize input and weights arrays (you may want to fill them with actual data)

    // Call the function
    feed_forward_simd_prefetch(N, input, weights, output);
    // feed_forward_simd_(N, input, weights, output);

    // Display or process the output as needed

    return 0;
}
