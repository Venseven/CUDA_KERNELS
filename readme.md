# Parallel Computing in Python and C++: SIMD, Multithreading, and CUDA
This repository demonstrates the implementation of a feedforward neural network layer using various parallel computing techniques in Python. It includes SIMD vector processing, Multithreading using OpenMP, and CUDA for GPU acceleration. Each method leverages different hardware capabilities to enhance computational performance.
## Table of Contents
1. [SIMD Vector Processing](#simd-vector-processing)
2. [Multithreading with OpenMP](#multithreading-with-openmp)
3. [CUDA for GPU Acceleration](#cuda-for-gpu-acceleration)

## SIMD Vector Processing
We explore SIMD (Single Instruction, Multiple Data) in C++ for parallel operations on data arrays, thereby harnessing CPU vectorization capabilities.

### SIMD Implementation in C++
```cpp
extern "C" {
    void feed_forward_simd(int N, const float* input, const float* weights, float* output) {
        #pragma omp simd
        for (int idx = 0; idx < N; ++idx) {
            output[idx] = tanh(input[idx] * weights[idx]);
        }
    }
}
```

## Multithreading with OpenMP
We demonstrate how OpenMP can be used for multithreading in C++, enabling the distribution of computational tasks across multiple CPU threads.

### Multithreading Implementation in C++
```cpp
extern "C" {
    void feed_forward_mult(int N, const float* input, const float* weights, float* output, int num_threads) {
        omp_set_num_threads(num_threads);
        #pragma omp parallel for
        for (int idx = 0; idx < N; ++idx) {
            output[idx] = tanh(input[idx] * weights[idx]);
        }
    }
}
```

## Combined Multithreading and SIMD
Here we combine multithreading and SIMD, leveraging both task-level and data-level parallelism for enhanced performance.

### Combined Implementation in C++
```cpp
extern "C" {
    void feed_forward_MultSIMD(int N, const float* input, const float* weights, float* output, int num_threads) {
        // omp_set_num_threads(num_threads); // Optional thread count setting
        #pragma omp parallel for 
        for (int idx = 0; idx < N; ++idx) {
            #pragma omp simd
            output[idx] = tanh(input[idx] * weights[idx]);
        }
    }
}
```

## CUDA for GPU Acceleration
Delving into CUDA, we demonstrate the application of this powerful GPU acceleration platform for large-scale data processing tasks.

### Key Highlights Across Techniques
- **SIMD Vector Processing**: Enhancing CPU performance through vectorized operations, integrated with Python using `ctypes`.
- **Multithreading with OpenMP**: Utilizing CPU cores to their fullest with parallel loop executions.
- **CUDA**: Direct implementation of CUDA kernels for handling computationally intensive tasks on NVIDIA GPUs.

Stay tuned for further updates and implementations. Contributions and suggestions are always welcome!


# Parallel Computing Techniques: Performance Analysis

This README presents a detailed analysis of the performance characteristics of different parallel computing approaches including CUDA, Multithreading, SIMD, and a combination of Multithreading & SIMD. The analysis is based on profiling data obtained from executing a feedforward layer implementation using these techniques.

## Performance Summary Table

| Technique                     | Execution Time (seconds) | Total Function Calls | Significant Function Calls |
|-------------------------------|-------------------------|----------------------|----------------------------|
| CUDA                          | 0.01084                 | 4002                 | data_ptr (3000 calls)      |
| Multithreading                | 0.37423                 | 18002                | from_param (3000 calls)    |
| SIMD                          | 0.33688                 | 18002                | from_param (3000 calls)    |
| Multithreading & SIMD Combined| 0.24423                 | 18002                | from_param (3000 calls)    |

## Detailed Analysis

### CUDA Profiling
- **Fast Execution**: CUDA shows the fastest execution with only 0.01084 seconds. This demonstrates the efficiency of GPU parallel processing for data-intensive tasks.
- **Lower Function Calls**: CUDA has fewer total function calls, indicating a more direct computation path and less overhead.

### Multithreading Profiling
- **Increased Overhead**: With the highest execution time of 0.37423 seconds, multithreading shows increased overhead compared to CUDA.
- **High Function Calls**: The higher number of function calls suggests more time spent in setting up and managing threads.

### SIMD Profiling
- **Moderate Performance**: SIMD provides a middle ground in performance, with an execution time of 0.33688 seconds.
- **Effective for Vectorizable Tasks**: The results indicate that SIMD is effective for tasks that can be vectorized but may not reach the performance levels of GPU processing.

### Combined Multithreading and SIMD
- **Improved Performance Over Multithreading Alone**: The combined approach shows better performance (0.24423 seconds) than using multithreading alone.
- **Balanced Approach**: This method balances task-level and data-level parallelism, showcasing a significant performance improvement.

### Conclusion
- **CUDA is Most Efficient for Parallelizable Tasks**: For operations that are highly parallelizable, CUDA provides the best performance.
- **Effective Use of CPU Parallelism**: While not as fast as CUDA, SIMD and combined techniques efficiently utilize CPU parallelism.
- **Selection Based on Task and Hardware**: The choice of technique should consider the nature of the task and available hardware resources.



