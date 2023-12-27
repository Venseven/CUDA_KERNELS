# Accelerating ML Inference in Python and C++: Utilizing SIMD, Multithreading, and CUDA for Parallel Computing
In this repository, we focus on constructing a feedforward neural network layer by harnessing a variety of parallel computing methods in Python. This includes leveraging SIMD vector processing, utilizing multithreading through OpenMP, and optimizing with CUDA for GPU enhancement. Each approach is specifically designed to optimize the unique advantages offered by different hardware systems, thereby enhancing overall computational effectiveness. 

My primary objective revolves around optimizing machine learning model inferences at the CPU core level. This is especially crucial for models deployed as services, aiming to achieve peak efficiency and performance. 

The current scope of this repository is centered on implementing fundamental CPU optimization techniques for a **singular feedforward layer**.
## Table of Contents
1. [SIMD Vector Processing](#simd-vector-processing)
2. [Multithreading with OpenMP](#multithreading-with-openmp)
3. [CUDA for GPU Acceleration](#cuda-for-gpu-acceleration)

## SIMD Vector Processing
We explore SIMD (Single Instruction, Multiple Data) in C++ for parallel operations on data arrays, thereby harnessing CPU vectorization capabilities. We also explore SIMD with data prefetching for enhanced cache efficiency.

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

### SIMD Implementation with Prefetching in C++
```cpp
#define PD 64
#define WILL_READ_AND_WRITE 1
#define LOCALITY_LOW 1

extern "C" {
    void feed_forward_simd_prefetch(int N, const float* input, const float* weights, float* output) {
        #pragma omp simd
        for (int idx = 0; idx < N; ++idx) {
            if ((idx % 32) == 0) {
                __builtin_prefetch(&input[idx + PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
                __builtin_prefetch(&weights[idx + PD], WILL_READ_AND_WRITE, LOCALITY_LOW);
            }
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

# Integrating C++ and Python

This section has the basic steps which involves integerating the above c++ snippets in python

### Steps to Integrate

1. **Compile C++ Code to Shared Library**: First, ensure that the C++ feed forward layer code is compiled to a shared library. In our case, it's compiled to `libfeedforward.so`.

2. **Load the Library in Python**: Use `ctypes.CDLL` to load the compiled shared library.
   ```python
   import ctypes
   import numpy as np

   lib = ctypes.CDLL('/path/to/libfeedforward.so')
   ```

3. **Define Argument and Return Types**: Before calling functions from the shared library, you need to define the argument and return types for these functions. This is crucial as it allows `ctypes` to correctly pass data between Python and C++.
   ```python
   lib.feed_forward_MultSIMD_prefetch.argtypes = [
       ctypes.c_int,  # N (number of elements)
       np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # input
       np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # weights
       np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),  # output
       ctypes.c_int   # num_threads if needed
   ]
   lib.feed_forward_MultSIMD_prefetch.restype = None
   ```

4. **Call the Function**: With the setup complete, you can now call the C++ function as if it were a Python function.
   ```python
   N = 1000  # Example array size
   input_data = np.random.randn(N).astype(np.float32)
   weights = np.random.randn(N).astype(np.float32)
   output = np.empty(N, dtype=np.float32)
   num_threads = 4  # Example number of threads

   lib.feed_forward_MultSIMD_prefetch(N, input_data, weights, output, num_threads)
   ```


# Parallel Computing Techniques: Performance Analysis

This README presents a detailed analysis of the performance characteristics of different parallel computing approaches including CUDA, Multithreading, SIMD, and a combination of Multithreading & SIMD. The analysis is based on profiling data obtained from executing a feedforward layer implementation using these techniques.

## Performance Analysis

In this section, we provide a detailed analysis of the computational performance of various parallel computing techniques, measured in MFLOPS (Million Floating Point Operations Per Second). The analysis helps in understanding the efficiency and effectiveness of each method.

### Performance Summary Table

| Technique                     | Execution Time (seconds) | MFLOPS              | 
|-------------------------------|-------------------------|---------------------|
| Multithreading                | 0.34823                 | 47.04               |   
| SIMD                          | 0.3388                  | 48.3                |   
| SIMD (with prefetching)       | 0.31                    | 51.2                |
| Multithreading & SIMD Combined| 0.42                    | 38.14               |  
| Multithreading & SIMD prefetch| 0.23                    | 68.29               |  


### Torch ELU optimization

| Technique                     | Execution Time (seconds) | MFLOPS             | 
|-------------------------------|-------------------------|---------------------|
| Torch ELU                     | 0.0711                  | 28.79               |   
| CUDA Kernel ELU               | 0.0096                  | 211.9               |   

### Detailed Analysis

- **Multithreading:**
  - **Total FLOPs**: Approximately 16.37 billion.
  - **Analysis**: Shows good parallelization, but per-operation efficiency is lower than SIMD.

- **SIMD:**
  - **Total FLOPs**: Approximately 16.36 billion.
  - **Analysis**: Better efficiency per operation due to effective vectorization.

- **SIMD with Prefetching:**
  - **Total FLOPs**: Approximately 15.87 billion.
  - **Analysis**: Prefetching improves cache utilization, leading to faster execution.

- **Multithreading & SIMD Combined:**
  - **Total FLOPs**: Approximately 16.02 billion.
  - **Analysis**: The combination doesn't yield the expected improvement, possibly due to overheads.

- **Multithreading & SIMD with Prefetching:**
  - **Total FLOPs**: Approximately 15.71 billion.
  - **Analysis**: The most efficient method, significantly enhancing performance.



