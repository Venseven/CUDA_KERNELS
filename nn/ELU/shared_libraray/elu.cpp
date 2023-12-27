#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void elu(
    const float* input,
    const float* output,
    float alpha
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (input[index] < 0)
    {
        output[index] = alpha * (exp(input[index]) - 1);
    }

    }