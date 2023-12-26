import cupy as cp
import torch
import timeit
import cProfile
import io
import pstats
from loguru import logger

# Configure loguru logger to write to a file
logger.add("CUDA_profiling.log", format="{time} {level} {message}", level="INFO", rotation="1 week")

# CUDA kernel code
cuda_code = '''
extern "C" __global__
void feed_forward_kernel(const float* input, const float* weights, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = tanh(input[idx] * weights[idx]);  // Example activation function
    }
}
'''

def task():
    # logger.info("Starting CUDA kernel execution.")
    for _ in range(1000):
        kernel((grid_size,), (block_size,), (input_tensor.data_ptr(), weights_tensor.data_ptr(), output_tensor.data_ptr(), input_tensor.numel()))
    # logger.info("CUDA kernel execution completed.")

if __name__ == "__main__":
    # Compile the CUDA kernel
    module = cp.RawModule(code=cuda_code)
    kernel = module.get_function('feed_forward_kernel')
    N = 8192
    # Example data
    input_tensor = torch.randn(N, device='cuda')
    weights_tensor = torch.randn(N, device='cuda')

    # Allocate output tensor
    output_tensor = torch.randn(N, device='cuda')
    
    block_size = 32  # Number of threads per block
    grid_size = (input_tensor.numel() + block_size - 1) // block_size
    logger.info(f"CUDA Block size: {block_size}, CUDA Grid Size: {grid_size}")

    # Measure and log the time taken by the CUDA kernel
    start_time = timeit.default_timer()
    task()
    logger.info(f"Time taken by CUDA: {timeit.default_timer() - start_time} seconds")

    # Profiling with cProfile and capturing the output
    profiler = cProfile.Profile()
    profiler.enable()
    task()
    profiler.disable()

    profile_output = io.StringIO()
    ps = pstats.Stats(profiler, stream=profile_output)
    ps.print_stats()
    profile_output.seek(0)
    logger.info(f"Profiling Output:\n{profile_output.read()}")

    # Convert output to PyTorch tensor
    output_tensor = output_tensor.cpu().numpy()
