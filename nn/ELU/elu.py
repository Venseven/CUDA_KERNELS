import cupy as cp
import torch
import timeit
import cProfile
import io
import pstats
from loguru import logger

# Configure loguru logger to write to a file
logger.add("ELU_profiling.log", format="{time} {level} {message}", level="INFO", rotation="1 week")

# CUDA kernel code
cuda_code = '''
extern "C" __global__ void elu(
    const float* input,
    float* output,
    float alpha
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (input[index] < 0)
    {
        output[index] = alpha * (exp(input[index]) - 1);
    }

    }
'''

def task():
    # logger.info("Starting CUDA kernel execution.")
    for _ in range(1000):
        kernel((grid_size,), (block_size,), (input_tensor.data_ptr(), output_tensor.data_ptr(), alpha))
    # logger.info("CUDA kernel execution completed.")


def tasks():
    # logger.info("Starting CUDA kernel execution.")
    for _ in range(1000):
        torch.nn.ELU(input_tensor)

if __name__ == "__main__":
    # Compile the CUDA kernel
    module = cp.RawModule(code=cuda_code)
    kernel = module.get_function('elu')
    N = 1024
    # Example data
    flops_per_operation = 2
    iterations = 1000
    total_flops = flops_per_operation * N * iterations
    input_tensor = torch.randn(N, device='cuda')
    # weights_tensor = torch.randn(N, device='cuda')

    # Allocate output tensor
    output_tensor = torch.randn(N, device='cuda')
    alpha = 0.001
    block_size = 32  # Number of threads per block
    grid_size = (input_tensor.numel() + block_size - 1) // block_size
    logger.info(f"CUDA Block size: {block_size}, CUDA Grid Size: {grid_size}")

    # Measure and log the time taken by the CUDA kernel
    start_time = timeit.default_timer()
    task()
    execution_time = timeit.default_timer() - start_time
    logger.info(f"Time taken by CUSTOM FUCNTION: {timeit.default_timer() - start_time} seconds")
    mflops = (total_flops / execution_time) / 1e6
    logger.info(f"mflops for custom function: {mflops}")
    # Profiling with cProfile and capturing the output
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = timeit.default_timer()
    tasks()
    execution_time = timeit.default_timer() - start_time
    profiler.disable()
    mflops = (total_flops / execution_time) / 1e6
    logger.info(f"mflops for torch ELU: {mflops}")
    logger.info(f"Time taken by torch ELU: {execution_time} seconds")
    profile_output = io.StringIO()
    ps = pstats.Stats(profiler, stream=profile_output)
    ps.print_stats()
    profile_output.seek(0)
    logger.info(f"Profiling Output:\n{profile_output.read()}")

    # Convert output to PyTorch tensor
    output_tensor = output_tensor.cpu().numpy()
