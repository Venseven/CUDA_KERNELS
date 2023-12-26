import ctypes
import numpy as np
import timeit
import cProfile
import io
import pstats
from loguru import logger

# Configure loguru logger to write to a file
logger.add("SIMD_prefetch_profiling.log", format="{time} {level} {message}", level="INFO", rotation="1 week")

def task():

    for _ in range(1000):
        lib.feed_forward_simd_prefetch(N, input_data, weights, output)

if __name__ == "__main__":

    # Load the shared library
    lib = ctypes.CDLL('/scratch/subramav/CUDA_KERNELS/feed_forward/OPENMP/SIMD/libfeedforward.so')

    # Define the argument and return types of the feed_forward function
    lib.feed_forward_simd_prefetch.argtypes = [ctypes.c_int, 
                                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')]
    lib.feed_forward_simd_prefetch.restype = None

    # Example data
    N = 8192
    flops_per_operation = 2
    iterations = 1000
    total_flops = flops_per_operation * N * iterations
    input_data = np.random.randn(N).astype(np.float32)
    weights = np.random.randn(N).astype(np.float32)
    output = np.empty(N, dtype=np.float32)

    # Call the C++ function
    start_time = timeit.default_timer()
    task()
    execution_time = timeit.default_timer() - start_time
    logger.info(f"Time taken by SIMD: {execution_time} seconds")

    mflops = (total_flops / execution_time) / 1e6
    logger.info(f"mflops : {mflops}")

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

