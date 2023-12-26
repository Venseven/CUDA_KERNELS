import ctypes
import numpy as np
import timeit
import io
import pstats
from loguru import logger
import cProfile

logger.add("MultSIMD_Prefetch_profiling.log", format="{time} {level} {message}", level="INFO", rotation="1 week")

def task():
    # logger.info("Starting CUDA kernel execution.")
    # Run the kernel
    for _ in range(1000):
        lib.feed_forward_MultSIMD_prefetch(N, input_data, weights, output, num_threads)    # logger.info("CUDA kernel execution completed.")

if __name__ == "__main__":

    # Load the shared library
    lib = ctypes.CDLL('/scratch/subramav/CUDA_KERNELS/feed_forward/OPENMP/SIMD/libfeedforward.so')

    # Define the argument and return types of the feed_forward function
    lib.feed_forward_MultSIMD_prefetch.argtypes = [ctypes.c_int, 
                                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                                ctypes.c_int]
    lib.feed_forward_MultSIMD_prefetch.restype = None

    # Example data
    N = 8192
    num_threads = N
    logger.info(f"Num Threads - {num_threads}")

    input_data = np.random.randn(N).astype(np.float32)
    weights = np.random.randn(N).astype(np.float32)
    output = np.empty(N, dtype=np.float32)

    # Call the C++ function
    start_time = timeit.default_timer()
    task()
    logger.info(f"Time taken by {num_threads} CPU threads: {timeit.default_timer() - start_time} seconds")
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
