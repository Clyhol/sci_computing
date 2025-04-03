import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from dask import array as da
import dask
from dask.distributed import Client


def dask_compute_mandelbrot(C, max_iters, threshold):
    z = np.zeros(C.shape, dtype=np.complex64) # populate z with starting points
    mset = np.zeros(C.shape, dtype=np.uint8) # NOTE uint16 may be too small for large max_iters
    
    for _ in range(max_iters):
        # iterate points which didn't diverge, where mask is true
        mask = np.abs(z) <= threshold
        z[mask] = z[mask]**2 + C[mask]
        mset[mask] += 1
    return mset

def dask_mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold, chunk_size):
    # chunk size can differ for real and imaginary part
    re = da.linspace(re_min, re_max, p_re, chunk_size)
    im = da.linspace(im_min, im_max, p_im, chunk_size)

    # create complex plane
    x, y = da.meshgrid(re, im, indexing="xy")
    C = x + 1j * y
    
    # using map_blocks to fix memory issues
    mset = da.map_blocks(dask_compute_mandelbrot, C, max_iters, threshold, dtype=np.uint16)
    
    return mset

if __name__ == "__main__":
    p_re = 5000
    p_im = 5000
    re_min, re_max = -2.0, 1.0
    im_min, im_max = -1.5, 1.5
    max_iters = 100
    threshold = 2
    chunk_size = [50, 85, 100, 250, 500, 1000, 2500]

    results = {}

    client = Client(address="10.92.1.232:8786") # start local cluster
    print("Computing Mandelbrot...")
    for size in chunk_size:
        start_time = timer()
        mset = dask_mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold, chunk_size)
        mset = mset.compute()
        runtime = timer() - start_time
        results[size] = runtime
        print(f"Chunk size: {size}, Runtime: {runtime}")

    client.close()
    
    plt.figure()
    for size in chunk_size:
        plt.plot(chunk_size, results[size])
    plt.xlabel("Chunk size")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs. Chunk size")
    plt.grid()
    plt.xscale("log")
    
