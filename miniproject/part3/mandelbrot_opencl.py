import numpy as np
import pyopencl as cl
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
from typing import Union

def mandelbrot_gpu(C:np.complex64, max_iters:np.int32, threshold:np.float32, pl_id:str, grid_size:Union[tuple[np.uint16], None] = None) -> np.ndarray[np.int32]:
    """This function computes the Mandelbrot set. It contains instructions for a openCL kernel, to compute the Mandelbrot set in parallel, using either CPU or GPU.

    Args:
        C (np.complex64): Complex mesh grid for coordinates in the Mandelbrot set.
        max_iters (np.int32): Max iterations to check while computing. Using int32 to match with default dtype in openCL kernel.
        threshold (np.float32): Threshold for divergence. Using float32 to enable comparison with absolute value complex64 values.
        pl_id (str): Platform ID. CPU or GPU.
        grid_size (tuple[np.uint16], optional): Grid size for the local work grid. Needs to be tuple with just one value. Defaults to None.

    Returns:
        np.ndarray[np.int32]: NDarray containing the number of iterations before divergence for each pixel.
    """    
    
    # opencl setup
    platforms = {"CPU": 0, "GPU": 1}
    platform = cl.get_platforms()[platforms[pl_id]]  # platform (CPU or GPU)
    device = [platform.get_devices()[0]]

    ctx = cl.Context(devices=device)
    queue = cl.CommandQueue(ctx)

    # prepare mesh grid and output
    C_flat = np.complex64(C.flatten()) # flatten to simplify C array indexing
    mset = np.zeros(C_flat.shape, dtype=np.int32)

    # create buffers for mesh and output
    mf = cl.mem_flags  # shortcut to use memory flags
    C_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C_flat)  # copy to host device
    mset_buff = cl.Buffer(ctx, mf.WRITE_ONLY, mset.nbytes)  # create buffer with size of mset in bytes

    # define kernel
    mandelbrot_kernel = """
    #include <pyopencl-complex.h>
    __kernel void compute_mandelbrot(
        __global const cfloat_t *C_buff,
        __global int *mset_buff,
        const uint max_iters,
        const float threshold
        )
    {
        private int gid = get_global_id(0);
        private cfloat_t z = cfloat_new(0.0f, 0.0f);
        private cfloat_t zsq;
        
        private cfloat_t C = cfloat_new(C_buff[gid].real, C_buff[gid].imag);
        private uint iter = 0;

        while(iter < max_iters){
            zsq = cfloat_mul(z, z);
            z = cfloat_add(zsq, C);
            float abs_z = cfloat_abs(z);
            if (abs_z >= threshold) {
                break;
            }
            iter++;
        }
        mset_buff[gid] = iter;
    }
    """
    
    # create program
    program = cl.Program(ctx, mandelbrot_kernel)
    program.build()

    # run program on chosen device
    program.compute_mandelbrot(
        queue,
        (C_flat.size,),
        grid_size,  # local work grid
        C_buff,
        mset_buff,
        np.uint32(max_iters),  # use uint32 for max_iters
        np.float32(threshold)  # use float32 for threshold
    )
    
    # copy results back from device
    cl.enqueue_copy(queue, mset, mset_buff)

    # reshape mset back into array shape
    mset = np.reshape(mset, C.shape)

    return mset


def create_mesh(re_min:int, re_max:int, im_min:int, im_max:int, p_re:int, p_im:int) -> np.ndarray[np.complex64]:
    """This function constructs a complex mesh grid from limits for the real and imaginary axis.

    Args:
        re_min (int): Minimum value for the real axis
        re_max (int): Maximum value for the real axis
        im_min (int): Minimum value for the imaginary axis
        im_max (int): Maximum value for the imaginary axis
        p_re (int): Resolution for the real axis. Number of points on the axis
        p_im (int): Resoultion for the imaginary axis. Number of points on the axis

    Returns:
        np.ndarray[np.complex64]: Complex 2D mesh grid, used for computing the Mandelbrot set
    """    

    re = np.linspace(re_min, re_max, p_re, dtype=np.complex64)
    im = np.linspace(im_min, im_max, p_im, dtype=np.complex64)
    C = re[np.newaxis, :] + 1j * im[:, np.newaxis] # create axis across columns for real part and rows for imaginary part
    
    return C.astype(np.complex64)

if __name__ == "__main__":
    
    ############# GRID SIZE COMP #############
    resolution = 20000
    # grid_sizes = [8, 16, 32, 64, 128, 256]
    grid_sizes = [(8,), (16,), (32,), (64,), (128,), (256,), (512,)]
    re_min, re_max = -2.0, 1.0
    im_min, im_max = -1.5, 1.5
    max_iters = 100
    threshold = 2

    # dictionary to store times in
    times = {
        "CPU": {size: [] for size in grid_sizes},
        "GPU": {size: [] for size in grid_sizes}
    }

    platform_names = ["CPU", "GPU"]

    for pl in platform_names:
        for size in grid_sizes:
            try:
                print(f"{pl}: grid size {size}")
                start = timer()
                C = create_mesh(re_min, re_max, im_min, im_max, resolution, resolution)
                mset = mandelbrot_gpu(C, max_iters, threshold, pl, size)
                times[pl][size] = timer() - start
            except cl.LogicError:
                print(f"Invalid work group size: {size}. Skipping...")
                times[pl][size] = -1
                continue
        print("#####################################")

    # plot results 

    # exclude fields with invalid work group size
    valid_cpu_data = {size: runtime for size, runtime in times["CPU"].items() if runtime != -1}
    valid_gpu_data = {size: runtime for size, runtime in times["GPU"].items() if runtime != -1}

    # convert from tuples to lists
    # e.g. [(16,), (32,)] -> [16,32]
    grid_sizes_cpu = [size[0] for size in valid_cpu_data.keys()]
    runtime_cpu = list(valid_cpu_data.values())

    grid_sizes_gpu = [size[0] for size in valid_gpu_data.keys()]
    runtime_gpu = list(valid_gpu_data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes_cpu, runtime_cpu, "o-", label="CPU")
    plt.plot(grid_sizes_gpu, runtime_gpu, 'o-', label="GPU")

    plt.xlabel("Local Work Grid Size")
    plt.ylabel("Runtime [s]")
    plt.title(f"CPU vs GPU Performance: Mandelbrot resolution {resolution} x {resolution}")
    plt.grid(True)
    plt.legend()
    plt.xscale("log", base=2)  # Log scale for better visualization
    
    ############# RESOLUTION COMP #############
    resolutions = [1000, 2000, 3000, 5000, 10000, 15000]
    re_min, re_max = -2.0, 1.0
    im_min, im_max = -1.5, 1.5
    max_iters = 100
    threshold = 2

    # dictionary to store times in
    times = {
        "CPU": {res: [] for res in resolutions},
        "GPU": {res: [] for res in resolutions}
    }

    platform_names = ["CPU", "GPU"]

    for pl in platform_names:
        for res in resolutions:
            print(f"{pl}: resolution {res}")
            start = timer()
            C = create_mesh(re_min, re_max, im_min, im_max, res, res)
            mset = mandelbrot_gpu(C, max_iters, threshold, pl)
            times[pl][res] = timer() - start

    # plot results
    plt.Figure(figsize=(10,6))

    plt.plot(resolutions, [times["CPU"][res] for res in resolutions], label = "CPU")
    plt.plot(resolutions, [times["GPU"][res] for res in resolutions], label = "GPU")

    plt.xlabel("Grid Resolution")
    plt.ylabel("Runtime [s]")

    plt.title("CPU vs GPU Mandelbrot Performance Comparison")
    plt.legend()
    plt.grid(True)