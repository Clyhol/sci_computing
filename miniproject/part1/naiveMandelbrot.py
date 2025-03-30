import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from numba import jit
from timeit import default_timer as timer
import line_profiler

# fastmath uses less accurate but faster operations
@line_profiler.profile
@jit(nopython = True, fastmath = True)
def compute_mandelbrot(c, max_iters, threshold):
    z = 0
    for n in range(max_iters):
        if abs(z) > threshold:
            return n
        else:
            z = z**2 + c
    return  max_iters # return max_iters if the point doesn't diverge after max_iters iterations

@line_profiler.profile
@jit(nopython = True, fastmath = True)
def mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold):
    re = np.linspace(re_min, re_max, p_re)
    im = np.linspace(im_min, im_max, p_im)
    
    mset = np.zeros((p_im, p_re)) # create an array to store the mandelbrot set
    
    for i in range(p_im): # rows (y-axis)
        for j in range(p_re): # columns (x-axis)
            c = re[j] +1j * im[i] # create a starting point
            mset[i, j] = compute_mandelbrot(c, max_iters, threshold) # store the number of iterations before diverge
    return mset


if __name__ == "__main__":
    p_re = 5000 # number of points in the real part
    p_im = 5000 # number of points in the imaginary part
    re_min, re_max = -2.0, 1.0
    im_min, im_max = -1.5, 1.5
    max_iters = 100
    threshold = 2
    outputdir = "miniproject/output"
    os.makedirs(outputdir, exist_ok=True)
    
    print("Computing Mandelbrot set...")
    start_time = timer()
    mset = mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold)
    print(f"Execution time: {timer() - start_time}s")
    
    plt.imshow(mset, extent=[re_min, re_max, im_min, im_max])
    plt.set_cmap('hot')
    plt.colorbar()
    plt.show()
    plt.imsave(os.path.join(outputdir, f"{os.path.basename(__file__).split(".")[0]}_{max_iters}_iters.pdf"), mset)
    
    # runtime for 100 iterations
    # JIT:      8.27s
    # No JIT:   203.71s 
    
            
    