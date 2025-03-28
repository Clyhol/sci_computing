import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from numba import jit
from timeit import default_timer as timer

def vector_mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold):
    re = np.linspace(re_min, re_max, p_re)
    im = np.linspace(im_min, im_max, p_im)
    C = re[np.newaxis, :] + 1j * im[:, np.newaxis] # create axis across columns for real part and rows for imaginary part
    
    z = np.copy(C) # populate z with starting points
    mset = np.zeros(C.shape) # array to store number of iterations before divergence
    
    for _ in tqdm(range(max_iters), desc="Computing Mandelbrot set"):
        mask = np.abs(z) <= threshold
        z[mask] = z[mask]**2 + C[mask] # perform operation on points that have not diverged
        mset += mask # add +1 to all points with true in the mask
    return mset

if __name__ == "__main__":
    p_re = 5000
    p_im = 5000
    re_min, re_max = -2.0, 1.0
    im_min, im_max = -1.5, 1.5
    max_iters = 100
    threshold = 2
    outputdir = "miniproject/output"
    os.makedirs(outputdir, exist_ok=True)
    
    mset = vector_mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold)
    
    plt.imshow(mset, extent=[re_min, re_max, im_min, im_max])
    plt.set_cmap('hot')
    plt.colorbar()
    plt.show()
    plt.imsave(os.path.join(outputdir, f"{os.path.basename(__file__).split(".")[0]}_{max_iters}_iters.pdf"), mset)
    
    # runtime for 100 iterations
    # No JIT:   18s
    
            
    