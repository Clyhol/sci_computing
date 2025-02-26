import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_mandelbrot(c, max_iters, threshold):
    z = 0
    for n in range(max_iters):
        if abs(z) > threshold:
            return n
        else:
            z = z**2 + c
    return  max_iters # return max_iters if the point doesn't diverge after max_iters iterations

def mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold):
    re = np.linspace(re_min, re_max, p_re) # real part
    im = np.linspace(im_min, im_max, p_im) # imaginary part
    
    mset = np.zeros((p_re, p_im)) # create an array to store the mandelbrot set
    
    for i in tqdm(range(p_im), desc="Calculating Mandelbrot set"): # rows (y-axis)
        for j in range(p_re): # columns (x-axis)
            c = re[j] +1j * im[i] # create a starting point
            mset[i, j] = compute_mandelbrot(c, max_iters, threshold) # store the number of iterations to diverge
    return mset


if __name__ == "__main__":
    p_re = 5000
    p_im = 5000
    re_min, re_max = -2.0, 1.0
    im_min, im_max = -1.5, 1.5
    max_iters = 100 # 1000 iters, 14:32 runtime. 100 iters, 2:32 runtime
    threshold = 2
    
    mset = mandelbrot_set(re_min, re_max, im_min, im_max, p_re, p_im, max_iters, threshold)
    
    plt.imshow(mset, extent=[re_min, re_max, im_min, im_max])
    plt.set_cmap('hot')
    plt.colorbar()
    plt.show()
    plt.imsave(f"{max_iters}_iters.pdf", mset)
    
    
            
    