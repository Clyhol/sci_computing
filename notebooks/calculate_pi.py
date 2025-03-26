import time
from dask.distributed import Client, wait
import matplotlib.pyplot as plt
import numpy as np

# this function returns the number of times the point fell
# within the unit circle out of n attempts
def in_circle(n, dummy):
    coords = np.random.rand(n,2) # random coordinates in (n,2) array
    count = 0
    for i in range(n):
        if (coords[i][0])**2 + (coords[i][1])**2 < 1:
            count += 1
    return count


# this function calculates the pi in parallel and measures
# the runtime
def parallel_pi(num_points, num_calls, client):
    np.random.seed(42)
       
    start = time.time()
    
    counts = client.map(in_circle, [num_points]*num_calls, range(num_calls))
    total = client.submit(sum, counts)
    wait(total) # wait for the total process to finish

    pi_estimate = 4*total.result()/num_points/num_calls
    
    stop = time.time()
    time_ex = stop-start # runtime
    
    return [pi_estimate, time_ex]

if __name__ == "__main__":
    num_cores = 20
    num_calls = [10, 100, 1000, 10000]
    client = Client("10.92.0.XXXX:8786")
    
    results = []
    for num in num_calls:

        pi_estimate, time_ex = parallel_pi(100000, num, client)
        results.append([pi_estimate, time_ex])
    
    client.close()
