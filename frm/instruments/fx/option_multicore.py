# -*- coding: utf-8 -*-

import os
import pathlib

if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) # path to ./frm/
    print(__file__.split('\\')[-1], os.getcwd())

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time

NUMBER_OF_CORES = 8
APPLY_CONTROL_VARIATES = True
APPLY_PARALLEL_PROCESSING = False

x0 = 10
dt = 1/365
drift = 0.05
speed = 0.3

nb_timesteps = 10
nb_assets = 1
nb_iterations = 1000 * 1000
chunk_size = nb_iterations // NUMBER_OF_CORES


def helper(start_iter, end_iter):
    np.random.seed()
    sqrt_dt = np.sqrt(dt)
    local_nb_iterations = end_iter - start_iter
    
    if APPLY_CONTROL_VARIATES:
        # https://en.wikipedia.org/wiki/Control_variates
        W_1 = np.random.randn(int(nb_timesteps), int(nb_assets), int(local_nb_iterations // 2))
        result_1 = x0 * np.cumprod(np.exp((drift - 0.5 * speed**2) * dt + speed * sqrt_dt * W_1), axis=1)
        W_2 = -W_1
        result_2 = x0 * np.cumprod(np.exp((drift - 0.5 * speed**2) * dt + speed * sqrt_dt * W_2), axis=1)
        result = np.append(result_1, result_2, axis=2)
    else:
        W = np.random.randn(nb_timesteps, nb_assets, local_nb_iterations)
        result = x0 * np.cumprod(np.exp((drift - 0.5 * speed**2) * dt + speed * sqrt_dt * W), axis=1)

    return result


if __name__ == '__main__':
    
    
    print('# of cores:', multiprocessing.cpu_count())
    
    
    t1 = time.time()
    
    with ProcessPoolExecutor(max_workers=NUMBER_OF_CORES) as executor:
        future_results = [executor.submit(helper, i, i + chunk_size) for i in range(0, nb_iterations, chunk_size)]
        results = [future.result() for future in future_results]
        final_result = np.concatenate(results, axis=2)

    print(final_result[1,-1,:].mean())
    t2 = time.time()
    print(t2-t1)    



    final_result = helper(0, nb_iterations)
        
    print(final_result[1,-1,:].mean())
    t3 = time.time()
    print(t3-t2)

# Talk with Ruchita
# 1. https://en.wikipedia.org/wiki/Antithetic_variates - can we do this for GBM simulations?
# 2. Any other variance reduction techneques applicable?
# 3. Is there a way we can check the simuluation converged? 
# 
#   - is there a magic number i.e 1m iterations 
#   - note this example is a simple example and will be used as the base for more complex option valuations
# My parallel processing isn't much quicker, I was hoping for a improvement similiar to the number of cores?

# How would this be built in effeciently and could it consider parallel processing?
