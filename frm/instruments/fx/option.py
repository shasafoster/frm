# -*- coding: utf-8 -*-


import os
import pathlib

os.chdir(pathlib.Path(__file__).parent.parent.resolve()) 

import numpy as np
import multiprocessing as mp
import time

NUMBER_OF_CORES = mp.cpu_count()
APPLY_CONTROL_VARIATES = False
APPLY_PARALLEL_PROCESSING = True

np.random.seed(1) # Set random seed so we get repeatable results

def GeometricBrownian(x0, 
                      dt, 
                      drift, 
                      speed, 
                      W, 
                      correlation_matrix=None):
    """
    Generate the path of a Geomtric Brownian motion.

        X(t) = X(0) + N(drift * t, speed**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    The random variable of the position at time t, X(t), has a normal distribution whose mean is
    the position at time t=0 + drift*t and whose variance is speed**2*t.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) * exp(N(drift * dt, speed**2 * dt)) 

    Arguments
    ---------
    x0 : float
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    dt : float
        The time step.
    drift : float
        Specifies the "drift" of the Brownian motion. 
    speed : float
        Specifies "speed" of the Brownian motion.  
    W : numpy array  
        Random Normal Variates

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """
    
    nb_timesteps = 365
    nb_assets = 1
    nb_iterations = 100000 
    
    
    def helper(x0, dt, drift, speed, nb_timesteps, nb_assets, nb_iterations):
        np.random.seed()
        
        sqrt_dt = np.sqrt(dt)
        
        if APPLY_CONTROL_VARIATES:
            # https://en.wikipedia.org/wiki/Control_variates
            W_1 = np.random.randn(nb_timesteps,nb_assets,int(nb_iterations / 2))
            result_1 = x0 * np.cumprod(np.exp((drift - 0.5 * speed**2) * dt + speed * sqrt_dt * W_1),axis=1)  
            W_2 = -W_1 
            result_2 = x0 * np.cumprod(np.exp((drift - 0.5 * speed**2) * dt + speed * sqrt_dt * W_2),axis=1)
            result = np.append(result_1, result_2, axis=2)
        else:
            W = np.random.randn(nb_timesteps,nb_assets,nb_iterations)
            result = x0 * np.cumprod(np.exp((drift - 0.5 * speed**2) * dt + speed * sqrt_dt * W),axis=1)
            
        return result
            
    if APPLY_PARALLEL_PROCESSING:
        result_list = []
        def log_result(result):
            # This is called whenever helper(i) returns a result.
            # result_list is modified only by the main process, not the pool workers.
            result_list.append(result)
        
        pool = mp.Pool(NUMBER_OF_CORES)
        for i in range(10):
            res = pool.apply_async(helper, args=(x0, dt, drift, speed, nb_timesteps, nb_assets, nb_iterations,), callback=log_result)
            print(res.get())
        pool.close()
        pool.join()
        result = result_list
    else:
        result = helper(x0, dt, drift, speed, nb_timesteps, nb_assets, nb_iterations)


    # https://en.wikipedia.org/wiki/Antithetic_variates

    return result


# Multi core
# https://agustinus.kristia.de/techblog/2016/06/13/parallel-monte-carlo/
#
#


if __name__ == '__main__':

    x0 = 10
    dt = 1/365
    drift = 0.05
    speed = 0.3
    
    
    t1 = time.time()
    
    result = GeometricBrownian(x0, dt, drift, speed, None)
    
    t2 = time.time()
    
    print(t2-t1)


    
    
    