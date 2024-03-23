# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
import numpy as np
from numba import jit, prange
import timeit
import time

from frm.frm.pricing_engine.monte_carlo_generic import generate_rand_nbs


def simulate_gbm_path(initial_px: np.array,
                      drift: np.array, 
                      forward_volatility: np.array,
                      timestep_length: np.array, 
                      rand_nbs: np.array):
    """
    Simulates the path of a Geometric Brownian Motion (GBM) over a defined period set.

    This function simulates the path of a stochastic process following GBM, given initial prices,
    drift rates, volatilities, and random numbers for each period. It supports vectorized operations
    allowing simulation of multiple paths in parallel.

    Parameters:
    initial_px (np.array): An array of initial prices for each simulation path.
    drift (np.array): An array of drift rates for each time step.
    forward_volatility (np.array): An array of volatilities for each time step.
    timestep_length (np.array): An array representing the length of each step
    rand_nbs (np.array): A 3D array of random numbers; shape=(number of time steps, number of random variables, number of simulations)

    Returns:
    np.array: A 2D array where each row represents a simulated GBM path and each column
              represents a time step in the path.

    Raises:
    AssertionError: If the shapes of period_length, period_drift, period_forward_volatility,
                    and rand_nbs do not align as expected.

    Notes:
    - The function assumes that period_length, period_drift, and period_forward_volatility have
      the same shape and that the number of periods matches the number of columns in rand_nbs.
    - It simulates the path by sequentially updating the price for each period using the GBM formula.
    """    
    
    assert timestep_length.ndim == 1
    
    # Reshape inputs to 2D arrays so to support parallel simulation of multiple underlyings 
    drift = np.atleast_1d(drift)
    forward_volatility = np.atleast_1d(forward_volatility)
    initial_px = np.atleast_1d(initial_px)
    
    if drift.ndim == 1:
        drift = drift.reshape(-1, 1)
    if forward_volatility.ndim == 1:
        forward_volatility = forward_volatility.reshape(-1, 1)  
    if initial_px.ndim == 1:
        initial_px = initial_px.reshape(1, -1)          
    
    # Shape checks
    nb_timesteps = timestep_length.shape[0]
    assert nb_timesteps == drift.shape[0]
    assert nb_timesteps == forward_volatility.shape[0]
    assert nb_timesteps == rand_nbs.shape[0]
    assert initial_px.shape[0] == 1
    
    nb_rand_vars = initial_px.shape[1]
    assert nb_rand_vars == drift.shape[1]
    assert nb_rand_vars == forward_volatility.shape[1]
    assert nb_rand_vars == rand_nbs.shape[1]
    
    # Initialise shape and set initial value of underlying(s)
    x = np.zeros((rand_nbs.shape[0]+1, rand_nbs.shape[1], rand_nbs.shape[2]))
    x[0,:,:] = initial_px
    
    # Simulate per geometric brownian motion process
    for i in range(0, nb_timesteps):
        x[i+1,:,:] = x[i,:,:] * np.exp((drift[i,:] - 0.5 * (forward_volatility[i,:] ** 2)) * timestep_length[i] + forward_volatility[i,:] * np.sqrt(timestep_length[i]) * rand_nbs[i,:,:])
        
    return x


@jit(nopython=True)
def simulate_gbm_path_numba(initial_px: np.array,
                            drift: np.array, 
                            forward_volatility: np.array,
                            timestep_length: np.array, 
                            rand_nbs: np.array):
    
    assert timestep_length.shape == drift.shape
    assert timestep_length.shape == forward_volatility.shape
    assert timestep_length.shape[0] == rand_nbs.shape[0]
    assert initial_px.shape == drift.shape
    
    nb_periods = len(timestep_length)
    
    x = np.zeros((rand_nbs.shape[0],nb_periods+1))
    x[:,0,] = initial_px
    
    for i in range(0, nb_periods):
        x[:,i+1] = x[:,i] * np.exp((drift[i] - 0.5 * (forward_volatility[i] ** 2)) * timestep_length[i] + forward_volatility[i] * np.sqrt(timestep_length[i]) * rand_nbs[:,i])
        
    return x
            

            
if __name__ == "__main__":
            

    initial_px = np.array([0.6629])
    drift = np.array([0.95*0.01, 0.55*0.01, -0.02*0.01, -0.34*0.01])
    forward_volatility = np.array([9.73*0.01, 10.10*0.01, 9.75*0.01, 10.97*0.01])
    timestep_length = np.array([0.512328767, 0.498630137, 0.501369863, 0.498630137])
    nb_periods=len(timestep_length)
    
    rand_nbs = generate_rand_nbs(nb_periods=nb_periods,
                                 nb_rand_vars=1,
                                 nb_simulations=1000 * 1000)
    
    
    # results = simulate_gbm_path_numba(initial_px=initial_px,
    #                                   drift=drift,
    #                                   forward_volatility=forward_volatility,
    #                                   timestep_length=timestep_length,
    #                                   rand_nbs=rand_nbs)
    
        
    t1 = time.time()
    results = simulate_gbm_path(initial_px=initial_px,
                                      drift=drift,
                                      forward_volatility=forward_volatility,
                                      timestep_length=timestep_length,
                                      rand_nbs=rand_nbs)
    
    # t2 = time.time()
    # results = simulate_gbm_path_numba(initial_px=initial_px,
    #                                   drift=drift,
    #                                   forward_volatility=forward_volatility,
    #                                   timestep_length=timestep_length,
    #                                   rand_nbs=rand_nbs)
    
    t3 = time.time()
    
    
    print("----------")
    print("GBM:", t2-t1)
    print("GBM numba:", t3-t2)
    #print("GBM numba_prange:", t4-t3)    
    
    
    
    
    # # Time the original function
    # time_original = timeit.timeit(
    #     """simulate_gbm_path(initial_px=initial_px,
    #                                       period_drift=period_drift,
    #                                       period_forward_volatility=period_forward_volatility,
    #                                       period_length=period_length,
    #                                       rand_nbs=rand_nbs)""",
    #     globals=globals(),
    #     number=10
    # )
    
    # # Time the Numba-optimized function (first call compiles the function)
    # results = simulate_gbm_path_numba(initial_px=initial_px,
    #                                   period_drift=period_drift,
    #                                   period_forward_volatility=period_forward_volatility,
    #                                   period_length=period_length,
    #                                   rand_nbs=rand_nbs)
    
    # time_numba = timeit.timeit(
    #     """simulate_gbm_path_numba(initial_px=initial_px,
    #                                       period_drift=period_drift,
    #                                       period_forward_volatility=period_forward_volatility,
    #                                       period_length=period_length,
    #                                       rand_nbs=rand_nbs)""",
    #     globals=globals(),
    #     number=10
    # )
    
    #print(f"Time taken by original function: {time_original:.6f} seconds")
    #print(f"Time taken by Numba-optimized function: {time_numba:.6f} seconds")

#%%


            
            
            
            

def simulate_gbm(x0, mu, σ, T, dt, rand_nbs=None, method=0):
    """
    Simulate Geometric Brownian Motion (GBM).
    
    Parameters:
    x0 (float): Initial value of the process.
    mu (float): Drift.
    σ (float): Volatility.
    T (float): Time end point
    dt (float): Time step size.
    no (np.ndarray, optional): Array of normally distributed pseudorand nbs. Defaults to None.
    method (int, optional): Method for simulation. Defaults to 0.
        0: Direct integration
        1: Euler scheme
        2: Milstein scheme 1st order
        3: Milstein scheme 2nd order

    Returns:
    np.ndarray: Vector containing the simulated trajectory of the GBM.
    
    References
    [1]: Rouah, F.D. (2013). Euler Scheme for the Black-Scholes Model.
    """
    
    if rand_nbs is None:
        rand_nbs = np.rand.normal(0, 1, int(np.ceil(T / dt)))
    
    rand_nbs = np.asarray(rand_nbs).flatten()
    
    if len(rand_nbs) != int(np.ceil(T / dt)):
        raise ValueError("Error: length(no) != n/dt")
    
    if method == 0:
        x = x0 * np.exp(np.cumsum((mu - 0.5 * σ ** 2) * dt + σ * np.sqrt(dt) * rand_nbs))    
    elif method == 1:
        x = x0 * np.cumprod(1 + mu * dt + σ * np.sqrt(dt) * rand_nbs)
    elif method == 2:
        x = x0 * np.cumprod(1 + mu * dt + σ * np.sqrt(dt) * rand_nbs +
                            0.5 * σ ** 2 * dt * (rand_nbs ** 2 - 1))
    elif method == 3:
        x = x0 * np.cumprod(1 + mu * dt + σ * np.sqrt(dt) * rand_nbs +
                            0.5 * σ ** 2 * dt * (rand_nbs ** 2 - 1) +
                            mu * σ * rand_nbs * (dt ** 1.5) + 0.5 * (mu ** 2) * (dt ** 2))

    return np.concatenate(([x0], x))


@jit(nopython=True, parallel=True)
def simulate_gbm_numba_prange(x0, mu, σ, T, dt, rand_nbs=None, method=0):
    steps = int(np.ceil(T / dt))
    
    if rand_nbs is None:
        rand_nbs = np.rand.normal(0, 1, steps)
    
    rand_nbs = np.asarray(rand_nbs).flatten()
    
    if len(rand_nbs) != steps:
        raise ValueError("Error: length(no) != n/dt")
    
    sqrt_dt = np.sqrt(dt)
    
    x = np.empty(steps)
    
    if method == 0:
        for i in prange(steps):
            x[i] = np.exp((mu - 0.5 * σ ** 2) * dt + σ * sqrt_dt * rand_nbs[i])
        x = np.cumprod(x)
    elif method == 1:
        for i in prange(steps):
            x[i] = 1 + mu * dt + σ * sqrt_dt * rand_nbs[i]
        x = np.cumprod(x)
    elif method == 2:
        for i in prange(steps):
            x[i] = 1 + mu * dt + σ * sqrt_dt * rand_nbs[i] + 0.5 * σ ** 2 * (dt * (rand_nbs[i] ** 2 - 1))
        x = np.cumprod(x)
    elif method == 3:
        for i in prange(steps):
            x[i] = 1 + mu * dt + σ * sqrt_dt * rand_nbs[i] + 0.5 * σ ** 2 * (dt * (rand_nbs[i] ** 2 - 1)) + \
                   mu * σ * rand_nbs[i] * (dt ** 1.5) + 0.5 * (mu ** 2) * (dt ** 2)
        x = np.cumprod(x)
    
    result = np.empty(steps + 1)
    result[0] = x0
    for i in prange(steps):
        result[i + 1] = x0 * x[i]
    
    return result



@jit(nopython=True)
def simulate_gbm_numba(x0, mu, sigma, T, dt, rand_nbs=None, method=0):
    
    steps = int(np.ceil(T / dt))
    
    if rand_nbs is None:
        rand_nbs = np.rand.normal(0, 1, steps)
    
    if len(rand_nbs) != steps:
        raise ValueError("Error: length(no) != steps/dt")
    
    x = np.empty(steps+1)
    x[0] = x0
    
    if method == 0:
        for i in range(1, steps+1):
            x[i] = x[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand_nbs[i-1])
    elif method == 1:  # euler_scheme
        for i in range(1, steps+1):
            x[i] = x[i-1] * (1 + mu * dt + sigma * np.sqrt(dt) * rand_nbs[i-1])
    elif method == 2:  # milstein_scheme_1st_order
        for i in range(1, steps+1):
            x[i] = x[i-1] * (1 + mu * dt + sigma * np.sqrt(dt) * rand_nbs[i-1] +
                            0.5 * sigma ** 2 * dt * (rand_nbs[i-1] ** 2 - 1))
    elif method == 3:  # milstein_scheme_2nd_order
        for i in range(1, steps+1):
            x[i] = x[i-1] * (1 + mu * dt + sigma * np.sqrt(dt) * rand_nbs[i-1] +
                            0.5 * sigma ** 2 * dt * (rand_nbs[i-1] ** 2 - 1) +
                            mu * sigma * rand_nbs[i-1] * (dt ** 1.5) + 0.5 * (mu ** 2) * (dt ** 2))    
    return x


#%%



if __name__ == "__main__":
    
    # Numba is much faster after it has been compiled.
    # Key is having a problem where it gets compiled

    import timeit
    import time

    # Parameters
    x0 = 100
    mu = 0.05
    σ = 0.2
    n = 10000
    dt = 0.01
    rand_nbs = np.rand.normal(0, 1, int(np.ceil(n / dt)))
    
    
    t1 = time.time()
    x = simulate_gbm(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
    t2 = time.time()
    y = simulate_gbm_numba(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
    t3 = time.time()
    y = simulate_gbm_numba_prange(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
    t4 = time.time()
    
    print("----------")
    print("GBM:", t2-t1)
    print("GBM numba:", t3-t2)
    print("GBM numba_prange:", t4-t3)    




    # Time the original function
    time_original = timeit.timeit(
        'simulate_gbm(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)',
        globals=globals(),
        number=100
    )
    
    # Time the Numba-optimized function (first call compiles the function)
    simulate_gbm_numba(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
    
    time_numba = timeit.timeit(
        'simulate_gbm_numba(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)',
        globals=globals(),
        number=100
    )

    # Time the Numba-optimized function (first call compiles the function)
    simulate_gbm_numba_prange(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
        
    time_numba_prange = timeit.timeit(
        'simulate_gbm_numba_prange(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)',
        globals=globals(),
        number=100
    )    
    
    # Display results
    print(f"Time taken by original function: {time_original:.6f} seconds")
    print(f"Time taken by Numba-optimized function: {time_numba:.6f} seconds")
    print(f"Time taken by Numba-Prange-optimized function: {time_numba_prange:.6f} seconds")












