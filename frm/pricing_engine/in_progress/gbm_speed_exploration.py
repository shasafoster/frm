# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
from numba import jit, prange
import timeit
import time


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
        rand_nbs = np.random.normal(0, 1, int(np.ceil(T / dt)))

    rand_nbs = np.asarray(rand_nbs).flatten()

    if len(rand_nbs) != int(np.ceil(T / dt)):
        raise ValueError("Error: length(no) != n/dt")

    if method == 0:
        x = x0 * np.exp(np.cumsum((mu - 0.5 * σ ** 2) *
                        dt + σ * np.sqrt(dt) * rand_nbs))
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
        rand_nbs = np.random.normal(0, 1, steps)

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
            x[i] = 1 + mu * dt + σ * sqrt_dt * rand_nbs[i] + \
                0.5 * σ ** 2 * (dt * (rand_nbs[i] ** 2 - 1))
        x = np.cumprod(x)
    elif method == 3:
        for i in prange(steps):
            x[i] = 1 + mu * dt + σ * sqrt_dt * rand_nbs[i] + 0.5 * σ ** 2 * (dt * (rand_nbs[i] ** 2 - 1)) + \
                mu * σ * rand_nbs[i] * (dt ** 1.5) + \
                0.5 * (mu ** 2) * (dt ** 2)
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
        rand_nbs = np.random.normal(0, 1, steps)

    if len(rand_nbs) != steps:
        raise ValueError("Error: length(no) != steps/dt")

    x = np.empty(steps+1)
    x[0] = x0

    if method == 0:
        for i in range(1, steps+1):
            x[i] = x[i-1] * np.exp((mu - 0.5 * sigma ** 2)
                                   * dt + sigma * np.sqrt(dt) * rand_nbs[i-1])
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



if __name__ == "__main__":

    # Numba is much faster after it has been compiled.
    # Key is having a problem where it gets compiled

    # Parameters
    x0 = 100
    mu = 0.05
    σ = 0.2
    n = 10000
    dt = 0.01
    rand_nbs = np.random.normal(0, 1, int(np.ceil(n / dt)))

    t1 = time.time()
    x = simulate_gbm(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
    t2 = time.time()
    y = simulate_gbm_numba(x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
    t3 = time.time()
    y = simulate_gbm_numba_prange(
        x0, mu, σ, n, dt, rand_nbs=rand_nbs, method=0)
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
    print(
        f"Time taken by Numba-Prange-optimized function: {time_numba_prange:.6f} seconds")