# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import time

from frm.pricing_engine.monte_carlo_generic import generate_rand_nbs


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
    drift = np.atleast_2d(drift)
    forward_volatility = np.atleast_2d(forward_volatility)
    initial_px = np.atleast_2d(initial_px)

    # if drift.ndim == 1:
    #     drift = drift.reshape(-1, 1)
    # if forward_volatility.ndim == 1:
    #     forward_volatility = forward_volatility.reshape(-1, 1)
    # if initial_px.ndim == 1:
    #     initial_px = initial_px.reshape(1, -1)

    if drift.shape[0] == 1:
        drift = drift.reshape(-1, 1)
    if forward_volatility.shape[0] == 1:
        forward_volatility = forward_volatility.reshape(-1, 1)
    if initial_px.shape[0] == 1:
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
    x[0, :, :] = initial_px

    # Simulate per geometric brownian motion process
    for i in range(0, nb_timesteps):
        x[i+1, :, :] = x[i, :, :] * np.exp((drift[i, :] - 0.5 * (forward_volatility[i, :] ** 2)) *
                                           timestep_length[i] + forward_volatility[i, :] * np.sqrt(timestep_length[i]) * rand_nbs[i, :, :])

    return x


if __name__ == "__main__":

    initial_px = np.array([0.6629])
    drift = np.array([0.95*0.01, 0.55*0.01, -0.02*0.01, -0.34*0.01])
    forward_volatility = np.array([9.73*0.01, 10.10*0.01, 9.75*0.01, 10.97*0.01])
    timestep_length = np.array([0.512328767, 0.498630137, 0.501369863, 0.498630137])
    nb_steps = len(timestep_length)

    rand_nbs = generate_rand_nbs(nb_steps=nb_steps,
                                 nb_rand_vars=1,
                                 nb_simulations=1000 * 1000)


    t1 = time.time()
    results = simulate_gbm_path(initial_px=initial_px,
                                drift=drift,
                                forward_volatility=forward_volatility,
                                timestep_length=timestep_length,
                                rand_nbs=rand_nbs)

    t2 = time.time()
    
    print("GBM:", t2-t1)




# %%



