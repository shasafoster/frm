# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from frm.pricing_engine.monte_carlo_generic import generate_rand_nbs, MAX_SIMULATIONS_PER_LOOP

def clewlow_strickland_1_factor_simulate(forward_curve, nb_simulations, segments_per_day, T, alpha, sigma):
    """
    Simulates spot prices using the Clewlow-Strickland one-factor model based on a provided forward curve.

    Parameters
    ----------
    forward_curve : numpy.ndarray
        A two-dimensional array where the first column contains time steps and the second column contains the corresponding forward prices.
    nb_simulations : int
        Number of simulation paths to generate.
    segments_per_day : int
        Number of extra simulation steps per forward curve step; higher values increase the model's temporal resolution.
    T : float
        Time horizon in days for the simulation.
    alpha : float
        Mean reversion speed of the model.
    sigma : float
        Volatility of the underlying asset.

    Returns
    -------
    spot_px_result : numpy.ndarray
        A two-dimensional array of simulated spot prices. Each row corresponds to a time step, and each column to a simulation path.
    
    Reference:
    Ahmos Sansom (2024). Clewlow and Strickland Commodity one factor spot model 
    (https://www.mathworks.com/matlabcentral/fileexchange/26969-clewlow-and-strickland-commodity-one-factor-spot-model), 
    MATLAB Central File Exchange. Retrieved April 28, 2024.        
    
    ##### To do #####
    Currently the code assumes the forward curve step has a constant increment.
    Want it to support a generic structure where the increment may change and may be denomintated in days, months or years. 
    The code simply discretises it generally. Need to figure out if it should discretise evenly from t=0 to T=T or discretize based on the forward curve granularity (which may change)
    
    Probably easiest to simulate the entire forward curve, and not include the T input that Atmos did. 
    The user can process the simulation results after.
    
    Likely the easiest way is to add extra granuarity between the timesteps provided by the user. 
    That gives the user more power as well - they can apply interpolation if they want rather than us assuming they want it. 
    
    May not be relevant for this function but days is likely the best data structure for functions to expect your data. 
    Easiest to convert to anything else given a year frac function 1st calls a day count function. 
    """

    nb_simulations = int(nb_simulations)
    segments_per_day = int(segments_per_day)
    
    a = alpha
    σ = sigma

    # this would need to be adjusted to the time between t and t+1 based on the forward curve
    dt = 1.0 / segments_per_day 
    
    
    
    nb_steps = T * segments_per_day # The Clewlow Strickland 1 Factor model requires granular discretisation in order to converge
    tau = (np.arange(nb_steps+1) * dt)
    
    f = interp1d(forward_curve[:,0], forward_curve[:,1], kind='linear') # Interpolate the forward curve based on the steps 
    forward_curve_interp = f(tau)
    index_spot = np.arange(T+1) * segments_per_day
    ln_forward_curve_interp = np.log(forward_curve_interp)
    
    nb_simulations_per_loop = int(min(nb_simulations, MAX_SIMULATIONS_PER_LOOP // nb_steps))
    nb_loops = (nb_simulations + nb_simulations_per_loop - 1) // nb_simulations_per_loop
    
    spot_px_result = np.zeros((T+1, nb_simulations))
    ln_spot_px = np.zeros((nb_steps+1, nb_simulations_per_loop))
    ln_spot_px[0, :] = ln_forward_curve_interp[0] # Set t=0 to the current spot price
    
    # These terms evaluate to constants hence can be calculated outside the monte carlo loop
    term1 = (ln_forward_curve_interp[1:] - ln_forward_curve_interp[:-1]) / dt
    term1 = np.concatenate(([np.nan], term1)) # Prefix with nan for t=0
    term3 = -0.25 * σ**2 * (1.0 + np.exp(-2.0 * a * tau[:-1]))
    term3 = np.concatenate(([np.nan], term3)) # Prefix with nan for t=0    

    for j in range(nb_loops):        
        rand_nbs = generate_rand_nbs(nb_steps=nb_steps, nb_rand_vars=1, nb_simulations=nb_simulations_per_loop, flag_apply_antithetic_variates=False, random_seed=j)

        for i in range(1, ln_spot_px.shape[0]):
            term2 = a * (ln_forward_curve_interp[i-1] - ln_spot_px[i-1,:])
            ln_spot_px[i,:] = ln_spot_px[i-1,:] + (term1[i] + term2 + term3[i]) * dt + rand_nbs[i-1,0,:] * σ * np.sqrt(dt)   

        spot_px = np.exp(ln_spot_px)        
        
        idx_start = j*nb_simulations_per_loop
        idx_end = min((j+1)*nb_simulations_per_loop, nb_simulations)
        idx = np.arange(idx_start,idx_end)
        
        if j < (nb_loops-1):    
            spot_px_result[:, idx] = spot_px[index_spot, :]
        else:
            index_spot_2d = index_spot[:, np.newaxis]
            mask = np.arange(0,spot_px.shape[1]) < (idx_end-idx_start)
            spot_px_result[:, idx] = spot_px[index_spot_2d, mask]

    return spot_px_result
    


            
    








