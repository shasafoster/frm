# -*- coding: utf-8 -*-
"""
@author: Shasa Foster
https://www.linkedin.com/in/shasafoster
"""

import numpy as np
from scipy.stats import norm

def normalcorr(C: np.array, 
               rand_nbs: np.array):
    """
    Generate correlated pseudo random normal variates using the Cholesky factorization.
    
    Parameters:
    C (np.ndarray): Correlation matrix.
    rand_nbs (np.ndarray): Matrix of normally distributed pseudorandom numbers.
    
    Returns:
    np.ndarray: Correlated pseudo random normal variates.
    """    
    
    M = np.linalg.cholesky(C).T
    return (M @ rand_nbs.T).T


def simulate_heston(s0: float, 
                    mu: float, 
                    v0: float, 
                    vv: float, 
                    kappa: float, 
                    theta: float, 
                    rho: float, 
                    tau: float, 
                    dt: float, 
                    rand_nbs: np.array=None, 
                    method: str='quadratic_exponential'):
    """
     Simulate trajectories of the spot price and volatility processes in the Heston model.
     
     Parameters:
     s0 (float): initial spot price.
     mu (float): drift.
     v0 (float): initial volatility.
     vv (float): volatility of volatility.
     kappa (float): speed of mean reversion for volatility.
     theta (float): long-term mean of volatility.
     rho (float): correlation between spot price and volatility.
     T (float): time end point
     dt (float): the time step size
     no (np.ndarray, optional): 2-column array of normally distributed pseudorandom numbers. Defaults to None.
     simulation_method (str, optional): Defaults to 'quadratic_exponential'.
         {'quadratic_exponential,'euler_with_absorption_of_volatility_process','euler_with_reflection_of_volatility_process'}
     Returns:
     np.ndarray: 2-column array containing the simulated trajectories of the spot price and volatility.
     """    
     
    np.random.seed(0)
        
    if rand_nbs is None:
        rand_nbs = np.random.normal(loc=0, scale=1, size=(int(tau / dt), 2))

    
    if len(rand_nbs[:, 0]) != int(tau / dt):
        raise ValueError('Size of rand_nbs is inappropriate. Length of rand_nbs[:, 0] should be equal to tau/dt.')
    
    t = np.arange(0, np.ceil(tau / dt) + 1)
    sizet = len(t)
    x = np.zeros((sizet, 2))
    
    C = np.array([[1, rho], [rho, 1]])
    u = normalcorr(C, rand_nbs) * np.sqrt(dt)
    
    if method == 'quadratic_exponential':
        x[0, :] = [np.log(s0), v0]
        phiC = 1.5
        
        for i in range(1, sizet):
            m = theta + (x[i - 1, 1] - theta) * np.exp(-kappa * dt)
            s2 = (x[i - 1, 1] * vv ** 2 * np.exp(-kappa * dt) / kappa * (1 - np.exp(-kappa * dt)) +
                  theta * vv ** 2 / (2 * kappa) * (1 - np.exp(-kappa * dt)) ** 2)
            phi = s2 / m ** 2
            gamma1 = gamma2 = 0.5
            K0 = -rho * kappa * theta / vv * dt
            K1 = gamma1 * dt * (kappa * rho / vv - 0.5) - rho / vv
            K2 = gamma2 * dt * (kappa * rho / vv - 0.5) + rho / vv
            K3 = gamma1 * dt * (1 - rho ** 2)
            K4 = gamma2 * dt * (1 - rho ** 2)
            
            if phi <= phiC:
                b2 = 2 / phi - 1 + np.sqrt(2 / phi * (2 / phi - 1))
                a = m / (1 + b2)
                x[i, 1] = a * (np.sqrt(b2) + rand_nbs[i - 1, 1]) ** 2
            else:
                p = (phi - 1) / (phi + 1)
                beta = (1 - p) / m
                if 0 <= norm.cdf(rand_nbs[i - 1, 1]) <= p:
                    x[i, 1] = 0
                elif p < norm.cdf(rand_nbs[i - 1, 1]) <= 1:
                    x[i, 1] = 1 / beta * np.log((1 - p) / (1 - norm.cdf(rand_nbs[i - 1, 1])))
            
            x[i, 0] = x[i - 1, 0] + mu * dt + K0 + K1 * x[i - 1, 1] + K2 * x[i, 1] + \
                      np.sqrt(K3 * x[i - 1, 1] + K4 * x[i, 1]) * rand_nbs[i - 1, 0]
        
        x[:, 0] = np.exp(x[:, 0])
    elif method[:5] == 'euler':
        x[0, :] = [s0, v0]
        for i in range(1, sizet):
            if method == 'euler_with_absorption_of_volatility_process':
                x[i - 1, 1] = max(x[i - 1, 1], 0) 
            elif method == 'euler_with_reflection_of_volatility_process':
                x[i - 1, 1] = abs(x[i - 1, 1])
                
            x[i, 1] = x[i - 1, 1] + kappa * (theta - x[i - 1, 1]) * dt + vv * np.sqrt(x[i - 1, 1]) * u[i - 1, 1]
            x[i, 0] = x[i - 1, 0] + x[i - 1, 0] * (mu * dt + np.sqrt(x[i - 1, 1]) * u[i - 1, 0])
    
    return x