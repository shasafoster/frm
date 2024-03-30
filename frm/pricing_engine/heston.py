# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
import numpy as np
from numba import jit, prange
from scipy.stats import norm
import timeit
import time

from frm.frm.pricing_engine.monte_carlo_generic import generate_rand_nbs


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


def simulate_heston_single(S0: float, 
                           mu: float, 
                           var0: float, 
                           vv: float, 
                           kappa: float, 
                           theta: float, 
                           rho: float, 
                           tau: float, 
                           rand_nbs: np.array,
                           method: str='quadratic_exponential'):
    
    """
      Simulate trajectories of the spot price and volatility processes in the Heston model.
     
      Parameters:
      S0 (float): initial spot price.
      mu (float): drift (r-q)
      var0 (float): initial variance.
      vv (float): volatility of volatility.
      kappa (float): speed of mean reversion for volatility.
      theta (float): long-run variance
      rho (float): correlation between spot price and volatility.
      tau (float): time end point
      dt (float): the time step size
      rand_nbs (np.ndarray, optional): 2-column array of normally distributed pseudorandom numbers. Defaults to None.
      simulation_method (str, optional): Defaults to 'quadratic_exponential'.
          {'quadratic_exponential,'euler_with_absorption_of_volatility_process','euler_with_reflection_of_volatility_process'}
      Returns:
      np.ndarray: 2-column array containing the simulated trajectories of the spot price and volatility.
      
      References:
      [1] Janek, A., Kluge, T., Weron, R., Wystup, U. (2010). "FX smile in the Heston model"
      """    
     
    dt = tau / nb_timesteps

    t = np.arange(0, nb_timesteps + 1)
    x = np.zeros((nb_timesteps + 1, 2))
    
    C = np.array([[1, rho], [rho, 1]])
    u = normalcorr(C, rand_nbs) * np.sqrt(dt)
    
    if method == 'quadratic_exponential':
        x[0, :] = [np.log(S0), var0]
        phiC = 1.5
        
        for i in range(1, nb_timesteps + 1):
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
        x[0, :] = [S0, v0]
        for i in range(1, nb_timesteps+1):
            if method == 'euler_with_absorption_of_volatility_process':
                x[i - 1, 1] = max(x[i - 1, 1], 0) 
            elif method == 'euler_with_reflection_of_volatility_process':
                x[i - 1, 1] = abs(x[i - 1, 1])
                
            x[i, 1] = x[i - 1, 1] + kappa * (theta - x[i - 1, 1]) * dt + vv * np.sqrt(x[i - 1, 1]) * u[i - 1, 1]
            x[i, 0] = x[i - 1, 0] + x[i - 1, 0] * (mu * dt + np.sqrt(x[i - 1, 1]) * u[i - 1, 0])
    
    return x


def simulate_heston(S0: float, 
                    mu: float, 
                    v0: float, 
                    vv: float, 
                    kappa: float, 
                    theta: float, 
                    rho: float, 
                    tau: float, 
                    rand_nbs: np.array,
                    method: str='quadratic_exponential'):
    """
     Simulate trajectories of the spot price and volatility processes in the Heston model.
     
     Parameters:
     S0 (float): initial spot price.
     mu (float): drift.
     v0 (float): initial volatility.
     vv (float): volatility of volatility.
     kappa (float): speed of mean reversion for volatility.
     theta (float): long-term mean of volatility.
     rho (float): correlation between spot price and volatility.
     tau (float): time end point
     nb_timesteps (float): the number of timesteps in the simulation
     rand_nbs (np.ndarray, optional): 2-column array of normally distributed pseudorandom numbers. Defaults to None.
     simulation_method (str, optional): Defaults to 'quadratic_exponential'.
         {'quadratic_exponential,'euler_with_absorption_of_volatility_process','euler_with_reflection_of_volatility_process'}
     Returns:
     np.ndarray: 2-column array containing the simulated trajectories of the spot price and volatility.
     """    
     
    np.random.seed(0)
    
    nb_timesteps = rand_nbs.shape[0]
    dt = tau / nb_timesteps

    t = np.arange(0, nb_timesteps + 1)
    x = np.zeros((nb_timesteps+1, 2, rand_nbs.shape[2]))
    
    C = np.array([[1, rho], [rho, 1]])
    u = normalcorr(C, rand_nbs) * np.sqrt(dt) # need to check this correlation works as expected
    
    if method == 'quadratic_exponential':
        x[0, 0, :] = np.log(S0)
        x[0, 1, :] = v0
        
        phiC = 1.5
        
        for i in range(1, nb_timesteps + 1):
            m = theta + (x[i - 1, 1, :] - theta) * np.exp(-kappa * dt)
            s2 = (x[i - 1, 1, :] * vv ** 2 * np.exp(-kappa * dt) / kappa * (1 - np.exp(-kappa * dt)) +
                  theta * vv ** 2 / (2 * kappa) * (1 - np.exp(-kappa * dt)) ** 2)
            phi = s2 / m ** 2
            gamma1 = gamma2 = 0.5
            K0 = -rho * kappa * theta / vv * dt
            K1 = gamma1 * dt * (kappa * rho / vv - 0.5) - rho / vv
            K2 = gamma2 * dt * (kappa * rho / vv - 0.5) + rho / vv
            K3 = gamma1 * dt * (1 - rho ** 2)
            K4 = gamma2 * dt * (1 - rho ** 2)
            
            mask = phi <= phiC
            b2 = np.full(mask.shape, np.nan)
            b2[mask] = 2 / phi[mask] - 1 + np.sqrt(2 / phi[mask] * (2 / phi[mask] - 1))
            a = np.full(mask.shape, np.nan)
            a[mask] = m[mask] / (1 + b2[mask])
            x[i, 1, mask] = a[mask] * (np.sqrt(b2[mask]) + rand_nbs[i - 1, 1, mask]) ** 2
            
            mask = np.logical_not(mask)
            p = np.full(mask.shape, np.nan)
            p[mask] = (phi[mask] - 1) / (phi[mask] + 1)
            beta = np.full(mask.shape, np.nan)
            beta[mask] = (1 - p[mask]) / m[mask]
            
            mask_0_p = np.logical_and(0 <= norm.cdf(rand_nbs[i - 1, 1, :]), norm.cdf(rand_nbs[i - 1, 1, :]) <= p)
            mask_net = np.logical_and(mask, mask_0_p)
            x[i, 1, mask_net] = 0
            
            mask_p_1 = np.logical_and(p < norm.cdf(rand_nbs[i - 1, 1, :]), norm.cdf(rand_nbs[i - 1, 1, :]) <= 1)
            mask_net = np.logical_and(mask, mask_p_1)
            x[i, 1, mask_net] = 1 / beta[mask_net] * np.log((1 - p[mask_net]) / (1 - norm.cdf(rand_nbs[i - 1, 1, mask_net])))
                        
            x[i, 0, :] = x[i - 1, 0, :] + mu * dt + K0 + K1 * x[i - 1, 1, :] + K2 * x[i, 1, :] + \
                      np.sqrt(K3 * x[i - 1, 1, :] + K4 * x[i, 1, :]) * rand_nbs[i - 1, 0, :]
        
        x[:, 0] = np.exp(x[:, 0])
    elif method[:5] == 'euler':
        x[0, 0, :] = S0
        x[0, 1, :] = v0
        
        for i in range(1, nb_timesteps + 1):
            if method == 'euler_with_absorption_of_volatility_process':
                x[i - 1, 1, :] = np.maximum(0, x[i - 1, 1, :])
            elif method == 'euler_with_reflection_of_volatility_process':
                x[i - 1, 1, :] = np.abs(x[i - 1, 1, :])

            x[i, 1, :] = x[i - 1, 1, :] + kappa * (theta - x[i - 1, 1, :]) * dt + vv * np.sqrt(x[i - 1, 1, :]) * u[i - 1, 1, :]
            x[i, 0, :] = x[i - 1, 0, :] + x[i - 1, 0, :] * (mu * dt + np.sqrt(x[i - 1, 1, :]) * u[i - 1, 0, :])
    
    return x


#%%

if __name__ == '__main__':
    
    # This is a check to demonstate the vectorised function gives the same result as the non vectorised function
    
    np.random.seed(0)
    
    S0 = 0.6629
    mu = 0
    v0 = 0.01030476434426229
    vv = 0.2992984338043174
    kappa = 1.5
    theta = 0.013836406947876231
    rho = -0.3432643651463818
    tau = 2
    nb_timesteps = 100
    
    rand_nbs = generate_rand_nbs(nb_timesteps=nb_timesteps,
                                 nb_rand_vars=2,
                                 nb_simulations=1 * 1000,
                                 flag_apply_antithetic_variates=False)
   
    
    t1 = time.time()
    
    result = simulate_heston(S0=S0,
        mu=mu,
        v0=v0,
        vv=vv,
        kappa=kappa,
        theta=theta,
        rho=rho, 
        tau=tau, 
        rand_nbs=rand_nbs,
        method='quadratic_exponential')
    
    t2 = time.time()

    if True:
        for i in range(rand_nbs.shape[2]):
            
            print(i, '/', rand_nbs.shape[2])
    
            rand_nbs_single = rand_nbs[:,:,i]
    
            results_single = simulate_heston_single(S0=S0,
                mu=mu,
                v0=v0,
                vv=vv,
                kappa=kappa,
                theta=theta,
                rho=rho, 
                tau=tau, 
                rand_nbs=rand_nbs_single,
                method='quadratic_exponential')
                
            check = (result[:,:,i] - results_single).sum()
            epsilon = 1e-8
            assert check < epsilon
            
        t3 = time.time()
        
        print('Runtime of vectorised function:', t2-t1)
        print('Runtime of for-loop function:', t3-t2)
    
    result_avg = result.mean(axis=2)






