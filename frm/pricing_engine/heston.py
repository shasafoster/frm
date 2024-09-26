# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 
    
import numpy as np
from scipy.stats import norm
from frm.pricing_engine.monte_carlo_generic import normal_corr


def simulate_heston_scalar(S0: float, 
                           mu: float, 
                           var0: float, 
                           vv: float, 
                           kappa: float, 
                           theta: float, 
                           rho: float, 
                           tau: float, 
                           rand_nbs: np.array,
                           method: str='quadratic_exponential') -> np.ndarray:
    
    """
    Simulate scalar (non-vectorised) trajectories of the spot price and volatility processes using the Heston model.
     
    Parameters:
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Drift term (r - q).
    var0 : float
        Initial variance.
    vv : float
        Volatility of volatility.
    kappa : float
        Speed of mean reversion for volatility.
    theta : float
        Long-run variance.
    rho : float
        Correlation between spot price and volatility.
    tau : float
        Time end point of the simulation.
    rand_nbs : np.ndarray
        Array of pseudorandom numbers for spot and volatility shocks. Shape=(# of timesteps, 2). 
    method : str, optional
        The method used for simulating variance paths. Defaults to 'quadratic_exponential'.
        Options:
        - 'quadratic_exponential'
        - 'euler_with_absorption_of_volatility_process'
        - 'euler_with_reflection_of_volatility_process'

    Returns:
    -------
    np.ndarray
        Simulated trajectories of spot price and variance. Shape is (nb_timesteps + 1, 2).

    References:
    ----------
    [1] Janek, A., Kluge, T., Weron, R., Wystup, U. (2010). "FX smile in the Heston model".
    """ 
    
    assert rand_nbs.shape[1] == 2
    nb_timesteps = rand_nbs.shape[0]
    dt = tau / nb_timesteps
    x = np.zeros((nb_timesteps + 1, rand_nbs.shape[1]))
    
    C = np.array([[1, rho], [rho, 1]])
    u = normal_corr(C, rand_nbs) * np.sqrt(dt)
    
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
        x[0, :] = [S0, var0]
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
                    var0: float, 
                    vv: float, 
                    kappa: float, 
                    theta: float, 
                    rho: float, 
                    tau: float, 
                    rand_nbs: np.array,
                    method: str='quadratic_exponential'):
    """
     Simulate trajectories of the spot price and volatility processes using the Heston model. Is Vectorised.


     
    Parameters:
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Drift term (r - q).
    var0 : float
        Initial variance.
    vv : float
        Volatility of volatility.
    kappa : float
        Speed of mean reversion for volatility.
    theta : float
        Long-run variance. 
    rho : float
        Correlation between spot price and volatility.
    tau : float
        Time end point of the simulation.
    rand_nbs : np.ndarray
        Array of pseudorandom numbers for spot and volatility shocks. Shape is (# of timesteps, 2, # of simulations).
    method : str, optional
        The method used for simulating variance paths. Defaults to 'quadratic_exponential'.
        Options:
        - 'quadratic_exponential'
        - 'euler_with_absorption_of_volatility_process'
        - 'euler_with_reflection_of_volatility_process'

    Returns:
    -------
    np.ndarray
        Simulated trajectories of spot price and variance. Shape is (nb_timesteps + 1, 2).

    References:
    ----------
    [1] Janek, A., Kluge, T., Weron, R., Wystup, U. (2010). "FX smile in the Heston model".
     """    
     
    assert rand_nbs.shape[1] == 2
    nb_timesteps = rand_nbs.shape[0]
    dt = tau / nb_timesteps
    x = np.zeros((nb_timesteps+1, rand_nbs.shape[1], rand_nbs.shape[2]))
    
    C = np.array([[1, rho], [rho, 1]])
    u = normal_corr(C, rand_nbs) * np.sqrt(dt) # need to check this correlation works as expected
    
    if method == 'quadratic_exponential':
        x[0, 0, :] = np.log(S0)
        x[0, 1, :] = var0
        
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
        x[0, 1, :] = var0
        
        for i in range(1, nb_timesteps + 1):
            if method == 'euler_with_absorption_of_volatility_process':
                x[i - 1, 1, :] = np.maximum(0, x[i - 1, 1, :])
            elif method == 'euler_with_reflection_of_volatility_process':
                x[i - 1, 1, :] = np.abs(x[i - 1, 1, :])

            x[i, 1, :] = x[i - 1, 1, :] + kappa * (theta - x[i - 1, 1, :]) * dt + vv * np.sqrt(x[i - 1, 1, :]) * u[i - 1, 1, :]
            x[i, 0, :] = x[i - 1, 0, :] + x[i - 1, 0, :] * (mu * dt + np.sqrt(x[i - 1, 1, :]) * u[i - 1, 0, :])
    
    return x







