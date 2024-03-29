# -*- coding: utf-8 -*

import numpy as np
from scipy.stats import norm


if __name__ == "__main__":
    import os
    import pathlib
    import sys
    
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())


#%% Test COS method function with the normal distribution

def cos_method(p, cf, dt, a, b, N=160):
    """
    PROBLEM STATEMENT: The charactheristic function (or moment generating function, or
    Laplace transform, from which to retrieve it) of a certain random variable X is known. 
    The objective is approximating f and F, respectively PDF and CDF of X.

    Parameters
    ----------
    p : TYPE
        point at which to evaluate the CDF (F) of f - must be in [a,b].
    cf : TYPE
        characteristic function provided as a function handle where the only variable is the point at which to evaluate it. 
    N : int
        Number of sums, per [1], N=160 the COS method quickly priced options with high accuracy
    dt : float
        Step size for approximating the integral
    a : float
        Lower integration bound
    b : float
        Upper integration bound

    Returns
    -------
    Fp : float
        CDF evaluated at p
    F : float
        CDF evaluated on pts
    f : float
        PDF evaluated on pts
    I : float
        Integral of the PDF over [a,b]: 
    pts : float
        points over which the CDF and PDF are sampled: a:dt:pts (and point p)    
        
        
    References:
    [1] Fang, Fang & Oosterlee, Cornelis. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. SIAM J. Scientific Computing. 31. 826-848. 10.1137/080718061. 
    """
    
    assert dt > 0.0    
    assert (a + dt) <= b
    assert p >= a
    assert p <= b
    
    pts = np.sort(np.unique(np.concatenate(([p], np.arange(a, b + dt, dt))))) # Array from [a, a+dt, a+2*dt, ..., b]
    dt_pts = np.append(np.diff(pts), dt)
    k = np.arange(N + 1) # Array from [0, 1, 2, ... N, N+1]

    w = np.append(0.5, np.ones(N))
    
    Fk = 2 / (b-a) * np.real(cf(k * np.pi / (b-a)) * np.exp(-1j * k * np.pi * a / (b-a))) # Equation 8 in [1] 
    C = lambda x: np.cos(k * np.pi * (x - a) / (b-a)) # Inner term of equation 11 in [1] 
    PDF_pts = np.array([np.sum(w * (Fk * C(x))) for x in pts]) # Equation 11 in [1]

    integral_of_a_to_b = np.sum(PDF_pts * dt_pts)
    if integral_of_a_to_b < norm.cdf(5):
        raise ValueError('The integral of the PDF over [a,b] is not close enough to 1. Please increase the range of [a,b]')
    
    CDF_pts = np.cumsum(PDF_pts * dt_pts)
    CDF_p = CDF_pts[np.where(pts == p)[0][0]]

    return CDF_p, CDF_pts, PDF_pts, integral_of_a_to_b, pts


def get_cos_truncation_range(model, L, model_param):
    """
    # This function returns the truncation range of the characteristic function 
    # Defined for COS method approximations of stochastic models as detailed in Appendix of Fang & Oosterlee (2008), in Table 11, page 21/21.
    # Note: c1, c2, ... cn are the 1st, 2nd, .. nth cumulant of ln(ST/K)
    # References:
    # [1] Fang, Fang & Oosterlee, Cornelis. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. SIAM J. Scientific Computing. 31. 826-848. 10.1137/080718061. 
        
    Parameters
    ----------
    model : str
        the type of stochastic model (geomentric_brownion_motion, heston, VG, CGMY).
    model_param : dict
        defining paramaters of the stochastic model

    Returns
    -------
    a : float
        lower truncation value.
    b : float
        upper truncation value.

    """
    
    # From Table 11 in the Appendix A of [1]
    if model == 'geomentric_brownion_motion':
        # GBM is defined by two parameters:
        # mu: the drift
        # σ: the volatility
        mu,σ,tau = model_param
        
        c1 = mu * tau
        c2 = (σ**2) * tau
        c4 = 0
     
    elif model == 'heston':
        # Remap the parameters per the symbols used in [1] so comparison to the formulae on page 21 of 21 is easier
        tau = model_param['tau'] # time-to-expiry
        mu = model_param['mu'] # drift mu. mu = r-q. For FXO mu = r_DOM - r_FOR
        u0 = model_param['var0'] # Initial variance.
        η  = model_param['vv'] # Volatility of volatility.
        λ = model_param['kappa'] # rate of mean reversion to the long-run variance
        u_bar = model_param['theta'] # Long-run variance
        rho = model_param['rho'] # Correlation.
                        
        c1 = mu * tau + (1- np.exp(-1 * λ * tau)) * (u_bar - u0) / (2 * λ) - 0.5 * u_bar * tau
        
        c2 = (1 / (8 * λ**3)) * (
            (η * tau * λ * np.exp(-λ * tau) * (u0 - u_bar) * (8 * λ * rho - 4 * η))
            + λ * rho * η * (1 - np.exp(-λ * tau)) * (16 * u_bar - 8 * u0)
            + 2 * u_bar * λ * tau * (-4 * λ * rho * η + η**2 + 4 * λ**2)
            + η**2 * ((u_bar - 2 * u0) * np.exp(-2 * λ * tau) + u_bar * (6 * np.exp(-λ * tau) - 7) + 2 * u0)
            + 8 * λ**2 * (u0 - u_bar) * (1 - np.exp(-λ * tau))
        )
                
        c4 = 0
        
        #print("SF: c1,c2", c1, c2)
          
    elif model == 'VG':
        # Detailed in paper but not priority for this project
        pass
    elif model == 'CGMY':
        # Detailed in paper but not priority for this project
        pass    
        
    a = c1 - L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4))) # Per Equantion 49 from [1]
    b = c1 + L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4))) # Per Equantion 49 from [1]

    return a,b







