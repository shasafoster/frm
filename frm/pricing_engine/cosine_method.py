# -*- coding: utf-8 -*

import numpy as np
from scipy.stats import norm

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


def chf_heston(r_d, r_f, tau, v0, vv, kappa, theta, rho):
    """
    Characteristic function for the Heston model.
    
    Parameters:
    - r_d (float): Risk-free domestic (quote) currency interest rate
    - r_f (float): Risk-free domestic (foreign) currency interest rate
    - tau (float): Time to maturity
    - v0 (float): Initial volatility
    - vv (float): Volatility of volatility
    - kappa (float): rate of mean reversion towards the long-term mean of variance process
    - theta (float): long-term mean of the variance process
    - rho (float): Correlation between price and volatility
    
    Returns:
    - function: Characteristic function taking φ (complex) as input and returning complex value
    """    
    
    D1 = lambda φ: np.sqrt((kappa - vv * rho * 1j * φ) ** 2 + (φ ** 2 + 1j * φ) * vv ** 2)
    g = lambda φ: (kappa - vv * rho * 1j * φ - D1(φ)) / (kappa - vv * rho * 1j * φ + D1(φ))
    C = lambda φ: (1 - np.exp(-D1(φ) * tau)) / (vv ** 2 * (1 - g(φ) * np.exp(-D1(φ) * tau)))
    A = lambda φ: (r_d - r_f) * 1j * φ * tau + kappa * theta * tau / vv ** 2 * (kappa - vv * rho * 1j * φ - D1(φ)) \
                   - 2 * kappa * theta / vv ** 2 * np.log((1 - g(φ) * np.exp(-D1(φ) * tau)) / (1 - g(φ)))
    return lambda φ: np.exp(A(φ) + C(φ) * v0)



#%% Coefficients for vanilla European options




def calculate_Uk_european_options(cp, a, b, k):
    """
    Compute Uk for valuing vanilla european call or put options under Heston using COS method.
    
    Parameters:
    - cp (int): 1 for call, -1 for put
    - a (float): Lower truncation boundary
    - b (float): Upper truncation boundary
    - k (np.array): Array of k values, [0, 1, 2, .... to N] where N is the number of expansion terms
    
    Returns:
    - np.array: Coefficients
    
    References:
    [1] Fang, Fang & Oosterlee, Cornelis. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. SIAM J. Scientific Computing. 31. 826-848. 10.1137/080718061. 
    """     

    def calc_chi(k, c, d):
        # Calculate χ, chi, Per equation 22 from [1] (page 6 of 21)
        χ = np.zeros(shape=k.shape)
        term1 = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
        term2 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)
        term3 = np.cos(k * np.pi  * (c - a) / (b - a)) * np.exp(c)
        term4 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a))
        term5 = k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
        χ = term1 * (term2 - term3 + term4 - term5)
        return χ
        
    def calc_psi(k, c, d):
        # Calculate Ψ, psi, per equation 23 from [1] (page 6 of 21)
        Ψ = np.zeros(shape=k.shape)        
        Ψ[1:] = (np.sin(k[1:] * np.pi * (d-a) / (b-a)) - np.sin(k[1:] * np.pi * (c-a) / (b-a))) * (b-a) / (k[1:] * np.pi)
        Ψ[0] = d - c
        return Ψ

    if cp == 1:
        # For call, c=0, d=b per equation 29 from [1] (page 7 of 21)
        χ = calc_chi(k=k, c=0, d=b)
        Ψ = calc_psi(k=k, c=0, d=b)      
        Uk_call = (2 / (b-a)) * (χ - Ψ)
        return Uk_call
    elif cp == -1:
        # For put, c=a, d]0, per equation 29 from [1] (page 7 of 21)
        χ = calc_chi(k=k, c=a, d=0)
        Ψ = calc_psi(k=k, c=a, d=0)
        Uk_put = (2 / (b-a)) * (-χ + Ψ)
        return Uk_put
    
    
def heston_cos_vanilla_european2(S0, tau, r_f, r_d, cp, K, v0, vv, kappa, theta, rho, N=160, L=3):
    
    """
    Computes the call or put option prices using the COS method.
    
    Parameters:
    - cp (int): 1 for call and -1 for put
    - S0 (float): Initial stock price
    - r (float): Interest rate
    - tau (float): Time to maturity
    - K (list or np.array): List of strike prices
    - N (int): Number of expansion terms (<160 should be sufficient per Fang, 2008)
    - L (float): Size of truncation domain
    
    Returns:
    - np.array: Option prices

    References:
    [1] Fang, Fang & Oosterlee, Cornelis. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. SIAM J. Scientific Computing. 31. 826-848. 10.1137/080718061. 
    """     

    x0 = np.log(S0 / K) # Per [1] in section 3.1 on page 6 of 21

    # Truncation domain
    a, b = -L * np.sqrt(tau), L * np.sqrt(tau)

    # Summation from k = 0 to k = N-1
    k = np.linspace(0, N-1, N).reshape(-1, 1)
    
    chf = chf_heston(r_f, r_d, tau, v0, vv, kappa, theta, rho)
    chf_result = chf((k*np.pi)/(b-a))      
    
    U = calculate_Uk_european_options(cp, a, b, k) # Per equation 29 of [1] (page 8 of 21)
    
    Σ = chf_result * U * np.exp(1j * k * np.pi * (x0 - a)/(b-a)) # Per the summation in equation 34 of [1] (page 8 of 21)
    Σ[0] = 0.5 * Σ[0] # Per page 3/21 of [1], "where Σ′ indicates that the first term in the summation is weighted by one-half"    
    v = K * np.exp(-(r_d) * tau) * np.real(sum(Σ)) # Per equation 34 of [1] (page 8 of 21)

    return v

#%% Misc 
# References:
# [1] Fang, Fang & Oosterlee, Cornelis. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. SIAM J. Scientific Computing. 31. 826-848. 10.1137/080718061. 

# Truncation range for COS method

# c1, c2, ... cn are the 1st, 2nd, .. nth cumulant of ln(ST/K)



def get_cos_truncation_range(model, model_param):

    # From Table 11 in the Appendix A of [1]
    if model == 'geomentric_brownion_motion':
        # GBM is defined by two parameters:
        # mu: the drift
        # σ: the volatility
        mu,σ,T = model_param
        
        c1 = mu * T
        c2 = (σ**2) * T
        c4 = 0
     
    elif model == 'heston':
        mu, kappa, theta, vv, v0, rho, T = model_param        
        # The Heston model is defined by six parameters 
        # - mu: the drift
        # - v0: Initial volatility.
        # - vv: Volatility of volatility.
        # - kappa: rate of mean reversion to the long-run variance
        # - theta: Long-run variance.
        # - lambda_: Market price of volatility risk.
        # - rho: Correlation.

        
        c1 = mu * T + (1- np.exp(-1 * kappa * T)) * (theta - v0) / (2 * kappa) - 0.5 * theta * T
        
        c2 = (1 / (8 * kappa**3)) * (
            (vv * T * kappa * np.exp(-kappa * T) * (v0 - theta) * (8 * rho - 4 * vv))
            + rho * vv * (1 - np.exp(-kappa * T)) * (16 * theta - 8 * v0)
            + 2 * theta * kappa * T * (-4 * kappa * rho + vv**2 + 4 * kappa**2)
            + vv**2 * ((theta - 2 * v0) * np.exp(-kappa * T) + theta * (6 * np.exp(-kappa * T) - 7) + 2 * v0)
            + 8 * kappa**2 * (v0 - theta) * (1 - np.exp(-kappa * T))
        )
        
        c4 = 0
          
    elif model == 'VG':
        # Detailed in paper but not priority for this project
        pass
    elif model == 'CGMY':
        # Detailed in paper but not priority for this project
        pass    
        
    L = 10
    a = c1 - L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4))) # Per Equantion 49 from [1]
    b = c1 + L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4))) # Per Equantion 49 from [1]

    return a,b




