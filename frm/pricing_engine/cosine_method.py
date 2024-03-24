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

#%%

def heston_cos_vanilla_european2(s0, tau, r_f, r_d, cp, K, v0, vv, kappa, theta, rho, N=160, L=10, calculate_via_put_call_parity=True):
    
    """
    Computes the call or put option prices using the COS method.
    
    Parameters:
    - cp (int): 1 for call and -1 for put
    - s0 (float): Initial stock price
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

    x0 = np.log(s0 / K) # Per [1] in section 3.1 on page 6 of 21

    if True:
        # Apply the truncation method per appendix 11 of [1] 
        mu = r_d - r_f # technically we should use the drift implied from market FX forward rate, but the IR parity forward rate should be close enough
        model_param =  {'tau': tau,
                        'mu': mu,
                        'v0': v0,
                        'vv': vv,
                        'kappa': kappa,
                        'theta': theta, 
                        'rho': rho}
        a, b = get_cos_truncation_range(model='heston', L=L, model_param=model_param)
    else:
        # This is a simpler truncation method
        # It produces a wider bound than the more complex method detailed in [1]
        a, b = -L * np.sqrt(tau), L * np.sqrt(tau)

    # Summation from k = 0 to k = N-1
    k = np.arange(N)
    
    u = (k*np.pi)/(b-a)
    chf = chf_heston_cos(u=u, tau=tau, r_f=r_f, r_d=r_d, v0=v0, vv=vv, kappa=kappa, theta=theta, rho=rho)
    
    Uk_call = calculate_Uk_european_options(cp=1, a=a, b=b, k=k) # Per equation 29 of [1] (page 8 of 21)
    Uk_put = calculate_Uk_european_options(cp=1, a=a, b=b, k=k) # Per equation 29 of [1] (page 8 of 21)
    
    Fk = np.real(chf  * np.exp(1j * k * np.pi * (x0 - a)/(b-a))) 
    Fk[0] = 0.5 * Fk[0] # Per page 3/21 of [1], "where Σ′ indicates that the first term in the summation is weighted by one-half"    
    
    # Note for the equations below, K can adjusted to be a vector, per equation 34 of [1] (page 8 of 21)
    if cp == -1 or calculate_via_put_call_parity:
        put_px = K * np.sum(np.multiply(Fk, Uk_put)) * np.exp(-r_d * tau) 
    if cp == 1 and calculate_via_put_call_parity:
        call_px = put_px + s0 * np.exp(-r_f * tau) - K * np.exp(-r_d * tau) # Method by put-call parity is more stable
    elif cp == 1 and not calculate_via_put_call_parity:
        call_px = K * np.sum(np.multiply(Fk, Uk_call)) * np.exp(-r_d * tau) 
    
    if cp == 1:
        return call_px
    elif cp == -1:
        return put_px
    


#%% Misc 


def get_cos_truncation_range(model, L, model_param):
    """
    # This function returns the truncation range for COS method approximations of stochastic models
    # as detailed in Appendix of Fang & Oosterlee (2008), in Table 11, page 21/21.
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
        mu = model_param['mu'] # drift
        u0 = model_param['v0'] # Initial volatility.
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
          
    elif model == 'VG':
        # Detailed in paper but not priority for this project
        pass
    elif model == 'CGMY':
        # Detailed in paper but not priority for this project
        pass    
        
    a = c1 - L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4))) # Per Equantion 49 from [1]
    b = c1 + L * np.sqrt(np.abs(c2) + np.sqrt(np.abs(c4))) # Per Equantion 49 from [1]

    return a,b


# def chf_heston_lambda(log_s0, tau, r_f, r_d, v0, vv, kappa, theta, rho):
#     """
#     Compute the characteristic function for the Heston model.
    
#     Parameters:
#     u (float or np.array): The argument of the characteristic function.
#     log_s0 (float): Natural Log of the initial spot price (in # of domestic currency units per 1 foreign currency unit)
#     log_K: Natural log of strike price (in # of domestic currency units per 1 foreign currency unit)
#     tau (float): Time to maturity (in years)
#     r_f (float): Foreign risk-free interest rate.
#     r_d (float): Domestic risk-free interest rate.
#     v0 (float): Initial volatility level.
#     vv (float): Volatility of volatility.
#     kappa (float): Rate of mean reversion of the variance
#     theta (float): Long-term level of volatility.
#     rho (float): Correlation between stock price and volatility.

#     Returns:
#         charFunc (float or np.array): Value of the characteristic function at u.
    
#     [1] S.Heston, (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options
#     [2] Albrecher, Hansjoerg & Mayer, Philipp & Schoutens, Wim & Tistaert, Jurgen. (2007). The Little Heston Trap. Wilmott. 83-92
#     """    
    
#     # Remap the parameters per the symbols used in [2] so comparison to the formulae on page 4 of 21 is easier
#     k = kappa 
#     η = theta 
#     λ = vv
#     σ0 = v0
#     del kappa, theta, vv, v0    
    
#     # Per definition in equation (1) of [2] on page 4/21
#     d = lambda u: np.sqrt((rho*λ*u*1j - k)**2 + λ**2 * (1j*u + u**2))
    
#     # There are two formula's for "g", in the Heston characteristic function because d, an input to g, has two roots. 
#     # Let "g1", be the presentation in the original Heston 1993 paper (on page 5 of 17 of [1])
#     # Let "g2", be the alternative, that is detailed in [2], "The little heston trap".
#     # g2 = 1 / g1. 
#     # We use g2, as using g1 leads to numerical problems based on most software implementations of calculation involving complex numbers        
#     g2 = lambda u: (k - rho *λ*1j*u - d(u)) / (k - rho*λ*1j*u + d(u))
    
#     # 1st inner exponential term on line 1 of 3, in equation (2) in [2]
#     A = lambda u: 1j*u * (log_s0 + (r_d - r_f) * tau) 
    
#     # 2nd inner exponential term on line 2 of 3, in equation (2) in [2]
#     B = lambda u: η*k*(λ**-2) * ((k - rho*λ*1j*u - d(u))*tau - 2*np.log((1 - g2(u)*np.exp(-d(u) * tau)) / (1 - g2(u))))
    
#     # 3rd inner exponential term on line 3 of 3, in equation (2) in [2]
#     C = lambda u: (σ0**2) * (λ**-2) * (k - rho*λ*1j*u - d(u)) * (1 - np.exp(-d(u) * tau)) / (1 - g2(u) * np.exp(-d(u) * tau)) 
    
#     chf = lambda u: np.exp(A(u) + B(u) + C(u))
    
#     return chf


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


def chf_heston_lambda(log_s0, tau, r_f, r_d, v0, vv, kappa, theta, rho):
    """
    Compute the characteristic function for the Heston model.
    
    Parameters:
    u (float or np.array): The argument of the characteristic function.
    log_s0 (float): Natural Log of the initial spot price (in # of domestic currency units per 1 foreign currency unit)
    log_K: Natural log of strike price (in # of domestic currency units per 1 foreign currency unit)
    tau (float): Time to maturity (in years)
    r_f (float): Foreign risk-free interest rate.
    r_d (float): Domestic risk-free interest rate.
    v0 (float): Initial volatility level.
    vv (float): Volatility of volatility.
    kappa (float): Rate of mean reversion of the variance
    theta (float): Long-term level of volatility.
    rho (float): Correlation between stock price and volatility.

    Returns:
        charFunc (float or np.array): Value of the characteristic function at u.
    
    [1] S.Heston, (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options
    [2] Albrecher, Hansjoerg & Mayer, Philipp & Schoutens, Wim & Tistaert, Jurgen. (2007). The Little Heston Trap. Wilmott. 83-92
    """    


def chf_heston_cos(u, tau, r_f, r_d, v0, vv, kappa, theta, rho):
    
    mu = r_d - r_f
    
    λ = kappa  # mean reversion speed to the long-run variance
    u_bar = theta # long-run variance
    η = vv # volatiltiy of the volatility
    u0 = v0 # initial variance
    
    d = np.sqrt(np.power(λ - 1j * rho * η * u, 2) + np.power(η,2) * (np.power(u,2) + u * 1j))
    g2 = (λ - 1j * rho * η * u - d) / (λ - 1j * rho * η * u + d)
    C = np.divide(λ * u_bar, np.power(η,2)) * ( (λ - 1j * rho * η * u - d) * tau - 2 * np.log(np.divide((1 - g2 * np.exp(-d * tau)) , (1-g2)) ))
    D = 1j * r_f * u * tau + u0 / np.power(η,2) * (np.divide((1 - np.exp(-d * tau)), (1 - g2 * np.exp(-d * tau)))) * (λ - 1j * rho * η * u - d) 
    phi = np.exp(D) * np.exp(C)
    
    return phi
