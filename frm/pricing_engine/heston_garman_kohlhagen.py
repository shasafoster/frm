# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    import sys
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())
   

from frm.frm.pricing_engine.garman_kohlhagen import gk_solve_implied_σ, gk_solve_strike
from frm.frm.pricing_engine.cosine_method_generic import get_cos_truncation_range
import numpy as np
import scipy.fft
import scipy  
from numba import jit, njit
from typing import Tuple
import warnings

VALID_HESTON_PRICING_METHODS = [
    'heston_analytical_1993',
    'heston_carr_madan_gauss_kronrod_quadrature',
    'heston_carr_madan_fft_w_simpsons',
    'heston_cosine'
]

def validate_input(var, var_name, validation_fn):
    if np.isscalar(var):
        if not validation_fn(var):
            raise ValueError(f"'{var_name}' has invalid values: {var}")
    else:
        if not np.all(validation_fn(var)):
            raise ValueError(f"'{var_name}' has invalid values: {var}")


def heston_fit_vanilla_fx_smile(
        Δ: np.array, 
        Δ_convention: str, 
        σ_market: np.array, 
        S0: float, 
        r_f: float, 
        r_d: float, 
        tau: float, 
        cp: np.array,
        pricing_method='carr_madan_gauss_kronrod_quadrature') -> Tuple[float, float, float, float, float, float, np.array, float]:
    """
    Fit the Heston model to the FX market implied volatility smile.
    
    Parameters:
    - Δ (np.array): Vector of spot delta values
    - Δ_convention (str): Delta convention ('prem-adj' or 'prem-adj-fwd')
    - σ_market (np.array): Vector of market implied volatilities
    - S0 (float): Spot price
    - r_f (float): Foreign interest rate
    - r_d (float): Domestic interest rate
    - tau (float): Time to maturity in years
    - cp (np.array): Vector of option types (1 for call, -1 for put)

    Returns:
    - Tuple: Initial variance (var0), vol of vol (vv), mean reversion (kappa), long-run mean variance (theta), market price of volatility risk (lambda_), correlation (rho), vector of implied volatilities (IV), sum of squared errors (SSE)

    References:
    [1] Janek, A., Kluge, T., Weron, R., Wystup, U. (2010). "FX smile in the Heston model"    

    The Heston SDE is:
        dS(t) = µ*S*dt + σ(t)*S*dW1(t)
        dσ(t) = kappa(theta - σ(t))*dt + vv*σ(t)*dW2(t)

    The Heston model is defined by six parameters 
    - var0: Initial variance.
    - vv: Volatility of volatility.
    - kappa: rate of mean reversion to the long-run variance
    - theta: Long-run variance.
    - lambda_: Market price of volatility risk.
    - rho: Correlation.

    The function initially estimates strikes using the Garman-Kohlhagen model. 
    It then optimizes the Heston parameters to minimize the sum of squared errors between market and model-implied volatilities.
    """
        
    def heston_vanilla_sse(param, var0, kappa, S0, tau, r_f, r_d, cp, K, σ_market):           
        """
        Compute the sum of squared errors (SSE) between market and model implied volatilities.
    
        Parameters:
        param (list): [vol of vol (vv), long-run variance (theta), correlation (rho)]
        var0 (float): Initial variance.
        kappa (float): Mean reversion speed to the long-run variance
        S0 (float): Spot price.
        tau (float): Time to maturity in years.
        r_f (float): Foreign interest rate (annualized).        
        r_d (float): Domestic interest rate (annualized).
        cp (array): Option types; 1 for call, -1 for put.
        K (array): Vector of strike prices.
        σ_market (array): Vector of market implied volatilities.

        Returns:
        float: Sum of squared errors between model and market implied volatilities.
        """

        vv, theta, rho = param
        if vv < 0.0 or theta < 0.0 or abs(rho) > 1.0:
            warnings.warn("Invalid value for vv, theta or rho encountered")
            return np.inf
                
        P = np.zeros(nb_strikes)
        IV = np.zeros(nb_strikes)
        
        if pricing_method == 'heston_cosine':
            P = heston_cosine_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=strikes, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho)
        else:
            # Integral required for each strike hence can't be vectorised
            for i in range(nb_strikes):        
                if pricing_method == 'heston_analytical_1993':
                    P[i] = heston1993_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=K[i], var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, lambda_=lambda_)
                elif pricing_method == 'heston_carr_madan_gauss_kronrod_quadrature':
                    P[i] = heston_carr_madan_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=K[i], var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, integration_method=0)
                elif pricing_method == 'heston_carr_madan_fft_w_simpsons':
                    P[i] = heston_carr_madan_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=K[i], var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, integration_method=1)
                else:
                    raise ValueError("Invalid 'pricing_method:", pricing_method)
            
        for i in range(nb_strikes):
            if P[i] < 0.0:
                IV[i] = -1.0
            else:
                IV[i] = gk_solve_implied_σ(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=strikes[i], X=P[i], σ_guess=σ_market[i])

        return np.sum((σ_market - IV)**2)


    # Input validation
    validate_input(tau, 'tau', lambda x: x > 0 and not np.isnan(x))
    validate_input(S0, 'S0', lambda x: x > 0 and not np.isnan(x))
    validate_input(r_f, 'r_f', lambda x: not np.isnan(x))
    validate_input(r_d, 'r_d', lambda x: not np.isnan(x))
    validate_input(cp, 'cp', lambda x: np.isin(x, [-1, 1]))
    if pricing_method not in VALID_HESTON_PRICING_METHODS:
        raise ValueError(f"'pricing_method' is invalid: {pricing_method}")

    # Calculate strikes for market deltas
    strikes = gk_solve_strike(S0=S0,tau=tau,r_f=r_f,r_d=r_d,σ=σ_market,Δ=Δ,Δ_convention=Δ_convention)

    nb_strikes = len(strikes)
    
    # Set the initial volatility, var0, to the implied ATM market volatility
    # var0 will NOT be solved in the calibration
    var0 = np.power(σ_market[nb_strikes // 2], 2)
    
    # Set mean reversion to the long-run variance to constant value = 1.5
    # The influence of mean reversion can be compensated by a the volatility of variance
    # kappa could be adjusted to enforce a valid Feller condition but this may detrimentally impact the fit to the market 
    # The Feller condition is often required to be violated in order to get a good fit to market data
    kappa = 1.5 
    
    # lambda_ is the market price of volatility risk
    # lambda_ will NOT be solved in the calibration
    # We set it to zero, given we are calibrating off option prices which already contain this feature
    # If calibrating of historical data, this field would need to be considered.
    lambda_ = 0 
    
    # Set initial values for vv, theta, rho (the parameters we are solving for)
    initparam = [2 * np.sqrt(var0), 2*var0, 0]

    res = scipy.optimize.minimize(lambda param: heston_vanilla_sse(param, var0=var0, kappa=kappa, S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=strikes, σ_market=σ_market), initparam)
    vv, theta, rho = res.x
    
    if 2 * kappa * theta - vv**2 <= 0.0:
        # In the Heston model, the Feller condition is often required to be violated in order to get a good fit to market data
        warnings.warn("Feller condition violated.") 

    # Calculate the Heston model implied volatilities so we can chart and compare them to σ_market
    P = np.zeros(nb_strikes)
    IV = np.zeros(nb_strikes)
    
    if pricing_method == 'heston_cosine':
        P = heston_cosine_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=strikes, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho)
    else:
        # Integral required for each strike hence can't be vectorised
        for i in range(nb_strikes):
            if pricing_method == 'heston_analytical_1993':
                P[i] = heston1993_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=strikes[i], var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, lambda_=lambda_)
            elif pricing_method == 'heston_carr_madan_gauss_kronrod_quadrature':
                P[i] = heston_carr_madan_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=strikes[i], var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, integration_method=0)
            elif pricing_method == 'heston_carr_madan_fft_w_simpsons':
                P[i] = heston_carr_madan_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=strikes[i], var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, integration_method=1)
            else:
                raise ValueError("Invalid 'pricing_method:", pricing_method)

    for i in range(nb_strikes):    
        IV[i] = gk_solve_implied_σ(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=strikes[i], X=P[i], σ_guess=σ_market[i])
 
    SSE = heston_vanilla_sse(res.x, var0=var0, kappa=kappa, S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=strikes, σ_market=σ_market)      
    
    if SSE == np.inf:
        warnings.warn("Calibration failed, SSE==inf")
        return None
    else:    
        return var0, vv, kappa, theta, rho, lambda_, IV, SSE

#%% Heston 1993 analytical pricing implementation (2nd version of the Characteristic function)

#@njit(fastmath=True, cache=True)
def heston_1993_fx_vanilla_european_integral(φ, m, S0, K, tau, r_f, r_d, var0, vv, kappa, theta, rho, lambda_):
    """
    Defines the integral for pricing an FX Vanilla European option per the analytic Heston 1993 formula
    This implementation uses the 2nd form of the Heston Characteristic function, detailed in Albrecher 2006 is used as it is more numerically stable.
    This in an auxiliary function for heston1993_price_fx_vanilla_european(), separated so @jit can be used
    
    Parameters:
    φ (complex): Point at which the auxiliary function is evaluated.
    m (int): Index for specific calculation (either 1 or 2).
    S0 (float): Spot price of the underlying, specified in # of units of domestic per 1 unit of foreign currency
    K (float): Strike price.
    var0 (float): Initial variance.
    vv (float): Volatility of volatility.
    r_f (float): Foreign interest rate.
    r_d (float): Domestic interest rate.
    tau (float): Time to maturity in years.
    kappa (float): Mean reversion speed to the long-run variance
    theta (float): Long-run variance
    lambda_ (float): Market price of volatility risk.
    rho (float): Correlation.

    Returns:
    float: Value of the auxiliary function at point `φ`.
    
    References:
    [1] S.Heston, (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options
    [2] H.Albrecher, P.Mayer, W.Schoutens, J.Tistaert (2006) The little Heston trap, Wilmott Magazine, January: 83–92.
    [3] A.Janek, T.Kluge, R.Weron, U.Wystup (2010) FX smile in the Heston model.
    """
    
    mu = r_d - r_f
    
    # x per equation (11) from Heston, 1993
    x = np.log(S0) 
    # a, u, b per equation (12) from Heston, 1993
    a = kappa * theta 
    u = [0.5, -0.5] 
    b = [kappa + lambda_ - rho * vv, kappa + lambda_] 
    
    σ = vv # In Heston, 1993, the volatility of the volatility is defined as σ 
    
    # d is per per equation (17) from Heston, 1993
    d = np.sqrt((rho *σ*φ*1j - b[m-1])**2 - σ**2 * (2*u[m-1]*φ*1j - φ**2))
    
    # There are two formula's for "g", in the Heston characteristic function because d, an input to g, has two roots. 
    # Let "g1", be the presentation in the original Heston 1993 paper (on page 5 of 17 of [1])
    # Let "g2", be the alternative, that is detailed in Albrecher 2006, "The Little Heston Trap". 
    # g2 = 1 / g1. 
    # We use g2, as using g1 leads to numerical problems based on most software implementations of calculation involving complex numbers
    g2 = (b[m-1] - rho * σ * φ * 1j - d) \
        / (b[m-1] - rho * σ * φ * 1j + d)

    # C, D and characteristic function f, per equation (17) from Heston, 1993
    C = mu * φ * 1j * tau + a / (vv ** 2) * ((b[m-1] - rho * σ * φ * 1j - d) * tau - 2 * np.log((1 - g2 * np.exp(-d * tau)) / (1 - g2)))
    D = (b[m-1] - rho * σ * φ * 1j - d) / (vv ** 2) * ((1 - np.exp(-d * tau)) / (1 - g2 * np.exp(-d * tau)))
    chf = np.exp(C + D * var0 + 1j * φ * x)
    
    # Function inside the integral in equation (18) from Heston, 1993
    F = np.real(np.exp(-1j * φ * np.log(K)) * chf / (1j * φ))
    return F


def heston1993_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K, var0, vv, kappa, theta, rho, lambda_):
    """
    Calculate the price of a European Vanilla FX option using the analytical Heston 1993 formulae
    The 2nd form of the Heston Characteristic function, detailed in Albrecher 2006 is used as it is more numerically stable.
    
    Parameters:
    S0 (float): FX spot price, specified in # of units of domestic per 1 unit of foreign currency
    tau (float): Time to expiry in years.
    r_d (float): Domestic interest rate.
    r_f (float): Foreign interest rate.    
    
        cp (int): Call (1) or Put (-1) option.

    k (float): Strike price.
    var0 (float): Initial variance.
    vv (float): Volatility of volatility.


    kappa (float): Mean reversion speed to the long-run variance
    theta (float): Long-run variance.
    lambda_ (float): Market price of volatility risk.
    rho (float): Correlation.

    Returns:
    float: Option price.

    Example:
    >>> heston_garman_kohlhagen(1, 1.03, 1, 0.01, 0.02, 0.05, 0.03, 0.25, 10, 0.01, 0, 0.5)

    References:
    [1] S.Heston, (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options
    [2] H.Albrecher, P.Mayer, W.Schoutens, J.Tistaert (2006) The little Heston trap, Wilmott Magazine, January: 83–92.
    [3] A.Janek, T.Kluge, R.Weron, U.Wystup (2010) FX smile in the Heston model.
    
    Written by Agnieszka Janek and Rafal Weron (2010.07.07)
    Revised by Rafal Weron (2010.10.08, 2010.12.27)
    """

    # Equation (18) from Heston, 1993
    P1 = 0.5 + 1/np.pi * scipy.integrate.quad(func=lambda φ: heston_1993_fx_vanilla_european_integral(φ=φ, m=1, S0=S0, K=K, tau=tau, r_f=r_f, r_d=r_d, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, lambda_=lambda_), a=0, b=np.inf, epsrel=1e-8)[0]
    P2 = 0.5 + 1/np.pi * scipy.integrate.quad(func=lambda φ: heston_1993_fx_vanilla_european_integral(φ=φ, m=2, S0=S0, K=K, tau=tau, r_f=r_f, r_d=r_d, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho, lambda_=lambda_), a=0, b=np.inf, epsrel=1e-8)[0]

    Pplus = (1 - cp) / 2 + cp * P1   # Pplus = N(d1)
    Pminus = (1 - cp) / 2 + cp * P2  # Pminus = N(d2)

    # tk is this for spot or forward Δ, matlab code assumed spot delta input?
    X = cp * (S0 * np.exp(-r_f * tau) * Pplus - K * np.exp(-r_d * tau) * Pminus) 
    
    return X

#%%  Heston model using Carr-Madan via: 
# (method=0) Gauss-Kronrod quadrature
# (method=1) Fast Fourier Transform + Simpson method 

def heston_carr_madan_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K, var0, vv, kappa, theta, rho, integration_method=0):    
    """
     Calculate European FX option price using the Heston model via Carr-Madan approach.
     
     Parameters:
     kappa, theta, vv, rho, var0 (float): Heston parameters.
     integration_method (int, optional): 0 for Gauss-Kronrod quadrature, 1 for FFT + Simpson's rule. Default is 0.
     
     Returns:
     float: Option price.
     
     References:
     [1] Albrecher et al. (2006) "The little Heston trap."
     [2] Carr, Madan (1998) "Option valuation using the Fast Fourier transform."
     [3] Janek et al. (2010) "FX smile in the Heston model."
     [4] Schmelzle (2010) "Option Pricing Formulae using Fourier Transform."
     
     Authors:
     Written by Agnieszka Janek (2010.07.23)
     Revised by Rafal Weron (2010.10.08)
     Revised by Agnieszka Janek and Rafal Weron (2010.10.21, 2010.12.27)
    """    
    
    if cp == 1:
        alpha = 0.75
    elif cp == -1:
        alpha = 1.75
    else:
        raise ValueError
    
    log_S0 = np.log(S0)
    log_K = np.log(K)
    
    if integration_method == 0:
        # Integrate using adaptive Gauss-Kronrod quadrature
        result, _ = scipy.integrate.quad(heston_fft_fx_vanilla_european_integral, 0, np.inf, args=(cp, log_S0, log_K, tau, r_f, r_d, var0, vv, kappa, theta, rho, alpha))
        y = np.exp(-cp * log_K * alpha) * result / np.pi
    elif integration_method == 1:
        # Fast Fourier Transform with Simpson's rule (as suggested in [2])
        N = 2**10
        eta = 0.25
        v = np.arange(0, N) * eta
        
        lambda_ = 2*np.pi/(N*eta) 
        b = N*lambda_/2 # Equation (20) per Carr-Madan, 1998
        ku = -b + lambda_ * np.array(range(0,N)) # Equation (19) per Carr-Madan, 1998
        
        u = v - (cp * alpha + 1) * 1j
        chf = chf_heston_albrecher2007(u=u, log_S0=log_S0, tau=tau, r_f=r_f, r_d=r_d, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho)
        F = chf * np.exp(-r_d * tau) / (alpha**2 + cp * alpha - v**2 + 1j * (cp * 2 * alpha + 1) * v)
        
        # Use Simpson's approximation to calculate FFT (see [2])
        simpson_weights = get_simpson_weights(N) 
        fft_func = np.exp(1j * b * v) * F * eta * simpson_weights
        payoff = np.real(scipy.fft.fft(fft_func))
        option_value = np.exp(-cp * ku * alpha) * payoff / np.pi
        
        y = np.interp(log_K, ku, option_value)
                
    return y


def get_simpson_weights(n):
    # 1/3, then alternating 2/3, 4/3, 2/3, 4/3, 2/3, 4/3, .... 
    weights = np.array([1] + [4, 2] * ((n-2) // 2) + [1])
    return weights / 3


@njit(fastmath=True, cache=True)
def heston_fft_fx_vanilla_european_integral(v, cp, log_S0, log_K, tau, r_f, r_d, var0, vv, kappa, theta, rho, alpha):
    """
    Auxiliary function for heston_carr_madan_fx_vanilla_european.

    Parameters:
        v: Evaluation points for auxiliary function.
        cp: Option type (1 for call, -1 for put).
        log_S0 (float): Natural Log of the initial spot price (in # of domestic currency units per 1 foreign currency unit)
        log_K: Natural log of strike price (in # of domestic currency units per 1 foreign currency unit)
        tau (float): Time to maturity (in years)
        r_f (float): Foreign risk-free interest rate.
        r_d (float): Domestic risk-free interest rate.
        var0 (float): Initial variance.
        vv (float): Volatility of volatility.
        kappa (float): Mean reversion speed to the longsrun variance
        theta (float): Long-run variance.
        rho (float): Correlation between stock price and volatility.
        alpha: Damping coefficient.

    Returns:
        The values of the auxiliary function evaluated at points v.
    """
    
    u = v - (cp * alpha + 1) * 1j
    chf = chf_heston_albrecher2007(u=u, log_S0=log_S0, tau=tau, r_f=r_f, r_d=r_d, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho)
    ftt_func = chf * np.exp(-r_d * tau) / (alpha**2 + cp * alpha - v ** 2 + 1j * (cp * 2 * alpha + 1) * v)
    return np.real(np.exp(-1j * v * log_K) * ftt_func)


@njit(fastmath=True, cache=True)
def chf_heston_albrecher2007(u, log_S0, tau, r_f, r_d, var0, vv, kappa, theta, rho):
    """
    Compute the characteristic function for the Heston model.
    
    Parameters:
    u (float or np.array): The argument of the characteristic function.
    log_S0 (float): Natural Log of the initial spot price (in # of domestic currency units per 1 foreign currency unit)
    tau (float): Time to maturity (in years)
    r_f (float): Foreign risk-free interest rate.
    r_d (float): Domestic risk-free interest rate.
    var0 (float): Initial variance.
    vv (float): Volatility of volatility.
    kappa (float): Mean reversion speed to the long-run variance
    theta (float): Long-run variance 
    rho (float): Correlation between stock price and volatility.

    Returns:
        charFunc (float or np.array): Value of the characteristic function at u.
    
    [1] S.Heston, (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options
    [2] Albrecher, Hansjoerg & Mayer, Philipp & Schoutens, Wim & Tistaert, Jurgen. (2007). The Little Heston Trap. Wilmott. 83-92
    
    """    
    
    # The symbol use in heston model academic literature is inconsistent
    # Remap the parameters per the symbols used in [2] so comparison to the formulae on page 4 of 21 is easier    
    k = kappa # mean reversion rate of the variance to the long run variance
    η = theta # the long run variance
    λ = vv # volatility of the variance
    σ0 = np.sqrt(var0) # initial volatility
    r = r_d
    q = r_f
         
    # Per definition in equation (1) of [2] on page 4/21
    d = np.sqrt((rho*λ*u*1j - k)**2 + λ**2 * (1j * u + u ** 2))

    # There are two formula's for "g", in the Heston characteristic function because d, an input to g, has two roots. 
    # Let "g1", be the presentation in the original Heston 1993 paper (on page 5 of 17 of [1])
    # Let "g2", be the alternative, that is detailed in [2], "The little heston trap".
    # g2 = 1 / g1. 
    # We use g2, as using g1 leads to numerical problems based on most software implementations of calculation involving complex numbers
    g2 = (k - rho * λ * 1j * u - d) \
      / (k - rho * λ * 1j * u + d)
      
    # 1st inner exponential term on line 1 of 3, in equation (2) in [2]      
    A = 1j * u * (log_S0 + (r - q) * tau)
    
    # 2nd inner exponential term on line 2 of 3, in equation (2) in [2]
    B = η*k*(λ**-2) * ((k - rho*λ*1j*u - d) * tau - 2 * np.log((1 - g2*np.exp(-d*tau)) / (1 - g2)))
    
    # 3rd inner exponential term on line 3 of 3, in equation (2) in [2]
    C = (σ0**2)*(λ**-2) * (k - rho*λ*1j*u - d) * (1 - np.exp(-d*tau)) / (1 - g2*np.exp(-d*tau)) 
    
    return np.exp(A + B + C)

# Note this function (chf_heston_fang2008) is NOT speed up by use of numba/jit
@njit(fastmath=True, cache=True)
def chf_heston_fang2008(u, tau, r_f, r_d, var0, vv, kappa, theta, rho):
    """
    Compute the characteristic function for the Heston model per Fang 2008
    
    Parameters:
    u (float or np.array): The argument values of the characteristic function.
    tau (float): Time to maturity (in years)
    r_f (float): Foreign risk-free interest rate.
    r_d (float): Domestic risk-free interest rate.
    var0 (float): Initial variance.
    vv (float): Volatility of volatility.
    kappa (float): Rate of mean reversion to the long-run variance
    theta (float): Long-run variance
    rho (float): Correlation between stock price and volatility.

    Returns:
        charFunc (float or np.array): Value of the characteristic function at u.
        
    References
    [1] Fang, Fang & Oosterlee, Cornelis. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. SIAM J. Scientific Computing. 31. 826-848. 10.1137/080718061. 
    [2] Albrecher, Hansjoerg & Mayer, Philipp & Schoutens, Wim & Tistaert, Jurgen. (2007). The Little Heston Trap. Wilmott. 83-92
    """    
    
    mu = r_d - r_f # mu = r - q
    
    # Map to the symbols used in [1] so for easier comparison to the paper
    λ = kappa  # mean reversion speed to the long-run variance
    u_bar = theta # long-run variance
    η = vv # volatiltiy of the volatility
    u0 = var0 # initial variance
    
    ω = u
    W = λ - 1j*rho*η*ω # helper term to make simpler code (not in [3])
    D = np.sqrt( W**2 + (ω**2 + 1j*ω) * (η**2))
    G2 = (W - D) / (W + D) # G2 is more stable per [2]
    exp_D_tau = np.exp(-D*tau)
    
    inner_exp_1 = 1j*ω*mu*tau + (u0/(η**2)) * ((1-exp_D_tau)/(1 - G2*exp_D_tau)) * (W-D)
    inner_exp_2 = (λ*u_bar)/(η**2) * ((tau * (W - D)) - 2*np.log( (1-G2*exp_D_tau) / (1-G2) ))
    
    chf = np.exp(inner_exp_1 + inner_exp_2)
    
    return chf

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


def heston_cosine_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K, var0, vv, kappa, theta, rho, N=160, L=10, calculate_via_put_call_parity=True):
    
    """
    Computes the call or put option prices using the COS method.
    
    Parameters:
    - S0 (float): Initial FX spot price
    - tau (float): time to expiry
    - r_f (float): Foreign risk-free interest rate.
    - r_d (float): Domestic risk-free interest rate.
    - cp (int): 1 for call and -1 for put
    - S0 (float): Initial stock price
    - r (float): Interest rate
    - tau (float): Time to maturity
    - K (list or np.array): List of strike prices
    - N (int): Number of expansion terms (<160 should be sufficient per Fang, 2008)
    - L (float): Size of truncation domain (for N=160, set L=10 per Fang, 2008)
    - calculate_via_put_call_parity (boolean)
    
    Returns:
    - np.array: Option prices

    References:
    [1] Fang, Fang & Oosterlee, Cornelis. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. SIAM J. Scientific Computing. 31. 826-848. 10.1137/080718061. 
    """     

    cp = np.atleast_1d(cp).astype(float)
    K = np.atleast_1d(K).astype(float)
    assert cp.shape == K.shape
    
    result = np.full(K.shape, np.nan)

    x0 = np.log(S0 / K) # Per [1] in section 3.1 on page 6 of 21

    if True:
        # Apply the truncation method per appendix 11 of [1] 
        mu = r_d - r_f # technically we should use the drift implied from market FX forward rate, but the IR parity forward rate should be close enough
        model_param =  {'tau': tau,
                        'mu': mu,
                        'var0': var0,
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
    chf = chf_heston_fang2008(u=u, tau=tau, r_f=r_f, r_d=r_d, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho)
    Fk = np.real(chf[:, np.newaxis]  * np.exp(1j * k[:, np.newaxis] * np.pi * (x0 - a)/(b-a))) 
    Fk[0] = 0.5 * Fk[0] # Per page 3/21 of [1], "where Σ′ indicates that the first term in the summation is weighted by one-half"        
    
    
    if calculate_via_put_call_parity or (cp == -1).any():
        Uk_put = calculate_Uk_european_options(cp=-1, a=a, b=b, k=k)[:, np.newaxis] # Per equation 29 of [1] (page 8 of 21)
    
    if not calculate_via_put_call_parity and (cp == 1).any():
        Uk_call = calculate_Uk_european_options(cp=1, a=a, b=b, k=k)[:, np.newaxis] # Per equation 29 of [1] (page 8 of 21)
        
    if calculate_via_put_call_parity:
        # Method by put-call parity is more stable
        put_px = K * np.multiply(Fk, Uk_put).sum(axis=0) * np.exp(-r_d * tau) 
        call_px = put_px + S0 * np.exp(-r_f * tau) - K * np.exp(-r_d * tau)
    else:
        if (cp == 1).any():
            call_px = K * np.multiply(Fk, Uk_call).sum(axis=0) * np.exp(-r_d * tau)      
        if (cp == -1).any():
            put_px = K * np.multiply(Fk, Uk_put).sum(axis=0) * np.exp(-r_d * tau) 
    
    result[cp == 1] = call_px[cp == 1]
    result[cp == -1] = put_px[cp == -1]
    
    return result
    



    