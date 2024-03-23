# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    import sys
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())
   

from frm.frm.pricing_engine.garman_kohlhagen import gk_price, gk_solve_implied_σ, gk_solve_strike
import numpy as np
import scipy.fft
import scipy.optimize 
from scipy.integrate import quad
from scipy.optimize import minimize, newton, root_scalar
from numba import jit
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
    - Tuple: Initial volatility (v0), vol of vol (vv), mean reversion (kappa), long-run mean (theta), market price of volatility risk (lambda_), correlation (rho), vector of implied volatilities (IV), sum of squared errors (SSE)

    References:
    [1] Janek, A., Kluge, T., Weron, R., Wystup, U. (2010). "FX smile in the Heston model"    
    Converted from MATLAB to Python by Shasa Foster (2023.09.02)

    The Heston SDE is:
        dS(t) = µ*S*dt + σ(t)*S*dW1(t)
        dσ(t) = kappa(theta - σ(t))*dt + vv*σ(t)*dW2(t)

    The Heston model is defined by six parameters 
    - v0: Initial volatility.
    - vv: Volatility of volatility.
    - kappa: rate of mean reversion to the long-run variance
    - theta: Long-run variance.
    - lambda_: Market price of volatility risk.
    - rho: Correlation.

    The function initially estimates strikes using the Garman-Kohlhagen model. 
    It then optimizes the Heston parameters to minimize the sum of squared errors between market and model-implied volatilities.
    """
        
    def heston_vanilla_sse(param, v0, kappa, S0, tau, r_f, r_d, cp, K, σ_market):           
        """
        Compute the sum of squared errors (SSE) between market and model implied volatilities.
    
        Parameters:
        param (list): [vol of vol (vv), long-run variance (theta), correlation (rho)]
        cp (array): Option types; 1 for call, -1 for put.
        S (float): Spot price.
        K (array): Vector of strike prices.
        v0 (float): Initial volatility.
        σ_market (array): Vector of market implied volatilities.
        r_d (float): Domestic interest rate (annualized).
        r_f (float): Foreign interest rate (annualized).
        tau (float): Time to maturity in years.
        kappa (float): Level of mean reversion.
    
        Returns:
        float: Sum of squared errors between model and market implied volatilities.

        Converted from MATLAB to Python by Shasa Foster (2023.09.02)
        """

        vv, theta, rho = param
        if vv < 0.0 or theta < 0.0 or abs(rho) > 1.0:
            warnings.warn("Invalid value for vv, theta or rho encountered")
            return np.inf
                
        P = np.zeros(nb_strikes)
        IV = np.zeros(nb_strikes)
        
        for i in range(nb_strikes):
            if pricing_method == 'heston_analytical_1993':
                P[i] = heston1993_price_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], v0, vv, kappa, theta, rho, lambda_)
            elif pricing_method == 'heston_carr_madan_gauss_kronrod_quadrature':
                P[i] = heston_carr_madan_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], v0, vv, kappa, theta, rho, integration_method=0)
            elif pricing_method == 'heston_carr_madan_fft_w_simpsons':
                P[i] = heston_carr_madan_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], v0, vv, kappa, theta, rho, integration_method=1)
            #else:
            #    raise ValueError("invalid 'pricing_method' value: ", pricing_method)
            # The paper Pricing European Options by Stable Fourier-Cosine Series Expansions details this clearly
            elif pricing_method == 'heston_cosine':
                P[i] = heston_cos_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], v0, vv, kappa, theta, rho)

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
    
    # Set the initial variance, v0, to the implied ATM market volatility
    # v0 will NOT be solved in the calibration
    v0 = (σ_market[nb_strikes // 2])**2 
    
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
    initparam = [2 * np.sqrt(v0), 2 * v0, 0]

    res = minimize(lambda param: heston_vanilla_sse(param, v0, kappa, S0, tau, r_f, r_d, cp, strikes, σ_market), initparam)
    vv, theta, rho = res.x
    
    if 2 * kappa * theta - vv**2 <= 0.0:
        # In the Heston model, the Feller condition is often required to be violated in order to get a good fit to market data
        warnings.warn("Feller condition violated.") 

    # Calculate the Heston model implied volatilities so we can chart and compare them to σ_market
    P = np.zeros(nb_strikes)
    IV = np.zeros(nb_strikes)

    # Integral required for each strike hence can't be vectorised
    for i in range(nb_strikes):
        if pricing_method == 'heston_analytical_1993':
            P[i] = heston1993_price_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], strikes[i], v0, vv, kappa, theta, rho, lambda_)
        elif pricing_method == 'heston_carr_madan_gauss_kronrod_quadrature':
            P[i] = heston_carr_madan_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], strikes[i], v0, vv, kappa, theta, rho, integration_method=0)
        elif pricing_method == 'heston_carr_madan_fft_w_simpsons':
            P[i] = heston_carr_madan_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], strikes[i], v0, vv, kappa, theta, rho, integration_method=1)
        elif pricing_method == 'heston_cosine':
            # The paper Pricing European Options by Stable Fourier-Cosine Series Expansions details this clearly
            P[i] = heston_cos_vanilla_european(S0, tau, r_f, r_d, cp[i], strikes[i], v0, vv, kappa, theta, rho)    
                   
        else:
            raise ValueError(f"'pricing_method' is invalid: {pricing_method}")        
        
        
        IV[i] = gk_solve_implied_σ(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp[i], K=strikes[i], X=P[i], σ_guess=σ_market[i])
 
    SSE = heston_vanilla_sse(res.x, v0, kappa, S0, tau, r_f, r_d, cp, strikes, σ_market)      
    
    
    if SSE == np.inf:
        warnings.warn("Calibration failed, SSE==inf")
        return None
    else:    
        return v0, vv, kappa, theta, rho, lambda_, IV, SSE

#%% Heston 1993 analytical pricing implementation (2nd version of the Characteristic function)

@jit(nopython=True)
def heston_1993_fx_vanilla_european_integral(φ, m, S, k, v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho):
    """
    Defines the integral for pricing an FX Vanilla European option per the analytic Heston 1993 formula
    This implementation uses the 2nd form of the Heston Characteristic function, detailed in Albrecher 2006 is used as it is more numerically stable.
    This in an auxiliary function for heston1993_price_fx_vanilla_european(), separated so @jit can be used
    
    Parameters:
    φ (complex): Point at which the auxiliary function is evaluated.
    m (int): Index for specific calculation (either 1 or 2).
    s (float): Spot price of the underlying.
    k (float): Strike price.
    v0 (float): Initial volatility.
    vv (float): Volatility of volatility.
    r_d (float): Domestic interest rate.
    r_f (float): Foreign interest rate.
    tau (float): Time to maturity in years.
    kappa (float): Mean reversion level.
    theta (float): Long-term variance.
    lambda_ (float): Market price of volatility risk.
    rho (float): Correlation.

    Returns:
    float: Value of the auxiliary function at point `φ`.
    
    References:
    [1] S.Heston, (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options
    [2] H.Albrecher, P.Mayer, W.Schoutens, J.Tistaert (2006) The little Heston trap, Wilmott Magazine, January: 83–92.
    [3] A.Janek, T.Kluge, R.Weron, U.Wystup (2010) FX smile in the Heston model.

    Converted from MATLAB to Python by Shasa Foster (2023.09.02)
    """
    
    # x per equation (11) from Heston, 1993
    x = np.log(S) 
    # a, u, b per equation (12) from Heston, 1993
    a = kappa * theta 
    u = [0.5, -0.5] 
    b = [kappa + lambda_ - rho * vv, kappa + lambda_] 
    
    # d is per per equation (17) from Heston, 1993
    d = np.sqrt((1j * rho * vv * φ - b[m-1]) ** 2 - vv ** 2 * (2 * u[m-1] * φ * 1j - φ ** 2))
    
    # There are two formula's for "g", in the Heston characteristic function because d, an input to g, has two roots. 
    # Let "g1", be the presentation in the original Heston 1993 paper.
    # Let "g2", be the alternative, that is detailed in Albrecher 2006, "The Little Heston Trap". 
    # g2 = 1 / g1. 
    # We use g2, as using g1 leads to numerical problems based on most software implementations of calculation involving complex numbers
    g2 = (b[m-1] - rho * vv * φ * 1j - d) \
        / (b[m-1] - rho * vv * φ * 1j + d)

    D = (b[m-1] - rho * vv * φ * 1j - d) / (vv ** 2) * ((1 - np.exp(-d * tau)) / (1 - g2 * np.exp(-d * tau)))
    C = (r_d - r_f) * φ * 1j * tau + a / (vv ** 2) * ((b[m-1] - rho * vv * φ * 1j - d) * tau - 2 * np.log((1 - g2 * np.exp(-d * tau)) / (1 - g2)))
   
    # f per equation (17) from Heston, 1993
    f = np.exp(C + D * v0 + 1j * φ * x)
    
    # Function inside the integral in equation (18) from Heston, 1993
    F = np.real(np.exp(-1j * φ * np.log(k)) * f / (1j * φ))
    return F


def heston1993_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K, v0, vv, kappa, theta, rho, lambda_):
    """
    Calculate the price of a European Vanilla FX option using the analytical Heston 1993 formulae
    The 2nd form of the Heston Characteristic function, detailed in Albrecher 2006 is used as it is more numerically stable.
    
    Parameters:
    cp (int): Call (1) or Put (-1) option.
    s (float): Spot price.
    k (float): Strike price.
    v0 (float): Initial volatility.
    vv (float): Volatility of volatility.
    r_d (float): Domestic interest rate.
    r_f (float): Foreign interest rate.
    tau (float): Time to maturity in years.
    kappa (float): Level of mean reversion to the long-run variance
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
    Converted from MATLAB to Python by Shasa Foster (2023.09.02)
    """

    # Equation (18) from Heston, 1993
    P1 = 0.5 + 1/np.pi * quad(lambda φ: heston_1993_fx_vanilla_european_integral(φ, 1, S0, K, v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho), 0, np.inf, epsrel=1e-8)[0]
    P2 = 0.5 + 1/np.pi * quad(lambda φ: heston_1993_fx_vanilla_european_integral(φ, 2, S0, K, v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho), 0, np.inf, epsrel=1e-8)[0]

    Pplus = (1 - cp) / 2 + cp * P1   # Pplus = N(d1)
    Pminus = (1 - cp) / 2 + cp * P2  # Pminus = N(d2)

    # tk is this for spot or forward Δ, matlab code assumed spot delta input?
    X = cp * (S0 * np.exp(-r_f * tau) * Pplus - K * np.exp(-r_d * tau) * Pminus) 
    
    return X

#%%  Heston model using Carr-Madan via: 
# (method=0) Gauss-Kronrod quadrature
# (method=1) Fast Fourier Transform + Simpson method 

def heston_carr_madan_fx_vanilla_european(S, tau, r_f, r_d, cp, K, v0, vv, kappa, theta, rho, integration_method=0):    
    """
     Calculate European FX option price using the Heston model via Carr-Madan approach.
     
     Parameters:
     kappa, theta, vv, rho, v0 (float): Heston parameters.
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
     Converted from MATLAB to Python by Shasa Foster (2023.09.02)
    """    
    
    if cp == 1:
        alpha = 0.75
    elif cp == -1:
        alpha = 1.75
    else:
        raise ValueError
    
    s0 = np.log(S)
    k = np.log(K)
    
    if integration_method == 0:
        # Integrate using adaptive Gauss-Kronrod quadrature
        result, _ = quad(heston_fft_fx_vanilla_european_integral, 0, np.inf, args=(cp, s0, k, tau, r_d, r_f, kappa, theta, vv, rho, v0, alpha))
        y = np.exp(-cp * k * alpha) * result / np.pi
    elif integration_method == 1:
        # Fast Fourier Transform with Simpson's rule (as suggested in [2])
        N = 2**10
        eta = 0.25
        v = np.arange(0, N) * eta
        
        lambda_ = 2*np.pi/(N*eta) 
        b = N*lambda_/2 # Equation (20) per Carr-Madan, 1998
        ku = -b + lambda_ * np.array(range(0,N)) # Equation (19) per Carr-Madan, 1998
        
        u = v - (cp * alpha + 1) * 1j
        d, g, A, B, C, charFunc = char_func(u, s0, k, tau, r_d, r_f, kappa, theta, vv, rho, v0)
        F = charFunc * np.exp(-r_d * tau) / (alpha**2 + cp * alpha - v**2 + 1j * (cp * 2 * alpha + 1) * v)
        
        # Use Simpson's approximation to calculate FFT (see [2])
        simpson_weights = get_simpson_weights(N) 
        fft_func = np.exp(1j * b * v) * F * eta * simpson_weights
        payoff = np.real(scipy.fft.fft(fft_func))
        option_value = np.exp(-cp * ku * alpha) * payoff / np.pi
        
        y = np.interp(k, ku, option_value)
                
    return y

def get_simpson_weights(n):
    # 1/3, then alternating 2/3, 4/3, 2/3, 4/3, 2/3, 4/3, .... 
    weights = np.array([1] + [4, 2] * ((n-2) // 2) + [1])
    return weights / 3

@jit(nopython=True)
def heston_fft_fx_vanilla_european_integral(v, cp, s0, k, T, r_d, r_f, kappa, theta, vv, rho, v0, alpha):
    """
    Auxiliary function for heston_carr_madan_fx_vanilla_european.

    Parameters:
        v: Evaluation points for auxiliary function.
        cp: Option type (1 for call, -1 for put).
        s0: Log of spot price.
        k: Log of strike price.
        T: Time to maturity (years).
        r: Domestic interest rate.
        rf: Foreign interest rate.
        kappa: Level of mean reversion.
        theta: Long-run variance.
        vv: Volatility of volatility.
        rho: Correlation coefficient.
        v0: Initial volatility.
        alpha: Damping coefficient.

    Returns:
        The values of the auxiliary function evaluated at points v.
    """
    
    u = v - (cp * alpha + 1) * 1j
    d, g, A, B, C, charFunc = char_func(u, s0, k, T, r_d, r_f, kappa, theta, vv, rho, v0)
    ftt_func = charFunc * np.exp(-r_d * T) / (alpha**2 + cp * alpha - v ** 2 + 1j * (cp * 2 * alpha + 1) * v)
    return np.real(np.exp(-1j * v * k) * ftt_func)

@jit
def char_func(u, s0, k, tau, r_d, r_f, kappa, theta, vv, rho, v0):
    """
    Compute the characteristic function for the Heston model.
    
    Parameters:
    u (float or np.array): The argument of the characteristic function.
    s0 (float): Initial spot price (in # of domestic currency units per 1 foreign currency unit)
    k (float): Strike price.
    tau (float): Time to maturity.
    r_d (float): Domestic risk-free interest rate.
    r_f (float): Foreign risk-free interest rate.
    kappa (float): Rate of mean reversion.
    theta (float): Long-term level of volatility.
    vv (float): Volatility of volatility.
    rho (float): Correlation between stock price and volatility.
    v0 (float): Initial volatility level.
    
    Returns:
    tuple: Tuple containing:
        d (float or np.array): A derived value used for further calculations.
        g (float or np.array): A derived value used for further calculations.
        A (float or np.array): A term in the characteristic function.
        B (float or np.array): A term in the characteristic function.
        C (float or np.array): A term in the characteristic function.
        charFunc (float or np.array): Value of the characteristic function at u.
    """    
    
    # Characteristic function (see [1])
    d = np.sqrt((rho * vv * u * 1j - kappa) ** 2 + vv ** 2 * (1j * u + u ** 2))
    g = (kappa - rho * vv * 1j * u - d) / (kappa - rho * vv * 1j * u + d)
    A = 1j * u * (s0 + (r_d - r_f) * tau)
    B = theta * kappa * vv ** -2 * ((kappa - rho * vv * 1j * u - d) * tau - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    C = v0 * vv ** -2 * (kappa - rho * vv * 1j * u - d) * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)) 
    return d, g, A, B, C, np.exp(A + B + C)


#%% COS method

def heston_cos_vanilla_european(S0, tau, r_f, r_d, cp, K, v0, vv, kappa, theta, rho, N=160, L=3):
    
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

    def chi_psi(a, b, c, d, k):
        """
        Compute chi and psi coefficients for COS method.
        
        Parameters:
        - a, b, c, d (float): Boundaries for truncation and integration
        - k (np.array): Array of k values
        
        Returns:
        - tuple (np.array, np.array): chi, psi
        """        
        psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/ (b - a))
        psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
        psi[0] = d - c
        
        chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
        expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d) - np.cos(k * np.pi  * (c - a) / (b - a)) * np.exp(c)
        expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
        chi = chi * (expr1 + expr2)
        
        return chi, psi 

    def call_put_coefficients(cp, a, b, k):
        """
        Compute coefficients for call or put options using COS method.
        
        Parameters:
        - cp (int): 1 for call, -1 for put
        - a (float): Lower truncation boundary
        - b (float): Upper truncation boundary
        - k (np.array): Array of k values
        
        Returns:
        - np.array: Coefficients
        """        
        c, d = (0, b) if cp == 1 else (a, 0)
        chi, psi = chi_psi(a, b, c, d, k)
        prefactor = 2 / (b - a)
        return prefactor * (chi - psi) if cp == 1 else prefactor * (-chi + psi)

    if K is not np.array:
        if np.isscalar(K):
            K = np.array([K]).reshape(1, 1)
        else:
            K = np.array(K).reshape(len(K), 1)

    x0 = np.log(S0 / np.array(K)).reshape(-1, 1)

    # Truncation domain
    a, b = -L * np.sqrt(tau), L * np.sqrt(tau)
    
    # Summation from k = 0 to k = N-1
    k = np.linspace(0, N-1, N).reshape(-1, 1)
    u = (k * np.pi / (b - a))
    
    H_k = call_put_coefficients(cp, a, b, k)

    mat = np.exp(1j * np.outer((x0 - a) , u))
    chf = chf_heston_cosine_model(r_f, r_d, tau, v0, vv, kappa, theta, rho)
    temp = chf(u) * H_k
    temp[0] = 0.5 * temp[0] # Per page 3/21 of [1], "where Σ′ indicates that the first term in the summation is weighted by one-half" 
    return np.exp(-r_d * tau) * K * np.real(mat.dot(temp))


def chf_heston_cosine_model(r_f, r_d, tau, v0, vv, kappa, theta, rho):
    """
    Characteristic function for the Heston model.
    
    Parameters:
    - r_f (float): Risk-free domestic (foreign) currency interest rate        
    - r_d (float): Risk-free domestic (quote) currency interest rate
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





    