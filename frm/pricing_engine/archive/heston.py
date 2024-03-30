# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    import sys
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())
   

from frm.pricing_engine.garman_kohlhagen import garman_kohlhagen
import numpy as np
import scipy.fft
import scipy.optimize 
from scipy.integrate import quad
from scipy.optimize import minimize, newton, root_scalar
from numba import jit
from typing import Tuple
import warnings
from scipy.stats import norm

def solve_implied_σ(X: float, S: float, K: float, r_d: float, r_f: float, tau: float, cp: int, σ_guess: float) -> float:
    """
    Solve the implied volatility using the Garman-Kohlhagen model for a European Vanilla FX option.

    Parameters:
    - X (float): Option price
    - S (float): Spot price
    - K (float): Strike price
    - r_d (float): Domestic interest rate
    - r_f (float): Foreign interest rate
    - tau (float): Time to maturity in years
    - cp (int): Option type (1 for call, -1 for put)
    - σ_guess (float): Initial guess for volatility

    Returns:
    - float: Implied volatility

    Attempts to find implied volatility using Newton's method initially and falls back to Brent's method in case of failure.
    """
    try:
        # Try Netwon's method first (it's faster but less robust)
        return newton(lambda σ: (garman_kohlhagen(S=S,σ=σ,r_f=r_f,r_d=r_d,tau=tau,cp=cp,K=K,task='px') - X), x0=σ_guess, tol=1e-4, maxiter=50)
    except RuntimeError:
        # Fallback to Brent's method
        return root_scalar(lambda σ: (garman_kohlhagen(S=S,σ=σ,r_f=r_f,r_d=r_d,tau=tau,cp=cp,K=K,task='px') - X), bracket=[0.0001, 2], method='brentq').root

def heston_fit_vanilla_fx_smile(
        Δ: np.array, 
        Δ_convention: str, 
        σ_market: np.array, 
        S: float, 
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
    - S (float): Spot price
    - r_f (float): Foreign interest rate
    - r_d (float): Domestic interest rate
    - tau (float): Time to maturity in years
    - cp (np.array): Vector of option types (1 for call, -1 for put)

    Returns:
    - Tuple: Initial volatility (v0), vol of vol (vv), mean reversion (kappa), long-run mean (theta), market price of volatility risk (lambda_), correlation (rho), vector of implied volatilities (IV), sum of squared errors (SSE)

    References:
    [1] Janek, A., Kluge, T., Weron, R., Wystup, U. (2010). "FX smile in the Heston model"

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
    
    def heston_vanilla_sse(param, cp, S, strikes, v0, σ_market, r_d, r_f, tau, kappa):        
        """
        Compute the sum of squared errors (SSE) between market and model implied volatilities.
    
        Parameters:
        param (list): [vol of vol (vv), long-run variance (theta), correlation (rho)]
        cp (array): Option types; 1 for call, -1 for put.
        S (float): Spot price.
        strikes (array): Vector of strike prices.
        v0 (float): Initial volatility.
        σ_market (array): Vector of market implied volatilities.
        r_d (float): Domestic interest rate (annualized).
        r_f (float): Foreign interest rate (annualized).
        tau (float): Time to maturity in years.
        kappa (float): Level of mean reversion.
    
        Returns:
        float: Sum of squared errors between model and market implied volatilities.
        """

        vv, theta, rho = param
        if vv < 0.0 or theta < 0.0 or abs(rho) > 1.0:
            warnings.warn("Invalid value for vv, theta or rho encountered")
            return np.inf
        P = np.zeros(nb_strikes)
        IV = np.zeros(nb_strikes)
        for i in range(nb_strikes):
            if pricing_method == 'analytical_1993':
                P[i] = heston1993_price_fx_vanilla_european(cp[i], S, strikes[i], v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho)
            elif pricing_method == 'carr_madan_fft_w_simpsons':
                P[i] = heston_carr_madan_fx_vanilla_european(cp[i], S, strikes[i], tau, r_d, r_f, kappa, theta, vv, rho, v0, integration_method=1)
            elif pricing_method == 'carr_madan_gauss_kronrod_quadrature':
                P[i] = heston_carr_madan_fx_vanilla_european(cp[i], S, strikes[i], tau, r_d, r_f, kappa, theta, vv, rho, v0, integration_method=0)
            elif pricing_method == 'lipton gauss-kronrod quadrature':
                pass # not working
                #P[i] = heston_lipton_price_fx_vanilla_european(cp[i], S, strikes[i], tau, r_d, r_f, kappa, theta, vv, rho, v0)
            
            if P[i] < 0:
                print("Reeeee")
                pass
            
            IV[i] = solve_implied_σ(P[i], S, strikes[i], r_d, r_f, tau, cp[i], σ_market[i])
        return np.sum((σ_market - IV)**2)

    # Calculate strikes for market deltas
    strikes = garman_kohlhagen(S=S,σ=σ_market,r_f=r_f, r_d=r_d,tau=tau,Δ=Δ,task='strike',Δ_convention=Δ_convention)
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

    res = minimize(lambda param: heston_vanilla_sse(param, cp, S, strikes, v0, σ_market, r_d, r_f, tau, kappa), initparam)
    vv, theta, rho = res.x
    
    if 2 * kappa * theta - vv**2 <= 0.0:
        # In the Heston model, the Feller condition is often required to be violated in order to get a good fit to market data
        warnings.warn("Feller condition violated.") 

    # Calculate the Heston model implied volatilities so we can chart and compare them to σ_market
    P = np.zeros(nb_strikes)
    IV = np.zeros(nb_strikes)
    for i in range(nb_strikes):
        P[i] = heston1993_price_fx_vanilla_european(cp[i], S, strikes[i], v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho)
        IV[i] = solve_implied_σ(P[i], S, strikes[i], r_d, r_f, tau, cp[i], σ_market[i])
 
    SSE = heston_vanilla_sse(res.x, cp, S, strikes, v0, σ_market, r_d, r_f, tau, kappa)
    
    if SSE == np.inf:
        # Calibration failed
        return None
    else:    
        return v0, vv, kappa, theta, lambda_, rho, IV, SSE

#%% Heston 1993 analytical pricing implementation (2nd version of the Characteristic function)

@jit(nopython=True)
def heston_1993_fx_vanilla_european_integral(phi, m, S, k, v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho):
    """
    Defines the integral for pricing an FX Vanilla European option per the analytic Heston 1993 formula
    This implementation uses the 2nd form of the Heston Characteristic function, detailed in Albrecher 2006 is used as it is more numerically stable.
    This in an auxiliary function for heston1993_price_fx_vanilla_european(), separated so @jit can be used
    
    Parameters:
    phi (complex): Point at which the auxiliary function is evaluated.
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
    float: Value of the auxiliary function at point `phi`.
    
    References:
    [1] S.Heston, (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options
    [2] H.Albrecher, P.Mayer, W.Schoutens, J.Tistaert (2006) The little Heston trap, Wilmott Magazine, January: 83–92.
    [3] A.Janek, T.Kluge, R.Weron, U.Wystup (2010) FX smile in the Heston model.
    """
    
    # x per equation (11) from Heston, 1993
    x = np.log(S) 
    # a, u, b per equation (12) from Heston, 1993
    a = kappa * theta 
    u = [0.5, -0.5] 
    b = [kappa + lambda_ - rho * vv, kappa + lambda_] 
    
    # d is per per equation (17) from Heston, 1993
    d = np.sqrt((1j * rho * vv * phi - b[m-1]) ** 2 - vv ** 2 * (2 * u[m-1] * phi * 1j - phi ** 2))
    
    # There are two formula's for "g", in the Heston characteristic function because d, an input to g, has two roots. 
    # Let "g1", be the presentation in the original Heston 1993 paper.
    # Let "g2", be the alternative, that is detailed in Albrecher 2006, "The Little Heston Trap". 
    # g2 = 1 / g1. 
    # We use g2, as using g1 leads to numerical problems based on most software implementations of calculation involving complex numbers
    g2 = (b[m-1] - rho * vv * phi * 1j - d) \
        / (b[m-1] - rho * vv * phi * 1j + d)

    D = (b[m-1] - rho * vv * phi * 1j - d) / (vv ** 2) * ((1 - np.exp(-d * tau)) / (1 - g2 * np.exp(-d * tau)))
    C = (r_d - r_f) * phi * 1j * tau + a / (vv ** 2) * ((b[m-1] - rho * vv * phi * 1j - d) * tau - 2 * np.log((1 - g2 * np.exp(-d * tau)) / (1 - g2)))
   
    # f per equation (17) from Heston, 1993
    f = np.exp(C + D * v0 + 1j * phi * x)
    
    # Function inside the integral in equation (18) from Heston, 1993
    F = np.real(np.exp(-1j * phi * np.log(k)) * f / (1j * phi))
    return F


def heston1993_price_fx_vanilla_european(cp, s, k, v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho):
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
    """

    # Equation (18) from Heston, 1993
    P1 = 0.5 + 1/np.pi * quad(lambda phi: heston_1993_fx_vanilla_european_integral(phi, 1, s, k, v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho), 0, np.inf, epsrel=1e-8)[0]
    P2 = 0.5 + 1/np.pi * quad(lambda phi: heston_1993_fx_vanilla_european_integral(phi, 2, s, k, v0, vv, r_d, r_f, tau, kappa, theta, lambda_, rho), 0, np.inf, epsrel=1e-8)[0]

    Pplus = (1 - cp) / 2 + cp * P1   # Pplus = N(d1)
    Pminus = (1 - cp) / 2 + cp * P2  # Pminus = N(d2)

    # tk is this for spot or forward Δ, matlab code assumed spot delta input?
    X = cp * (s * np.exp(-r_f * tau) * Pplus - k * np.exp(-r_d * tau) * Pminus) 
    
    return X

#%%  Heston model using Carr-Madan via 
# (method=0) Gauss-Kronrod quadrature
# (method=1) Fast Fourier Transform + Simpson method 

def heston_carr_madan_fx_vanilla_european(cp, S, K, T, r_d, r_f, kappa, theta, vv, rho, v0, integration_method=0):
    """
     Calculate European FX option price using the Heston model via Carr-Madan approach.
     
     Parameters:
     cp (int): 1 for call, -1 for put.
     S, K, T (float): Spot price, strike, and time to maturity.
     r_d, r_f (float): Domestic and foreign interest rates.
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
    """    
    
    if cp == 1:
        alpha = 0.75
    elif cp == -1:
        alpha = 1.75
    
    s0 = np.log(S)
    k = np.log(K)
    
    if integration_method == 0:
        # Integrate using adaptive Gauss-Kronrod quadrature
        result, _ = quad(heston_fft_fx_vanilla_european_integral, 0, np.inf, args=(cp, s0, k, T, r_d, r_f, kappa, theta, vv, rho, v0, alpha))
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
        d, g, A, B, C, charFunc = char_func(u, s0, k, T, r_d, r_f, kappa, theta, vv, rho, v0)
        F = charFunc * np.exp(-r_d * T) / (alpha**2 + cp * alpha - v**2 + 1j * (cp * 2 * alpha + 1) * v)
        
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
    Auxiliary function for HESTONFFTVANILLA.

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
    s0 (float): Initial stock price.
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
    charFunc = np.exp(A + B + C)
    return d, g, A, B, C, charFunc



#%%

def call_put_option_price_cos_method(cf, cp, S0, r, tau, K, N, L):
    """
    Computes the call or put option prices using the COS method.
    
    Parameters:
    - cf (function): Characteristic function
    - cp (int): 1 for call and -1 for put
    - S0 (float): Initial stock price
    - r (float): Interest rate
    - tau (float): Time to maturity
    - K (list or np.array): List of strike prices
    - N (int): Number of expansion terms
    - L (float): Size of truncation domain
    
    Returns:
    - np.array: Option prices
    """    
    
    # cf - Characteristic function as a functon, in the book denoted by phi
    # cp - C for call and P for put
    # S0 - Initial stock price
    # r - Interest rate (constant)
    # tau - Time to maturity
    # K - List of strikes
    # N - Number of expansion terms
    # L - Size of truncation domain (typ.:L=8 or L=10)

    # Reshape K to become a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])

    x0 = np.log(S0 / np.array(K)).reshape(-1, 1)

    # Truncation domain
    a, b = -L * np.sqrt(tau), L * np.sqrt(tau)
    
    # Summation from k=0 to k=N-1
    k = np.linspace(0, N-1, N).reshape(-1, 1)
    u = k * np.pi / (b - a)
    
    H_k = call_put_coefficients(cp, a, b, k)

    mat = np.exp(1j * np.outer((x0 - a) , u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))

    return value


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
    coef = chi_psi(a, b, c, d, k)
    
    chi, psi = coef['chi'], coef['psi']
    prefactor = 2 / (b - a)
    
    return prefactor * (chi - psi) if cp == 1 else prefactor * (-chi + psi)


@jit(nopython=True)
def chi_psi(a, b, c, d, k):
    """
    Compute chi and psi coefficients for COS method.
    
    Parameters:
    - a, b, c, d (float): Boundaries for truncation and integration
    - k (np.array): Array of k values
    
    Returns:
    - dict: {'chi': np.array, 'psi': np.array}
    """        
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/ (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d) - np.cos(k * np.pi  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    return {"chi": chi, "psi": psi}


def bs_call_option_price(cp, S_0, K, σ, tau, r):
    """
    Computes Black-Scholes call or put option prices.
    
    Parameters:
    - cp (int): 1 for call, -1 for put
    - S_0 (float): Initial stock price
    - K (list or np.array): List of strike prices
    - σ (float): Volatility
    - tau (float): Time to maturity
    - r (float): Interest rate
    
    Returns:
    - np.array: Option prices
    """        
    K = np.array(K).reshape(-1, 1)
    d1 = (np.log(S_0 / K) + (r + 0.5 * σ ** 2) * tau) / (σ * np.sqrt(tau))
    d2 = d1 - σ * np.sqrt(tau)

    if cp == 1:
        return norm.cdf(d1) * S_0 - norm.cdf(d2) * K * np.exp(-r * tau)
    elif cp == -1:
        return norm.cdf(-d2) * K * np.exp(-r * tau) - norm.cdf(-d1) * S_0


# Implied volatility method
def implied_volatility(cp, market_price, K, T, S_0, r):
    """
    Calculates implied volatility using Newton's method.
    
    Parameters:
    - cp (int): 1 for call, -1 for put
    - market_price (float): Observed market price of option
    - K (float): Strike price
    - T (float): Time to maturity
    - S_0 (float): Initial stock price
    - r (float): Interest rate
    
    Returns:
    - float: Implied volatility
    """        
    func = lambda sigma: (bs_call_option_price(cp, S_0, K, sigma, T, r) - market_price) ** 1.0
    return scipy.optimize.newton(func, 0.7, tol=1e-5)


def chf_heston_model(r, tau, kappa, gamma, vbar, v0, rho):
    """
    Characteristic function for Heston model.
    
    Parameters:
    - r (float): Interest rate
    - tau (float): Time to maturity
    - kappa (float): Mean reversion rate
    - gamma (float): Volatility of volatility
    - vbar (float): Long-term average volatility
    - v0 (float): Initial variance
    - rho (float): Correlation coefficient
    
    Returns:
    - function: Characteristic function for Heston model
    """        
    D1 = lambda u: np.sqrt((kappa - gamma * rho * 1j * u) ** 2 + (u ** 2 + 1j * u) * gamma ** 2)
    g = lambda u: (kappa - gamma * rho * 1j * u - D1(u)) / (kappa - gamma * rho * 1j * u + D1(u))
    C = lambda u: (1 - np.exp(-D1(u) * tau)) / (gamma ** 2 * (1 - g(u) * np.exp(-D1(u) * tau)))
    A = lambda u: r * 1j * u * tau + kappa * vbar * tau / gamma ** 2 * (kappa - gamma * rho * 1j * u - D1(u)) \
                 - 2 * kappa * vbar / gamma ** 2 * np.log((1 - g(u) * np.exp(-D1(u) * tau)) / (1 - g(u)))
    return lambda u: np.exp(A(u) + C(u) * v0)



#%%  Heston model using the Lewis-Lipton formula (something wrong with it)

# def heston_lipton_fx_vanilla_european_integral(v, S, K, T, r_d, r_f, kappa, theta, vv, rho, v0):
#     """
#     Auxiliary function used by heston_vanilla_lipton for numerical integration.
#     This in an auxiliary function for heston1993_price_fx_vanilla_european(), separated so @jit can be used
    
#     Parameters:
#     v (float): Variable for integration.
#     S (float): Spot price.
#     K (float): Strike price.
#     T (float): Time to maturity.
#     r (float): Domestic interest rate.
#     rf (float): Foreign interest rate.
#     kappa (float): Rate of mean reversion.
#     theta (float): Long-term volatility level.
#     vv (float): Volatility of volatility.
#     rho (float): Correlation between Wiener processes.
#     v0 (float): Initial volatility.
    
#     Returns:
#     float: Value of the auxiliary function at point v.
#     """        
#     X = np.log(S / K) + (r_d - r_f) * T
#     kappa_hat = kappa - rho * vv / 2
#     zeta = np.sqrt(v ** 2 * vv ** 2 * (1 - rho ** 2) + 2 * 1j * v * vv * rho * kappa_hat + kappa_hat ** 2 + vv ** 2 / 4)
#     psi_plus = -(1j * kappa * rho * vv + kappa_hat) + zeta
#     psi_minus = (1j * kappa * rho * vv + kappa_hat) + zeta
#     alpha = -kappa * theta / (vv ** 2) * (psi_plus * T + 2 * np.log((psi_minus + psi_plus * np.exp(-zeta * T)) / (2 * zeta)))
#     beta = (1 - np.exp(-zeta * T)) / (psi_minus + psi_plus * np.exp(-zeta * T))
#     payoff = np.real(np.exp((-1j * v + 0.5) * X + alpha - (v ** 2 + 0.25) * beta * v0) / (v ** 2 + 0.25))
#     return payoff

# def heston_lipton_price_fx_vanilla_european(cp, S, K, T, r_d, r_f, kappa, theta, sigma, rho, v0):
#     """
#     Calculate European FX option price using the Heston model and Lewis-Lipton formula.
    
#     Parameters:
#     phi (int): 1 for call option, -1 for put option.
#     S (float): Spot price.
#     K (float): Strike price.
#     T (float): Time to maturity.
#     r_d (float): Domestic interest rate.
#     r_f (float): Foreign interest rate.
#     kappa (float): Rate of mean reversion.
#     theta (float): Long-term volatility level.
#     sigma (float): Volatility of volatility.
#     rho (float): Correlation between Wiener processes.
#     v0 (float): Initial volatility.
    
#     Returns:
#     float: Option price.
#     """    
#     C = np.exp(-r_f * T) * S - np.exp(-r_d * T) * K / np.pi * quad(heston_lipton_fx_vanilla_european_integral, 0, np.inf, args=(S, K, T, r_d, r_f, kappa, theta, sigma, rho, v0), epsrel=1e-8)[0]
#     if cp == 1:  # call option
#         X = C
#     else:  # put option
#         X = C - S * np.exp(-r_f * T) + K * np.exp(-r_d * T)
#     return X

#%%


