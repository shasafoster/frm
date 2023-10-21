#%%
"""
Created on Thu Nov 30 2018
Merton Model and convergence obtained with the COS method
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import enum
import scipy.optimize as optimize
from numba import jit


v0 = 0.0175 # intial variance
vv = 0.5751 # volatility of volatility 
kappa = 1.5768 # rate of mean reversion towards the long-term mean of variance process
theta = 0.0398 # long-term mean of the variance process
rho =-0.5711 # correlation


def chf_heston_model(r, tau, v0, vv, kappa, theta, rho):
    """
    Characteristic function for the Heston model.
    
    Parameters:
    - r (float): Risk-free interest rate
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
    A = lambda φ: r * 1j * φ * tau + kappa * theta * tau / vv ** 2 * (kappa - vv * rho * 1j * φ - D1(φ)) \
                 - 2 * kappa * theta / vv ** 2 * np.log((1 - g(φ) * np.exp(-D1(φ) * tau)) / (1 - g(φ)))
    return lambda φ: np.exp(A(φ) + C(φ) * v0)


def heston_cos_vanilla_european(cp, S0, r, tau, K, N, L, v0, vv, kappa, theta, rho):
    """
    Computes the call or put option prices using the COS method.
    
    Parameters:
    - cp (int): 1 for call and -1 for put
    - S0 (float): Initial stock price
    - r (float): Interest rate
    - tau (float): Time to maturity
    - K (list or np.array): List of strike prices
    - N (int): Number of expansion terms
    - L (float): Size of truncation domain
    - v0 (float): Initial volatility
    - vv (float): Volatility of volatility
    - kappa (float): rate of mean reversion towards the long-term mean of variance process
    - theta (float): long-term mean of the variance process
    - rho (float): Correlation between price and volatility    
    
    Returns:
    - np.array: Option prices
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
        # Reshape K to become a column vector
        K = np.array(K).reshape([len(K),1]) 
        
    x0 = np.log(S0 / np.array(K)).reshape(-1, 1)

    # Truncation domain
    a, b = -L * np.sqrt(tau), L * np.sqrt(tau)
    
    # Summation from k = 0 to k = N-1
    k = np.linspace(0, N-1, N).reshape(-1, 1)
    u = k * np.pi / (b - a)
    
    H_k = call_put_coefficients(cp, a, b, k)

    mat = np.exp(1j * np.outer((x0 - a) , u))
    cf = chf_heston_model(r, tau, v0, vv, kappa, theta, rho)
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    return np.exp(-r * tau) * K * np.real(mat.dot(temp))



def bs_call_option_price(cp, S_0, K, σ, tau, r):
    K = np.array(K).reshape(-1, 1)
    d1 = (np.log(S_0 / K) + (r + 0.5 * σ ** 2) * tau) / (σ * np.sqrt(tau))
    d2 = d1 - σ * np.sqrt(tau)

    if cp == 1:
        return norm.cdf(d1) * S_0 - norm.cdf(d2) * K * np.exp(-r * tau)
    elif cp == -1:
        return norm.cdf(-d2) * K * np.exp(-r * tau) - norm.cdf(-d1) * S_0


# Implied volatility method
def implied_volatility(cp, market_price, K, T, S_0, r):
    func = lambda sigma: (bs_call_option_price(cp, S_0, K, sigma, T, r) - market_price) ** 1.0
    return optimize.newton(func, 0.7, tol=1e-5)





cp = 1
S0 = 100
r = 0.00
tau = 1

K = np.linspace(50,150,10000)
K = np.array(K).reshape([len(K),1])

# COS method settings
L = 3


v0 = 0.0175 # intial variance
vv = 0.5751 # volatility of volatility 
kappa = 1.5768 # rate of mean reversion towards the long-term mean of variance process
theta = 0.0398 # long-term mean of the variance process
rho =-0.5711 # correlation


# The COS method -- Reference value
N = 1000
valCOSRef = heston_cos_vanilla_european(cp, S0, r, tau, K, N, L, v0, vv, kappa, theta, rho)

plt.figure(1)
plt.plot(K,valCOSRef,'-k')
plt.xlabel("strike, K")
plt.ylabel("Option Price")
plt.grid()

# For different numbers of integration terms

N = [32,64,96,128,160,400]
maxError = []
for n in N:
    valCOS = heston_cos_vanilla_european(cp, S0, r, tau, K, n, L, v0, vv, kappa, theta, rho)
    errorMax = np.max(np.abs(valCOS - valCOSRef))
    maxError.append(errorMax)
    print('maximum error for n ={0} is {1}'.format(n,errorMax))
        

