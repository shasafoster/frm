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


def call_put_option_price_cos_method(cf, cp, S0, r, tau, K, N, L):
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
    c, d = (0, b) if cp == 1 else (a, 0)
    coef = chi_psi(a, b, c, d, k)
    
    chi, psi = coef['chi'], coef['psi']
    prefactor = 2 / (b - a)
    
    return prefactor * (chi - psi) if cp == 1 else prefactor * (-chi + psi)


@jit(nopython=True)
def chi_psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/ (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0))
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d) - np.cos(k * np.pi  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    return {"chi": chi, "psi": psi}


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


def chf_heston_model(r, tau, kappa, gamma, vbar, v0, rho):
    D1 = lambda u: np.sqrt((kappa - gamma * rho * 1j * u) ** 2 + (u ** 2 + 1j * u) * gamma ** 2)
    g = lambda u: (kappa - gamma * rho * 1j * u - D1(u)) / (kappa - gamma * rho * 1j * u + D1(u))
    C = lambda u: (1 - np.exp(-D1(u) * tau)) / (gamma ** 2 * (1 - g(u) * np.exp(-D1(u) * tau)))
    A = lambda u: r * 1j * u * tau + kappa * vbar * tau / gamma ** 2 * (kappa - gamma * rho * 1j * u - D1(u)) \
                 - 2 * kappa * vbar / gamma ** 2 * np.log((1 - g(u) * np.exp(-D1(u) * tau)) / (1 - g(u)))
    return lambda u: np.exp(A(u) + C(u) * v0)



cp = 1
S0 = 100
r = 0.00
tau = 1

K = np.linspace(50,150,10000)
K = np.array(K).reshape([len(K),1])

# COS method settings
L = 3

kappa = 1.5768
gamma = 0.5751
vbar = 0.0398
rho =-0.5711
v0 = 0.0175

# Compute ChF for the Heston model
cf = chf_heston_model(r,tau,kappa,gamma,vbar,v0,rho)

# The COS method -- Reference value
N = 5000
valCOSRef = call_put_option_price_cos_method(cf, cp, S0, r, tau, K, N, L)

plt.figure(1)
plt.plot(K,valCOSRef,'-k')
plt.xlabel("strike, K")
plt.ylabel("Option Price")
plt.grid()

# For different numbers of integration terms

N = [32,64,96,128,160,400]
maxError = []
for n in N:
    valCOS = call_put_option_price_cos_method(cf, cp, S0, r, tau, K, n, L)
    errorMax = np.max(np.abs(valCOS - valCOSRef))
    maxError.append(errorMax)
    print('maximum error for n ={0} is {1}'.format(n,errorMax))
        

