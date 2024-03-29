# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:49:52 2024

@author: shasa
"""

## Title:        All Functions
## Author:       Elisa FLeissner, Lars Stauffenegger
## Email:        elisa.fleissner@student.unisg.ch,
##               lars.stauffenegger@student.unisg.ch,
## Place, Time:  Zürich, 24.03.19
## Description:  All Functions for Option Pricing with Black Scholes, COS Method and Heston Model
## Improvements: -
## Last changes: -


# In[1]: Packages, Settings
import numpy as np
from scipy.special import erf


# In[2]: Standard Normal Cumulative Distribution Function
def StdNormCdf(z):
    phi = 0.5 * (1 + erf(z/np.sqrt(2)))
    return phi


# In[3]: Black Scholes Model
def blackScholes(S, K, r, tau, sigma, q):
    S = S * np.exp(-q * tau)
    d1 = np.divide((np.log(np.divide(S, K)) + (r + 1/2 * np.power(sigma, 2)) * tau), (sigma * np.sqrt(tau)))
    d2 = d1 - sigma * np.sqrt(tau)
    call  = np.multiply(S, StdNormCdf(d1)) - np.multiply(np.multiply(K, np.exp(-r * tau)), StdNormCdf(d2))
    put  = call + np.multiply(K, np.exp(-r * tau)) - S
    return call, put


# In[4]: Truncation Range based on Fitt et al. (2010)
def truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol):
        c1 = mu * tau + (1 - np.exp(-lm * tau)) * (v_bar - sigma)/(2 * lm) - v_bar * tau / 2

        c2 = 1/(8 * np.power(lm,3)) * (volvol * tau * lm * np.exp(-lm * tau)  * (sigma - v_bar) * (8 * lm * rho - 4 * volvol) \
            + lm * rho * volvol * (1 - np.exp(-lm * tau)) * (16 * v_bar - 8 * sigma) \
            + 2 * v_bar * lm * tau * (-4 * lm * rho * volvol + np.power(volvol,2) + 4 * np.power(lm,2)) \
            + np.power(volvol,2) * ((v_bar - 2 * sigma) * np.exp(-2 * lm * tau) + v_bar * (6 * np.exp(-lm * tau) - 7) + 2 * sigma) \
            + 8 * np.power(lm,2) * (sigma - v_bar) * (1 - np.exp(-lm * tau)))

        a = c1 - L * np.sqrt(np.abs(c2))
        b = c1 + L * np.sqrt(np.abs(c2))
        return a, b




# In[6]: Characteristic Functions
def charFuncBSM(s, mu, sigma, T):
    phi = np.exp((mu - 0.5 * np.power(sigma,2)) * 1j * np.multiply(T,s) - 0.5 * np.power(sigma,2) * T * np.power(s,2))
    return phi

def charFuncHestonFO(mu, r, u, tau, sigma, v_bar, lm, rho, volvol):
    d = np.sqrt(np.power(lm - 1j * rho * volvol * u, 2) + np.power(volvol,2) * (np.power(u,2) + u * 1j))
    g = (lm - 1j * rho * volvol * u - d) / (lm - 1j * rho * volvol * u + d)
    C = np.divide(lm * v_bar, np.power(volvol,2)) * ( (lm - 1j * rho * volvol * u - d) * tau - 2 * np.log(np.divide((1 - g * np.exp(-d * tau)) , (1-g)) ))
    D = 1j * r * u * tau + sigma / np.power(volvol,2) * (np.divide((1 - np.exp(-d * tau)), (1 - g * np.exp(-d * tau)))) * (lm - 1j * rho * volvol * u - d) 
    phi = np.exp(D) * np.exp(C)
    return phi


## Title:        Option Pricing
## Author:       Elisa FLeissner, Lars Stauffenegger
## Email:        elisa.fleissner@student.unisg.ch,
##               lars.stauffenegger@student.unisg.ch,
## Place, Time:  Zürich, 24.03.19
## Description:  Option Pricing with Black Scholes, COS Method and Heston Model
## Improvements: -
## Last changes: -
