# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 15:24:02 2023

@author: shasa
"""

## Title:        Option Pricing
## Author:       Elisa FLeissner, Lars Stauffenegger
## Email:        elisa.fleissner@student.unisg.ch,
##               lars.stauffenegger@student.unisg.ch,
## Place, Time:  Zürich, 24.03.19
## Description:  Option Pricing with Black Scholes, COS Method and Heston Model
## Improvements: -
## Last changes: -


# In[1]: Packages
import quandl
import numpy as np
import AllFunctions as func
import matplotlib.pyplot as plt


# In[2]: Data
# Import from quandl
quandl.ApiConfig.api_key = "mrMTRoAdPycJSyzyjxPN"
ticker     = "AAPL"
database   = "EOD"
identifier = database + "/" + ticker
stockData  = quandl.get(identifier, rows = 500)

# Return and Volatility
logReturn = np.log(stockData.Close) - np.log(stockData.Close.shift(1))
logReturn.drop(logReturn.index[:1], inplace = True)
tradingDaysCount   = 252
annualisedMean     = np.mean(logReturn) * tradingDaysCount
annualisedVariance = np.var(logReturn) * tradingDaysCount
annualisedStdDev   = np.sqrt(annualisedVariance)
lastPrice          = stockData.Close.tail(1)


# In[2]: Parameter
# Volvol and rho according to Fang, 2010, p. 30
r      = 0                  # assumption Risk-free rate
mu     = r #annualisedMean  # Mean rate of drift
sigma  = annualisedStdDev   # Initial Vola of underyling at time 0; also called u0 or a
S0     = lastPrice[0]       # Today's stock price
tau    = 30 / 365           # Time to expiry in years
q      = 0                  # Divindend Yield
lm     = 1.5768             # The speed of mean reversion
v_bar  = annualisedVariance # Mean level of variance of the underlying
volvol =  0.5751            # Volatility of the volatiltiy process
rho    = -0.5711            # Covariance between the log stock and the variance process

# Range of Strikes
mini    = int(S0 * 0.8)
maxi    = int(S0 * 1.2)
K       = np.arange(mini, maxi, dtype = np.float)

# Truncation Range
L       = 120
a, b    = func.truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol)
bma     = b-a

# Number of Points
N       = 15
k       = np.arange(np.power(2,N))

# Input for the Characterstic Function Phi
u       = k * np.pi/bma


# In[3]: Black Scholes Option Pricing
C_BS, P_BS = func.blackScholes(S0, K, r, tau, sigma, q)
print(C_BS)


# In[4]: COS-FFT Value Function for Put
UkPut  = 2 / bma * ( func.cosSer1(a,b,a,0,k) - func.cosSerExp(a,b,a,0,k) )
UkCall = 2 / bma * ( func.cosSerExp(a,b,0,b,k) - func.cosSer1(a,b,0,b,k) )


# In[5]: COS with BS-Characterstic Function
charactersticFunctionBS = func.charFuncBSM(u, mu, sigma, tau)

C_COS = np.zeros((np.size(K)))

for m in range(0,np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(np.multiply(charactersticFunctionBS, addIntegratedTerm))
    Fk[0]=0.5 * Fk[0] 
    C_COS[m] = K[m] * np.sum(np.multiply(Fk,UkCall)) * np.exp(-r * tau)
    
print (C_COS)


# In[6]: COS with Fang & Oosterlee (2008) Version of Heston's Characteristic Function
charactersticFunctionHFO = func.charFuncHestonFO(mu, r, u, tau, sigma, v_bar, lm, rho, volvol)

C_COS_HFO = np.zeros((np.size(K)))
P_COS_HFO = np.zeros((np.size(K)))
C_COS_PCP = np.zeros((np.size(K)))

for m in range(0, np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(charactersticFunctionHFO * addIntegratedTerm)
    Fk[0] = 0.5 * Fk[0]						
    C_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkCall)) * np.exp(-r * tau)
    P_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkPut)) * np.exp(-r * tau)
    C_COS_PCP[m] = P_COS_HFO[m] + S0 * np.exp(-q * tau) - K[m] * np.exp(-r * tau)

print(C_COS_HFO)
print(P_COS_HFO)
print(C_COS_PCP)


# In[7]: Plotting
plt.plot(K, C_BS, "g.", K, C_COS, "b.", K, C_COS_HFO, "r.")
plt.axvline(x = S0)
plt.show()
print("C_BS = green, C_COS = blue, C_COS_HFO = red")

## End

#%%

import numpy as np

cp = 1
S0 = 100
r = 0.00
tau = 1

K = np.linspace(50,150)
K = np.array(K).reshape([len(K),1])

# COS method settings
L = 3

kappa = 1.5768
gamma = 0.5751
vbar = 0.0398
rho =-0.5711
v0 = 0.0175


#%%








