
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

    os.chdir(pathlib.Path(__file__).parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.pricing_engine.cosine_method import calculate_Uk_european_options, get_cos_truncation_range, chf_heston_cos

import quandl
import numpy as np
import AllFunctions as func
import matplotlib.pyplot as plt


# In[2]: Data
# Import from quandl
quandl.ApiConfig.api_key = "YNMkiy_Ncc2PsePgzAvg"
ticker     = "AAPL"
database   = "WIKI"
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
mu     = 0 #r #annualisedMean  # Mean rate of drift
sigma  = annualisedStdDev   # Initial Vola of underyling at time 0; also called u0 or a
S0     = lastPrice[0]       # Today's stock price
tau    = 30 / 365           # Time to expiry in years
q      = 0                  # Divindend Yield
lm     = 1.5768             # The speed of mean reversion
v_bar  = annualisedVariance # Mean level of variance of the underlying
volvol =  0.5751            # Volatility of the volatiltiy process
rho    = -0.5711            # Covariance between the log stock and the variance process

r      = 0                  # assumption Risk-free rate
mu     = -0.0026899999999999997 #r #annualisedMean  # Mean rate of drift
sigma  = 0.1217400016428454   # Initial Vola of underyling at time 0; also called u0 or a
S0     = 1.2779     # Today's stock price
tau    = 2 #30 / 365           # Time to expiry in years
q      = 0                  # Divindend Yield
lm     = 1.5             # The speed of mean reversion
v_bar  = 0.017132031 # Mean level of variance of the underlying
volvol =  0.293475456           # Volatility of the volatiltiy process
rho    = 0.232453505225961         # Covariance between the log stock and the variance process


# Range of Strikes
mini    = int(S0 * 0.8 * 100)
maxi    = int(S0 * 1.2 * 100)
K       = np.arange(mini, maxi, dtype = np.float) / 100

# Truncation Range
L       = 10
a, b    = func.truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol)


model_param =  {'tau': tau,
                'mu': mu,
                'v0': sigma,
                'vv': volvol,
                'kappa': lm,
                'theta': v_bar, 
                'rho': rho}
a_, b_ = get_cos_truncation_range(model='heston', L=L, model_param=model_param)

# Number of Points
k = np.arange(160) # Per Fang, for an L of 10, this should be fine at 160

# Input for the Characterstic Function Phi
u = k * np.pi/(b-a)

# In[3]: Black Scholes Option Pricing
C_BS, P_BS = func.blackScholes(S0, K, r, tau, sigma, q)
print(C_BS)


# In[4]: COS-FFT Value Function for Put

UkCall = calculate_Uk_european_options(cp=1, a=a, b=b, k=k)
UkPut = calculate_Uk_european_options(cp=-1, a=a, b=b, k=k)

# In[5]: COS with BS-Characterstic Function
charactersticFunctionBS = func.charFuncBSM(u, mu, sigma, tau)

C_COS = np.zeros((np.size(K)))

for m in range(0,np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/(b-a))
    Fk = np.real(np.multiply(charactersticFunctionBS, addIntegratedTerm))
    Fk[0]=0.5 * Fk[0] 
    C_COS[m] = K[m] * np.sum(np.multiply(Fk,UkCall)) * np.exp(-r * tau)
    
print (C_COS)

# In[6]: COS with Fang & Oosterlee (2008) Version of Heston's Characteristic Function

charactersticFunctionHFO = chf_heston_cos(u=u, tau=tau, r_f=r, r_d=q, v0=sigma, vv=volvol, kappa=lm, theta=v_bar, rho=rho)


C_COS_HFO = np.zeros((np.size(K)))
P_COS_HFO = np.zeros((np.size(K)))
C_COS_PCP = np.zeros((np.size(K)))

for m in range(0, np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/(b-a))
    Fk = np.real(charactersticFunctionHFO * addIntegratedTerm)
    Fk[0] = 0.5 * Fk[0]						
    C_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkCall)) * np.exp(-r * tau) # European call option price
    P_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkPut)) * np.exp(-r * tau) # European put option price
    C_COS_PCP[m] = P_COS_HFO[m] + S0 * np.exp(-q * tau) - K[m] * np.exp(-r * tau) # By put-call parity

print(C_COS_HFO)
print(P_COS_HFO)
print(C_COS_PCP)


# In[7]: Plotting
plt.plot(K, C_BS, "g*", label='C_BlackScholes')
plt.plot(K, C_COS, "b--", label='C_COS_BlackScholes')
plt.plot(K, C_COS_HFO, "r.", label='C_COS_Heston_FangOsterlee')
plt.axvline(x=S0, label='S0')
plt.legend(loc='best')  # Adds a legend with the best location
plt.show()

## End