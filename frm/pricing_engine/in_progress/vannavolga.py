# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# https://gpt4all.io/index.html


import os
import pandas as pd
import numpy as np
import scipy as sc

os.chdir('C:/Users//OneDrive/Finance/OTC')

from otc.schedule.tenor import tenor_to_date_offset
from otc.calendars.calendar import get_calendar
from otc.schedule.daycounter import DayCounter



#%% Import and clean data

file_path = 'C:/Users//OneDrive/Finance/OTC/otc/notebooks/fx_data.xlsx'
sheet_name = 'fx_data_validation'  
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols='A:H')

#%%

df['alpha'] = -1 * sc.stats.norm.ppf(0.25 * np.exp(df['base_ccy_depo_rates'] * df['years_to_maturity'])) 
df['K_atm'] = df['fx_spot'] * np.exp((df['base_ccy_depo_rates'] - df['term_ccy_depo_rates'] + 0.5 * df['σ_atm'] ** 2) * df['years_to_maturity']) 
df['K_25Δcall'] = df['fx_spot'] * np.exp(df['alpha'] * df['σ_25Δcall'] * np.sqrt(df['years_to_maturity']) + (df['base_ccy_depo_rates'] - df['term_ccy_depo_rates'] + 0.5 * df['σ_25Δcall'] ** 2) * df['years_to_maturity'])
df['K_25Δput'] = df['fx_spot'] * np.exp(-1 * df['alpha'] * df['σ_25Δput'] * np.sqrt(df['years_to_maturity']) + (df['base_ccy_depo_rates'] - df['term_ccy_depo_rates'] + 0.5 * df['σ_25Δput'] ** 2) * df['years_to_maturity'])
assert (df['K_25Δcall'] > df['K_atm']).all(), 'K_25Δcall is less than K_atm'
assert (df['K_25Δput'] < df['K_atm']).all(), 'K_25Δput is greater than than K_atm'
        
#%%


def d1(F,K,σ,t):
    dplus = (np.log(F/K) + 0.5 * (σ**2) * t) / (σ*np.sqrt(t))
    return dplus

def d2(F,K,σ,t):
    dminus = d1(F,K,σ,t) - σ * np.sqrt(t)
    return dminus


def VannaVolgaImpliedVol(F,K,t,K1,K2,K3,σ1,σ2,σ3):
    y1 = (np.log(K2/K) * np.log(K3/K)) / (np.log(K2/K1) * np.log(K3/K1))
    y2 = (np.log(K/K1) * np.log(K3/K)) / (np.log(K2/K1) * np.log(K3/K2))
    y3 = (np.log(K/K1) * np.log(K/K2)) / (np.log(K3/K1) * np.log(K3/K2))

    P = (y1*σ1 + y2*σ2 + y3 *σ3) - σ2
    
    Q = y1 * d1(F,K1,σ1,t) * d2(F,K1,σ1,t) * ((σ1-σ2)**2) \
      + y2 * d1(F,K2,σ2,t) * d2(F,K2,σ2,t) * ((σ2-σ2)**2) \
      + y3 * d1(F,K3,σ3,t) * d2(F,K3,σ3,t) * ((σ3-σ2)**2)
    
    d1d2 = d1(F,K,σ2,t) * d2(F,K,σ2,t)
    
    σ = σ2 + (-σ2 + np.sqrt(σ2**2 + d1d2 *(2*σ2*P+Q)))/(d1d2)
    
    return σ

#%%
strike = (1+np.arange(-0.20,0.21,0.001))*df['fx_spot'].iloc[0]

strikes_matrix = np.tile(strike,[len(df),1])

VVImpliedVol = np.zeros((len(df),len(strike)),dtype=float)
F = df['fx_spot'] * np.exp(np.multiply(df['base_ccy_depo_rates'] - df['term_ccy_depo_rates'],df['years_to_maturity']))

#%%

for i in range(len(strike)):
    VVImpliedVol[:,i]=VannaVolgaImpliedVol(F,strikes_matrix[:,i],df['years_to_maturity'],
        df['K_25Δput'], df['K_atm'], df['K_25Δcall'],
        df['σ_25Δput'], df['σ_atm'], df['σ_25Δcall'])
    
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X = df['years_to_maturity']
Y = strike
X, Y = np.meshgrid(X,Y)

Z = VVImpliedVol.T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Strike')
ax.set_zlabel('Volatility')
surf = ax.plot_surface(X, Y, Z, cmap=cm.gnuplot,linewidth=0)
plt.show()

