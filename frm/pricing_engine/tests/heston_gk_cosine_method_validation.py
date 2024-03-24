# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os
    import pathlib
    import sys
    
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.pricing_engine.heston_garman_kohlhagen import heston1993_price_fx_vanilla_european, heston_fit_vanilla_fx_smile, heston_cos_vanilla_european, heston_carr_madan_fx_vanilla_european
from frm.frm.pricing_engine.cosine_method import heston_cos_vanilla_european2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
import pandas as pd

#% STF2hes07 - fit smile

# Initialize
standalone = 0  # set to 0 to make plots as seen in STF2
delta_plt = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) # forward deltas
Δ = np.array([-0.1, -0.25, 0.5, 0.25, 0.1])
r_d = np.array([0.31100, 0.32875, 0.49781, 0.70075, 1.08, 1.08]) * 0.01
r_f = np.array([0.58, 0.631, 0.884, 1.131, 1.399, 1.399]) * 0.01
tau = np.array([7 / 365, 1 / 12, 3 / 12, 6 / 12, 1, 2])
S = 1.2779
cp = np.array([-1, -1, 1, 1, 1])
Δ_convention = 'regular_forward_Δ'

tenors = ['1W','1M','3M','6M','1Y','2Y']

σ_market_set = [
    np.array([13.948, 13.192, 13.35, 14.042, 15.4355]) * 0.01,      # 1W 
    np.array([12.45, 12.249, 12.7, 13.749, 15.225]) * 0.01,         # 1M
    np.array([12.6885, 12.3945, 13.0495, 14.5445, 16.666]) * 0.01,  # 3M
    np.array([13.688, 12.899, 13.55, 15.299, 18.1275]) * 0.01,      # 6M
    np.array([13.6705, 12.668, 13.187, 14.718, 17.2705]) * 0.01,    # 1Y
    np.array([12.139, 11.784, 12.174, 13.284, 14.764]) * 0.01       # 2Y
]

start_time = time()
results_gk = []
results_cos = []

#pricing_method = 'heston_analytical_1993'
pricing_method_gk = 'heston_carr_madan_gauss_kronrod_quadrature'
#pricing_method = 'heston_carr_madan_fft_w_simpsons'
pricing_method_cos = 'heston_cosine'


# # Main loop for various smiles
# for i, σ_market in enumerate(σ_market_set):
#     if i > 5:
#         #delta_spot = np.exp(-r_f[i] * tau[i]) * delta
#         t1 = time()
#         v0, vv, kappa, theta, rho, lambda_, IV_gk, SSE = heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ_market, S, r_f[i], r_d[i], tau[i], cp, pricing_method=pricing_method_gk)        
#         t2 = time()
#         results_gk.append([v0, vv, kappa, theta, rho, lambda_, IV_gk, SSE])
       
#         t3 = time()
#         v0, vv, kappa, theta, rho, lambda_, IV_cos, SSE = heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ_market, S, r_f[i], r_d[i], tau[i], cp, pricing_method=pricing_method_cos)
#         t4 = time()
#         results_cos.append([v0, vv, kappa, theta, rho, lambda_, IV_cos, SSE])
        
#         # Displaying output
#         print(f'=== {tenors[i]} calibration results ===')
#         print(f'v0, vv, kappa, theta, rho: {v0, vv, kappa, theta, rho}')
#         #print(f'[IV (10, 25, ATM, 75, 90), SSE] * 100%: {IV*100, SSE*100}')
        
#         print(t2-t1)
#         print(t4-t3)
        
#         # Plotting
#         plt.figure(i+1)
#         plt.plot(delta_plt * 100, σ_market * 100, 'ko-', linewidth=1)
#         plt.plot(delta_plt * 100, IV_gk * 100, 'rs--', linewidth=1)
#         plt.plot(delta_plt * 100, IV_cos * 100, 'gs--', linewidth=1)
#         plt.legend([f'{tenors[i]} smile', 'Heston fit'], loc='upper left')
#         plt.xlabel('Delta [%]')
#         plt.ylabel('Implied volatility [%]')
#         plt.xticks(delta_plt * 100)
#         #plt.title(pricing_method)
#         plt.show()

# end_time = time()
# print(f"Elapsed time: {end_time - start_time:.2f} seconds")



i = 2
rf = r_f[i] # 0.01131
rd = r_d[i] # 0.0070075
K = np.array([1.03413767, 1.15064198, 1.28873193, 1.466967  , 1.6959051 ])
cp = [-1, -1, 1, 1, 1]

# Inputs
v0 = 0.1217400016428454
vv = 0.293475456
kappa = 1.5
theta = 0.017132031
rho = 0.232453505225961

S0 = 1.2779
tau = 0.5



print('Heston1993', heston1993_price_fx_vanilla_european(S0, tau, rf, rd, cp[i], K[i], v0, vv, kappa, theta, rho, 0))
print('CM_GQ:', heston_carr_madan_fx_vanilla_european(S0, tau, rf, rd, cp[i], K[i], v0, vv, kappa, theta, rho, integration_method=0))
print('CM_FFT:', heston_carr_madan_fx_vanilla_european(S0, tau, rf, rd, cp[i], K[i], v0, vv, kappa, theta, rho, integration_method=1))
print('COS:', heston_cos_vanilla_european(S0, tau, rf, rd, cp[i], K[i], v0, vv, kappa, theta, rho, N=160, L=10))
print('COS2:', heston_cos_vanilla_european2(S0, tau, rf, rd, cp[i], K[i], v0, vv, kappa, theta, rho, N=160, L=10))


#heston1993_price_fx_vanilla_european(s0=1.2, tau=2, r_f=0.001, r_d=0.0011, cp=1, K=1, v0=0.1, vv=0.0001, kappa=0, theta=0.1**2, rho=0, lambda_=0)

#Heston1993 0.010856228973634544
#CM_GQ: 0.010856228973635803
#CM_FFT: 0.010921155228074013
#chf version 1 <function chf_heston_cosine_model.<locals>.<lambda> at 0x000002B192032200>
#COS: [[0.01669337]]
#COS2: [0.00124444]

#print(K[i]/S)
#print(np.log(K[i]/S))

#%%







