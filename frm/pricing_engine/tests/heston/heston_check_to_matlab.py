# -*- coding: utf-8 -*-
# This test script is a check to "STF2hes07.m" which is a support to
# "FX smile in the Heston model" by A Janek, 2010.
# Source: https://ideas.repec.org/c/wuu/hscode/zip10001.html

if __name__ == "__main__":
    import os
    import pathlib
    import sys
    
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.pricing_engine.heston_garman_kohlhagen import heston1993_price_fx_vanilla_european, heston_fit_vanilla_fx_smile
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
import pandas as pd


# Initialize
standalone = 0  # set to 0 to make plots as seen in STF2
delta_plt = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) # forward deltas
#Δ = np.array([-0.1, -0.25, 0.5, 0.25, 0.1])
Δ = np.array([0.1, 0.25, 0.5, -0.25, -0.1])
r_d = np.array([0.31100, 0.32875, 0.49781, 0.70075, 1.08, 1.08]) * 0.01
r_f = np.array([0.58, 0.631, 0.884, 1.131, 1.399, 1.399]) * 0.01
tau = np.array([7 / 365, 1 / 12, 3 / 12, 6 / 12, 1, 2])
S = 1.2779
#cp = np.array([-1, -1, 1, 1, 1])
cp = np.array([1, 1, 1, -1, -1])
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

pricing_method = 'heston_analytical_1993'
#pricing_method = 'heston_carr_madan_gauss_kronrod_quadrature'
#pricing_method = 'heston_carr_madan_fft_w_simpsons'
#pricing_method = 'heston_cosine'


# Main loop for various smiles
for i, σ_market in enumerate(σ_market_set):
    if i >= 5:
        #delta_spot = np.exp(-r_f[i] * tau[i]) * delta
        var0, vv, kappa, theta, rho, lambda_, IV, SSE = heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ_market, S, r_f[i], r_d[i], tau[i], cp, pricing_method=pricing_method)        
        results_gk.append([tenors[i], var0, vv, kappa, theta, rho, lambda_, IV, SSE])

        # Displaying output
        print(f'=== {tenors[i]} calibration results ===')
        print(f'v0, vv, kappa, theta, rho: {var0, vv, kappa, theta, rho}')
        print(f'[IV (10, 25, ATM, 75, 90), SSE] * 100%: {IV*100, SSE*100}')
        
        # Plotting
        plt.figure(i+1)
        plt.plot(delta_plt * 100, σ_market * 100, 'ko-', linewidth=1)
        plt.plot(delta_plt * 100, IV * 100, 'rs--', linewidth=1)
        plt.legend([f'{tenors[i]} smile', 'Heston fit'], loc='upper left')
        plt.xlabel('Delta [%]')
        plt.ylabel('Implied volatility [%]')
        plt.xticks(delta_plt * 100)
        plt.title(pricing_method)
        plt.show()

end_time = time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")


