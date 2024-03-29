# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os
    import pathlib
    import sys
    
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.pricing_engine.heston_garman_kohlhagen import heston1993_price_fx_vanilla_european, heston_fit_vanilla_fx_smile, heston_cosine_price_fx_vanilla_european, heston_carr_madan_price_fx_vanilla_european
from frm.frm.pricing_engine.heston_test import heston_cosine_method_SLV

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
import pandas as pd
import time


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


i = 4
rf = r_f[i] # 0.01131
rd = r_d[i] # 0.0070075
K = np.array([1.03413767, 1.15064198, 1.28873193, 1.466967  , 1.6959051 ]) 
cp = [-1, -1, 1, 1, 1]

# Inputs
vol0 = 0.1217400016428454
vv = 0.293475456
kappa = 1.5
theta = 0.017132031
rho = 0.232453505225961

S0 = 1.2779
tau = 2
dp = 6

for i in range(5):
    heston_1993 = heston1993_price_fx_vanilla_european(S0, tau, rf, rd, cp[i], K[i], vol0**2, vv, kappa, theta, rho, 0)
    cm_quad = heston_carr_madan_price_fx_vanilla_european(S0, tau, rf, rd, cp[i], K[i], vol0**2, vv, kappa, theta, rho, integration_method=0)
    cm_fft = heston_carr_madan_price_fx_vanilla_european(S0, tau, rf, rd, cp[i], K[i], vol0**2, vv, kappa, theta, rho, integration_method=1)
    COS = heston_cosine_price_fx_vanilla_european(S0, tau, rf, rd, cp[i], K[i], vol0**2, vv, kappa, theta, rho, N=160, L=10)
    
    print('Original, 1993:', round(heston_1993,dp))
    print('Carr Maddan Gauss Quad:', round(cm_quad,dp))
    print('Carr Maddan FTT:', round(cm_fft,dp))
    print('COS:', round(COS,dp))
    print('')







