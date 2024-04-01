# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os
    import pathlib
    import sys
    
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.pricing_engine.heston_garman_kohlhagen import heston1993_price_fx_vanilla_european, heston_cosine_price_fx_vanilla_european, heston_carr_madan_price_fx_vanilla_european
import numpy as np
import matplotlib.pyplot as plt
from time import time

#% Sample input:
S0 = 1.2;
tau = 0.5;
r_d = 0.022;
r_f = 0.018;
kappa = 1.5;
theta = 0.015;
vv = 0.2
rho = 0.05;
var0 = 0.01;
K = np.linspace(1.1,1.3,81)
# Tolerances
dp = 7
tol = 1 / (10**dp)

cp = np.ones(shape=K.shape) 


# The COS method can be applied to multiple strikes
COS_vectorised = heston_cosine_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K, var0, vv, kappa, theta, rho, N=160, L=10)

heston_1993 = []
cm_quad = []
cm_fft = []
COS = []

for i in range(len(K)):
        heston_1993.append(heston1993_price_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], var0, vv, kappa, theta, rho, 0))
        cm_quad.append(heston_carr_madan_price_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], var0, vv, kappa, theta, rho, integration_method=0))
        cm_fft.append(heston_carr_madan_price_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], var0, vv, kappa, theta, rho, integration_method=1))
        COS.append(heston_cosine_price_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], var0, vv, kappa, theta, rho, N=160, L=10))
         
        print(i,'-----------------------')
        print('Original, 1993:', round(heston_1993[i],dp))
        print('Carr Maddan Gauss Quad:', round(cm_quad[i],dp))
        print('Carr Maddan FTT:', round(cm_fft[i],dp))
        print('COS:', round(COS[i].item(),dp))
        print('COS Vectorised:', round(COS_vectorised[i],dp))
        print('')
    
        assert np.abs(COS[i].item() - heston_1993[i]) < tol
        assert np.abs(COS[i].item() - cm_quad[i]) < tol
        assert np.abs(COS[i].item() - cm_fft[i]) < (tol * 10000) # The FFT method is less accurate
        assert np.abs(COS[i].item() - COS_vectorised[i]) < tol

# Plotting - these plot's match STF2hes03.m
plt.figure(1)
plt.plot(K, COS, 'r--', linewidth=1)
plt.plot(K, heston_1993, 'kx', linewidth=1)
plt.xlabel('Strike, K')
plt.ylabel('Option price')
plt.show()


#%% Speed Test

t1 = time()
for i in range(1000):
    COS_vectorised = heston_cosine_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K, var0, vv, kappa, theta, rho, N=160, L=10)
t2 = time()
print("COS: ", t2-t1)


t1 = time()
for j in range(1000):
    for i in range(len(K)):
        cm_quad.append(heston_carr_madan_price_fx_vanilla_european(S0, tau, r_f, r_d, cp[i], K[i], var0, vv, kappa, theta, rho, integration_method=0))
t2 = time()
print("CM: ", t2-t1)