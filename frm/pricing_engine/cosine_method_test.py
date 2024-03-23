# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import os
    import pathlib
    import sys
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from frm.pricing_engine.cosine_method import cos_method


#%% Test cos_method() with Standard Normal Distribution
# PDF and CDF of a standard normal distribution
mu = 0
sig = 1
cf = lambda s: np.exp(1j * s * mu - 0.5 * sig**2 * s**2)

z_score_90 = 1.644854

Fp, F, f, I, pts = cos_method(p=z_score_90, cf=cf, dt=0.0001, a=-10, b=10, N=100)

print(f'CDF(p): {Fp:.4f}')
assert round(Fp,4) == 0.95
print(f'Integral: {I:.4f}')
assert round(I,4) == 1.0

plot_xp = np.arange(-10, 10.1, 0.1)
plt.subplot(1, 2, 1)
plt.plot(pts, f, 'b', linewidth=1.2)
plt.grid(True)
plt.plot(plot_xp, 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((plot_xp - mu) / sig)**2), '--r')
plt.legend(['COS method', 'Normal pdf'])

plt.subplot(1, 2, 2)
plt.plot(pts, F, 'b')
plt.grid(True)
plt.plot(plot_xp, 0.5 * (1 + erf((plot_xp - mu) / (sig * np.sqrt(2)))), '--r')
plt.legend(['COS method', 'Normal cdf'])

# Test cos_method() with a Normal(8, 2) distribution
mu = 8
sig = 2
cf = lambda s: np.exp(1j * s * mu - 0.5 * sig**2 * s**2)

Fp, _, _, _, _ = cos_method(p=8, cf=cf, dt=0.0001, a=0, b=16, N=100)
print(f'CDF(p): {Fp:.4f}')
assert round(Fp,4) == 0.5

# Compute expected value using MGF
a = 5
b = 8
mgf = lambda t: np.exp(3 * t) * (1 - 5 * b * t)**-a
cf = lambda t: mgf(1j * t)

_, _, f, _, pts = cos_method(p=0, cf=cf, dt=0.05, a=0, b=1200, N=200)
m = np.sum(f * pts * 0.05)

print(f'Mean: {m:.2f}')
assert round(m,4) == 203.0

#%%

