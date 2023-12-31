# -*- coding: utf-8 -*-
"""
@author: Shasa Foster
https://www.linkedin.com/in/shasafoster
"""

if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
from frm.frm.market_data.ir_zero_curve import ZeroCurve
from frm.frm.pricing_engine.garman_kohlhagen import gk_price, gk_solve_implied_σ, gk_solve_strike
from frm.frm.pricing_engine.heston_garman_kohlhagen import heston_fit_vanilla_fx_smile, heston1993_price_fx_vanilla_european, heston_carr_madan_fx_vanilla_european
from frm.frm.market_data.fx_volatility_surface import FXVolatilitySurface

from frm.frm.schedule.tenor import calc_tenor_date
from frm.frm.schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS
from frm.frm.schedule.business_day_calendar import get_calendar        
from frm.frm.utilities.utilities import convert_column_type    

import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import fsolve, root_scalar
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline 
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Literal
import warnings
import re
import matplotlib.pyplot as plt

curve_date = pd.Timestamp(2023,6,30) 
curve_ccy = 'audusd'

# All data is from IDD, 30 June 2023, London 8am from FX module
tenor_name = ['spot', '1d', '1w', '2w', '3w', '1m', '2m', '3m', '6m', '9m', '1y', '15m', '18m', '2y', '3y', '4y', '5y', '7y', '10y',]
fxf = [0.6629, 0.6629176, 0.66302406, 0.66314942, 0.66327973, 0.66352705, 0.66403999, 0.66456628, 0.66612154, 0.66730573, 0.66796243, 0.66809219, 0.66789518, 0.66675039, 0.66306281, 0.65906951, 0.6524625, 0.6358184, 0.6083719,]
fx_forward_curve = pd.DataFrame(np.transpose([tenor_name,fxf]), columns=['tenor_name','fx_forward_rate'])
fx_forward_curve['fx_forward_rate'] = fx_forward_curve['fx_forward_rate'].astype(float)
del tenor_name, fxf

tenor_name = ['1d', '1w', '2w', '3w', '1m', '2m', '3m', '6m', '9m', '1y', '15m', '18m', '2y', '3y', '4y', '5y', '7y', '10y',]
r_d = [0.05055, 0.05059, 0.05063, 0.05065, 0.05142, 0.05221, 0.0527, 0.0539, 0.05432, 0.05381, 0.05248, 0.05122, 0.04812, 0.04373, 0.04087, 0.039, 0.0369, 0.0355,]
r_f = [0.04156, 0.04153, 0.0415, 0.04138, 0.04194, 0.04274, 0.04335, 0.04479, 0.04595, 0.0466, 0.04674, 0.04673, 0.04578, 0.04427, 0.04295, 0.04285, 0.04362, 0.04493,]
foreign_zero_data = pd.DataFrame(np.transpose([tenor_name,r_f]),columns=['tenor_name','zero_rate'])
foreign_zero_data['zero_rate'] = foreign_zero_data['zero_rate'].astype(float)
domestic_zero_data = pd.DataFrame(np.transpose([tenor_name,r_d]),columns=['tenor_name','zero_rate'])
domestic_zero_data['zero_rate'] = domestic_zero_data['zero_rate'].astype(float)

zc_f = ZeroCurve(curve_date=pd.Timestamp(2023,6,30), curve_ccy=curve_ccy[:3],zero_data=foreign_zero_data, extrapolation_method='flat')
zc_d = ZeroCurve(curve_date=pd.Timestamp(2023,6,30), curve_ccy=curve_ccy[3:],zero_data=domestic_zero_data, extrapolation_method='flat')
del tenor_name, r_d, r_f, foreign_zero_data, domestic_zero_data

Δ_convention = ['regular_spot_Δ','regular_spot_Δ','regular_spot_Δ','regular_spot_Δ','regular_spot_Δ','regular_spot_Δ','regular_spot_Δ','regular_forward_Δ','regular_forward_Δ','regular_forward_Δ','regular_forward_Δ','regular_forward_Δ','regular_forward_Δ']

σ_x = np.array(['1W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y'])
σ_y = ['σ_5Δput', 'σ_10Δput', 'σ_15Δput', 'σ_20Δput', 'σ_25Δput', 'σ_30Δput', 'σ_atmΔneutral', 'σ_30Δcall', 'σ_25Δcall', 'σ_20Δcall', 'σ_15Δcall', 'σ_10Δcall', 'σ_5Δcall']
σ = np.array([
    [0.11943, 0.11656, 0.11481, 0.11350, 0.11240, 0.11140, 0.10868, 0.10722, 0.10704, 0.10683, 0.10663, 0.10643, 0.10628],
    [0.11145, 0.10786, 0.10568, 0.10405, 0.10271, 0.10152, 0.09814, 0.09598, 0.09559, 0.09516, 0.09471, 0.09421, 0.09365],
    [0.11514, 0.10990, 0.10683, 0.10457, 0.10275, 0.10116, 0.09684, 0.09441, 0.09407, 0.09368, 0.09331, 0.09296, 0.09274],
    [0.11834, 0.11200, 0.10832, 0.10564, 0.10350, 0.10165, 0.09670, 0.09400, 0.09364, 0.09323, 0.09287, 0.09256, 0.09249],
    [0.12402, 0.11599, 0.11141, 0.10812, 0.10550, 0.10326, 0.09745, 0.09440, 0.09404, 0.09365, 0.09335, 0.09318, 0.09349],
    [0.12996, 0.12006, 0.11455, 0.11065, 0.10758, 0.10496, 0.09848, 0.09535, 0.09508, 0.09481, 0.09470, 0.09486, 0.09587],
    [0.13546, 0.12361, 0.11718, 0.11270, 0.10920, 0.10624, 0.09922, 0.09610, 0.09596, 0.09585, 0.09599, 0.09657, 0.09847],
    [0.14159, 0.12824, 0.12123, 0.11645, 0.11273, 0.10961, 0.10150, 0.09831, 0.09833, 0.09846, 0.09893, 0.10004, 0.10306],
    [0.14683, 0.13215, 0.12452, 0.11932, 0.11530, 0.11190, 0.10300, 0.09934, 0.09930, 0.09943, 0.09998, 0.10126, 0.10474],
    [0.15161, 0.13618, 0.12808, 0.12254, 0.11823, 0.11457, 0.10488, 0.10076, 0.10065, 0.10071, 0.10117, 0.10236, 0.10568],
    [0.15477, 0.13875, 0.13032, 0.12455, 0.12005, 0.11620, 0.10600, 0.10166, 0.10155, 0.10160, 0.10206, 0.10325, 0.10660],
    [0.16703, 0.14603, 0.13573, 0.12888, 0.12360, 0.11909, 0.10750, 0.10369, 0.10407, 0.10478, 0.10615, 0.10877, 0.11528],
    [0.17042, 0.14966, 0.13948, 0.13267, 0.12739, 0.12283, 0.11100, 0.10683, 0.10711, 0.10774, 0.10904, 0.11157, 0.11787],
])
σ_pillar = pd.DataFrame(σ,columns=σ_y)
σ_pillar.insert(loc=0, column='tenor_name', value=σ_x)
σ_pillar.insert(loc=1, column='Δ_convention', value=Δ_convention)

K = [
    [0.6453, 0.6495, 0.6523, 0.6544, 0.6562, 0.6578, 0.6631, 0.6683, 0.6698, 0.6714, 0.6733, 0.6757, 0.6793],
    [0.6278, 0.6365, 0.6421, 0.6464, 0.6500, 0.6532, 0.6638, 0.6740, 0.6769, 0.6802, 0.6839, 0.6887, 0.6957],
    [0.6146, 0.6271, 0.6349, 0.6409, 0.6459, 0.6503, 0.6646, 0.6782, 0.6821, 0.6865, 0.6916, 0.6981, 0.7079],
    [0.6030, 0.6189, 0.6288, 0.6363, 0.6425, 0.6479, 0.6654, 0.6820, 0.6868, 0.6922, 0.6986, 0.7066, 0.7188],
    [0.5785, 0.6016, 0.6159, 0.6268, 0.6357, 0.6435, 0.6677, 0.6908, 0.6978, 0.7056, 0.7148, 0.7266, 0.7447],
    [0.5584, 0.5879, 0.6060, 0.6196, 0.6307, 0.6404, 0.6698, 0.6978, 0.7066, 0.7164, 0.7281, 0.7433, 0.7674],
    [0.5405, 0.5759, 0.5974, 0.6134, 0.6265, 0.6379, 0.6713, 0.7032, 0.7137, 0.7255, 0.7396, 0.7581, 0.7882],
    [0.4890, 0.5369, 0.5663, 0.5882, 0.6063, 0.6220, 0.6737, 0.7243, 0.7397, 0.7572, 0.7787, 0.8077, 0.8570],
    [0.4506, 0.5075, 0.5426, 0.5691, 0.5911, 0.6103, 0.6737, 0.7366, 0.7559, 0.7781, 0.8056, 0.8433, 0.9088],
    [0.4190, 0.4823, 0.5222, 0.5525, 0.5778, 0.6000, 0.6738, 0.7477, 0.7705, 0.7970, 0.8299, 0.8751, 0.9545],
    [0.3919, 0.4598, 0.5032, 0.5364, 0.5643, 0.5888, 0.6711, 0.7545, 0.7805, 0.8108, 0.8487, 0.9012, 0.9941],
    [0.3387, 0.4174, 0.4673, 0.5057, 0.5379, 0.5664, 0.6621, 0.7625, 0.7953, 0.8345, 0.8852, 0.9586, 1.1006],
    [0.289813, 0.3709, 0.4245, 0.4667, 0.5028, 0.5351, 0.6471, 0.7691, 0.8098, 0.859, 0.9232, 1.0178, 1.2043],
]
K_x = np.array(['1W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y'])
K_y = ['k_5Δput', 'k_10Δput', 'k_15Δput', 'k_20Δput', 'k_25Δput', 'k_30Δput', 'k_atmΔneutral', 'k_30Δcall', 'k_25Δcall', 'k_20Δcall', 'k_15Δcall', 'k_10Δcall', 'k_5Δcall']
Δ_K_pillar = pd.DataFrame(K,columns=K_y)
Δ_K_pillar.insert(loc=0, column='tenor_name', value=K_x)
Δ_K_pillar.insert(loc=1, column='Δ_convention', value=Δ_convention)
del σ_x, σ_y, σ, K_x, K_y, K


        
        

curve_date = pd.Timestamp(2023,6,30)
curve_ccy = 'audusd'

smile_interpolation_method = 'heston_carr_madan_gauss_kronrod_quadrature'
#smile_interpolation_method = 'cubic_spline'


vs = FXVolatilitySurface(
    curve_date=curve_date,
    curve_ccy=curve_ccy,
    fx_forward_curve=fx_forward_curve,
    foreign_zero_curve=zc_f,
    domestic_zero_curve=zc_d,
    σ_pillar=σ_pillar,
    smile_interpolation_method=smile_interpolation_method)

if 'heston' in smile_interpolation_method:
    vs.fit_and_plot_smile()
    pass

del fx_forward_curve, zc_f, zc_d, σ_pillar, Δ_K_pillar, Δ_convention

#%%

results = vs.simulate_fx_rate_path()


#%%
from frm.frm.pricing_engine.monte_carlo_generic import generate_rand_nbs
from frm.frm.pricing_engine.geometric_brownian_motion import simulate_gbm_path

date_grid = vs.σ_pillar.index
tau = vs.daycounter.year_fraction(vs.curve_date,date_grid).values
tau = np.insert(tau, 0, 0)
dt = tau[1:] - tau[:-1] 

fx_forward_rates_t2 = vs.interp_fx_forward_curve(date_grid).values
fx_forward_rates_t1 = np.zeros(shape=fx_forward_rates_t2.shape)
fx_forward_rates_t1[0] = vs.fx_spot_rate
fx_forward_rates_t1[1:] = fx_forward_rates_t2[:-1].copy()
period_drift = np.log(fx_forward_rates_t2 / fx_forward_rates_t1) / dt

mask = np.logical_and.reduce([date_grid >= vs.σ_pillar.index.min(), date_grid <= vs.σ_pillar.index.max()])
σ_atm = np.full(date_grid.shape, np.nan)
if mask.any():
    σ_atm[mask] = vs.Δ_σ_daily['σ_atmΔneutral'][date_grid[mask]]

t1 = tau[:-1]
t2 = tau[1:]
σ_atm_t1 = np.insert(σ_atm[:-1], 0, σ_atm[0])
σ_atm_t2 = σ_atm
σ_forward = vs.forward_volatility(t1, σ_atm_t1, t2, σ_atm_t2)
σ_forward[0] = σ_atm[0]

df_gbm_monte_carlo_inputs = pd.DataFrame()
df_gbm_monte_carlo_inputs['dates'] = np.insert(date_grid.values, 0, vs.curve_date)
df_gbm_monte_carlo_inputs['tau'] = tau
df_gbm_monte_carlo_inputs['dt'] = np.insert(dt, 0, 0)
df_gbm_monte_carlo_inputs['fx_forward_rates'] = np.insert(fx_forward_rates_t2, 0, vs.fx_spot_rate)
df_gbm_monte_carlo_inputs['drift'] = np.insert(period_drift, 0, np.nan)
df_gbm_monte_carlo_inputs['atm_volatility'] = np.insert(σ_atm, 0, σ_atm[0])
df_gbm_monte_carlo_inputs['forward_atm_volatility'] = np.insert(σ_forward, 0, np.nan)


#%%

method = 'geometric_brownian_motion'

if method == 'geometric_brownian_motion':
    nb_of_periods = len(date_grid)
    rand_nbs = generate_rand_nbs(nb_of_periods=nb_of_periods,
                                 nb_of_simulations=None,
                                 )

    
    initial_px = vs.fx_spot_rate
    period_drift = period_drift
    period_forward_volatility = σ_forward
    period_length = dt        


    results = simulate_gbm_path(initial_px=initial_px,
                      period_drift=period_drift,
                      period_forward_volatility=period_forward_volatility,
                      period_length=period_length,
                      rand_nbs=rand_nbs)
    
    results_avg = results.mean(axis=0)

#%%

# cp = -1
# K = 0.5

# expiry_datetimeindex = pd.DatetimeIndex([pd.Timestamp(2028,7,3)])

# σ = vs.interp_σ_surface(expiry_datetimeindex, K=K ,cp=cp)
# X = vs.price_fx_vanilla_european(expiry_datetimeindex=expiry_datetimeindex, K=K ,cp=cp)

# print(σ, X * 1e6)

# # Displaying output
# #print(f'=== {tenors[i]} calibration results ===')
# print(f'v0, vv, kappa, theta, rho: {v0, vv, kappa, theta, rho}')
# print(f'[IV (10, 25, ATM, 75, 90), SSE] * 100%: {IV*100, SSE*100}')

# # Plotting
# i = 1
# plt.figure(i+1)
# plt.plot(delta_plt * 100, σ_market * 100, 'ko-', linewidth=1)
# plt.plot(delta_plt * 100, IV * 100, 'rs--', linewidth=1)
# plt.legend([f'{tenors[i]} smile', 'Heston fit'], loc='upper left')
# plt.xlabel('Delta [%]')
# plt.ylabel('Implied volatility [%]')
# plt.xticks(delta_plt * 100)
# plt.title(pricing_method)
# plt.show()


#%%

#strike = np.array([0.55,0.60])
#interp = vs.interp_σ_surface(expiry_datetimeindex, strike)


#%%
# import time

# t1 = time.time()

# dates = pd.date_range(min(vol.Δ_σ_pillar.index),max(vol.Δ_σ_pillar.index),freq='d')[:100]
# strikes = np.array([0.6 for _ in range(len(dates))])


# interp = vol_surface.interp_σ_surface(dates, strikes)
# values = pd.DataFrame([v.item() for v in list(interp.values())])
