# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    

import numpy as np
from numba import jit, prange
import scipy
from scipy.stats import norm
import timeit
import time
import pandas as pd
from frm.frm.market_data.ir_zero_curve import ZeroCurve
from frm.frm.pricing_engine.hull_white_1_factor import calc_theta
    
df = pd.read_excel('./frm/frm/pricing_engine/hull_white/zero_data.xlsx')

# running this - working through getting really clean functions
zc = ZeroCurve(curve_date=pd.Timestamp(2023,6,30),
               zero_data=df[['years','discount_factor']],
               interpolation_method='cubic_spline_on_zero_rates')

years = df['years']
years_linspace = np.linspace(min(years),max(years),100)

dt = 1e-5



f_plus_dt = zc.instantaneous_forward_rate(years=years_linspace+dt)
f_minus_dt = zc.instantaneous_forward_rate(years=years_linspace-dt)
df_dt = (f_plus_dt - f_minus_dt) / (2 * dt)
 
f = zc.instantaneous_forward_rate(years=years_linspace)

α = 0.19    # speed of convergence - Not required in Ho Lee Model
σ = 0.0196  # vol    

θ = df_dt + α * f + (σ**2)/(α*2)*(1-np.exp(-2*α*years_linspace))

theta = calc_theta(zc, α,σ,100)

# dt = 1e-5

# def forward_rate(tck, years):
    
#     scipy.interpolate.splev(x=years_linspace, tck=tck, der=1) # der=1 gets the 1st derivative



# f_t = 


# first_deriv = scipy.interpolate.splev(x=years_linspace, tck=tck, der=1) # der=1 gets the 1st derivative
# second_deriv = scipy.interpolate.splev(x=years_linspace, tck=tck, der=2) # der=1 gets the 1st derivative        
