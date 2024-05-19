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
    
df = pd.read_excel('./frm/frm/pricing_engine/hull_white/zero_data.xlsx')

#%%

zc = ZeroCurve(curve_date=pd.Timestamp(2023,6,30),
               zero_data=df[['tenor_date','discount_factor']],
               interpolation_method='cubic_spline_on_zero_rates')


#%%

tck = zc.cubic_spline_definition

# Define the term structure for θ
years = df['years']
years_linspace = np.linspace(min(years),max(years),100)
interp = scipy.interpolate.splev(x=years_linspace, tck=tck, der=0) # der=1 gets the 1st derivative


#%%

# dt = 1e-5

# def forward_rate(tck, years):
    
#     scipy.interpolate.splev(x=years_linspace, tck=tck, der=1) # der=1 gets the 1st derivative



# f_t = 


# first_deriv = scipy.interpolate.splev(x=years_linspace, tck=tck, der=1) # der=1 gets the 1st derivative
# second_deriv = scipy.interpolate.splev(x=years_linspace, tck=tck, der=2) # der=1 gets the 1st derivative        
