# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
import numpy as np
import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.hw1f import HullWhite1Factor
from frm.enums import DayCountBasis, ZeroCurveInterpMethod, TermRate
from frm.utils import get_busdaycal, year_frac
from frm.pricing_engine.black76_bachelier import black76_price
import scipy


# TODO link in with TF quant finance.

fp = 'C:/Users/shasa/Documents/frm_private/tests_private/test_optionlet_support_20240628.xlsm'
curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')

zero_curve = ZeroCurve(curve_date=curve_date,
                       pillar_df=pd.read_excel(io=fp, sheet_name='DF_3M'),
                       day_count_basis=DayCountBasis.ACT_365,
                       cal=busdaycal,
                       interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR)
hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=0.05, vol=0.05)
hw1f.setup_theta(num=50)
hw1f.calc_error_for_theta_fit(print_results=True)

#%%

day_count_basis = DayCountBasis.ACT_365
K = 0.0447385
cp = 1
d1 = pd.Timestamp('2025-04-01')
d2 = pd.Timestamp('2025-07-01')
t1 = year_frac(curve_date, d1, day_count_basis)
t2 = year_frac(curve_date, d2, day_count_basis)

F = zero_curve.get_forward_rates(d1, d2, TermRate.SIMPLE)
discount_factor = zero_curve.get_discount_factors(d2)

tau = year_frac(curve_date, d1, day_count_basis)
annuity_factor = year_frac(d1, d2, day_count_basis) * discount_factor

vol_sln = 0.1326
#vol_sln = 0
black76_px = black76_price(F=F, tau=tau, cp=cp, K=K, vol_sln=vol_sln, ln_shift=+0.02, annuity_factor=annuity_factor)['price'][0]
print(f'Black76: {100e6*black76_px:,.2f}')

#%%

# Enforce positive mean reversion level - if negative, the HW1F model is not the right model.
MEAN_REV_LVL_BOUNDS = [(0.001 / 100, 50 / 100)]  # 0.001% to 50%.
VOL_BOUNDS = [(0.001 / 100, 100 / 100)]  # 0.001% to 100%



#%% Solve both params

def obj_func_mean_rev_lvl_and_vol(params):
    hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=params[0], vol=params[1])
    hw1f.setup_theta()
    hw1f_px = hw1f.price_optionlet(effective_years=t1, termination_years=t2, K=K, cp=cp)
    # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
    # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
    return ((black76_px - hw1f_px)/black76_px)**2

options = {'xtol': (1e-5)**2,   # tighter parameter tolerance, 0.01%^2
           'gtol': 0,           # ignore gradient tolerance
           'maxiter': 2000}     # more iterations for thorough exploration
res = scipy.optimize.minimize(
    fun=obj_func_mean_rev_lvl_and_vol,
    x0=[0.01, 0.05],
    bounds=MEAN_REV_LVL_BOUNDS + VOL_BOUNDS,
    method='trust-constr',
    options=options)

if res.success:
    mean_rev_lvl = res.x[0]
    vol = res.x[1]
    print(f'Mean Rev Lvl: {mean_rev_lvl:.4%}, Vol: {vol:.4%}')
else:
    print(res)
    raise ValueError('Optimization failed')

hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=mean_rev_lvl, vol=vol)
hw1f.setup_theta()
hw1f_px = hw1f.price_optionlet(effective_years=t1, termination_years=t2, K=K, cp=cp)
print(f'HW1F: {100e6*hw1f_px:,.2f}')
print(f'Black76: {100e6*black76_px:,.2f}')

#%% Solve vol only

mean_rev_lvl = 0.000001

def obj_func_vol(vol):
    hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=mean_rev_lvl, vol=vol.item())
    hw1f.setup_theta()
    hw1f_px = hw1f.price_optionlet(effective_years=t1, termination_years=t2, K=K, cp=cp)
    # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
    # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
    return ((black76_px - hw1f_px)/black76_px)**2

options = {'xtol': (1e-5)**2,   # tighter parameter tolerance, 0.01%^2
           'gtol': 0,           # ignore gradient tolerance
           'maxiter': 500}      # more iterations for thorough exploration
res = scipy.optimize.minimize(
    fun=obj_func_vol,
    x0=np.atleast_1d(0.05),
    bounds=VOL_BOUNDS,
    method='trust-constr',
    options=options)

if res.success:
    vol = res.x[0]
    print(f'Mean Rev Lvl: {mean_rev_lvl:.4%}, Vol: {vol:.4%}')
else:
    print(res)
    raise ValueError('Optimization failed')

hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=mean_rev_lvl, vol=vol)
hw1f.setup_theta()
hw1f_px = hw1f.price_optionlet(effective_years=t1, termination_years=t2, K=K, cp=cp)
print(f'HW1F: {100e6*hw1f_px:,.2f}')
print(f'Black76: {100e6*black76_px:,.2f}')


#%% Solve mean_rev_lvl only

vol = (85.71/10000) * t1

def obj_func_mean_rev_lvl(mean_rev_lvl):
    hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=mean_rev_lvl.item(), vol=vol)
    hw1f.setup_theta()
    hw1f_px = hw1f.price_optionlet(effective_years=t1, termination_years=t2, K=K, cp=cp)
    # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
    # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
    return ((black76_px - hw1f_px)/black76_px)**2

options = {'xtol': (1e-5)**2,   # tighter parameter tolerance, 0.01%^2
           'gtol': 0,           # ignore gradient tolerance
           'maxiter': 500}      # more iterations for thorough exploration
res = scipy.optimize.minimize(
    fun=obj_func_mean_rev_lvl,
    x0=np.atleast_1d(0.01),
    bounds=MEAN_REV_LVL_BOUNDS,
    method='trust-constr',
    options=options)

if res.success:
    mean_rev_lvl = res.x[0]
    print(f'Mean Rev Lvl: {mean_rev_lvl:.4%}, Vol: {vol:.4%}')
else:
    print(res)
    raise ValueError('Optimization failed')

hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=mean_rev_lvl, vol=vol)
hw1f.setup_theta()
hw1f_px = hw1f.price_optionlet(effective_years=t1, termination_years=t2, K=K, cp=cp)
print(f'HW1F: {100e6*hw1f_px:,.2f}')
print(f'Black76: {100e6*black76_px:,.2f}')


# For short tenors, the volatility dominates, and for long tenors, the mean reversion level dominates.

# Under the risk neutral measure, if jointly calibrating to caps/floors and swaptions, the principle is to calibrate:
# (i) the short-rate volatility to expiry's
# (ii) the mean reversion levels to maturities.

# Maturity = expiry + tenor for swaptions and maturity for caps/floors.
# I.e. for a 1Y2Y swaption and a 1M3Y cap (both have an approximate 3Y maturity).


