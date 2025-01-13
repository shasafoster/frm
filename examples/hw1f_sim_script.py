# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.hw1f import HullWhite1Factor
from frm.utils import year_frac
from frm.enums import DayCountBasis, ZeroCurveInterpMethod, CompoundingFreq

# ESTR swap curve on 1 April 2024
curve_date = pd.Timestamp('2024-04-01')
df = pd.DataFrame({
    'tenor': ['ON', 'SW', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '12M', '15M', '18M', '21M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '11Y', '12Y', '15Y', '20Y', '25Y', '30Y'],
    'date': pd.to_datetime(['2-Apr-2024', '10-Apr-2024', '17-Apr-2024', '24-Apr-2024', '3-May-2024', '3-Jun-2024', '3-Jul-2024', '5-Aug-2024', '3-Sep-2024', '3-Oct-2024', '4-Nov-2024', '3-Dec-2024', '3-Jan-2025', '3-Feb-2025', '3-Mar-2025', '3-Apr-2025', '3-Jul-2025', '3-Oct-2025', '5-Jan-2026', '7-Apr-2026', '5-Apr-2027', '3-Apr-2028', '3-Apr-2029', '3-Apr-2030', '3-Apr-2031', '5-Apr-2032', '4-Apr-2033', '3-Apr-2034', '3-Apr-2035', '3-Apr-2036', '4-Apr-2039', '4-Apr-2044', '5-Apr-2049', '3-Apr-2054']),
    'discount_factor': [0.999892, 0.999026, 0.998266, 0.997514, 0.996546, 0.993222, 0.99014, 0.98688, 0.984079, 0.981287, 0.978453, 0.975944, 0.973358, 0.970875, 0.968705, 0.966373, 0.959921, 0.954107, 0.948336, 0.942805, 0.922607, 0.903406, 0.884216, 0.864765, 0.845061, 0.824882, 0.804566, 0.783991, 0.763235, 0.742533, 0.683701, 0.605786, 0.54803, 0.500307]
})
df['years'] = year_frac(curve_date, df['date'], DayCountBasis.ACT_ACT)

zero_curve = ZeroCurve(curve_date=curve_date,
               pillar_df=df[['years','discount_factor']],
               interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR)

# HW1F model parameters
short_rate_mean_rev_lvl = 0.05 # Standard values are 1%-10% annualized
short_rate_vol = 0.0196 # Standard values are 1%-10% annualized

# Chosen settings
grid_length = 50

hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=short_rate_mean_rev_lvl, vol=short_rate_vol)
hw1f.setup_theta(num=grid_length)

#%%

years_grid = np.linspace(zero_curve.pillar_df['years'].min(), zero_curve.pillar_df['years'].max(),grid_length)

hw1f_instantaneous_forward_rate  = [hw1f.get_instantaneous_forward_rate(t) for t in years_grid]
hw1f_zero_rates = [hw1f.zero_curve.get_zero_rates(years=t, compounding_freq=CompoundingFreq.CONTINUOUS) for t in years_grid]
hw1f_discount_factors = [hw1f.zero_curve.get_discount_factors(years=t) for t in years_grid]

plt.figure()
plt.plot(years_grid, hw1f_instantaneous_forward_rate, label ='HW1F model instantaneous forward rates')
plt.plot(years_grid, hw1f_zero_rates, label='HW1F model zero rates')
plt.plot(zero_curve.pillar_df['years'], zero_curve.pillar_df['cczr'], marker='.', linestyle='None', label='Source data zero rates')
plt.xlabel(r'Maturity $T$ ')
plt.title(r'Plot of interest rate term structure')
plt.grid(True)
plt.legend()
plt.ylabel('Rates (%')
plt.show()

#%%

tau = 5
nb_simulations = 10000
nb_steps = tau*252

results = hw1f.simulate(tau=tau,
                        nb_steps=nb_steps,
                        nb_simulations=nb_simulations,
                        flag_apply_antithetic_variates=True,
                        random_seed=1500)






print(hw1f.calc_discount_factor_by_solving_ODE(0, 1))
print(hw1f.get_discount_factor2(0, 1))
print(hw1f.zero_curve.get_discount_factors(years=1)[0])
