# -*- coding: utf-8 -*-
import os

from frm.enums.term_structures import TermRate

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.hull_white_1_factor_class import HullWhite1Factor
from frm.enums.utils import DayCountBasis, CompoundingFrequency
from frm.utils.business_day_calendar import get_busdaycal
from prettytable import PrettyTable
from scipy.stats import norm

fp = 'C:/Users/shasa/Documents/frm_private/tests_private/term_structures/test_optionlet_support_20240628.xlsm'
curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')

zero_curve = ZeroCurve(curve_date=curve_date,
                       data=pd.read_excel(io=fp, sheet_name='DF_3M'),
                       day_count_basis=DayCountBasis.ACT_365,
                       busdaycal=busdaycal,
                       interpolation_method='cubic_spline_on_zero_rates')


hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=0.001, vol=0.2)
hw1f.setup_theta()
#hw1f.calc_error_for_theta_fit(print_results=True)


t1 = 0.7589
t2 =  1.00822

d1 = pd.Timestamp('2025-04-01')
d2 = pd.Timestamp('2025-07-01')

#print("ZeroCurve F:",hw1f.zero_curve.get_forward_rates(period_start=d1, period_end=d2, forward_rate_type=TermRate.SIMPLE)[0])
#print("HW1F F:",hw1f.get_forward_rate(t1, t2))

K = 0.04
annuity_factor = 0.2383
cp = 1

px = 100e6 * hw1f.price_optionlet(t1, t2, K, cp, annuity_factor)
print('optionlet px: {:,.0f}'.format(px))

#%%%



σ = hw1f.vol
α = hw1f.mean_rev_lvl
term = t2 - t1  # Day count fraction; adjust based on your convention if necessary

# Discount factors
DF_t1 = hw1f.get_discount_factor(0, t1)
DF_t2 = hw1f.get_discount_factor(0, t2)

# Calculate bond price volatility between the optionlet start and end date
σP = σ * np.sqrt((1 - np.exp(-2 * α * t1)) / (2 * α)) * hw1f.calc_B(t1, t2)

# Calculate d1 and d2
# TODO this bit with F_i doesn't match exactly to [1]. Check.
F_i = hw1f.get_forward_rate(t1, t2)
h = (1 / σP) * np.log(F_i / K) + 0.5 * σP

h2 = (1 / σP) * np.log(DF_t2 * (1 + K * term) / DF_t1) + 0.5 * σP

d1 = h2
d2 = d1 - σP

# Caplet price using the Black-like formula
# Given similarity to Black76, calibration to caps could calibrate α and σ by matching σP and h.

annuity_factor = term * DF_t2
caplet_price = annuity_factor * (F_i * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))

print('Caplet price: ', 100e6 * caplet_price)











#%%

ti = t2
ti_1 = t1
sigma = hw1f.vol
a = hw1f.mean_rev_lvl
t = 0
N = 100e6
X = 0.04


P_t_ti = 0.966571588
P_t_ti_1 = 0.955911215



tau_i = ti - ti_1

# Compute parameters for Hull-White model
B_t_ti = (1 - np.exp(-a * (ti - ti_1))) / a
sigma_p_i = sigma * np.sqrt((1 - np.exp(-2 * a * (ti - t))) / (2 * a)) * B_t_ti

#%%

# Calculate hi
hi = (1 / sigma_p_i) * np.log(P_t_ti / (P_t_ti_1 * (1 + X * tau_i))) + (sigma_p_i / 2)

# Caplet price using the formula (3.42)
caplet_price = N * (P_t_ti_1 * norm.cdf(-hi + sigma_p_i) - (1 + X * tau_i) * P_t_ti * norm.cdf(-hi))

# Floorlet price using analogous formula (3.43)
floorlet_price = N * ((1 + X * tau_i) * P_t_ti * norm.cdf(hi) - P_t_ti_1 * norm.cdf(hi - sigma_p_i))
