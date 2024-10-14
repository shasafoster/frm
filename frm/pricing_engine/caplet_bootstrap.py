# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
import re
from frm.pricing_engine.sabr import solve_alpha, calc_ln_vol_for_strike, fit_sabr_params_to_smile
from frm.utils.business_day_calendar import get_busdaycal
from frm.utils.daycount import year_fraction
from frm.enums.utils import DayCountBasis, PeriodFrequency
from frm.utils.tenor import clean_tenor, tenor_to_date_offset
from frm.utils.utilities import convert_column_to_consistent_data_type
from frm.utils.schedule import get_schedule, get_payment_dates
from frm.term_structures.zero_curve import ZeroCurve




fp = './tests_private/term_structures/ir_vol_surface.xlsm'

cap_floor_quotes = pd.read_excel(io=fp, sheet_name='CapFloor')
discount_factors = pd.read_excel(io=fp, sheet_name='DFs')

cap_floor_quotes.columns = cap_floor_quotes.columns.str.lower().str.strip()

swap_index_tenor = '3m'
curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')
day_count_basis = DayCountBasis.ACT_365


zero_curve = ZeroCurve(curve_date=curve_date,
                       data=discount_factors,
                       day_count_basis=day_count_basis,
                       busdaycal=busdaycal,
                       interpolation_method='linear_on_log_of_discount_factors')


# This code block is my "ATM Quote" helper that feeds into a bootstrapper function.

settlement_date = np.busday_offset(curve_date.to_numpy().astype('datetime64[D]'), offsets=1, roll='following', busdaycal=busdaycal)
effective_date_np = (settlement_date + tenor_to_date_offset(swap_index_tenor)).to_numpy().astype('datetime64[D]')
effective_date = np.busday_offset(effective_date_np, offsets=0, roll='following', busdaycal=busdaycal)

cap_floor_quotes['tenor'] = cap_floor_quotes['tenor'].apply(clean_tenor)
cap_floor_quotes['settlement_date'] = settlement_date
cap_floor_quotes['effective_date'] = effective_date

for i, row in cap_floor_quotes.iterrows():
    date_offset = tenor_to_date_offset(row['tenor'])
    maturity_date_np = (curve_date + date_offset).to_numpy().astype('datetime64[D]')
    cap_floor_quotes.at[i, 'termination_date'] = np.busday_offset(maturity_date_np, offsets=0, roll='following', busdaycal=busdaycal)

cap_floor_quotes['expiry_years'] = year_fraction(curve_date, cap_floor_quotes['termination_date'], day_count_basis)
cap_floor_quotes['term_years'] = year_fraction(cap_floor_quotes['effective_date'], cap_floor_quotes['termination_date'], day_count_basis)

cap_floor_quotes = convert_column_to_consistent_data_type(cap_floor_quotes)

# Two methods of specifying quotes
# 1. Quotes relative to the atm forward rate (e.g. ATM, ATM+/-50bps, ATM+/-100bps...)
# 2. Absolute quotes (e.g. 2.5%, 3.0%, 3.5%...)

# Code block for method 1
quote_str_map = dict()
for col_name in cap_floor_quotes.columns:

    # Convert column name to common data format
    bps_quote = r'[+-]?\s?\d+\s?(bps|bp)'
    percentage_quote = r'[+-]?\s?\d+(\.\d+)?\s?%'
    atm_quote = '(a|at)[ -]?(t|the)[ -]?(m|money)[ -]?(f|forward)?'

    if re.search(bps_quote, col_name):
        v = round(float(col_name.replace('bps', '').replace('bp', '').replace(' ', '')) / 10000,8)
        new_col_name = (str(int(v * 10000)) if (round(v * 10000,8)).is_integer() else str(round(v * 10000, 8))) + 'bps'
        cap_floor_quotes = cap_floor_quotes.rename(columns={col_name: new_col_name})
        quote_str_map[new_col_name] = v

    elif re.search(percentage_quote, col_name):
        v = round(float(col_name.replace('%', '').replace(' ', '')) / 100,8)
        new_col_name = (str(int(v * 100)) if (round(v * 100,8)).is_integer() else str(round(v * 100, 8))) + 'bps'
        cap_floor_quotes = cap_floor_quotes.rename(columns={col_name: new_col_name})
        quote_str_map[new_col_name] = v
    elif re.search(atm_quote, col_name):
        new_col_name = 'atm'
        cap_floor_quotes = cap_floor_quotes.rename(columns={col_name: new_col_name})
        quote_str_map[new_col_name] = 0

normal_vols = cap_floor_quotes.loc[cap_floor_quotes['field']=='normal_vol', :]
forward_rates = cap_floor_quotes.loc[cap_floor_quotes['field']=='forward_rate', :]
lognormal_vols = cap_floor_quotes.loc[cap_floor_quotes['field']=='lognormal_vol', :]
del normal_vols['field'], forward_rates['field'], lognormal_vols['field']

lognormal_vols.reset_index(drop=True, inplace=True)
forward_rates.reset_index(drop=True, inplace=True)
normal_vols.reset_index(drop=True, inplace=True)


lognormal_vols.loc[:,quote_str_map.keys()] = lognormal_vols.loc[:,quote_str_map.keys()].astype('float64') / 100
forward_rates.loc[:,quote_str_map.keys()] = forward_rates.loc[:,quote_str_map.keys()].astype('float64') / 100
normal_vols.loc[:,quote_str_map.keys()] = normal_vols.loc[:,quote_str_map.keys()].astype('float64') / 10000

#date_columns = ['tenor','settlement_date', 'effective_date', 'termination_date', 'expiry_years', 'term_years']
#quote_columns = sorted(quote_columns)
#lognormal_vols = lognormal_vols[date_columns + quote_str]
#forward_rates = forward_rates[date_columns + quote_str]
#normal_vols = normal_vols[date_columns + quote_str]

#%% Iterate up the term structure and solve.
ln_shift = 0.02

# Create the optionlet schedule for the term structure.
N = len(lognormal_vols) - 1
effective_date = lognormal_vols.loc[N,'effective_date']
termination_date = lognormal_vols.loc[N,'termination_date']
schedule = get_schedule(start_date=effective_date,
                        end_date=termination_date,
                        frequency=PeriodFrequency.QUARTERLY,
                        busdaycal=busdaycal)
schedule['payment_dates'] = get_payment_dates(schedule=schedule, busdaycal=busdaycal)
schedule['coupon_term'] = year_fraction(schedule['period_start'], schedule['period_end'], day_count_basis)
schedule['discount_factors'] = zero_curve.get_discount_factors(dates=schedule['payment_dates'])
schedule['alpha'] = np.nan
schedule['beta'] = np.nan
schedule['rho'] = np.nan
schedule['volvol'] = np.nan
schedule['atm_sln_vol'] = np.nan
schedule['atm_vol'] = np.nan


# Solve the SABR parameters for the 1st pillar point.
# As the solve is on the volatilities, the annuity factor is not required.
beta = 1.0
i = 0
tau = lognormal_vols.loc[i,'expiry_years']
vols_sln = lognormal_vols.loc[i,quote_str_map.keys()].astype('float64').values
vols_n = normal_vols.loc[i,quote_str_map.keys()].astype('float64').values
K = forward_rates.loc[i,quote_str_map.keys()].astype('float64').values
F = forward_rates.loc[i,'atm']
params, res = fit_sabr_params_to_smile(tau=tau, F=F, ln_shift=ln_shift, K=K, vols_sln=vols_sln, beta=beta)
params = {k: round(v, 8) for k, v in zip(['alpha', 'beta', 'rho', 'volvol'], params)}

mask = schedule['period_end'] <= lognormal_vols.loc[i,'termination_date']
schedule.loc[mask, 'alpha'] = params['alpha']
schedule.loc[mask, 'beta'] = params['beta']
schedule.loc[mask, 'rho'] = params['rho']
schedule.loc[mask, 'volvol'] = params['volvol']
schedule.loc[mask, 'atm_sln_vol'] = vols_sln[K == F][0]
schedule.loc[mask, 'atm_vol'] = vols_n[K == F][0]

#%%

i = 1
tau = lognormal_vols.loc[i,'expiry_years']
vols_sln = lognormal_vols.loc[i,quote_str_map.keys()].astype('float64').values
vols_n = normal_vols.loc[i,quote_str_map.keys()].astype('float64').values
K = forward_rates.loc[i,quote_str_map.keys()].astype('float64').values
F = forward_rates.loc[i,'atm']



#%%





print(params)
print(res)

# 2nd pillar point.
# Step 1. Price the option lets from t=1 to t=1st pillar point.
i = 1







# Solve the shortest tenor SABR smile. This defines the smile for the 0-shortest tenor. (i.e 0 to 1Y).

# Price the next tenor cap/floor. Get price X.
# X = A + B = [Price of cap from 0Y to 1Y] + [Price of caplets from 1Y to 2Y].
# A is known and X is known. Solve for B.
# B defines the caplet smile from 1Y to 2Y.
# B could be a:
# (i) single set of SABR parameters or
# This is achieved by solving the SABR parameters that produce valuations of the caplets that sum to X.
# (ii) different SABR parameters for each day in the term.
# The daily term structure is achieved by interpolating the atm vol between the 1Y and 2Y tenors.
# The SABR parameters at the pillar point caplet (21M) are solved.
# SABR parameters for the 12M-21M are interpolated from the 12M SABR fit and the 21M SABR fit.

# So, in the fit function I need:
# 1. The short end pillar SABR definition.
# 2. The definition of the caplets over the term.

# We are building up a dataframe.




