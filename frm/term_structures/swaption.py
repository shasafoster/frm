# -*- coding: utf-8 -*-
import os

from Demos.mmapfile_demo import offset

from frm.enums.term_structures import TermRate
from frm.utils.tenor import tenor_to_date_offset, clean_tenor

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
# from dataclasses import dataclass, field, InitVar
# import datetime as dt
# import numpy as np
# import pandas as pd
# import scipy
# import re
# from frm.pricing_engine.black import black76, bachelier, normal_vol_to_black76_sln, black76_sln_to_normal_vol, black76_sln_to_normal_vol_analytical, normal_vol_atm_to_black76_sln_atm, VOL_SLN_BOUNDS, VOL_N_BOUNDS
# from frm.pricing_engine.sabr import solve_alpha_from_sln_vol, calc_sln_vol_for_strike_from_sabr_params
# from frm.utils.daycount import year_fraction, day_count
# from frm.enums.utils import DayCountBasis, PeriodFrequency, RollConvention, StubType, DayRoll, TimingConvention
# from frm.enums.term_structures import TermRate
# from frm.utils.tenor import clean_tenor, tenor_to_date_offset
# from frm.utils.utilities import convert_column_to_consistent_data_type
# from frm.utils.schedule import Schedule, get_schedule, add_period_length_to_schedule
# from frm.term_structures.zero_curve import ZeroCurve
# from typing import Optional, Union, List
# import time
# import numbers
# import concurrent.futures as cf

import numpy as np
import pandas as pd
import numbers
import time
from prettytable import PrettyTable

from frm.utils.business_day_calendar import get_busdaycal
from frm.utils.daycount import DayCountBasis
from frm.utils.schedule import PeriodFrequency, Schedule
from frm.term_structures.zero_curve import ZeroCurve
from frm.utils.daycount import year_fraction, day_count
from frm.enums.utils import DayCountBasis, RollConvention
from frm.pricing_engine.sabr import solve_alpha_from_sln_vol, calc_sln_vol_for_strike_from_sabr_params, fit_sabr_params_to_sln_smile
from frm.term_structures.interest_rate_option_helpers import standardise_relative_quote_col_names, standardise_atmf_quote_col_names

fp = './tests_private/term_structures/test_optionlet_support_20240628.xlsm'
#idd_quote_details = pd.read_excel(io=fp, sheet_name='Quotes')
discount_factors_3m = pd.read_excel(io=fp, sheet_name='DF_3M')
discount_factors_6m = pd.read_excel(io=fp, sheet_name='DF_6M')

curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')
day_count_basis = DayCountBasis.ACT_365

zero_curve_3m = ZeroCurve(curve_date=curve_date,
                          data=discount_factors_3m,
                          day_count_basis=DayCountBasis.ACT_365,
                          busdaycal=busdaycal,
                          interpolation_method='linear_on_log_of_discount_factors')
zero_curve_6m = ZeroCurve(curve_date=curve_date,
                          data=discount_factors_6m,
                          day_count_basis=DayCountBasis.ACT_365,
                          busdaycal=busdaycal,
                          interpolation_method='linear_on_log_of_discount_factors')

swaption_quotes = pd.read_excel(io=fp, sheet_name='Swaption1Y')
vol_ln_df = swaption_quotes.loc[swaption_quotes['field']=='lognormal_vol', :].reset_index(drop=True)
df = vol_ln_df


fixed_frequency = PeriodFrequency.QUARTERLY
settlement_delay = 1
ln_shift = 0.01




# Set the swaption expiry date, and the underlying swap effective date and swap termination date.
df['expiry'] = df['expiry'].apply(clean_tenor)
df['expiry_date'] = np.busday_offset(
    dates=(curve_date + df['expiry'].apply(tenor_to_date_offset)).to_numpy().astype('datetime64[D]'),
    offsets=0, roll='following', busdaycal=busdaycal)

df['swap_effective_date'] = np.busday_offset(dates=df['expiry_date'].to_numpy().astype('datetime64[D]'),
    offsets=settlement_delay, roll='following', busdaycal=busdaycal)

df['swap_term'] = df['swap_term'].apply(clean_tenor)
df['swap_termination_date'] = np.busday_offset(
    dates=(df['swap_effective_date'] + df['swap_term'].apply(tenor_to_date_offset)).to_numpy().astype('datetime64[D]'),
    offsets=settlement_delay, roll='following', busdaycal=busdaycal)

df['expiry_years'] = year_fraction(curve_date, df['expiry_date'], day_count_basis)

# Convert the quote columns to standardised format.
col_name_update_atmf = standardise_atmf_quote_col_names(col_names=list(df.columns))
df.rename(columns=col_name_update_atmf, inplace=True)
col_name_update_relative, col_name_adj_to_forward = standardise_relative_quote_col_names(col_names=list(df.columns))
df.rename(columns=col_name_update_relative, inplace=True)
quote_columns = list(col_name_update_atmf.values()) + list(col_name_update_relative.values())

for i, row in df.iterrows():
    df.loc[i,'fixed_freq'] = PeriodFrequency.from_value(row['fixed_freq'])
    df.loc[i, 'float_freq'] = PeriodFrequency.from_value(row['float_freq'])



#%%

# Functions

# 1. Swaption quote helper - cleans up the swaption quotes dataframe.
# 2. Swaption term structure term rate bootstrapper.
# TODO:
#  Need a 'term rate swap curve'
#  Composed of zero curves and term fixings.
#




# Calculate the par swap rate for each swap term / expiry combo. This is a bit slow (0.2s) for 14 rows.
# Could be optimised by creating the longest schedule over each fixed frequency then indexing.
df['F'] = np.nan
for i, row in df.iterrows():

    frequency = PeriodFrequency.from_value(row['fixed_freq'])

    fixed_schedule = Schedule(start_date=row['swap_effective_date'],
                             end_date=row['swap_termination_date'],
                             frequency=frequency,
                             busdaycal=busdaycal,
                             add_fixing_dates=True)
    fixed_schedule.add_period_length_to_schedule(day_count_basis=day_count_basis)

    if frequency == PeriodFrequency.QUARTERLY:
        fixed_schedule.df['forward_rate'] = zero_curve_3m.get_forward_rates(period_start=fixed_schedule.df['period_start'],
                                                                        period_end=fixed_schedule.df['period_end'],
                                                                        forward_rate_type=TermRate.SIMPLE)
    elif frequency == PeriodFrequency.SEMIANNUAL:
        fixed_schedule.df['forward_rate'] = zero_curve_6m.get_forward_rates(period_start=fixed_schedule.df['period_start'],
                                                                        period_end=fixed_schedule.df['period_end'],
                                                                        forward_rate_type=TermRate.SIMPLE)

    fixed_schedule.df['discount_factor'] = zero_curve_3m.get_discount_factors(dates=fixed_schedule.df['payment_date'])
    fixed_schedule.df['annuity_factor'] = fixed_schedule.df['period_years'] * fixed_schedule.df['discount_factor']
    df.loc[i,'F'] = (fixed_schedule.df['forward_rate'] * fixed_schedule.df['annuity_factor'] / fixed_schedule.df[
        'annuity_factor'].sum()).sum()

# Setup strikes dataframe
strikes_df = df[quote_columns].copy()
strikes_df[quote_columns] = np.nan
strikes_df.loc[:,'atmf'] = df['F'].values
for col, adj in col_name_adj_to_forward.items():
    strikes_df[col] = strikes_df['atmf'] + adj


# Each loop is independent, so can be parallelised. 0.27s for 14 rows.
df[['alpha_', 'beta_', 'rho_', 'volvol_']] = np.nan
for (i,vols_sln_row),(j,strikes_row) in zip(df.iterrows(),strikes_df.iterrows()):
    result = fit_sabr_params_to_sln_smile(tau=vols_sln_row['expiry_years'],
                                          F=vols_sln_row['F'],
                                          K=strikes_row[quote_columns],
                                          vols_sln=vols_sln_row[quote_columns],
                                          ln_shift=ln_shift,
                                          beta=vols_sln_row['beta'])

    (alpha, beta, rho, volvol), res = result
    if res.success:
        df.loc[i, ['alpha_', 'beta_', 'rho_', 'volvol_']] = alpha, beta, rho, volvol
    else:
        print(res)
        raise ValueError(f"SABR calibration of expiry-swap term {vols_sln_row['expiry']}{vols_sln_row['swap_term']} failed.")



df.to_clipboard()


#%%

# Process swaption_quotes into nice format.

# INPUTS
# Dataframe
#  columns names:
#   'expiry': a string of the tenor swaption expiry (e.g. '3M', '1Y', '5Y')
#   'swap_term': a tenor string of the tenor swaption expiry (e.g. '3M', '1Y', '5Y')
#    column names for the strikes of the swaption quotes:
#    (i) atmf and/or
#    (ii) (a) relative quotes to the forward rate (e.g. '-100bps', '-50bps', '0bps', '50bps', '100bps') or
#         (b) absolute quotes (e.g. '2.5%', '3.0%', '3.5%')
#         Note only (a) or (b) can be specified, not (a) and (b) together.
#   fixed leg frequency, day count basis
#   floating leg frequency, day count basis
# curve_date
# busdaycal
# roll_convention
# zero_curve
# swap_settle_delay_after_expiry
# settlement_date

# To do
# Format the swaption quotes into a nice format
# validate, settlement date, settlement delay
# calculate the forward rate for each swap term / expiry combo (adjust for the fixed/float rate day count & freq)
# calculate the expiry, effective date, termination date for swap term / expiry combo
# calculate the annuity factor for each swap term / expiry combo (if needed).

# SABR fit function.
# Fit sabr params to the swaption quotes.

# Interp function

# Interp term structure for any term and expiry. Linear interp on beta, rho, volvol and vol_n_atmf.
# alpha and vol_sln_atmf are analytically calculated.

# Swaption pricer for any swaption.
# Given annuity factor is the dimension of variation of schedule, requiring a fixed/float schedule object/df would be easy.
# THis is a later priority, once we map how to define swap legs and a trade/swap object (Pay/Rcv leg is where I am thinking currently).
