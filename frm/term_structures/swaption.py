# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from frm.enums import TermRate, PeriodFreq, DayCountBasis
from frm.utils import tenor_to_date_offset, clean_tenor, get_busdaycal, year_frac, Schedule, workday
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.sabr import fit_sabr_params_to_sln_smile
from frm.term_structures.interest_rate_option_helpers import standardise_relative_quote_col_names, standardise_atmf_quote_col_names

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))


fp = 'C:/Users/shasa/Documents/frm_private/tests_private/test_optionlet_support_20240628.xlsm'
#idd_quote_details = pd.read_excel(io=fp, sheet_name='Quotes')
discount_factors_3m = pd.read_excel(io=fp, sheet_name='DF_3M')
discount_factors_6m = pd.read_excel(io=fp, sheet_name='DF_6M')

curve_date = pd.Timestamp('2024-06-28')
cal = get_busdaycal('AUD')
day_count_basis = DayCountBasis.ACT_365

zero_curve_3m = ZeroCurve(curve_date=curve_date,
                          pillar_df=discount_factors_3m,
                          day_count_basis=DayCountBasis.ACT_365,
                          cal=cal,
                          interp_method='linear_on_ln_discount')
zero_curve_6m = ZeroCurve(curve_date=curve_date,
                          pillar_df=discount_factors_6m,
                          day_count_basis=DayCountBasis.ACT_365,
                          cal=cal,
                          interp_method='linear_on_ln_discount')

swaption_quotes = pd.read_excel(io=fp, sheet_name='Swaption1Y')
vol_sln_df = swaption_quotes.loc[swaption_quotes['field']=='lognormal_vol', :].reset_index(drop=True)


fixed_frequency = PeriodFreq.QUARTERLY
settlement_delay = 1
ln_shift = 0.01


# Convert the quote columns to standardised format.
col_name_update_atmf = standardise_atmf_quote_col_names(col_names=list(vol_sln_df.columns))
vol_sln_df.rename(columns=col_name_update_atmf, inplace=True)
col_name_update_relative, col_name_adj_to_forward = standardise_relative_quote_col_names(col_names=list(vol_sln_df.columns))
vol_sln_df.rename(columns=col_name_update_relative, inplace=True)
quote_columns = list(col_name_update_atmf.values()) + list(col_name_update_relative.values())


#%%

df = vol_sln_df[['expiry', 'swap_term', 'frequency']].copy()

# Set the swaption expiry date, and the underlying swap effective date and swap termination date.
df['expiry'] = df['expiry'].apply(clean_tenor)
df['expiry_date'] = workday(curve_date + df['expiry'].apply(tenor_to_date_offset), 0, cal)
df['swap_effective_date'] = workday(df['expiry_date'], settlement_delay, cal)
df['swap_term'] = df['swap_term'].apply(clean_tenor)
df['swap_termination_date'] = workday(df['swap_effective_date'] + df['swap_term'].apply(tenor_to_date_offset), settlement_delay, cal)
df['expiry_years'] = year_frac(curve_date, df['expiry_date'], day_count_basis)

for i, row in df.iterrows():
    df.loc[i,'frequency'] = PeriodFreq.from_value(row['frequency'])


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

    freq = row['frequency']

    fixed_schedule = Schedule(start_date=row['swap_effective_date'],
                             end_date=row['swap_termination_date'],
                             freq=freq,
                             cal=cal)
    fixed_schedule.add_period_yearfrac(day_count_basis=day_count_basis)

    if freq == PeriodFreq.QUARTERLY:
        fixed_schedule.df['forward_rate'] = zero_curve_3m.get_forward_rates(period_start=fixed_schedule.df['period_start'],
                                                                            period_end=fixed_schedule.df['period_end'],
                                                                            forward_rate_type=TermRate.SIMPLE)
    elif freq == PeriodFreq.SEMIANNUAL:
        fixed_schedule.df['forward_rate'] = zero_curve_6m.get_forward_rates(period_start=fixed_schedule.df['period_start'],
                                                                            period_end=fixed_schedule.df['period_end'],
                                                                            forward_rate_type=TermRate.SIMPLE)

    fixed_schedule.df['discount_factor'] = zero_curve_3m.get_discount_factors(dates=fixed_schedule.df['payment_date'])
    fixed_schedule.df['annuity_factor'] = fixed_schedule.df['period_yearfrac'] * fixed_schedule.df['discount_factor']
    df.loc[i,'F'] = (fixed_schedule.df['forward_rate'] * fixed_schedule.df['annuity_factor'] \
                     / fixed_schedule.df['annuity_factor'].sum()).sum()

# Setup strikes dataframe
strikes_df = vol_sln_df[quote_columns].copy()
strikes_df[quote_columns] = np.nan
strikes_df.loc[:,'atmf'] = df['F'].values
for col, adj in col_name_adj_to_forward.items():
    strikes_df[col] = strikes_df['atmf'] + adj

#%%

# Each loop is independent, so can be parallelised. 0.27s for 14 rows.
df[['alpha', 'beta', 'rho', 'volvol']] = np.nan
for (i,sabr_row),(j,strikes_row),(k,vol_sln_row) in zip(df.iterrows(),strikes_df.iterrows(),vol_sln_df.iterrows()):
    result = fit_sabr_params_to_sln_smile(tau=sabr_row['expiry_years'],
                                          F=sabr_row['F'],
                                          K=strikes_row[quote_columns],
                                          vols_sln=vol_sln_row[quote_columns],
                                          ln_shift=ln_shift,
                                          beta=vol_sln_row['beta'])

    (alpha, beta, rho, volvol), res = result
    if res.success:
        df.loc[i, ['alpha', 'beta', 'rho', 'volvol']] = alpha, beta, rho, volvol
    else:
        print(res)
        raise ValueError(f"SABR calibration of expiry-swap term {vol_sln_row['expiry']}{vol_sln_row['swap_term']} failed.")

df.to_clipboard()


#%%%



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
