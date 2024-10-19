# -*- coding: utf-8 -*-
import os
from lib2to3.pgen2.tokenize import printtoken

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
import scipy
import re
from frm.pricing_engine.sabr import fit_sabr_params_to_smile
from frm.pricing_engine.black import black76, bachelier, normal_vol_to_black76_ln, black76_ln_to_normal_vol, black76_ln_to_normal_vol_analytical
from frm.pricing_engine.sabr import solve_alpha, calc_ln_vol_for_strike

from frm.utils.business_day_calendar import get_busdaycal
from frm.utils.daycount import year_fraction
from frm.enums.utils import DayCountBasis, PeriodFrequency
from frm.enums.term_structures import TermRate
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
    termination_date = settlement_date + date_offset
    last_optionlet_expiry_date = termination_date - tenor_to_date_offset(swap_index_tenor)
    last_optionlet_expiry_date_np = last_optionlet_expiry_date.to_numpy().astype('datetime64[D]')
    termination_date_date_np = termination_date.to_numpy().astype('datetime64[D]')
    cap_floor_quotes.at[i, 'last_optionlet_expiry_date'] = np.busday_offset(last_optionlet_expiry_date_np, offsets=0, roll='following',busdaycal=busdaycal)
    cap_floor_quotes.at[i, 'termination_date'] = np.busday_offset(termination_date_date_np, offsets=0, roll='following', busdaycal=busdaycal)

cap_floor_quotes['term_years'] = year_fraction(cap_floor_quotes['effective_date'], cap_floor_quotes['termination_date'], day_count_basis)
cap_floor_quotes['last_optionlet_expiry_years'] = year_fraction(curve_date, cap_floor_quotes['last_optionlet_expiry_date'], day_count_basis)

cap_floor_quotes = convert_column_to_consistent_data_type(cap_floor_quotes)

first_columns = ['tenor', 'settlement_date', 'effective_date', 'last_optionlet_expiry_date','termination_date', 'term_years','last_optionlet_expiry_years']
column_order = first_columns + [col for col in cap_floor_quotes.columns if col not in first_columns]
cap_floor_quotes = cap_floor_quotes[column_order]



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

normal_vols = cap_floor_quotes.loc[cap_floor_quotes['field']=='normal_vol', :].copy()
forward_rates = cap_floor_quotes.loc[cap_floor_quotes['field']=='forward_rate', :].copy()
lognormal_vols = cap_floor_quotes.loc[cap_floor_quotes['field']=='lognormal_vol', :].copy()
del normal_vols['field'], forward_rates['field'], lognormal_vols['field']

lognormal_vols.reset_index(drop=True, inplace=True)
forward_rates.reset_index(drop=True, inplace=True)
normal_vols.reset_index(drop=True, inplace=True)

lognormal_vols.loc[:,quote_str_map.keys()] = round(lognormal_vols.loc[:,quote_str_map.keys()].astype('float64') / 100, 10)
forward_rates.loc[:,quote_str_map.keys()] = forward_rates.loc[:,quote_str_map.keys()].astype('float64') / 100
normal_vols.loc[:,quote_str_map.keys()] = round(normal_vols.loc[:,quote_str_map.keys()].astype('float64') / 10000, 10)

lognormal_vols['ln_shift'] = 0.02
lognormal_vols['F'] = forward_rates.loc[0,'atm']
normal_vols['F'] = forward_rates.loc[0,'atm']

cp_df = pd.DataFrame(
    data=np.where(forward_rates.loc[:, quote_str_map.keys()] >= forward_rates.loc[:, 'atm'].values.reshape(-1, 1), -1, 1),
    columns=list(quote_str_map.keys()))

# Create the optionlet schedule for the term structure.
N = len(lognormal_vols) - 1
effective_date = lognormal_vols.loc[N,'effective_date']
termination_date = lognormal_vols.loc[N,'termination_date']
optionlet_df = get_schedule(start_date=effective_date,
                            end_date=termination_date,
                            frequency=PeriodFrequency.QUARTERLY,
                            busdaycal=busdaycal)
optionlet_df['payment_dates'] = get_payment_dates(schedule=optionlet_df, busdaycal=busdaycal)
optionlet_df['coupon_term'] = year_fraction(optionlet_df['period_start'], optionlet_df['period_end'], day_count_basis)
optionlet_df['discount_factors'] = zero_curve.get_discount_factors(dates=optionlet_df['payment_dates'])
optionlet_df['expiry_years'] = year_fraction(curve_date, optionlet_df['period_start'], day_count_basis)
optionlet_df['F'] = zero_curve.get_forward_rates(period_start=optionlet_df['period_start'],
                                                 period_end=optionlet_df['period_end'],
                                                 forward_rate_type=TermRate.SIMPLE)
optionlet_df['alpha'] = np.nan
optionlet_df['beta'] = np.nan
optionlet_df['rho'] = np.nan
optionlet_df['volvol'] = np.nan
optionlet_df['vol_sln_atm'] = np.nan
optionlet_df['ln_shift'] = 0.02
optionlet_df['vol_n_atm'] = np.nan
optionlet_df['black76_px'] = '→'
optionlet_df[list(quote_str_map.keys())] = np.nan


#% Price the first option based on the flat lognormal volatility.
beta = 1.0
i = 0

def capfloor_px_from_scalar_sln_quote(optionlet_df: pd.DataFrame,
                                      lognormal_vols_df: pd.DataFrame,
                                      K_df: pd.DataFrame,
                                      cp_df: pd.DataFrame,
                                      quote_columns: list,
                                      index: int) -> pd.DataFrame:

    # Index option quote parameters
    ln_shift = lognormal_vols_df.loc[index,'ln_shift']
    vol_sln = lognormal_vols_df.loc[index,quote_columns].astype('float64').values
    K = K_df.loc[index,quote_columns].astype('float64').values
    cp = cp_df.loc[index,quote_columns].astype('float64').values

    optionlet_pxs = np.full(shape=(len(optionlet_df), len(quote_columns)), fill_value=np.nan)

    for i,row in optionlet_df.iterrows():
        black76_forward_px = black76(F=row['F'],
                                     tau=row['expiry_years'],
                                     r=0, # Annuity factor is applied later
                                     cp=cp,
                                     K=K,
                                     vol_sln=vol_sln,
                                     ln_shift=ln_shift)['price']

        optionlet_pxs[i,:] = black76_forward_px * row['coupon_term'] * row['discount_factors']

    capfloor_pxs = optionlet_pxs.sum(axis=0)
    return pd.DataFrame(data=capfloor_pxs.reshape(1, -1), columns=list(quote_columns))

capfloor_pxs_df = capfloor_px_from_scalar_sln_quote(
    optionlet_df=optionlet_df,
    lognormal_vols_df=lognormal_vols,
    K_df=forward_rates,
    cp_df=cp_df,
    quote_columns=list(quote_str_map.keys()),
    index=0)

capfloor_pxs_df = capfloor_px_from_scalar_sln_quote(
    optionlet_df=optionlet_df,
    lognormal_vols_df=lognormal_vols,
    K_df=forward_rates,
    cp_df=cp_df,
    quote_columns=['atm'],
    index=0)


def vol_sse(param: float,
            optionlet_df: pd.DataFrame,
            target: float,
            sse_multiplier: int = 1e4):
    """SSE function for solving the pillar point normal volatility for the optionlet term structure"""

    bachelier_atm_cap_px = 0

    for i, row in optionlet_df.iterrows():

        if pd.isna(optionlet_df.loc[0, 'vol_n_atm']):
            # For the first pillar point, use a flat normal volatility over the tenor.
            vol_n = param
        else:
            # For subsequent pillar points, interpolate the normal volatility between the last and current pillar point.
            if pd.notna(row['vol_n_atm']):
                vol_n = row['vol_n_atm']
            else:
                last_valid_index = optionlet_df['vol_n_atm'].last_valid_index()
                X = [optionlet_df.loc[last_valid_index, 'expiry_years'],
                     optionlet_df.loc[len(optionlet_df) - 1, 'expiry_years']]
                Y = [optionlet_df.loc[last_valid_index, 'vol_n_atm'], param.item()]
                x = row['expiry_years']
                vol_n = np.interp(x, X, Y)

        bachelier_forward_px = bachelier(F=row['F'],
                                         tau=row['expiry_years'],
                                         r=0,
                                         cp=1,
                                         K=row['F'],
                                         vol_n=vol_n)['price']

        bachelier_atm_cap_px += bachelier_forward_px * row['coupon_term'] * row['discount_factors']

    SSE = sse_multiplier * (target - bachelier_atm_cap_px) ** 2
    return SSE


#% Price the at-the-money option based on the lognormal volatility quote and solve the equivalent normal volatility.

#index = 0
#mask = optionlet_df['period_end'] <= lognormal_vols.loc[index, 'termination_date']

for index in range(len(lognormal_vols)):

    mask_option_term = optionlet_df['period_end'] <= lognormal_vols.loc[index, 'termination_date']

    if index == 0:
        mask_optionlets_to_solve = mask_option_term
    else:
        mask_optionlets_to_solve = np.logical_and(
            mask_option_term,
            np.logical_not(optionlet_df['period_end'] <= lognormal_vols.loc[max(0,index-1), 'termination_date']))

    pillar_optionlet_index = np.where(mask_optionlets_to_solve.values)[0][-1] if np.any(mask_optionlets_to_solve.values) else None


    # Value the cap/floor using the scalar lognormal at-the-money volatility quote.
    capfloor_atm_px = capfloor_px_from_scalar_sln_quote(
        optionlet_df=optionlet_df[mask_option_term],
        lognormal_vols_df=lognormal_vols,
        K_df=forward_rates,
        cp_df=cp_df,
        quote_columns=['atm'],
        index=index).at[0, 'atm']


    # Value the cap/floor using the optionlet term structure of normal volatilities.

    x0 = black76_ln_to_normal_vol_analytical(
            F = lognormal_vols.loc[index,'F'],
            tau = lognormal_vols.loc[index,'last_optionlet_expiry_years'],
            K = lognormal_vols.loc[index,'F'],
            vol_sln = lognormal_vols.loc[index,'atm'],
            ln_shift = lognormal_vols.loc[index,'ln_shift'])

    res = scipy.optimize.minimize(fun=lambda param: vol_sse(param=param, optionlet_df=optionlet_df.loc[mask_option_term,:], target=capfloor_atm_px),
                                  x0=x0,
                                  bounds=[(0, None)])

    if res.success:
        vol_n_atm = res.x.item()
        if index == 0:
            optionlet_df.loc[mask_option_term,'vol_n_atm'] = vol_n_atm
        else:
            # The pillar point optionlet normal volatility is set the solved value.
            # Intra-pillar dates are linearly interpolated between the pillar points.
            last_valid_index = optionlet_df['vol_n_atm'].last_valid_index()
            X = [optionlet_df.loc[last_valid_index, 'expiry_years'], optionlet_df.loc[pillar_optionlet_index, 'expiry_years']]
            Y = [optionlet_df.loc[last_valid_index, 'vol_n_atm'], vol_n_atm]
            x = optionlet_df.loc[mask_optionlets_to_solve, 'expiry_years']
            vol_n = np.interp(x, X, Y)
    else:
        raise ValueError('Optimisation failed to converge.')


    for i,row in optionlet_df.loc[mask_optionlets_to_solve,:].iterrows():
        optionlet_df.loc[i,'vol_sln_atm'] = normal_vol_to_black76_ln(
            F=row['F'],
            tau=row['expiry_years'],
            K=row['F'],
            vol_n=row['vol_n_atm'],
            ln_shift=row['ln_shift'])



#%%
#
# for i,_ in lognormal_vols.iterrows():
#     mask = optionlet_df['period_end'] <= lognormal_vols.loc[i, 'termination_date']
#
#     # Calculate the at-the-money (ATM) option price from the Black76 model.
#     capfloor_atm_px = capfloor_px_from_scalar_sln_quote(
#         optionlet_df=optionlet_df,
#         capfloor_lognormal_vols_df=lognormal_vols,
#         quote_columns=['atm'],
#         index=i).at[0,'atm']
#
#     black76_atm_cap_px = 0
#
#     for _,row in optionlet_df[mask].iterrows():
#         black76_forward_px = black76(F=row['F'],
#                                      tau=row['expiry_years'],
#                                      r=0,
#                                      cp=1,
#                                      K=row['forward_rate'],
#                                      vol_sln=lognormal_vols.loc[i,'atm'],
#                                      ln_shift=row['ln_shift'])['price']
#
#         black76_atm_cap_px += black76_forward_px * row['coupon_term'] * row['discount_factors']
#
#
# def vol_sse(param, optionlet_df, sse_multiplier=1e4):
#     bachelier_atm_cap_px = 0
#
#     for _,row in optionlet_df[mask].iterrows():
#         bachelier_forward_px = bachelier(F=row['F'],
#                                      tau=row['expiry_years'],
#                                      r=0,
#                                      cp=1,
#                                      K=row['forward_rate'],
#                                      vol_n=param)['price']
#
#         bachelier_atm_cap_px += bachelier_forward_px * row['coupon_term'] * row['discount_factors']
#
#     SSE =  sse_multiplier * (black76_atm_cap_px - bachelier_atm_cap_px) ** 2
#     return SSE
#
# x0 = black76_ln_to_normal_vol_analytical(
#         F = lognormal_vols.loc[0,'F'],
#         tau = lognormal_vols.loc[0,'last_optionlet_expiry_years'],
#         K = lognormal_vols.loc[0,'F'],
#         vol_sln = lognormal_vols.loc[0,'atm'],
#         ln_shift = lognormal_vols.loc[0,'ln_shift'])
#
# res = scipy.optimize.minimize(fun=lambda param: vol_sse(param), x0=x0, bounds=[(0, None)])
#
# if res.success:
#     vol_n_atm = res.x.item()
# else:
#     raise ValueError('Optimisation failed to converge.')
#
# # Setup the SABR parameters for the optionlet schedule.
# sabr_option_let_params = pd.DataFrame(columns=['alpha', 'beta', 'rho', 'volvol','vol_sln_atm','ln_shift', 'vol_n_atm', 'F','expiry_years'])
# sabr_option_let_params['beta'] = 1.0
# sabr_option_let_params['F'] = optionlet_current['forward_rate']
# sabr_option_let_params['expiry_years'] = optionlet_current['expiry_years']
# sabr_option_let_params['ln_shift'] = optionlet_current['ln_shift']
# sabr_option_let_params['vol_n_atm'] = vol_n_atm
#
# for i,row in sabr_option_let_params.iterrows():
#     sabr_option_let_params.loc[i,'vol_sln_atm'] = normal_vol_to_black76_ln(
#         F=row['F'],
#         tau=row['expiry_years'],
#         K=row['F'],
#         vol_n=row['vol_n_atm'],
#         ln_shift=row['ln_shift'])
#
#
#
# #%%
#
# def vol_sse(param, beta=None, sse_multiplier=1e8):
#     # beta, rho, volvol from outer
#
#     if len(param) == 2:
#         rho, volvol = param
#     elif len(param) == 3:
#         beta, rho, volvol = param
#
#     for i,row in optionlet_current.iterrows():
#         F = row['forward_rate']
#         tau = row['expiry_years']
#         ln_shift = row['ln_shift']
#
#         alpha = solve_alpha(
#             tau=tau,
#             F=F,
#             beta=beta,
#             rho=rho,
#             volvol=volvol,
#             vol_sln_atm=sabr_option_let_params.loc[i,'vol_sln_atm'],
#             ln_shift=ln_shift)
#
#         vol_sln_sabr = calc_ln_vol_for_strike(
#             tau=tau,
#             F=F,
#             alpha=alpha,
#             beta=beta,
#             rho=rho,
#             volvol=volvol,
#             K=K,
#             ln_shift=ln_shift)
#
#         black76_forward_px = black76(F=row['forward_rate'],
#                                      tau=row['expiry_years'],
#                                      r=0,
#                                      cp=cp,
#                                      K=K,
#                                      vol_sln=vol_sln_sabr,
#                                      ln_shift=ln_shift)['price']
#
#         optionlet_pvs[i,:] = black76_forward_px * row['coupon_term'] * row['discount_factors']
#
#     optionlet_pv = optionlet_pvs.sum(axis=0)
#
#     SSE = sse_multiplier * sum((option_flat_vol_pv - optionlet_pv) ** 2)
#
#     return SSE
#
#
# optionlet_pvs = np.full(shape=optionlet_current[quote_str_map.keys()].shape, fill_value=np.nan)
#
# # params = (beta), rho, volvol
# # beta has a valid range of 0≤β≤1
# # rho has a valid range of -1≤ρ≤1
# # volvol has a valid range of 0<v≤∞
# beta = 1.0
# x0 = np.array([0.00, 0.10]) if beta is not None else np.array([0.0, 0.0, 0.1])
# bounds = [(-1.0, 1.0), (0.0001, None)] if beta is not None else [(-1.0, 1.0), (-1.0, 1.0), (0.0001, None)]
#
# res = scipy.optimize.minimize(
#     fun=lambda param: vol_sse(param=param, beta=beta),
#     x0=x0,
#     bounds=bounds)
#
# if res.success:
#     beta, rho, volvol = (beta, *res.x) if beta is not None else res.x
# else:
#     raise ValueError('Optimisation failed to converge.')
#
# sabr_option_let_params['beta'] = beta
# sabr_option_let_params['rho'] = rho
# sabr_option_let_params['volvol'] = volvol
#
# #%%
#
# for i,row in sabr_option_let_params.iterrows():
#     sabr_option_let_params.loc[i,'alpha'] = solve_alpha(
#         tau=row['expiry_years'],
#         F=row['F'],
#         beta=row['beta'],
#         rho=row['rho'],
#         volvol=row['volvol'],
#         vol_sln_atm=row['vol_sln_atm'],
#         ln_shift=row['ln_shift'])
#
# #%% Next step is to try the 2Y
#
#
#







def vol_see(param, tau, F, ln_shift, K, vols_sln, vol_sln_atm, beta=None):
    if beta is None:
        beta, rho, volvol = param
    else:
        rho, volvol = param

    for i,row in sabr_option_let_params.iterrows():
        sabr_option_let_params.loc[i,'alpha'] = solve_alpha(tau=tau,
                                                            F=F, beta=beta,
                                                            rho=rho,
                                                            volvol=volvol,
                                                            vol_sln_atm=vol_sln_atm,
                                                            ln_shift=ln_shift)

    sabr_vols = calc_ln_vol_for_strike(tau=tau,F=F,alpha=alpha,beta=beta,rho=rho,volvol=volvol,K=K, ln_shift=ln_shift)
    return sum((vols_sln - sabr_vols) ** 2)


# Index the at-the-money (ATM) volatility
#mask_atm = K == F
#if mask_atm.sum() != 1:
#    raise ValueError('ATM strike must be unique and present.')
#vol_sln_atm = vols_sln[mask_atm].item()



# params = (beta), rho, volvol
# beta has a valid range of 0≤β≤1
# rho has a valid range of -1≤ρ≤1
# volvol has a valid range of 0<v≤∞
x0 = np.array([0.00, 0.10]) if beta is not None else np.array([0.0, 0.0, 0.1])
bounds = [(-1.0, 1.0), (0.0001, None)] if beta is not None else [(-1.0, 1.0), (-1.0, 1.0), (0.0001, None)]

res = scipy.optimize.minimize(
    fun=lambda param: vol_sse(param, tau=tau, F=F, ln_shift=ln_shift, K=K, vols_sln=vols_sln, vol_sln_atm=vol_sln_atm,
                              beta=beta),
    x0=x0,
    bounds=bounds)

beta, rho, volvol = (beta, *res.x) if beta is not None else res.x
alpha = solve_alpha(tau=tau, F=F, beta=beta, rho=rho, volvol=volvol, vol_sln_atm=vol_sln_atm, ln_shift=ln_shift)

return (alpha, beta, rho, volvol), res

#%%
beta = 1.0
i = 0
tau = lognormal_vols.loc[i,'last_optionlet_expiry_years']
vols_sln = lognormal_vols.loc[i,quote_str_map.keys()].astype('float64').values
vols_n = normal_vols.loc[i,quote_str_map.keys()].astype('float64').values
K = forward_rates.loc[i,quote_str_map.keys()].astype('float64').values
F = forward_rates.loc[i,'atm']
params, res = fit_sabr_params_to_smile(tau=tau, F=F, ln_shift=ln_shift, K=K, vols_sln=vols_sln, beta=beta)
params = {k: round(v, 10) for k, v in zip(['alpha', 'beta', 'rho', 'volvol'], params)}

mask = optionlet_df['period_end'] <= lognormal_vols.loc[i,'termination_date']
optionlet_df.loc[mask, 'ln_shift'] = ln_shift
optionlet_df.loc[mask, 'alpha'] = params['alpha']
optionlet_df.loc[mask, 'beta'] = params['beta']
optionlet_df.loc[mask, 'rho'] = params['rho']
optionlet_df.loc[mask, 'volvol'] = params['volvol']
# TODO As atm_sln_vol is a function of the forward rate, normal_vol and tau, the atm_sln_vol should be resolved for each the optionlet forward rate.
# TODO Then alpha should be resolved as alpha is a function of atm_sln_vol, beta, rho, volvol, K and F.
optionlet_df.loc[mask, 'atm_sln_vol'] = vols_sln[K == F][0]
optionlet_df.loc[mask, 'atm_n_vol'] = vols_n[K == F][0]

for i,row in optionlet_df[mask].iterrows():

    K = forward_rates.loc[i,quote_str_map.keys()].astype('float64').values
    cp = np.where(K >= F, -1, 1)

    black76_forward_px = black76(F=F,
                                 tau=row['expiry_years'],
                                 r=0,
                                 cp=cp,
                                 K=K,
                                 vol_sln=lognormal_vols.loc[i,quote_str_map.keys()].astype('float64').values,
                                 ln_shift=row['ln_shift'])['price']

    optionlet_px = black76_forward_px * row['coupon_term'] * row['discount_factors']
    optionlet_df.loc[i,quote_str_map.keys()] = optionlet_px * 100e6

#%%
# We minimise function that:
# Interpolates the SABR parameters between the final and last pillar point. to get the daily term structure.
# Prices the optionlets from the 1st pillar point to the 2nd pillar point.
# Sums the optionlet prices to get the cap price. Compares this cap price to the cap prices from the market volatilities.
# SSE = Cap prices from market volatilities - Cap prices from model volatilities.

i = 1

mask_prior = optionlet_df['period_end'] <= lognormal_vols.loc[i-1,'termination_date']
mask_current = optionlet_df['period_end'] <= lognormal_vols.loc[i,'termination_date']
mask_current_pillar = np.logical_and(mask_current, ~mask_prior)

pillar_index_last = optionlet_df[mask_prior].index[-1]
pillar_index_current = optionlet_df[mask_current].index[-1]



last_period_sabr_params = optionlet_df.loc[mask_prior, ['alpha', 'beta', 'rho', 'volvol']].iloc[-1]

# Initial with the last period's SABR parameters as the first guess in the calibration.
optionlet_df.loc[mask_current_pillar, ['alpha', 'beta', 'rho', 'volvol']] = last_period_sabr_params.values



# Linearly interpolate the normal volatilities between the last and current pillar point.
X = [optionlet_df.loc[mask_prior,'expiry_years'].iloc[-1], optionlet_df.loc[mask_current,'expiry_years'].iloc[-1]]
Y = [optionlet_df.loc[mask_prior,'atm_n'].iloc[-1], optionlet_df.loc[mask_current,'atm_n'].iloc[-1]]
x = optionlet_df.loc[mask_current_pillar,'expiry_years'].values
y = np.interp(x, X, Y)

#optionlet_df.loc[mask_current_pillar,'atm_sln_vol'] = lognormal_vols.loc[i,'atm'].astype('float64')
#optionlet_df.loc[mask_current_pillar,'atm_n_vol'] = normal_vols.loc[i,'atm'].astype('float64')


#%%

mask = optionlet_df['period_end'] <= lognormal_vols.loc[i,'termination_date']
#tau = lognormal_vols.loc[i,'expiry_years']
vols_sln = lognormal_vols.loc[i,quote_str_map.keys()].astype('float64').values
vols_n = normal_vols.loc[i,quote_str_map.keys()].astype('float64').values
K = forward_rates.loc[i,quote_str_map.keys()].astype('float64').values
F = forward_rates.loc[i,'atm']



# 1. Price the optionlets from t=0 to t=1st pillar point.
for i,row in optionlet_df[mask].iterrows():
    pass


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




