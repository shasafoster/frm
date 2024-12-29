# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
import numpy as np
import pandas as pd
import re

from frm.utils.daycount import year_frac
from frm.enums import DayCountBasis, PeriodFreq, TermRate
from frm.utils import clean_tenor, tenor_to_date_offset, convert_column_to_consistent_data_type, make_schedule
from frm.term_structures.zero_curve import ZeroCurve

from typing import Optional






def process_capfloor_quotes(
        curve_date: pd.Timestamp,
        vol_ln_df: pd.DataFrame,
        ln_shift: float,
        day_count_basis: DayCountBasis,
        zero_curve: ZeroCurve,
        optionlet_date_offset: pd.DateOffset,
        settlement_date: Optional[pd.Timestamp]=None,
        settlement_delay: Optional[int]=1,
        busdaycal: Optional[np.busdaycalendar]=np.busdaycalendar()
):

    if settlement_date is None:
        settlement_date = np.busday_offset(curve_date.to_numpy().astype('datetime64[D]'), offsets=settlement_delay, roll='following', busdaycal=busdaycal)

    effective_date_np = (settlement_date + optionlet_date_offset).to_numpy().astype('datetime64[D]')
    effective_date = np.busday_offset(effective_date_np, offsets=0, roll='following', busdaycal=busdaycal)

    vol_ln_df['tenor'] = vol_ln_df['tenor'].apply(clean_tenor)
    vol_ln_df['settlement_date'] = settlement_date
    vol_ln_df['effective_date'] = effective_date

    for i, row in vol_ln_df.iterrows():
        date_offset = tenor_to_date_offset(row['tenor'])
        termination_date = settlement_date + date_offset
        last_optionlet_expiry_date = termination_date - optionlet_date_offset
        last_optionlet_expiry_date_np = last_optionlet_expiry_date.to_numpy().astype('datetime64[D]')
        termination_date_date_np = termination_date.to_numpy().astype('datetime64[D]')
        vol_ln_df.at[i, 'last_optionlet_expiry_date'] = np.busday_offset(last_optionlet_expiry_date_np, offsets=0, roll='following',busdaycal=busdaycal)
        vol_ln_df.at[i, 'termination_date'] = np.busday_offset(termination_date_date_np, offsets=0, roll='following', busdaycal=busdaycal)

    vol_ln_df['term_years'] = year_frac(vol_ln_df['effective_date'], vol_ln_df['termination_date'], day_count_basis)
    vol_ln_df['last_optionlet_expiry_years'] = year_frac(curve_date, vol_ln_df['last_optionlet_expiry_date'], day_count_basis)
    vol_ln_df['F'] = np.nan
    vol_ln_df['ln_shift'] = ln_shift

    vol_ln_df = convert_column_to_consistent_data_type(vol_ln_df)

    first_columns = ['tenor', 'settlement_date', 'effective_date', 'last_optionlet_expiry_date','termination_date', 'term_years','last_optionlet_expiry_years','F','ln_shift']
    column_order = first_columns + [col for col in vol_ln_df.columns if col not in first_columns]
    vol_ln_df = vol_ln_df[column_order]

    # Two methods of specifying quotes, same functions can be used for Caps/Floors and Swaptions
    # 1. Quotes relative to the atm forward rate (e.g. ATM, ATM+/-50bps, ATM+/-100bps...)
    # 2. Absolute quotes (e.g. 2.5%, 3.0%, 3.5%...)

    col_name_update_dict, quote_str_map = standardise_relative_quote_col_names(col_names=list(vol_ln_df.columns))
    vol_ln_df.rename(columns=col_name_update_dict, inplace=True)
    vol_ln_df.reset_index(drop=True, inplace=True)

    N = len(vol_ln_df) - 1
    effective_date = vol_ln_df.loc[N,'effective_date']
    termination_date = vol_ln_df.loc[N,'termination_date']
    optionlet_df = make_schedule(start_date=effective_date,
                                end_date=termination_date,
                                frequency=PeriodFreq.QUARTERLY,
                                busdaycal=busdaycal)
    optionlet_df['coupon_term'] = year_frac(optionlet_df['period_start'], optionlet_df['period_end'], day_count_basis)
    optionlet_df['discount_factors'] = zero_curve.get_discount_factors(dates=optionlet_df['payment_dates'])
    optionlet_df['annuity_factor'] = optionlet_df['coupon_term'] * optionlet_df['discount_factors']
    optionlet_df['expiry_years'] = year_frac(curve_date, optionlet_df['period_start'], day_count_basis)
    optionlet_df['F'] = zero_curve.get_forward_rates(period_start=optionlet_df['period_start'],
                                                     period_end=optionlet_df['period_end'],
                                                     forward_rate_type=TermRate.SIMPLE)
    optionlet_df['vol_n_atm'] = np.nan
    optionlet_df['vol_sln_atm'] = np.nan
    optionlet_df['ln_shift'] = ln_shift
    optionlet_df['alpha'] = np.nan
    optionlet_df['beta'] = np.nan
    optionlet_df['rho'] = np.nan
    optionlet_df['volvol'] = np.nan

    # Calculate the forward rate (pre lognormal shift) for the cap/floor quotes
    for i,row in vol_ln_df.iterrows():
        mask = (optionlet_df['period_end'] <= vol_ln_df.loc[i,'termination_date'])
        vol_ln_df.loc[i,'F'] = (optionlet_df.loc[mask, 'F'] * optionlet_df.loc[mask, 'annuity_factor']).sum() / optionlet_df.loc[mask, 'annuity_factor'].sum()

    # Setup a strike dataframe
    strikes_df = vol_ln_df.copy()
    strikes_df[list(quote_str_map.keys())] = np.nan
    for column in strikes_df[quote_str_map.keys()].columns:
        strikes_df[column] = strikes_df['F'] + quote_str_map[column]

    # Setup a call/put flag dataframe
    call_put_df = vol_ln_df.copy()
    call_put_df[list(quote_str_map.keys())] = np.nan
    call_put_df[list(quote_str_map.keys())] = np.where(strikes_df[list(quote_str_map.keys())].values > call_put_df['F'].values[:, None], 1, -1)


    capfloor_surf = dict()
    capfloor_surf['vol_ln_df'] = vol_ln_df
    capfloor_surf['strikes'] = strikes_df
    capfloor_surf['call_put_df'] = call_put_df
    capfloor_surf['optionlet_df'] = optionlet_df
    capfloor_surf['zero_curve'] = zero_curve
    capfloor_surf['quote_columns'] = list(quote_str_map.keys())

    return capfloor_surf



def settlement_date_helper(curve_date: pd.Timestamp,
                           settlement_date: Optional[pd.Timestamp],
                           settlement_delay: Optional[int],
                           busdaycal: Optional[np.busdaycalendar]=np.busdaycalendar()):

        if settlement_date is None and settlement_delay is None:
            raise ValueError('Either settlement_date or settlement_delay must be provided.')

        elif settlement_date is None and settlement_delay is not None:
            settlement_date = np.busday_offset(curve_date.to_numpy().astype('datetime64[D]'),
                                                    offsets=settlement_delay,
                                                    roll='following',
                                                    busdaycal=busdaycal)
        elif settlement_date is not None and settlement_delay is None:
            assert settlement_date >= curve_date, f"settlement_date {settlement_date} must be on or after curve_date {curve_date}."

        return settlement_date


def standardise_atmf_quote_col_names(col_names: list[str]):
    col_name_update = dict()
    atm_quote = '(a|at)[ -]?(t|the)[ -]?(m|money)[ -]?(f|forward)?'

    for col_name in col_names:
        if re.search(atm_quote, col_name, re.IGNORECASE):
            new_col_name = 'atmf'
            col_name_update[col_name] = new_col_name

    assert len(col_name_update) == 1, f"Expected 1 ATMF quote, found {len(col_name_update)}."

    return col_name_update


def standardise_relative_quote_col_names(col_names: list[str]):
    col_name_update = dict()
    col_name_adj_to_forward = dict()
    bps_quote = r'[+-]?\s?\d+\s?(bps|bp)'
    percentage_quote = r'[+-]?\s?\d+(\.\d+)?\s?%'

    # ATMF quote
    col_name_update_atmf = standardise_atmf_quote_col_names(col_names=col_names)
    col_name_update.update(col_name_update_atmf)
    col_name_adj_to_forward.update({'atmf': 0})

    # Relative Quotes
    for col_name in col_names:
        if re.search(bps_quote, col_name, re.IGNORECASE):
            v = round(float(col_name.replace('bps', '').replace('bp', '').replace(' ', '')) / 10000,8)
            new_col_name = (str(int(v * 10000)) if (round(v * 10000,8)).is_integer() else str(round(v * 10000, 8))) + 'bps'
            col_name_update[col_name] = new_col_name
            col_name_adj_to_forward[new_col_name] = v

        elif re.search(percentage_quote, col_name, re.IGNORECASE):
            v = round(float(col_name.replace('%', '').replace(' ', '')) / 100,8)
            new_col_name = (str(int(v * 100)) if (round(v * 100,8)).is_integer() else str(round(v * 100, 8))) + 'bps'
            col_name_update[col_name] = new_col_name
            col_name_adj_to_forward[new_col_name] = v

    return col_name_update, col_name_adj_to_forward


def standardise_absolute_quote_col_names(col_names: list[str]):
    col_name_update = dict()
    col_name_strike = dict()
    bps_quote = r'[+-]?\s?\d+\s?(bps|bp)'
    percentage_quote = r'[+-]?\s?\d+(\.\d+)?\s?%'

    for col_name in col_names:
        if re.search(bps_quote, col_name, re.IGNORECASE):
            v = round(float(col_name.replace('bps', '').replace('bp', '').replace(' ', '')) / 10000,8)
            new_col_name = (str(int(v * 10000)) if (round(v * 10000,8)).is_integer() else str(round(v * 10000, 8))) + 'bps'
            col_name_update[col_name] = new_col_name
            col_name_strike[new_col_name] = v

        elif re.search(percentage_quote, col_name, re.IGNORECASE):
            v = round(float(col_name.replace('%', '').replace(' ', '')) / 100,8)
            new_col_name = (str(int(v * 100)) if (round(v * 100,8)).is_integer() else str(round(v * 100, 8))) + 'bps'
            col_name_update[col_name] = new_col_name
            col_name_strike[new_col_name] = v

    return col_name_update, col_name_strike