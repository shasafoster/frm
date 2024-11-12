# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import re
from typing import Union
import warnings
from frm.utils.utilities import convert_column_to_consistent_data_type
from frm.utils.tenor import clean_tenor, tenor_to_date_offset

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

VALID_DELTA_CONVENTIONS = ['regular_spot','regular_forward','premium_adjusted_spot','premium_adjusted_forward']



def fx_term_structure_helper(df: pd.DataFrame,
                             curve_date: pd.Timestamp,
                             spot_offset: int,
                             cal: np.busdaycalendar,
                             rate_set_date_str: str) -> pd.DataFrame:

    df['tenor'] = df['tenor'].apply(clean_tenor)

    mandatory_columns_at_least_one_of = [['tenor',rate_set_date_str,'delivery_date']]
    for column_set in mandatory_columns_at_least_one_of:
        assert sum([column in df.columns for column in column_set]) >= 1, f'Specify at least one of {column_set}'

    for column in [rate_set_date_str,'delivery_date']:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = df[column].astype('datetime64[ns]')

    # Input hierarchy is:
    # 1. Use the delivery date if it is available.
    # 2. Otherwise, calculate the delivery date from the rate-set date ('fixing_date' for forwards, 'expiry_date' for options
    # 3. Otherwise, calculate the rate-set date and delivery date from the tenor
    for i, row in df.iterrows():
        if pd.notna(row['delivery_date']):
            pass
        elif pd.notna(row[rate_set_date_str]):
            rate_set_date_np = row[rate_set_date_str].astype('datetime64[D]')
            df.at[i,'delivery_date'] = np.busday_offset(rate_set_date_np, offsets=-1*spot_offset, roll='preceding',busdaycal=cal)
        elif pd.notna(row['tenor']):
            date_offset = tenor_to_date_offset(row['tenor'])
            rate_set_date = curve_date + date_offset
            rate_set_date_np = rate_set_date.to_numpy().astype('datetime64[D]')
            df.at[i,rate_set_date_str] = np.busday_offset(rate_set_date_np, offsets=0, roll='following',busdaycal=cal)
            df.at[i,'delivery_date'] = np.busday_offset(rate_set_date_np, offsets=spot_offset, roll='following',busdaycal=cal)
        else:
            raise ValueError("'delivery_date', '{rate_set_date_str}' and 'tenor' are all missing")

    if 'tenor' in df.columns:
        time_column_order = ['tenor'] + [rate_set_date_str,'delivery_date']
    else:
        time_column_order = [rate_set_date_str,'delivery_date']
    column_order = time_column_order + [col for col in df.columns if col not in time_column_order]

    df = df[column_order]
    df = df.sort_values('delivery_date').reset_index(drop=True)

    return df


def fx_forward_curve_helper(fx_forward_curve_df: pd.DataFrame,
                            curve_date: pd.Timestamp,
                            spot_offset: int,
                            cal: np.busdaycalendar) -> pd.DataFrame:

    fx_forward_curve_df = convert_column_to_consistent_data_type(fx_forward_curve_df)

    mandatory_columns = ['fx_forward_rate']
    for column in mandatory_columns:
        assert column in fx_forward_curve_df.columns, f'{column} is missing in fx_forward_curve'

    fx_forward_curve_df = fx_term_structure_helper(df=fx_forward_curve_df,
                                                   curve_date=curve_date,
                                                   spot_offset=spot_offset,
                                                   cal=cal,
                                                   rate_set_date_str='fixing_date')
    return fx_forward_curve_df
    
    
def validate_ccy_pair(ccy_pair):

    ccy_pair = ccy_pair.lower().replace('/','').strip()
    assert len(ccy_pair) == 6, ccy_pair
    foreign_ccy = ccy_pair[:3] # Also known as the base currency
    domestic_ccy = ccy_pair[3:] # Also known as the quote currency

    if foreign_ccy == 'usd' and domestic_ccy in {'aud', 'eur', 'gbp', 'nzd'}:
        warnings.warn(
            "non conventional fx market, market convention is 'usd' as the domestic currency, i.e. the the quote currency, for 'aud/usd', 'eur/usd', 'gbp/usd' and 'nzd/usd' pairs")
    elif domestic_ccy == 'usd' and foreign_ccy not in {'aud', 'eur', 'gbp', 'nzd'}:
        warnings.warn(
            "non conventional fx market, market convention is for 'usd' to be the foreign currency, i.e. the base currency, (except for 'aud/usd', 'eur/usd', 'gbp/usd' and 'nzd/usd' pairs)")

    return ccy_pair, domestic_ccy, foreign_ccy





def check_delta_convention(df: pd.DataFrame, ccy_pair: str) -> pd.DataFrame:

    bool_cond = np.logical_and(ccy_pair in ['audusd', 'nzdeur', 'gbpusd', 'nzdusd'],
                               df['delta_convention'].str.contains('premium_adjusted'))
    df.loc[bool_cond, 'warnings'] += 'the regular delta is the market delta_convention for this currency pair\n'
    if bool_cond.any():
        warnings.warn('The delta_convention is premium_adjusted for the following tenors: '
                      f'{df.loc[bool_cond, "tenor"].unique()} for the currency pair {ccy_pair}. '
                      'The market convention is to use the regular delta. '
                      'Please ensure that the delta_convention is consistent with the market convention.')

    conditions_for_premium_adjusted_delta_convention = [
        np.full(len(df), np.logical_not(ccy_pair in ['audusd', 'nzdeur', 'gbpusd', 'nzdusd', 'usdaud', 'usdeur', 'usdgbp', 'usdnzd'])),
        np.full(len(df), 'usd' in ccy_pair),
        df['delta_convention'].str.contains('regular').values]
    bool_cond = np.logical_and.reduce(conditions_for_premium_adjusted_delta_convention)
    df.loc[bool_cond, 'warnings'] += 'premium adjusted delta is the market delta_convention for this currency pair\n'

    if bool_cond.any():
        warnings.warn('The delta_convention is regular for the following tenors: '
                      f'{df.loc[bool_cond, "tenor"].unique()} for the currency pair {ccy_pair}. '
                      'The market convention is to use the premium adjusted delta. '
                      'Please ensure that the delta_convention is consistent with the market convention.')


    return df


def clean_vol_quotes_column_names(vol_quotes_df):
    patterns = {
        r'^(0?[1-9]|[1-4]\d)[_ ]?(delta|Δ)[_ ]?(call|c|put|p)$': lambda
            m: f'{m.group(1)}_delta_{"call" if m.group(3) in ["call", "c"] else "put"}',
        r'^(0?[1-9]|[1-4]\d)[_ ]?(delta|Δ)[_ ]?(risk[_ ]?reversal|rr|butterfly|bf)$': lambda
            m: f'{m.group(1)}_delta_{m.group(3).replace("_", "").replace(" ", "").replace("rr", "riskreversal").replace("bf", "butterfly")}',
        r'^(a|at)[_ ]?(t|the)[_ ]?(m|money)[_ ]?(delta|Δ)[_ ]?neutral$': lambda _: 'atm_delta_neutral',
        r'^(a|at)[_ ]?(t|the)[_ ]?(m|money)[_ ]?(f|fwd|forward)$': lambda _: 'atm_forward'
    }

    for col_name in vol_quotes_df.columns:
        for pattern, func in patterns.items():
            match = re.match(pattern, col_name)
            if match:
                new_col_name = func(match)
                vol_quotes_df.rename(columns={col_name: new_col_name}, inplace=True)
                break

    return vol_quotes_df


def solve_call_put_quotes_from_strategy_quotes(vol_quotes_df):

    risk_reversal_deltas = sorted([col_name.split('_')[0] for col_name in vol_quotes_df.columns if 'riskreversal' in col_name])
    butterfly_deltas = sorted([col_name.split('_')[0] for col_name in vol_quotes_df.columns if 'butterfly' in col_name])

    assert 'atm_delta_neutral' in vol_quotes_df.columns
    assert risk_reversal_deltas == butterfly_deltas

    for delta in butterfly_deltas:

        if f'{delta}_delta_put' not in vol_quotes_df.columns:
            vol_quotes_df[f'{delta}_delta_put'] = vol_quotes_df['atm_delta_neutral'] \
                                                        + vol_quotes_df[f'{delta}_delta_butterfly'].values \
                                                        - 0.5 * vol_quotes_df[f'{delta}_delta_riskreversal'].values
        if f'{delta}_delta_call' not in vol_quotes_df.columns:
            vol_quotes_df[f'{delta}_delta_call'] = vol_quotes_df['atm_delta_neutral'] \
                                                         + vol_quotes_df[f'{delta}_delta_butterfly'].values \
                                                         + 0.5 * vol_quotes_df[f'{delta}_delta_riskreversal'].values

        vol_quotes_df.drop(labels=[f'{delta}_delta_butterfly', f'{delta}_delta_riskreversal'], axis=1, inplace=True)

    return vol_quotes_df


def get_delta_smile_quote_details(df: pd.DataFrame,
                                  call_put_pattern: str = r'^(0?[1-9]|[1-4]\d)[_ ]?(delta|Δ)[_ ]?(call|c|put|p)$',
                                  atm_delta_neutral_column_pattern: str = r'^(a|at)[_ ]?(t|the)[_ ]?(m|money)[_ ]?(delta|Δ)[_ ]?neutral$') -> pd.DataFrame:
    quotes_column_names = []
    for col_name in df.columns:
        if re.match(call_put_pattern, col_name):
            quotes_column_names.append(col_name)
        elif re.match(atm_delta_neutral_column_pattern, col_name):
            quotes_column_names.append(col_name)

    call_put_flag = np.array([1 if 'call' in col_name else -1 if 'put' in col_name else 1 if col_name == 'atm_delta_neutral' else np.nan for col_name in quotes_column_names])

    signed_delta = np.array(
        [0.5 if col_name == 'atm_delta_neutral' else call_put_flag[i] * float(col_name.split('_')[0]) / 100 for i, col_name
         in enumerate(quotes_column_names)]
    )
    delta_smile_quote_details = pd.DataFrame({'quotes_column_names': quotes_column_names, 'quotes_call_put_flag': call_put_flag, 'quotes_signed_delta': signed_delta})
    delta_smile_quote_details['order'] = 100 / delta_smile_quote_details['quotes_signed_delta']
    delta_smile_quote_details = delta_smile_quote_details.sort_values('order', ascending=True).reset_index(drop=True)
    delta_smile_quote_details.drop(labels='order', axis=1, inplace=True)

    return delta_smile_quote_details


def interp_fx_forward_curve_df(
        fx_forward_curve_df: pd.DataFrame,
        dates: Union[pd.Series, pd.DatetimeIndex] = None,
        date_type: str = None,
        flat_extrapolation: bool = True) -> pd.Series:

    assert date_type in ['fixing_date', 'delivery_date']

    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)
    elif isinstance(dates, pd.DatetimeIndex):
        pass
    else:
        raise ValueError("'dates' must be a pandas Series or DatetimeIndex")

    unique_dates = pd.DatetimeIndex(dates.drop_duplicates())
    combined_dates = unique_dates.union(pd.DatetimeIndex(fx_forward_curve_df[date_type]))
    df = pd.Series(fx_forward_curve_df['fx_forward_rate'].values, index=fx_forward_curve_df[date_type].values)

    result = df.reindex(combined_dates.values).copy()
    start_date = fx_forward_curve_df[date_type].min()
    end_date = fx_forward_curve_df[date_type].max()

    if flat_extrapolation:
        try:
            result = result.interpolate(method='time', limit_area='inside').ffill().bfill()
        except ValueError:
            raise ValueError("Interpolation failed, check for missing values in the input data.")
        # Find out of range dates and warn
        out_of_range_dates = unique_dates[(unique_dates < start_date) | (unique_dates > end_date)]
        for date in out_of_range_dates:
            warnings.warn(
                f"Date {date} is outside the {date_type} range {start_date} - {end_date}, flat extrapolation applied.")
    else:
        result = result.interpolate(method='time', limit_area='inside')

    return result.reindex(dates).values


