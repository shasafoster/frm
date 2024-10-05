# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 
    

import numpy as np
import pandas as pd
import re
from typing import Optional, Union
import warnings

from frm.utils.utilities import convert_column_to_consistent_data_type
from frm.utils.tenor import clean_tenor, tenor_to_date_offset
from frm.utils.daycount import year_fraction



def get_fx_spot_spot_offset(validated_ccy_pair: str) -> int:

    if len(validated_ccy_pair) == 6:
        # http://www.londonfx.co.uk/valdates.html
        # https://web.archive.org/web/20240213223033/http://www.londonfx.co.uk/valdates.html

        if validated_ccy_pair in {'usdcad','cadusd',
                                  'usdphp','phpusd',
                                  'usdrub','rubusd',
                                  'usdtry','tryusd'}:
            return 1
        else:
            return 2




def fx_term_structure_helper(df: pd.DataFrame,
                             curve_date: pd.Timestamp,
                             spot_offset: int,
                             busdaycal: np.busdaycalendar,
                             rate_set_date_str: str) -> pd.DataFrame:

    df['tenor'] = df['tenor'].apply(clean_tenor)

    mandatory_columns_at_least_one_of = [['tenor',rate_set_date_str,'delivery_date']]
    for column_set in mandatory_columns_at_least_one_of:
        assert sum([column in df.columns for column in column_set]) >= 1, f'Specify at least one of {column_set}'

    for column in [rate_set_date_str,'delivery_date']:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = df[column].astype('datetime64[ns]')

    # Hierarchy is to use the delivery date if it is available, otherwise use the fixing date, otherwise use the tenor.
    for i, row in df.iterrows():

        if pd.notna(row['delivery_date']):
            pass
        elif pd.notna(row[rate_set_date_str]):
            rate_set_date_np = row[rate_set_date_str].astype('datetime64[D]')
            df.at[i,'delivery_date'] = np.busday_offset(rate_set_date_np, offsets=-1*spot_offset, roll='preceding',busdaycal=busdaycal)
        elif pd.notna(row['tenor']):
            date_offset = tenor_to_date_offset(row['tenor'])
            rate_set_date = curve_date + date_offset
            rate_set_date_np = rate_set_date.to_numpy().astype('datetime64[D]')
            df.at[i,rate_set_date_str] = np.busday_offset(rate_set_date_np, offsets=0, roll='following',busdaycal=busdaycal)
            df.at[i,'delivery_date'] = np.busday_offset(rate_set_date_np, offsets=spot_offset, roll='following',busdaycal=busdaycal)
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
                            busdaycal: np.busdaycalendar) -> pd.DataFrame:

    fx_forward_curve_df = convert_column_to_consistent_data_type(fx_forward_curve_df)

    mandatory_columns = ['fx_forward_rate']
    for column in mandatory_columns:
        assert column in fx_forward_curve_df.columns, f'{column} is missing in fx_forward_curve'

    fx_forward_curve_df = fx_term_structure_helper(df=fx_forward_curve_df,
                                                   curve_date=curve_date,
                                                   spot_offset=spot_offset,
                                                   busdaycal=busdaycal,
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


def check_curve_date_spot_date_spot_offset_consistency(curve_date: pd.Timestamp,
                                                       spot_date: pd.Timestamp,
                                                       spot_offset: int,
                                                       busdaycal: np.busdaycalendar):


    implied_spot_offset = calc_implied_spot_offset(curve_date, spot_date, busdaycal)
    if implied_spot_offset != spot_offset:
        raise ValueError(f"The implied spot offset (from 'curve_date', 'spot_date' and 'busdaycal') is {implied_spot_offset} "
                         f"which is inconsistent with the provided spot_offset of {spot_offset}. \n"
                         f"Please ensure that the 'spot_offset' is consistent with the 'curve_date', 'spot_date' and 'busdaycal'.")


def calc_implied_spot_offset(curve_date: pd.Timestamp,
                             spot_date: pd.Timestamp,
                             busdaycal: np.busdaycalendar) -> int:
    spot_offset = 0
    date = curve_date
    while date < spot_date:
        spot_offset += 1
        date += pd.DateOffset(days=1)
        date_np = date.to_numpy().astype('datetime64[D]')
        date = np.busday_offset(date_np, offsets=0, roll='following', busdaycal=busdaycal)
    return spot_offset


def resolve_fx_curve_dates(
        ccy_pair: str,
        busdaycal: np.busdaycalendar,
        curve_date: Optional[pd.Timestamp] = None,
        spot_offset: Optional[int] = None,
        spot_date: Optional[pd.Timestamp] = None):

    if curve_date is not None and spot_date is not None and spot_offset is not None:
        # All three dates are specified, check for consistency
        check_curve_date_spot_date_spot_offset_consistency(curve_date, spot_date, spot_offset, busdaycal)
    elif spot_offset is None and curve_date is not None and spot_date is not None:
        # Spot offset is not specified, calculate it based on curve date and spot date
        spot_offset = calc_implied_spot_offset(curve_date, spot_date, busdaycal)
    else:
        # Only one of {curve_date, spot_date} are specified, get spot_offset per market convention from ccy_pair
        spot_offset = get_fx_spot_spot_offset(ccy_pair)

    if spot_date is None:
        curve_date_np = curve_date.to_numpy().astype('datetime64[D]')
        spot_date = pd.Timestamp(
            np.busday_offset(curve_date_np, offsets=spot_offset, roll='following', busdaycal=busdaycal))
    elif curve_date is None:
        spot_date_np = spot_date.to_numpy().astype('datetime64[D]')
        curve_date = pd.Timestamp(
            np.busday_offset(spot_date_np, offsets=-spot_offset, roll='preceding', busdaycal=busdaycal))

    return curve_date, spot_offset, spot_date


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


def clean_and_extract_vol_smile_columns(df: pd.DataFrame,
                                        call_put_pattern: str = r'^(0?[1-9]|[1-4]\d)[_ ]?(delta|Δ)[_ ]?(call|c|put|p)$',
                                        atm_delta_neutral_column_pattern: str = r'^atm[_ ]?(delta|Δ)[_ ]?neutral$'):
    quotes_column_names = []
    for col_name in df.columns:
        match = re.match(call_put_pattern, col_name)
        if match:
            # Extract components from match groups
            delta_value = match.group(1)  # 1 or 2 digits
            option_type = match.group(3)  # "call", "c", "put", or "p"

            # Convert "c" to "call" and "p" to "put"
            option_type_full = 'call' if option_type in ['call', 'c'] else 'put'

            # Return the column name in the new format
            new_col_name = f'{delta_value}_delta_{option_type_full}'
            df.rename(columns={col_name: new_col_name}, inplace=True)
            quotes_column_names.append(f'{delta_value}_delta_{option_type_full}')

        elif re.match(atm_delta_neutral_column_pattern, col_name):
            new_col_name = 'atm_delta_neutral'
            df.rename(columns={col_name: new_col_name}, inplace=True)
            quotes_column_names.append(new_col_name)

    call_put_flag = np.array([1 if 'call' in col_name else -1 if 'put' in col_name else 0 for col_name in quotes_column_names])
    signed_delta = np.array(
        [0.5 if col_name == 'atm_delta_neutral' else call_put_flag[i] * float(col_name.split('_')[0]) / 100 for i, col_name
         in enumerate(quotes_column_names)]
    )
    quote_details = pd.DataFrame({'quotes_column_names': quotes_column_names, 'quotes_call_put_flag': call_put_flag, 'quotes_signed_delta': signed_delta})
    quote_details['order'] = 100 / quote_details['quotes_signed_delta']
    quote_details = quote_details.sort_values('order', ascending=True).reset_index(drop=True)
    quote_details.drop(labels='order', axis=1, inplace=True)

    return df, quote_details


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


def forward_volatility(t1: Union[float, np.array],
                       vol_t1: Union[float, np.array],
                       t2: Union[float, np.array],
                       vol_t2: Union[float, np.array]) -> Union[float, np.array]:
    """
    Calculate the forward volatility from time t1 to t2.
    The forward volatility is based on the consistency condition:
    vol_t1**2 * t1 + vol_t1_t2**2 * (t2- t1) = vol_t1**2 * t2


    Parameters:
    - t1 (float): Time to first maturity (in years). Must be less than t2.
    - vol_t1 (float): (Annualised) Volatility to expiry 1 for a given delta
    - t2 (float): Time to second maturity (in years). Must be greater than t1.
    - vol_t2 (float): (Annualised) Volatility to expiry 2 for the same given delta

    Returns:
    - np.array: Forward volatility from time t1 to t2
    """

    tau = t2 - t1
    if np.any(tau == 0):
        mask = tau == 0
        warnings.warn(f"t2 and t1 are equal. NaN will be returned for these values: t1 {t1[mask]}, t2 {t2[mask]}")
    elif np.any(tau < 0):
        raise ValueError("t2 is less than t1.")

    result = (vol_t2 ** 2 * t2 - vol_t1 ** 2 * t1) / tau
    if np.any(result < 0):
        raise ValueError("Negative value encountered under square root.")

    return np.sqrt(result)


def flat_forward_interp(t1: Union[float, np.array],
                        vol_t1: Union[float, np.array],
                        t2: Union[float, np.array],
                        vol_t2: Union[float, np.array],
                        t: Union[float, np.array]) -> Union[float, np.array]:
    """
    Interpolate volatility at a given time 't' using flat forward interpolation.

    Parameters:
    - t1 (Union[float, array]): Time to first expiry in years. Must be less than t2.
    - vol_t1 (float): (Annualised) Volatility to expiry 1 for a given delta
    - t2 (Union[float, array]): Time to second expiry in years. Must be greater than t1.
    - vol_t2 (float): (Annualised) Volatility to expiry 2 for the same given delta
    - t (Union[float, array]): Time at which to interpolate the volatility. Must be between t1 and t2.

    Returns:
    - array: Interpolated volatility at time 't'
    """
    vol_t12 = np.zeros(shape=vol_t1.shape)
    mask_not_equal_t = (t1 != t2).flatten()
    vol_t12[mask_not_equal_t,:] = forward_volatility(t1[mask_not_equal_t], vol_t1[mask_not_equal_t,:],
                                                     t2[mask_not_equal_t], vol_t2[mask_not_equal_t,:])

    return np.sqrt((vol_t1 ** 2 * t1 + vol_t12 ** 2 * (t - t1)) / t)


