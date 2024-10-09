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


VALID_DELTA_CONVENTIONS = ['regular_spot','regular_forward','premium_adjusted_spot','premium_adjusted_forward']




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


# Old validation function for reading in vol surface from excel
# def fx_σ_input_helper(df):
#     for i, column_name in enumerate(['errors', 'warnings', 'internal_id']):
#         if column_name not in df.columns:
#             df.insert(loc=i, column=column_name, value='')
#
#     # % mandatory columns validation
#     mandatory_columns = [
#         'errors',
#         'warnings',
#         'internal_id',
#         'curve_date',
#         'curve_ccy',
#         'Δ_convention',
#     ]
#     df = df.dropna(axis=0, subset=mandatory_columns)  # drop rows with blanks in mandatory columns
#     missing_mandatory_columns = [col for col in mandatory_columns if col not in df.columns.to_list()]
#     if len(missing_mandatory_columns) > 0:
#         df['errors'] += f'missing mandatory columns: {missing_mandatory_columns}\n'
#         return df
#
#     # Validate  column data
#     valid_field_values = {'delta_convention': VALID_DELTA_CONVENTIONS,
#                           'day_count_basis': VALID_DAY_COUNT_BASIS + [np.nan, '']}
#
#     # Enforce to list of valid values
#     for column_name in mandatory_columns:
#         if column_name in valid_field_values.keys():
#             bool_cond = df[column_name].isin(valid_field_values[column_name])
#             df.loc[np.logical_not(bool_cond), 'errors'] = 'invalid value for ' + column_name
#
#     # fx option specific curve_ccy and delta_convention validation
#     field = 'curve_ccy'
#     bool_cond = df[field].isin(['usdaud', 'usdeur', 'usdgbp', 'usdnzd'])
#     df.loc[bool_cond, 'warnings'] += field + ' value is not per common market convention\n'
#
#     bool_cond = np.logical_and(df['curve_ccy'].isin(['audusd', 'nzdeur', 'gbpusd', 'nzdusd']),
#                                df['Δ_convention'].str.contains('premium_adjusted'))
#     df.loc[bool_cond, 'warnings'] += 'the regular delta is the market delta_convention for this currency pair\n'
#
#     bool_cond = np.logical_and.reduce([
#         np.logical_not(
#             df['curve_ccy'].isin(['audusd', 'nzdeur', 'gbpusd', 'nzdusd', 'usdaud', 'usdeur', 'usdgbp', 'usdnzd'])),
#         df['curve_ccy'].str.contains('usd'),
#         df['Δ_convention'].str.contains('regular')])
#     df.loc[bool_cond, 'warnings'] += 'premium adjusted delta is the market delta_convention for this currency pair\n'
#
#     date_columns = ['tenor_name', 'tenor_date']
#     optional_columns = ['day_count_basis', 'tenor_years', 'base_ccy', 'quote_ccy']
#
#     for column_name in optional_columns:
#         if column_name not in df.columns:
#             df[column_name] = np.nan
#         else:
#             # Enforce to list of valid values
#             if column_name in valid_field_values.keys():
#                 bool_cond = df[column_name].isin(valid_field_values[column_name])
#                 df.loc[np.logical_not(bool_cond), 'errors'] = 'invalid value for ' + column_name
#
#     user_input_columns = [v for v in df.columns if v not in mandatory_columns + date_columns + optional_columns]
#
#     # drop user input columns if they are all nan
#     mask = df[user_input_columns].isna().all()
#     cols_to_drop = mask[mask].index.tolist()
#     df = df.drop(columns=cols_to_drop)
#
#     # Validate volatility input columns
#     not_nan_user_input_columns = [v for v in df.columns if v not in mandatory_columns + date_columns + optional_columns]
#
#     invalid_columns = []
#     valid_volatility_input_columns = []
#     for i, v in enumerate(not_nan_user_input_columns):
#         pattern1 = r'^σ_(\d{1,2})Δ(call|put)$'
#         pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
#         atm_column_names = ['σ_atmΔneutral', 'σ_atmf']
#         if (re.match(pattern1, v) and 1 <= int(re.match(pattern1, v).group(1)) <= 99) \
#                 or (re.match(pattern2, v) and 1 <= int(re.match(pattern2, v).group(1)) <= 99) \
#                 or v in atm_column_names:
#             valid_volatility_input_columns.append(v)
#         else:
#             invalid_columns.append(v)
#
#     df = df.drop(columns=invalid_columns)
#
#     for col in invalid_columns:
#         msg = 'user added column' + "'" + col + "'" + ' does not ' \
#               + 'match regex pattern ' + "'" + pattern1 + "', or pattern " + "'" + pattern2 + "'," \
#               + ' and is not in the allowed list (' + ', '.join(atm_column_names) + ')\n'
#
#         bool_cond = df[col].isnotna()
#         df.loc[bool_cond, 'errors'] += msg
#
#     # Enforce only one type of σ quote input
#     pattern_call_put = r'^σ_(\d{1,2})Δ(call|put)$'
#     pattern_strategy = r'^σ_(\d{1,2})Δ(bf|rr)$'
#     cols_call_put = df.filter(regex=pattern_call_put).columns
#     cols_strategy = df.filter(regex=pattern_strategy).columns
#     mask_call_put = df[cols_call_put].apply(lambda x: x.notna().any(), axis=1)
#     mask_strategy = df[cols_strategy].apply(lambda x: x.notna().any(), axis=1)
#     if 'σ_atmf' in df.columns:
#         mask_atmf = df['σ_atmf'].apply(lambda x: x.notna().any(), axis=1)
#
#         mask = np.logical_and.reduce([mask_call_put, mask_strategy, np.logical_not(mask_atmf)])
#         df.loc[
#             mask, 'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-strategy quotes; specify only one volatility input type per row\n'
#
#         mask = np.logical_and.reduce([mask_call_put, mask_atmf, np.logical_not(mask_strategy)])
#         df.loc[
#             mask, 'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'
#
#         mask = np.logical_and.reduce([mask_strategy, mask_atmf, np.logical_not(mask_call_put)])
#         df.loc[
#             mask, 'errors'] = 'row has non-nan values for i) σ-strategy quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'
#
#         mask = np.logical_and.reduce([mask_strategy, mask_atmf, mask_call_put])
#         df.loc[
#             mask, 'errors'] = 'row has non-nan values for i) σ-strategy quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'
#
#         mask = mask_call_put & mask_strategy & mask_atmf
#         df.loc[
#             mask, 'errors'] = 'row has non-nan values for across i) Δ-σ quotes, ii) σ-strategy quotes and iii) σ-atmf quotes; input only one volatility input type per row\n'
#     else:
#         mask = np.logical_and.reduce([mask_call_put, mask_strategy])
#         df.loc[
#             mask, 'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-strategy quotes; input only one volatility input type per row\n'
#
#     # Enforce numeric types for strategy quotes
#     for col in valid_volatility_input_columns:
#         bool_cond = df[col].apply(lambda x: not isinstance(x, (float, int)) and not pd.isna(x))
#         df.loc[bool_cond, 'errors'] += col + ' has an invalid type\n'
#
#         # Call and put quotes validation
#     for col in (cols_call_put.to_list() + atm_column_names):
#         if col in df.columns:
#             bool_cond = df[col].apply(lambda x: isinstance(x, (float, int)) and x <= 0.0)
#             df.loc[bool_cond, 'errors'] += col + ' must have a positive value\n'
#
#             bool_cond = df[col].apply(
#                 lambda x: isinstance(x, (float, int)) and x > 2)  # unlikely for volatility quote to be > 200%
#             df.loc[bool_cond, 'warnings'] += col + ' value is unusually large'
#
#     for col in cols_strategy.to_list():
#         bool_cond = df[col].apply(
#             lambda x: isinstance(x, (float, int)) and abs(x) > 0.25)  # unlikely for strategy spread to be > 25%
#         df.loc[bool_cond, 'warnings'] += col + ' value is unusually large'
#
#     def extract_numbers(df, suffix):
#         pattern = rf'^σ_(\d{{1,2}})Δ{suffix}$'
#         cols = df.filter(regex=pattern).columns
#         return [col.split('_')[1].split('Δ')[0] for col in cols]
#
#     Δ_list = list(set(extract_numbers(df, 'rr') + extract_numbers(df, 'bf')))
#
#     for Δ in Δ_list:
#         atm = 'σ_atmΔneutral'
#         bf = 'σ_' + Δ + 'Δbf'
#         rr = 'σ_' + Δ + 'Δrr'
#
#         for v in ['call', 'put']:
#             column_name = 'σ_' + Δ + 'Δ' + v
#             if column_name not in df.columns:
#                 df[column_name] = np.nan
#
#         for i, row in df.iterrows():
#
#             if bf and rr in row.index:
#                 if atm in row.index:
#                     if pd.notna(row[bf]) and pd.notna(row[rr]) and pd.notna(row[atm]):
#                         if isinstance(row[bf], (float, int)) and isinstance(row[rr], (float, int)) \
#                                 and isinstance(row[atm], (float, int)) and row[atm] > 0:
#                             df.at[i, 'σ_' + Δ + 'Δcall'] = row[bf] + row[atm] + 0.5 * row[rr]
#                             df.at[i, 'σ_' + Δ + 'Δput'] = row[bf] + row[atm] - 0.5 * row[rr]
#
#                 if pd.isna(row[bf]) and pd.notna(row[rr]):
#                     df.loc[i, 'errors'] += bf + ' value is absent\n'  # add comment if butterfly is n/a
#                 elif pd.isna(row[rr]) and pd.notna(row[bf]):
#                     df.loc[i, 'errors'] += rr + ' value is absent\n'  # add comment if risk reversal is n/a
#                 elif (pd.notna(row[bf]) or pd.notna(row[rr])) and pd.isna(row[atm]):
#                     df.loc[i, 'errors'] += atm + ' value is absent\n'  # add comment if at-the-money is n/a
#
#             elif bf in row.index and rr not in row.index:
#                 if rr not in row.index and pd.notna(row[bf]):
#                     df.loc[i, 'errors'] += bf + ' value is present but column ' + rr + ' is absent\n'
#                 if atm not in row.index and pd.notna(row[bf]):
#                     df.loc[i, 'errors'] += bf + ' value is present but column ' + atm + ' is absent\n'
#
#             elif rr in row.index and bf not in row.index:
#                 if bf not in row.index and pd.notna(row[rr]):
#                     df.loc[i, 'errors'] += rr + ' value is present but column ' + bf + ' is absent\n'
#                 if atm not in row.index and pd.notna(row[rr]):
#                     df.loc[i, 'errors'] += rr + ' value is present but column ' + atm + ' is absent\n'
#
#                     # Drop σ-strategy quote columns
#     pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
#     cols_to_drop = df.filter(regex=pattern2).columns
#     df = df.drop(columns=cols_to_drop)
#
#     # Drop all nan columns
#     df = df.dropna(axis=1, how='all')
#
#     return df


