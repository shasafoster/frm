# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
from frm.frm.market_data.ir_zero_curve import ZeroCurve
from frm.frm.pricing_engine.garman_kohlhagen import gk_price, gk_solve_implied_σ, gk_solve_strike
from frm.frm.pricing_engine.heston_garman_kohlhagen import heston_fit_vanilla_fx_smile, heston1993_price_fx_vanilla_european, heston_carr_madan_price_fx_vanilla_european

from frm.frm.schedule.tenor import calc_tenor_date, get_spot_offset
from frm.frm.schedule.daycounter import VALID_DAY_COUNT_BASIS
from frm.frm.schedule.business_day_calendar import get_calendar        
from frm.frm.utilities.utilities import convert_column_to_consistent_data_type, clean_input_dataframe, move_col_after, copy_errors_and_warnings_to_input

import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import fsolve, root_scalar
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline 
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Literal
import warnings
import re
import matplotlib.pyplot as plt

VALID_DELTA_CONVENTIONS = ['regular_spot_delta','regular_forward_delta','premium_adjusted_spot_delta','premium_adjusted_forward_delta'] 


def interp_fx_forward_curve(fx_forward_curve: pd.DataFrame, 
                            dates: Union[pd.Series, pd.DatetimeIndex]=None,
                            date_type: str=None,
                            flat_extrapolation: bool=True) -> pd.Series:
    
    assert date_type in ['expiry_date','delivery_date']
    
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)
    elif isinstance(dates, pd.DatetimeIndex):
        pass
    else:
        raise ValueError("'dates' must be a pandas Series or DatetimeIndex")
        
    unique_dates = dates.drop_duplicates()
    combined_dates = unique_dates.union(pd.DatetimeIndex(fx_forward_curve[date_type]))
    df = pd.Series(fx_forward_curve['fx_forward_rate'].values, index=fx_forward_curve[date_type].values)
    
    result = df.reindex(combined_dates.values).copy()
    start_date = fx_forward_curve[date_type].min()
    end_date = fx_forward_curve[date_type].max()
    
    if flat_extrapolation:
        try:                    
            result = result.interpolate(method='time', limit_area='inside').ffill().bfill()
        except:
            pass
        # Find out of range dates and warn
        out_of_range_dates = unique_dates[(unique_dates < start_date) | (unique_dates > end_date)]
        for date in out_of_range_dates:
            warnings.warn(f"Date {date} is outside the {date_type} range {start_date} - {end_date}, flat extrapolation applied.")
    else:
        result = result.interpolate(method='time', limit_area='inside')
    
    result = result.reindex(dates)
    
    return result
    

def fx_σ_input_helper(df):
    
    df_input = df.copy()

    for i,column_name in enumerate(['errors','warnings','internal_id']):
        if column_name not in df.columns:
            df.insert(loc=i,column=column_name,value='')

    #% mandatory columns validation
    mandatory_columns = [
        'errors',
        'warnings',
        'internal_id',
        'curve_date',
        'curve_ccy',
        'Δ_convention',
    ]
    df = df.dropna(axis=0, subset=mandatory_columns) # drop rows with blanks in mandatory columns
    missing_mandatory_columns = [col for col in mandatory_columns if col not in df.columns.to_list()]
    if len(missing_mandatory_columns) > 0:
        df['errors'] += f'missing mandatory columns: {missing_mandatory_columns}\n'
        return df

    # Validate  column data
    valid_field_values =  {'delta_convention': VALID_DELTA_CONVENTIONS,
                           'day_count_basis': VALID_DAY_COUNT_BASIS + [np.nan,'']}
    
    # Enforce to list of valid values
    for column_name in mandatory_columns:
        if column_name in valid_field_values.keys():
            bool_cond = df[column_name].isin(valid_field_values[column_name])
            df.loc[np.logical_not(bool_cond),'errors'] = 'invalid value for ' + column_name
    
    # fx option specific curve_ccy and delta_convention validation
    field = 'curve_ccy'
    bool_cond = df[field].isin(['usdaud','usdeur','usdgbp','usdnzd'])       
    df.loc[bool_cond,'warnings'] += field + ' value is not per common market convention\n' 

    bool_cond = np.logical_and(df['curve_ccy'].isin(['audusd','nzdeur','gbpusd','nzdusd']),df['Δ_convention'].str.contains('premium_adjusted'))
    df.loc[bool_cond,'warnings'] += 'the regular delta is the market delta_convention for this currency pair\n' 

    bool_cond = np.logical_and.reduce([
        np.logical_not(df['curve_ccy'].isin(['audusd','nzdeur','gbpusd','nzdusd','usdaud','usdeur','usdgbp','usdnzd'])),
        df['curve_ccy'].str.contains('usd'),
        df['Δ_convention'].str.contains('regular')])
    df.loc[bool_cond,'warnings'] += 'premium adjusted delta is the market delta_convention for this currency pair\n' 

    date_columns = ['tenor_name','tenor_date']
    optional_columns = ['day_count_basis','tenor_years','base_ccy','quote_ccy']

    for column_name in optional_columns:
        if column_name not in df.columns:
            df[column_name] = np.nan
        else:
            # Enforce to list of valid values
            if column_name in valid_field_values.keys():
                bool_cond = df[column_name].isin(valid_field_values[column_name])
                df.loc[np.logical_not(bool_cond),'errors'] = 'invalid value for ' + column_name

    user_input_columns = [v for v in df.columns if v not in mandatory_columns + date_columns + optional_columns]

    # drop user input columns if they are all nan
    mask = df[user_input_columns].isna().all()
    cols_to_drop = mask[mask].index.tolist()
    df = df.drop(columns=cols_to_drop)

    # Validate volatility input columns
    not_nan_user_input_columns = [v for v in df.columns if v not in mandatory_columns + date_columns + optional_columns]

    invalid_columns = []
    valid_volatility_input_columns = []
    for i,v in enumerate(not_nan_user_input_columns):
        pattern1 = r'^σ_(\d{1,2})Δ(call|put)$'
        pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
        atm_column_names = ['σ_atmΔneutral','σ_atmf']
        if (re.match(pattern1, v) and 1 <= int(re.match(pattern1, v).group(1)) <= 99) \
            or (re.match(pattern2, v) and 1 <= int(re.match(pattern2, v).group(1)) <= 99) \
            or v in atm_column_names:   
            valid_volatility_input_columns.append(v)
        else:
            invalid_columns.append(v)
            
    df = df.drop(columns=invalid_columns)
    
    for col in invalid_columns:
        msg = 'user added column' + "'" + col + "'" + ' does not ' \
            + 'match regex pattern ' + "'" + pattern1 + "', or pattern " + "'" + pattern2 + "'," \
            + ' and is not in the allowed list (' + ', '.join(atm_column_names) + ')\n'   
            
        bool_cond = df[col].isnotna()
        df.loc[bool_cond,'errors'] += msg

    # Enforce only one type of σ quote input
    pattern_call_put = r'^σ_(\d{1,2})Δ(call|put)$'
    pattern_strategy = r'^σ_(\d{1,2})Δ(bf|rr)$'
    cols_call_put = df.filter(regex=pattern_call_put).columns
    cols_strategy = df.filter(regex=pattern_strategy).columns
    mask_call_put = df[cols_call_put].apply(lambda x: x.notna().any(), axis=1)
    mask_strategy = df[cols_strategy].apply(lambda x: x.notna().any(), axis=1)
    if 'σ_atmf' in df.columns:
        mask_atmf = df['σ_atmf'].apply(lambda x: x.notna().any(), axis=1)
        
        mask = np.logical_and.reduce([mask_call_put, mask_strategy, np.logical_not(mask_atmf)])
        df.loc[mask,'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-strategy quotes; specify only one volatility input type per row\n'

        mask = np.logical_and.reduce([mask_call_put, mask_atmf, np.logical_not(mask_strategy)])
        df.loc[mask,'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'                

        mask = np.logical_and.reduce([mask_strategy, mask_atmf, np.logical_not(mask_call_put)])
        df.loc[mask,'errors'] = 'row has non-nan values for i) σ-strategy quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'               
        
        mask = np.logical_and.reduce([mask_strategy, mask_atmf, mask_call_put])
        df.loc[mask,'errors'] = 'row has non-nan values for i) σ-strategy quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'      
        
        mask = mask_call_put & mask_strategy & mask_atmf
        df.loc[mask,'errors'] = 'row has non-nan values for across i) Δ-σ quotes, ii) σ-strategy quotes and iii) σ-atmf quotes; input only one volatility input type per row\n'
    else:
        mask = np.logical_and.reduce([mask_call_put, mask_strategy])
        df.loc[mask,'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-strategy quotes; input only one volatility input type per row\n'
        
    # Enforce numeric types for strategy quotes
    for col in valid_volatility_input_columns:
        bool_cond = df[col].apply(lambda x: not isinstance(x, (float, int)) and not pd.isna(x))
        df.loc[bool_cond,'errors'] += col + ' has an invalid type\n' 
        
    # Call and put quotes validation
    for col in (cols_call_put.to_list() + atm_column_names):
        if col in df.columns:
            bool_cond = df[col].apply(lambda x: isinstance(x, (float, int)) and x <= 0.0)
            df.loc[bool_cond,'errors'] += col + ' must have a positive value\n'  
            
            bool_cond = df[col].apply(lambda x: isinstance(x, (float, int)) and x > 2) # unlikely for volatility quote to be > 200% 
            df.loc[bool_cond,'warnings'] += col + ' value is unusually large' 
     
    for col in cols_strategy.to_list():
        bool_cond = df[col].apply(lambda x: isinstance(x, (float, int)) and abs(x) > 0.25) # unlikely for strategy spread to be > 25%
        df.loc[bool_cond,'warnings'] += col + ' value is unusually large' 
            
    def extract_numbers(df, suffix):
        pattern = rf'^σ_(\d{{1,2}})Δ{suffix}$'
        cols = df.filter(regex=pattern).columns
        return [col.split('_')[1].split('Δ')[0] for col in cols]
    
    Δ_list = list(set(extract_numbers(df, 'rr') + extract_numbers(df, 'bf')))

    for Δ in Δ_list:
        atm = 'σ_atmΔneutral'
        bf = 'σ_' + Δ + 'Δbf'
        rr = 'σ_' + Δ + 'Δrr'
        
        for v in ['call','put']:
            column_name = 'σ_' + Δ + 'Δ' + v
            if column_name not in df.columns:
                df[column_name] = np.nan
        
        for i,row in df.iterrows():

            if bf and rr in row.index:     
                if atm in row.index:
                    if pd.notna(row[bf]) and pd.notna(row[rr]) and pd.notna(row[atm]):
                        if isinstance(row[bf], (float, int)) and isinstance(row[rr], (float, int)) \
                            and isinstance(row[atm], (float, int)) and row[atm] > 0:
                                                      
                                df.at[i,'σ_' + Δ + 'Δcall'] = row[bf] + row[atm] + 0.5 * row[rr]
                                df.at[i,'σ_' + Δ + 'Δput'] = row[bf] + row[atm] - 0.5 * row[rr]
               
                if pd.isna(row[bf]) and pd.notna(row[rr]):
                    df.loc[i,'errors'] += bf + ' value is absent\n' # add comment if butterfly is n/a
                elif pd.isna(row[rr]) and pd.notna(row[bf]):
                    df.loc[i,'errors'] += rr + ' value is absent\n' # add comment if risk reversal is n/a
                elif (pd.notna(row[bf]) or pd.notna(row[rr])) and pd.isna(row[atm]):
                    df.loc[i,'errors'] += atm + ' value is absent\n' # add comment if at-the-money is n/a
                    
            elif bf in row.index and rr not in row.index:
                if rr not in row.index and pd.notna(row[bf]):
                    df.loc[i,'errors'] += bf + ' value is present but column ' + rr + ' is absent\n'
                if atm not in row.index and pd.notna(row[bf]):
                    df.loc[i,'errors'] += bf + ' value is present but column ' + atm + ' is absent\n'                    
                    
            elif rr in row.index and bf not in row.index:
                if bf not in row.index and pd.notna(row[rr]):
                    df.loc[i,'errors'] += rr + ' value is present but column ' + bf + ' is absent\n'
                if atm not in row.index and pd.notna(row[rr]):
                    df.loc[i,'errors'] += rr + ' value is present but column ' + atm + ' is absent\n'                         
          
    # Drop σ-strategy quote columns
    pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
    cols_to_drop = df.filter(regex=pattern2).columns
    df = df.drop(columns=cols_to_drop)
    
    # Drop all nan columns
    df = df.dropna(axis=1, how='all')
        
    return df



    
    
    
    