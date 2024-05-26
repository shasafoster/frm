# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from frm.frm.schedule.business_day_calendar import get_calendar
from frm.frm.schedule.tenor import calc_tenor_date, get_spot_offset
from frm.frm.schedule.daycounter import DayCounter

def convert_column_to_consistent_data_type(df: pd.DataFrame):
    for col in df.columns:
        if df[col].apply(isinstance, args=(float,)).all():
            df[col] = pd.to_numeric(df[col])
            
    return df


def copy_errors_and_warnings_to_input(df_processed, df_input):
    
    for v in ['errors','warnings']:
    
        internal_ids = df_processed.loc[df_processed[v] != '','internal_id'].to_list()
        for i in internal_ids:
            mask_processed = df_processed['internal_id'] == i
            mask_input = df_input['internal_id'] == i
            
            df_input.loc[mask_input,v] = df_processed.loc[mask_processed,v].iloc[0]
        
    return df_input


def clean_input_dataframe(df):    
    
    if 'internal_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'internal_id'})
        
    df = df.dropna(subset=[col for col in df.columns if col != 'internal_id'], how='all')
    df.reset_index(drop=True, inplace=True)
    
    df = df.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    if 'errors' not in df.columns:
        df.insert(0, 'errors', value='')
    if 'warnings' not in df.columns:
        df.insert(1, 'warnings', value='')
    
    return df


def move_col_after(df, col_to_move, ref_col):
    cols = df.columns.tolist()
    cols.insert(cols.index(ref_col) + 1, cols.pop(cols.index(col_to_move)))
    return df[cols]




def generic_market_data_input_cleanup_and_validation(df : pd.DataFrame,
                                                     spot_offset: bool=True):
    df_input = df.copy()

    # mandatory column validation  
    mandatory_columns = [
        'curve_date',
        'curve_ccy',
    ]
    df = df.dropna(axis=0, subset=mandatory_columns) # drop rows with blanks in mandatory columns
    missing_mandatory_columns = [col for col in mandatory_columns if col not in df.columns.to_list()]
    if len(missing_mandatory_columns) > 0:
        df['errors'] += f'missing mandatory columns: {missing_mandatory_columns}\n'
        return df

    # tenor input validation 
    if 'tenor_date' not in df.columns and 'tenor_name' not in df.columns:
        df['errors'] += f'a tenor input via tenor_name or tenor_date is mandatory\n'
    elif 'tenor_date' not in df.columns and 'tenor_name' in df.columns:
        df['tenor_date'] = np.nan
        df = move_col_after(df=df, col_to_move='tenor_date', ref_col='tenor_name')

    df['calendar'] = np.nan
    df['base_ccy'] = np.nan
    df['quote_ccy'] = np.nan
    df = move_col_after(df=df, col_to_move='base_ccy', ref_col='curve_ccy')
    df = move_col_after(df=df, col_to_move='quote_ccy', ref_col='curve_ccy')    
    
    # create a dictionary of all holiday calendars required
    curve_ccy_cal_dict = {}
    for curve_ccy in df['curve_ccy'].dropna().unique():
        if len(curve_ccy) == 3:
            curve_ccy_cal_dict[curve_ccy] = get_calendar(ccys=curve_ccy)
        elif len(curve_ccy) == 6:
            curve_ccy_cal_dict[curve_ccy] = get_calendar(ccys=[curve_ccy[:3],curve_ccy[3:]])    

    # row level validation
    for i,row in df.iterrows():
        
        field = 'curve_date'
        if not isinstance(pd.Timestamp(row[field]), pd.Timestamp):
            df.at[i,'errors'] += field + ' value is not a valid input\n'
            
        if pd.isna(row['tenor_name']) and pd.isna(row['tenor_date']):
            df.at[i,'errors'] += 'a tenor input in tenor_name or tenor_date is mandatory\n'
    
        if pd.isna(row['tenor_name']) and not pd.isna(row['tenor_date']):
            if not isinstance(pd.Timestamp(row['tenor_date']), pd.Timestamp):
                df.at[i,'errors'] += 'tenor_date' + ' value is not a valid input\n'

        field = 'curve_ccy'
        if not isinstance(row[field], str):
            df.at[i,'errors'] += field + ' value is not a valid input\n'
        else:
            if len(row[field]) not in {3,6}:
                df.at[i,'errors'] += field + ' value is not a valid input\n'
            else:
                if len(row[field]) == 3:
                    df.at[i,'calendar'] = curve_ccy_cal_dict[row[field]]
                elif len(row[field]) == 6:
                    df.at[i,'calendar'] = curve_ccy_cal_dict[row[field]]
                    df.at[i,'base_ccy'] = row[field][:3]
                    df.at[i,'quote_ccy'] = row[field][3:]  

    for i,row in df.iterrows():        
        if pd.isna(row['tenor_date']) and pd.notna(row['tenor_name']): 
            tenor_date, tenor_name_cleaned, spot_date = calc_tenor_date(row['curve_date'], row['tenor_name'], row['curve_ccy'], row['calendar'], spot_offset)
            df.at[i,'tenor_date'] = tenor_date
            df.at[i,'tenor_name'] = tenor_name_cleaned
        
    df = df.drop(['calendar'], axis=1)  
    
    if 'day_count_basis' not in df.columns:
        day_counter = DayCounter()
        df['day_count_basis'] = day_counter.day_count_basis
        df['tenor_years'] = day_counter.year_fraction(df['curve_date'], df['tenor_date'])
    else:
        df['tenor_years'] = np.nan
        for i,row in df.iterrows():
            day_counter = DayCounter(day_count_basis=row['day_count_basis'])
            df.at[i,'day_count_basis'] = day_counter.day_count_basis
            df.at[i,'tenor_years'] = day_counter.year_fraction(df.at[i,'curve_date'], df.at[i,'tenor_date'])
        
    df = move_col_after(df, 'day_count_basis', 'tenor_date')
    df = move_col_after(df, 'tenor_years', 'day_count_basis')
    
    # If column values are a consistent type, set the dataframe column type to that
    for col in df.columns:
        if df[col].apply(isinstance, args=(float,)).all():
            df[col] = pd.to_numeric(df[col])
    
    # In code we use Δ in all instances, not the word 'delta'
    df = df.applymap(lambda x: x.replace('delta', 'Δ') if isinstance(x, str) else x)
    df.columns = [col.replace('delta', 'Δ') for col in df.columns]             
                       
    return df




    