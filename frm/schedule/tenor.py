# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

import re 
import unicodedata
import logging
import numpy as np
import pandas as pd
from pandas import DateOffset
from typing import Union
import datetime as dt
  

def get_spot_offset(curve_ccy: str=None) -> int:
    
    if curve_ccy == None:
        return 2
    
    if not isinstance(curve_ccy, str):
        logging.error("function input 'curve_ccy' must be a string")
        raise TypeError("function input 'curve_ccy' must be a string")    
    
    if len(curve_ccy) == 6:
        # http://www.londonfx.co.uk/valdates.html
        if curve_ccy in {'usdcad','cadusd',
                         'usdphp','phpusd',
                         'usdrub','rubusd',
                         'usdtry','tryusd'}:
            return 1
        else:
            return 2

    # Spot offset for interest rate products
    elif len(curve_ccy) == 3:    
        if curve_ccy in {'aud','cad'}:
            return 1
        elif curve_ccy in {'nzd','jpy','usd','eur'}:
            return 2
        elif curve_ccy in {'gbp'}:
            return 0
        else:
            return 2
        
    else:
        return 2
        #logging.error("invalid 'curve_ccy' value: " + curve_ccy)      
        #raise ValueError("invalid 'curve_ccy' value: " + curve_ccy)


@np.vectorize
def clean_tenor(tenor: str) -> str:
    if not isinstance(tenor, str):
        logging.error("function input 'tenor' must be a string")
        raise TypeError("function input 'tenor' must be a string")      
    
    tenor = unicodedata.normalize('NFKD', tenor)

    tenor = tenor.lower().replace(' ','').replace('/','').replace('\n', '').replace('\r', '')
    
    tenor = tenor.replace('days','d')
    tenor = tenor.replace('day','d')
    tenor = tenor.replace('weeks','w')
    tenor = tenor.replace('week','w')
    tenor = tenor.replace('months','m')
    tenor = tenor.replace('month','m')
    tenor = tenor.replace('mon','m')
    tenor = tenor.replace('years','y')    
    tenor = tenor.replace('year','y')
    tenor = tenor.replace('yrs','y')        
    tenor = tenor.replace('yr','y')     
    
    tenor = tenor.replace('overnight','on') 
    tenor = tenor.replace('tomorrownext','tn')
    tenor = tenor.replace('tomnext','sn')  
    tenor = tenor.replace('spotweek','sw')    
    tenor = tenor.replace('spotnext','sn')     
    tenor = tenor.replace('spot','sp') 
    return tenor 
   

@np.vectorize
def tenor_name_to_date_offset(tenor_name: str) -> pd.DateOffset:    
    if not isinstance(tenor_name, str):
        logging.error("function input 'tenor' must be a string")
        raise TypeError("function input 'tenor' must be a string")      
        
    misc_tenors_offset = {
        'on' : DateOffset(days=1), # 1 day (overnight)
        'tn' : DateOffset(days=2), # 2 days (tomorrow next)    
        'sp'  : DateOffset(days=0), # spot date
        'sn' : DateOffset(days=1), # 1 day (spot next)
        'sw' : DateOffset(days=5), # 5 days (spot week)        
        }
    
    if tenor_name in misc_tenors_offset.keys():
        offset = misc_tenors_offset[tenor_name]
    else:
        # Identity tenors specified in integer days only; 1D, 30D , 360D
        if re.search('^\d+d$',tenor_name) is not None:
            offset = DateOffset(days=int(tenor_name[:-1]))
        # Identity tenors specified in integer weeks only; 1W, 52W, 104W
        elif re.search('^\d+w$',tenor_name) is not None:
            offset = DateOffset(weeks=int(tenor_name[:-1]))
        # Identity tenors specified in integer months only; 1M, 12M, 120M
        elif re.search('^\d+m$',tenor_name) is not None:
            offset = DateOffset(months=int(tenor_name[:-1]))       
        # Identity tenors specified in integer years only; 1Y, 10Y, 100Y
        elif re.search('^\d+y$',tenor_name) is not None:
            offset = DateOffset(years=int(tenor_name[:-1]))   
        # Identity tenors specified in integer years and integer monthly only; 1Y3M, 10Y6M, 100Y1M
        elif re.search('^\d+y\d+m$',tenor_name) is not None:
            years,months = tenor_name[:-1].split('y')
            total_months = int(years)*12 + int(months)
            offset = DateOffset(months=total_months)   
        else: 
            logging.error("invalid 'tenor' value: " + tenor_name)      
            raise ValueError("invalid 'tenor' value: " + tenor_name)
        
    return offset

@np.vectorize
def offset_market_data_date(curve_date: np.datetime64,
                spot_date: np.datetime64,
                tenor_name: str) -> np.datetime64:    
    
    # offset from 
    # (i) the curve_date for the ON and TN tenors and from;
    # (ii) the spot date for all other tenors   
    
    date_offset = tenor_name_to_date_offset(tenor_name)
    
    if tenor_name in {'on'}:
        return curve_date + date_offset.item() 
    else:
        return spot_date + date_offset.item()

@np.vectorize
def get_tenor_effective_date(tenor_name, curve_date, spot_date):
    if tenor_name in {'on'}:
        return curve_date 
    else:
        return spot_date 

def calc_tenor_date(curve_date: pd.Timestamp,
                    tenor_name: Union[str, np.ndarray], 
                    curve_ccy: str=None, 
                    holiday_calendar: np.busdaycalendar=None,
                    spot_offset: bool=True):
        
    if holiday_calendar == None or pd.isna(holiday_calendar):
        holiday_calendar = np.busdaycalendar()
    
    if type(curve_date) == dt.date:
        curve_date = np.datetime64(curve_date)
    else:
        curve_date = np.datetime64(curve_date.date())
            
    if spot_offset:
        spot_date = np.busday_offset(curve_date, offsets=get_spot_offset(curve_ccy), roll='following', busdaycal=holiday_calendar)
    else: 
        spot_date = curve_date
        
    cleaned_tenor_name = clean_tenor(tenor_name)
    if cleaned_tenor_name.size == 1:
        cleaned_tenor_name = cleaned_tenor_name.item()
    
    offset_date = offset_market_data_date(curve_date, spot_date, cleaned_tenor_name)
    
    holiday_rolled_offset_date = np.busday_offset(offset_date.astype('datetime64[D]'), offsets=0, roll='following', busdaycal=holiday_calendar)
    
    if holiday_rolled_offset_date.shape == ():
        holiday_rolled_offset_date = pd.Timestamp(holiday_rolled_offset_date.item()) # For scalar
    else:
        holiday_rolled_offset_date = pd.DatetimeIndex(holiday_rolled_offset_date) # For array
    
    
    
    return holiday_rolled_offset_date, cleaned_tenor_name, pd.Timestamp(spot_date)  
        
