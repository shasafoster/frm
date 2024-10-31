# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import re 
import unicodedata
import logging
import pandas as pd
from pandas import DateOffset
  

# def get_spot_offset(curve_ccy: str=None) -> int:
#
#     if curve_ccy is None:
#         return 2
#
#     if not isinstance(curve_ccy, str):
#         logging.error("function input 'curve_ccy' must be a string")
#         raise TypeError("function input 'curve_ccy' must be a string")
#
#     if len(curve_ccy) == 6:
#         # http://www.londonfx.co.uk/valdates.html
#         if curve_ccy in {'usdcad','cadusd',
#                          'usdphp','phpusd',
#                          'usdrub','rubusd',
#                          'usdtry','tryusd'}:
#             return 1
#         else:
#             return 2
#
#     # Spot offset for interest rate products
#     elif len(curve_ccy) == 3:
#         if curve_ccy in {'aud','cad'}:
#             return 1
#         elif curve_ccy in {'nzd','jpy','usd','eur'}:
#             return 2
#         elif curve_ccy in {'gbp'}:
#             return 0
#         else:
#             return 2
#
#     else:
#         return 2
#         #logging.error("invalid 'curve_ccy' value: " + curve_ccy)
#         #raise ValueError("invalid 'curve_ccy' value: " + curve_ccy)


def clean_tenor(tenor: str) -> str:
    if not isinstance(tenor, str):
        raise TypeError(f"'tenor' {tenor} must be a string. Instead is type {type(tenor)}")
    
    tenor = unicodedata.normalize('NFKD', tenor)
    tenor = tenor.lower().replace(' ','').replace('/','').replace('\n', '').replace('\r', '')

    replacements = {
        'd': ['days', 'day'],
        'w': ['weeks', 'week'],
        'm': ['months', 'month', 'mon'],
        'y': ['years', 'year', 'yrs', 'yr'],
        'on': ['overnight'],
        'tn': ['tomorrownext', 'tomnext'],
        'sw': ['spotweek'],
        'sn': ['spotnext'],
        'sp': ['spot']
    }

    pattern = re.compile('|'.join(map(re.escape, [val for sublist in replacements.values() for val in sublist])))
    tenor = pattern.sub(lambda match: next(k for k, v in replacements.items() if match.group(0) in v), tenor)

    return tenor


def tenor_to_date_offset(tenor: str) -> pd.DateOffset:    
    if not isinstance(tenor, str):
        raise TypeError("'tenor' must be a string")      
        
    misc_tenors_offset = {
        'on' : DateOffset(days=1), # 1 day (overnight)
        'tn' : DateOffset(days=2), # 2 days (tomorrow next)
        'sp'  : DateOffset(days=0), # spot date
        'sn' : DateOffset(days=1), # 1 day (spot next)
        'sw' : DateOffset(days=5), # 5 days (spot week)        
        }
    
    if tenor in misc_tenors_offset.keys():
        offset = misc_tenors_offset[tenor]
    else:
        # Identity tenors specified in integer days only; 1D, 30D , 360D
        if re.search('^\d+d$',tenor) is not None:
            offset = DateOffset(days=int(tenor[:-1]))
        # Identity tenors specified in integer weeks only; 1W, 52W, 104W
        elif re.search('^\d+w$',tenor) is not None:
            offset = DateOffset(weeks=int(tenor[:-1]))
        # Identity tenors specified in integer months only; 1M, 12M, 120M
        elif re.search('^\d+m$',tenor) is not None:
            offset = DateOffset(months=int(tenor[:-1]))       
        # Identity tenors specified in integer years only; 1Y, 10Y, 100Y
        elif re.search('^\d+y$',tenor) is not None:
            offset = DateOffset(years=int(tenor[:-1]))   
        # Identity tenors specified in integer years and integer monthly only; 1Y3M, 10Y6M, 100Y1M
        elif re.search('^\d+y\d+m$',tenor) is not None:
            years,months = tenor[:-1].split('y')
            total_months = int(years)*12 + int(months)
            offset = DateOffset(months=total_months)   
        else: 
            logging.error(f"invalid 'tenor' value: {tenor}")      
            raise ValueError(f"invalid 'tenor' value: {tenor}" )
        
    return offset


# @np.vectorize
# def offset_market_data_date(
#         curve_date: np.datetime64,
#         spot_date: np.datetime64,
#         tenor: str) -> np.datetime64:
#     # offset from
#     # (i) the curve_date for the ON and TN tenors and from;
#     # (ii) the spot date for all other tenors
#
#     date_offset = tenor_to_date_offset(tenor)
#
#     if tenor in {'on', 'tn'}: # This is not true for IRS with 1 day settlement delay
#         return curve_date + date_offset.item()
#     else:
#         return spot_date + date_offset.item()
#
#
# @np.vectorize
# def get_tenor_effective_date(tenor, curve_date, spot_date):
#     if tenor in {'on'}:
#         return curve_date
#     else:
#         return spot_date
#
#
#
#
#
# # FX vol quote to expiry, which is the difference between the tenor and the curve date
#
# def get_tenor_settlement_date(
#         curve_date: pd.Timestamp,
#         tenor: Union[str, np.ndarray],
#         busdaycal: np.busdaycalendar=np.busdaycalendar(),
#         ):
#
#     if isinstance(curve_date, dt.date):
#         curve_date = np.datetime64(curve_date)
#     else:
#         curve_date = np.datetime64(curve_date.date())
#
#     cleaned_tenor = clean_tenor(tenor)
#     if cleaned_tenor.size == 1:
#         cleaned_tenor = cleaned_tenor.item()
#
#     offset_date = offset_market_data_date(curve_date, spot_date, cleaned_tenor)
#
#     holiday_rolled_offset_date =
#
#     if holiday_rolled_offset_date.shape == ():
#         holiday_rolled_offset_date = pd.Timestamp(holiday_rolled_offset_date.item()) # For scalar
#     else:
#         holiday_rolled_offset_date = pd.DatetimeIndex(holiday_rolled_offset_date) # For array
#
#
#
#     return holiday_rolled_offset_date, cleaned_tenor, pd.Timestamp(spot_date)

