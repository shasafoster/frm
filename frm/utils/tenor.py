# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import re 
import unicodedata
import logging
import numpy as np
import pandas as pd
from pandas import DateOffset
from typing import Optional


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

#
# np.busday_offset(
#     dates=(df['effective_date'] + tenor_date_offset).to_numpy().astype('datetime64[D]'), offsets=0,
#     roll='following', busdaycal=busdaycal))

def workday(dates: [pd.Timestamp, pd.DatetimeIndex],
            offset: int,
            cal: np.busdaycalendar) -> pd.Timestamp:

    # Convert to datetime64[D] to use in np.busday_offset
    dates = dates.to_numpy().astype('datetime64[D]')
    return np.busday_offset(dates=dates, offsets=offset, roll='following', busdaycal=cal)




def check_day_offset_consistency(
        date: pd.Timestamp,
        offset_date: pd.Timestamp,
        day_offset: int,
        busdaycal: np.busdaycalendar):

    implied_offset = calc_implied_business_day_offset(date, offset_date, busdaycal)
    if implied_offset != day_offset:
        raise ValueError(f"The implied offset is {implied_offset} "
                         f"which is inconsistent with the provided day offset of {day_offset}.")


def calc_implied_business_day_offset(
        date: pd.Timestamp,
        offset_date: pd.Timestamp,
        cal: np.busdaycalendar) -> int:
    assert date >= offset_date, "date must be greater than or equal to offset_date"

    offset = 0
    while date < offset_date:
        offset += 1
        date += pd.DateOffset(days=1)
        date_np = date.to_numpy().astype('datetime64[D]')
        date = np.busday_offset(date_np, offsets=0, roll='following', busdaycal=cal)
    return offset


def resolve_fx_curve_dates(
        ccy_pair: str,
        cal: np.busdaycalendar,
        curve_date: Optional[pd.Timestamp] = None,
        spot_offset: Optional[int] = None,
        spot_date: Optional[pd.Timestamp] = None):

    if curve_date is not None and spot_date is not None and spot_offset is not None:
        # All three dates are specified, check for consistency
        check_day_offset_consistency(date=curve_date, offset_date=spot_date, offset=spot_offset, busdaycal=cal)
    elif spot_offset is None and curve_date is not None and spot_date is not None:
        # Spot offset is not specified, calculate it based on curve date and spot date
        spot_offset = calc_implied_business_day_offset(curve_date, spot_date, cal)
    else:
        # Only one of {curve_date, spot_date} are specified, get spot_offset per market convention from ccy_pair
        spot_offset = get_fx_spot_day_offset(ccy_pair)

    if spot_date is None:
        curve_date_np = curve_date.to_numpy().astype('datetime64[D]')
        spot_date = pd.Timestamp(
            np.busday_offset(curve_date_np, offsets=spot_offset, roll='following', busdaycal=cal))
    elif curve_date is None:
        spot_date_np = spot_date.to_numpy().astype('datetime64[D]')
        curve_date = pd.Timestamp(
            np.busday_offset(spot_date_np, offsets=-spot_offset, roll='preceding', busdaycal=cal))

    return curve_date, spot_offset, spot_date


def get_fx_spot_day_offset(validated_ccy_pair: str) -> int:

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


def get_ir_settlement_offset(ccy: str) -> int:
    if ccy in {'usd', 'cad'}:
        return 2
    elif ccy in {'aud', 'nzd', 'jpy', 'eur'}:
        return 2
    elif ccy in {'gbp'}:
        return 0
    else:
        return 2