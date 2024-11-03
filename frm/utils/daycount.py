# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import calendar
import datetime as dt
import numpy as np
import pandas as pd
from frm.enums import DayCountBasis

            
def convert_to_same_shape_DatetimeIndex(start_date, end_date):
    start_dti = to_datetimeindex(start_date)
    end_dti = to_datetimeindex(end_date)
    
    if len(start_dti) == 1 and len(end_dti) == 1:
        scalar_output = True
    else:
        scalar_output = False
    
    if len(start_dti) == 1 and len(end_dti) > 1:
        start_dti = pd.DatetimeIndex([start_dti.values[0] for _ in range(len(end_dti))])
    elif len(start_dti) > 1 and len(end_dti) == 1:
        end_dti = pd.DatetimeIndex([end_dti.values[0] for _ in range(len(start_dti))])
    
    return start_dti, end_dti, scalar_output


def day_count(start_date,
              end_date, 
              day_count_basis: DayCountBasis,
              is_end_date_on_termination: bool=None)->np.array:
    # If the start_date and end_date are scalars assumption is end_date is the termination date
    # If end_date is a vector, the final value of the vector is assumed to be the termination date
    
    # References
    # [1] The excel file "30-360-2006ISDADefs" sourced from https://www.isda.org/2008/12/22/30-360-day-count-conventions/
    #     Saved to  WayBackMachine on 23 September 2024, https://web.archive.org/web/20240923055727/https://www.isda.org/2008/12/22/30-360-day-count-conventions/
        
    start_dti, end_dti, scalar_output = convert_to_same_shape_DatetimeIndex(start_date, end_date)
    
    assert (start_dti <= end_dti).all()
    
    if day_count_basis in {DayCountBasis.ACT_360, DayCountBasis.ACT_365, DayCountBasis.ACT_ACT, DayCountBasis.ACT_366}:
        # Act = the actual number of days between the dates
        result = (end_dti - start_dti).days.values
    
    elif day_count_basis == DayCountBasis._30_360:
        # Logic for "30/360" / "360/360" / "Bond Basis" is defined in tab "30-360 Bond Basis" in reference [1]
        
        # If (DAY1=31), Set D1=30, Otherwise set D1=DAY1
        DAY1 = start_dti.day
        d1 = np.where(DAY1 == 31, 30, DAY1) 	
        
        # If (DAY2=31) and (DAY1=30 or 31), Then set D2=30, Otherwise set D2=DAY2	
        DAY2 = end_dti.day
        mask = np.logical_and(d1 == 30, DAY2 == 31)
        d2 = np.where(mask, 30, DAY2)
        
        result = 360*(end_dti.year - start_dti.year) \
               + 30*(end_dti.month - start_dti.month) \
               + d2 - d1
        result = result.values
    
    elif day_count_basis == DayCountBasis._30E_360:
        # Logic for "30/360E" / "Eurobond Basis" is defined in tab "30E-360 Eurobond" in reference [1]
        
        # If (DAY1=31), Set D1=30, Otherwise set D1=DAY1
        DAY1 = start_dti.day
        d1 = np.where(DAY1 == 31, 30, DAY1) 	            

        # If (DAY2=31), Then set D2=30, Otherwise set D2=DAY2	
        DAY2 = end_dti.day
        d2 = np.where(DAY2 == 31, 30, DAY2) 	
        
        result = 360 * (end_dti.year - start_dti.year) \
               + 30 * (end_dti.month - start_dti.month) \
               + d2 - d1
        result = result.values
               
    elif day_count_basis == DayCountBasis._30E_360_ISDA:
        # Logic for 30E/360 (ISDA)" is defined in tab "30E-360 ISDA" in [1]
        
        # Returns TRUE/FALSE if datetime is the last day of February
        def _is_last_day_of_feb(datetime):
            if datetime.month == 2:
                _, last_of_month = calendar.monthrange(datetime.year, datetime.month)
                if datetime.day == last_of_month:
                    return True
            return False

        def _apply_logic(start_date, end_date, is_end_date_on_termination):
            # If (DAY1=31) or (DAY1 is last day of February), Set D1=30, Otherwise set D1=DAY1	
            if start_date.day == 31 or _is_last_day_of_feb(start_date):
                d1 = 30
            else:
                d1 = start_date.day
    
            # If (DAY2=31) or (DAY2 is last day of February but not the Termination Date), Then set D2=30, Otherwise set D2=DAY2
            if end_date.day == 31 or (_is_last_day_of_feb(end_date) and not is_end_date_on_termination):
                d2 = 30
            else:
                d2 = end_date.day
    
            result = 360 * (end_date.year - start_date.year) \
                     + 30 * (end_date.month - start_date.month) \
                     + d2 - d1
            return result
        
        if is_end_date_on_termination is None:
            N = len(end_dti)
            is_end_date_on_termination = [False if i < (N-1)  else True for i in range(N)]
        else:
            is_end_date_on_termination = np.atleast_1d(is_end_date_on_termination)

        result = np.array([_apply_logic(start, end, flag) for start, end, flag in zip(start_dti, end_dti, is_end_date_on_termination)])               
    else:
        raise ValueError
    
    if scalar_output:
        return result.item()
    else:
        return result


def year_frac(start_date,
              end_date,
              day_count_basis: DayCountBasis,
              is_end_date_on_termination: bool=None)->np.array:
    
    # If the start_date and end_date are scalars assumption is end_date is the termination date
    # If end_date is a vector, the final value of the vector is assumed to be the termination date

    if day_count_basis in {DayCountBasis._30_360, DayCountBasis._30E_360, DayCountBasis._30E_360_ISDA, DayCountBasis.ACT_360}:
        return day_count(start_date, end_date, day_count_basis, is_end_date_on_termination) / 360.0
    
    elif day_count_basis == DayCountBasis.ACT_365:
        return day_count(start_date, end_date, day_count_basis, is_end_date_on_termination) / 365.0        

    elif day_count_basis == DayCountBasis.ACT_366:
        return day_count(start_date, end_date, day_count_basis, is_end_date_on_termination) / 366.0 
    
    elif day_count_basis == DayCountBasis.ACT_ACT:
        start_dti, end_dti, scalar_output  = convert_to_same_shape_DatetimeIndex(start_date, end_date)
        assert (start_dti <= end_dti).all()
        
        start_year = start_dti.year
        end_year = end_dti.year
        year_1_diff = 365 + start_year.map(calendar.isleap)
        year_2_diff = 365 + end_year.map(calendar.isleap) 
        
        total_sum = end_year - start_year - 1
        diff_first = pd.DatetimeIndex([dt.datetime(v + 1, 1, 1) for v in start_year]) - start_dti
        total_sum += diff_first.days / year_1_diff
        diff_second = end_dti - pd.DatetimeIndex([dt.datetime(v, 1, 1) for v in end_year]) 
        total_sum += diff_second.days / year_2_diff

        if scalar_output:
            return total_sum.item()
        else:
            return total_sum
    else:
        raise ValueError


def to_datetimeindex(date_object) -> 'pd.DatetimeIndex':
    """
    Converts a date-like object to a pandas DatetimeIndex.

    Parameters:
    - date_object: Can be of type pd.DatetimeIndex, pd.Timestamp, np.datetime64, dt.date, dt.datetime,
                   pd.Series, or list. Unsupported types will raise a ValueError.

    Returns:
    - pd.DatetimeIndex: The corresponding DatetimeIndex for the provided date-like object.

    Raises:
    - ValueError: If the input type is not supported.

    Supported types:
    - pd.DatetimeIndex: Returned directly.
    - pd.Timestamp: Converted to a DatetimeIndex containing one element.
    - np.datetime64: Converted to a DatetimeIndex containing one element.
    - dt.date / dt.datetime: Converted to a DatetimeIndex containing one element.
    - pd.Series or list: Converted to a DatetimeIndex.
    """        
    if isinstance(date_object, pd.DatetimeIndex):
        return date_object
    if isinstance(date_object, pd.Timestamp):
        return pd.DatetimeIndex([date_object.to_pydatetime()])
    elif isinstance(date_object, np.datetime64):
        return pd.DatetimeIndex([pd.to_datetime(date_object)])
    elif isinstance(date_object, dt.date) or isinstance(date_object, dt.datetime):
        return pd.DatetimeIndex([dt.datetime(date_object.year,date_object.month,date_object.day)])
    elif isinstance(date_object, pd.Series) or isinstance(date_object, list):
        return pd.DatetimeIndex(date_object)
    else:
        raise ValueError("Unsupported type", type(date_object), date_object)
      


if __name__ == "__main__":
    # Example usage
    start_date = pd.date_range(start='2024-01-01', end='2024-12-01', freq='ME')
    end_date = pd.date_range(start='2024-02-29', end='2024-12-31', freq='ME')
    
    day_count_basis = DayCountBasis.from_value('act/365')
    days = day_count(start_date, end_date, day_count_basis)
    years = year_frac(start_date, end_date, day_count_basis)
    
    for d1, d2, day_count, year_frac in zip(start_date, end_date, days, years):
        print(d1.date(), d2.date(), day_count.item(), round(year_frac.item(),6))

