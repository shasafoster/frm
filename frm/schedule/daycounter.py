# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import calendar
import datetime as dt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Literal, Union

VALID_DAY_COUNT_BASIS = ['30/360','30e/360','30e/360_isda','act/360','act/365','act/act']
VALID_DAY_COUNT_BASIS_TYPES = Union[tuple(Literal[i] for i in VALID_DAY_COUNT_BASIS)]


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
    if type(date_object) == pd.DatetimeIndex:
        return date_object
    if type(date_object) == pd.Timestamp:
        return pd.DatetimeIndex([date_object.to_pydatetime()])
    elif type(date_object) == np.datetime64:
        return pd.DatetimeIndex([pd.to_datetime(date_object)])
    elif type(date_object) == dt.date or type(date_object) == dt.datetime:
        return pd.DatetimeIndex([dt.datetime(date_object.year,date_object.month,date_object.day)])
    elif type(date_object) == pd.Series or type(date_object) == list:
        return pd.DatetimeIndex(date_object)
    else:
        raise ValueError("Unsupported type", type(date_object), date_object)
      
        
@dataclass
class DayCounter:
    day_count_basis: Optional[VALID_DAY_COUNT_BASIS_TYPES] = 'act/act'
    VALID_DAY_COUNT_BASIS = VALID_DAY_COUNT_BASIS 
    
    # References
    # [1] The excel file "30-360-2006ISDADefs" sourced from https://www.isda.org/2008/12/22/30-360-day-count-conventions/
    #     Saved to  WayBackMachine on 23 September 2024, https://web.archive.org/web/20240923055727/https://www.isda.org/2008/12/22/30-360-day-count-conventions/
    
    def __post_init__(self):
        if self.day_count_basis in [None, np.nan, '']:
            self.day_count_basis = 'act/act' 
        elif self.day_count_basis not in self.VALID_DAY_COUNT_BASIS:
            raise ValueError(f"Invalid day_count_basis. Must be one of {VALID_DAY_COUNT_BASIS}")

    def clean_up(self, start_date, end_date):
        start_date_dtidx = to_datetimeindex(start_date)
        end_date_dtidx = to_datetimeindex(end_date)
        
        if len(start_date_dtidx) == 1 or len(end_date_dtidx) == 1:
            scalar_output = True
        else:
            scalar_output = False
        
        if len(start_date_dtidx) == 1 and len(end_date_dtidx) > 1:
            start_date_dtidx = pd.DatetimeIndex([start_date_dtidx.values[i] for i in range(len(end_date_dtidx))])
        elif len(start_date_dtidx) > 1 and len(end_date_dtidx) == 1:
            end_date_dtidx = pd.DatetimeIndex([end_date_dtidx.values[i] for i in range(len(start_date_dtidx))])
        
        return start_date_dtidx, end_date_dtidx, scalar_output


    def day_count(self, 
                  start_date,
                  end_date, 
                  is_end_date_on_termination: bool=None)->np.array:
        # If the start_date and end_date are scalars assumption is end_date is the termination date
        # If end_date is a vector, the final value of the vector is assumed to be the termination date
        
        start_DatetimeIndex, end_DatetimeIndex, scalar_output = self.clean_up(start_date, end_date)
        
        assert (start_DatetimeIndex <= end_DatetimeIndex).all()
        
        if self.day_count_basis in {'act/360','act/365','act/act'}:
            # Act = the actual number of days between the dates
            result = (end_DatetimeIndex - start_DatetimeIndex).days.values
        
        elif self.day_count_basis == '30/360':
            # Logic for "30/360" / "360/360" / "Bond Basis" is defined in tab "30-360 Bond Basis" in reference [1]
            
            # If (DAY1=31), Set D1=30, Otherwise set D1=DAY1
            DAY1 = start_DatetimeIndex.day
            d1 = np.where(DAY1 == 31, 30, DAY1) 	
            
            # If (DAY2=31) and (DAY1=30 or 31), Then set D2=30, Otherwise set D2=DAY2	
            DAY2 = end_DatetimeIndex.day
            mask = np.logical_and(d1 == 30, DAY2 == 31)
            d2 = np.where(mask, 30, DAY2)
            
            result = 360*(end_DatetimeIndex.year - start_DatetimeIndex.year) \
                   + 30*(end_DatetimeIndex.month - start_DatetimeIndex.month) \
                   + d2 - d1
            result = result.values
        
        elif self.day_count_basis == '30e/360':
            # Logic for "30/360E" / "Eurobond Basis" is defined in tab "30E-360 Eurobond" in reference [1]
            
            # If (DAY1=31), Set D1=30, Otherwise set D1=DAY1
            DAY1 = start_DatetimeIndex.day
            d1 = np.where(DAY1 == 31, 30, DAY1) 	            

            # If (DAY2=31), Then set D2=30, Otherwise set D2=DAY2	
            DAY2 = end_DatetimeIndex.day
            d2 = np.where(DAY2 == 31, 30, DAY2) 	
            
            result = 360 * (end_DatetimeIndex.year - start_DatetimeIndex.year) \
                   + 30 * (end_DatetimeIndex.month - start_DatetimeIndex.month) \
                   + d2 - d1
            result = result.values
                   
        elif self.day_count_basis == '30e/360_isda':
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
            
            if is_end_date_on_termination == None:
                N = len(end_DatetimeIndex)
                is_end_date_on_termination = [False if i < (N-1)  else True for i in range(N)]
            else:
                is_end_date_on_termination = np.atleast_1d(is_end_date_on_termination)

            result = np.array([_apply_logic(start, end, flag) for start, end, flag in zip(start_DatetimeIndex, end_DatetimeIndex, is_end_date_on_termination)])               
        else:
            raise ValueError
        
        if scalar_output:
            return result.item()
        else:
            return result


    def year_fraction(self, 
                      start_date,
                      end_date,
                      is_end_date_on_termination: bool=None)->np.array:
        
        # If the start_date and end_date are scalars assumption is end_date is the termination date
        # If end_date is a vector, the final value of the vector is assumed to be the termination date

        if self.day_count_basis in {'30/360','30e/360','30e/360_isda', 'act/360'}:
            return self.day_count(start_date, end_date, is_end_date_on_termination) / 360.0
        
        elif self.day_count_basis == 'act/365':
            return self.day_count(start_date, end_date, is_end_date_on_termination) / 365.0        
        
        elif self.day_count_basis == 'act/act':
            start_DatetimeIndex, end_DatetimeIndex, scalar_output  = self.clean_up(start_date, end_date)
            assert (start_DatetimeIndex <= end_DatetimeIndex).all()
            
            start_year = start_DatetimeIndex.year
            end_year = end_DatetimeIndex.year
            year_1_diff = 365 + start_year.map(calendar.isleap)
            year_2_diff = 365 + end_year.map(calendar.isleap) 
            
            total_sum = end_year - start_year - 1
            diff_first = pd.DatetimeIndex([dt.datetime(v + 1, 1, 1) for v in start_year]) - start_DatetimeIndex
            total_sum += diff_first.days / year_1_diff
            diff_second = end_DatetimeIndex - pd.DatetimeIndex([dt.datetime(v, 1, 1) for v in end_year]) 
            total_sum += diff_second.days / year_2_diff

            if scalar_output:
                return total_sum.item()
            else:
                return total_sum



if __name__ == "__main__":
    # Example usage
    start_date = pd.date_range(start='2024-01-01', end='2024-12-01', freq='ME')
    end_date = pd.date_range(start='2024-02-29', end='2024-12-31', freq='ME')
    
    daycounter_obj = DayCounter('act/365')
    days = daycounter_obj.day_count(start_date, end_date)
    years = daycounter_obj.year_fraction(start_date, end_date)
    
    for d1, d2, day_count, year_frac in zip(start_date, end_date, days, years):
        print(d1.date(), d2.date(), day_count.item(), round(year_frac.item(),6))

