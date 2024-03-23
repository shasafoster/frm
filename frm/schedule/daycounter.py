# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())


import calendar
import datetime as dt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Literal, Union

# VALID_DAY_COUNT_BASIS_TYPES = Union[Literal['30/360'],
#                               Literal['30e/360'], 
#                               Literal['30e/360 isda'], 
#                               Literal['act/360'], 
#                               Literal['act/365'], 
#                               Literal['act/act']]

VALID_DAY_COUNT_BASIS = ['30/360','30e/360','30e/360_isda','act/360','act/365','act/act']
VALID_DAY_COUNT_BASIS_TYPES = Union[tuple(Literal[i] for i in VALID_DAY_COUNT_BASIS)]

@dataclass
class DayCounter:
    day_count_basis: Optional[VALID_DAY_COUNT_BASIS_TYPES] = 'act/act'
    VALID_DAY_COUNT_BASIS = VALID_DAY_COUNT_BASIS 
    
    def __post_init__(self):
        if self.day_count_basis in [None, np.nan, '']:
            self.day_count_basis = 'act/act' 
        elif self.day_count_basis not in self.VALID_DAY_COUNT_BASIS:
            raise ValueError(f"Invalid day_count_basis. Must be one of {VALID_DAY_COUNT_BASIS}")
           
    def to_datetime_or_datetimeindex(self, date_object) -> 'dt.datetime, pd.DatetimeIndex':
        if type(date_object) == pd.Timestamp:
            return date_object.to_pydatetime()
        elif type(date_object) == np.datetime64:
            return pd.to_datetime(date_object)
        elif type(date_object) == dt.date:
            return dt.datetime(date_object.year,date_object.month,date_object.day)
        elif type(date_object) == pd.Series or type(date_object) == list:
            return pd.DatetimeIndex(date_object)
        else:
            return date_object


    def clean_up(self, start_date, end_date):
        start_date = self.to_datetime_or_datetimeindex(start_date)
        end_date = self.to_datetime_or_datetimeindex(end_date)
        
        if type(start_date) == dt.datetime and type(end_date) == pd.DatetimeIndex:
            start_date = pd.DatetimeIndex([start_date for _ in range(len(end_date))])
        elif type(start_date) == pd.DatetimeIndex and type(end_date) == dt.datetime:
            end_date = pd.DatetimeIndex([end_date for _ in range(len(start_date))])
        
        return start_date, end_date

    def day_count(self, 
                  start_date,
                  end_date, 
                  is_end_date_on_termination: bool=False):
        
        start_date, end_date = self.clean_up(start_date, end_date)
        
        if self.day_count_basis in {'act/360','act/365','act/act'}:
            return (end_date - start_date).days
        
        elif self.day_count_basis == '30/360':
            d1 = np.minimum(30, start_date.day)
            
            # these lines need to work for dates and datetimeindex
            #d2 = np.minimum(d1, end_date.day) if d1 == 30 else end_date.day
            d2 = end_date.day.values
            idx = d1 == 30
            d2[idx] = np.minimum(d1, end_date.day).values[idx]
                 
            return 360*(end_date.year - start_date.year)\
                   + 30*(end_date.month - start_date.month)\
                   + d2 - d1
        
        elif self.day_count_basis == '30e/360':
            d1 = np.minimum(30, pd.Series(start_date.day))
            d2 = np.minimum(30, pd.Series(end_date.day))
            return 360 * (pd.Series(end_date.year) - pd.Series(start_date.year)) \
                   + 30 * (pd.Series(end_date.month) - pd.Series(start_date.month)) \
                   + d2 - d1
                   

        elif self.day_count_basis == '30e/360 isda':
    
            def _is_last_day_of_feb(date):
                if date.month == 2:
                    _, last_of_month = calendar.monthrange(date.year, date.month)
                    if date.day == last_of_month:
                        return True
                return False

            if start_date.day == 31 or _is_last_day_of_feb(start_date):
                d1 = 30
            else:
                d1 = start_date.day

            if end_date.day == 31 or (_is_last_day_of_feb(end_date)
                                     and not is_end_date_on_termination):
                d2 = 30
            else:
                d2 = end_date.day

            return 360 * (end_date.year - start_date.year) \
                   + 30 * (end_date.month - start_date.month) \
                   + d2 - d1




    def year_fraction(self, 
                      start_date,
                      end_date,
                      is_end_date_on_termination: bool=False):
        
       

        if self.day_count_basis in {'30/360','30e/360','30e/360 isda', 'act/360'}:
            return self.day_count(start_date, end_date, is_end_date_on_termination) / 360.0
        
        elif self.day_count_basis == 'act/act':
            start_date, end_date = self.clean_up(start_date, end_date)
            
            start_year = pd.Series(start_date.year)
            end_year = pd.Series(end_date.year)
            year_1_diff = 365 + start_year.map(calendar.isleap)
            year_2_diff = 365 + end_year.map(calendar.isleap) 
            
            total_sum = end_year - start_year - 1
            diff_first = pd.Series([dt.datetime(v + 1, 1, 1) for v in start_year]) - start_date
            total_sum += diff_first.dt.days / year_1_diff
            diff_second = end_date - pd.Series([dt.datetime(v, 1, 1) for v in end_year]) 
            total_sum += diff_second.dt.days / year_2_diff

            if type(start_date) is pd.DatetimeIndex:
                return total_sum
            elif type(start_date) is dt.datetime:
                return total_sum[0]

        elif self.day_count_basis == 'act/365':
            return self.day_count(start_date, end_date, is_end_date_on_termination) / 365.0
        