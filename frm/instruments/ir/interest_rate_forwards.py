# -*- coding: utf-8 -*-


import os
import pathlib

if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) # path to ./frm/ 
    print(__file__.split('\\')[-1], os.getcwd()) 

import numpy as np
import pandas as pd    
from dataclasses import dataclass, field, InitVar 
from typing import Optional
import datetime as dt
import calendar

from market_data.iban_ccys import VALID_CCYS
from schedule.business_day_calendar import get_calendar
from schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS
        
#%%    

#import warnings
#warnings.formatwarning = custom_formatwarning


def isIMMDate(date: dt.date):
           # Returns boolean on if date is an IMM date
           # IMM (International Monetary Market) dates are the third Wednesday of March, June, September and December 
           # (i.e., between the 15th and 21st, whichever such day is a Wednesday).
           if date.month in {3,6,9,12} and 15 <= date.day <= 21 and date.weekday() == 2:
               return True
           else:
               return False


def getIMMDate(year: int, month: int):
    # Get date of the 3rd Wednesday in the specified month and year 
    c = calendar.Calendar()
    monthcal = c.monthdatescalendar(year,month)
    
    imm_date = [day for week in monthcal for day in week if \
                     day.weekday() == calendar.WEDNESDAY and \
                     day.month == month][2]
    return imm_date 


@dataclass
class IRFuture():
    effective_date: Optional[pd.Timestamp]=None
    maturity_date: Optional[pd.Timestamp]=None
    price: Optional[float] = None
    futures_interest_rate: Optional[float] = None
    price_date: Optional[pd.Timestamp] = None
    convexity: float = 0.0

    day_count_basis: InitVar[Optional[VALID_DAY_COUNT_BASIS]] = None
    imm_delivery_year: InitVar[Optional[int]] = None
    imm_delivery_month: InitVar[Optional[int]] = None
    
    daycounter: DayCounter = field(init=False)
    forward_interest_rate: float = field(init=False)
    

    def __post_init__(self, day_count_basis, imm_delivery_year, imm_delivery_month):
        
        self.daycounter = DayCounter(day_count_basis)
        
        if self.effective_date is None and self.maturity_date is None \
            and self.imm_delivery_year is not None and self.imm_delivery_month is not None:
            maturity_year = imm_delivery_year
            maturity_month = imm_delivery_month
            effective_month = 12 if maturity_month == 3 else maturity_month - 3
            effective_year = maturity_year - 1 if maturity_month == 3 else maturity_year 
            
            self.effective_date = getIMMDate(effective_year,effective_month)
            self.maturity_date = getIMMDate(maturity_year,maturity_month)
        
        #if not isIMMDate(self.effective_date): warnings.warn("UserWarning: " + "effective_date " + self.effective_date.strftime('%Y-%m-%d-%a') + " is not an IMM date",UserWarning)
        #if not isIMMDate(self.maturity_date): warnings.warn("UserWarning: " + "maturity_date " + self.maturity_date.strftime('%Y-%m-%d-%a') + " is not an IMM date",UserWarning)
        if self.price is None and self.futures_interest_rate is None: raise ValueError("'price' or 'futures_interest_rate' must be specified")
        if self.effective_date >= self.maturity_date: raise ValueError("'effective_date' is equal to or after the 'maturity_date'")
        if self.price_date is not None and self.price_date > self.maturity_date: raise ValueError("'price_date' is equal to or after the 'maturity_date'")
        if self.price is None: self.price = 100 - self.futures_interest_rate * 100
        if self.futures_interest_rate is None: self.futures_interest_rate = (100 - self.price) / 100
        
        self.forward_interest_rate = self.futures_interest_rate + self.convexity    
        

@dataclass
class FRA():
    contract_rate: float
    transaction_date: pd.Timestamp  
    forward_starting: int = 2
    spot_date: Optional[pd.Timestamp]=None  
    wait_period: Optional[int]=None 
    contract_period: Optional[int]=None 
    effective_date: Optional[pd.Timestamp]=None
    maturity_date: Optional[pd.Timestamp]=None
    currency_holidays: Optional[VALID_CCYS]=None
    city_holidays: Optional[str]=None
    
    day_count_basis: InitVar[Optional[VALID_DAY_COUNT_BASIS]] = None
    fra_str: InitVar[Optional[str]] = None
    holiday_calendar: InitVar[Optional[np.busdaycalendar]]=None
    
    def __post_init__(self, day_count_basis, fra_str, holiday_calendar):

        if holiday_calendar is None: 
            holiday_calendar = get_calendar(self.currency_holidays, self.city_holidays)         

        if self.spot_date is None: 
            date = np.datetime64(self.transaction_date + pd.DateOffset(days=self.forward_starting),'D')
            np.busday_offset(dates=date, offsets=0, roll='following', busdaycal=holiday_calendar)
            
        if fra_str is not None:
            contract_period, wait_period = fra_str.upper().split('X')
            contract_period = int(contract_period)
            wait_period = int(wait_period)
        
        if self.effective_date is None and self.maturity_date is None:
            self.effective_date = self.spot_date + pd.DateOffset(months=self.wait_period) 
            self.maturity_date = self.spot_date + pd.DateOffset(months=self.contract_period)
                    
        self.daycounter = DayCounter(day_count_basis)


