# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 
    
    
from enum import Enum
import numpy as np
import pandas as pd
from frm.utils.enums import PeriodFrequency, CompoundingFrequency, DayCountBasis, RollConvention
from frm.utils.daycount import year_fraction
from frm.utils.schedule import schedule
from frm.utils.business_day_calendar import get_busdaycal



def zero_rate_from_discount_factor(
        years, 
        discount_factor, 
        compounding_frequency: CompoundingFrequency):
    
    if compounding_frequency == CompoundingFrequency.CONTINUOUS:
            return (- np.log(discount_factor) / years)    
    elif compounding_frequency == CompoundingFrequency.SIMPLE:
        return (((1.0 / discount_factor) - 1.0) / years)       
    elif compounding_frequency in {
            CompoundingFrequency.DAILY, 
            CompoundingFrequency.MONTHLY,
            CompoundingFrequency.QUARTERLY,
            CompoundingFrequency.SEMIANNUAL,
            CompoundingFrequency.ANNUAL}:
        periods_per_year = compounding_frequency.periods_per_year
        return periods_per_year * ((1.0 / discount_factor) ** (1.0 / (periods_per_year * years)) - 1.0)
    else:
        raise ValueError(f"Invalid compounding_frequency {compounding_frequency}")
    

def discount_factor_from_zero_rate(
        years, 
        zero_rate, 
        compounding_frequency: CompoundingFrequency):
    
    if compounding_frequency == CompoundingFrequency.CONTINUOUS:
        return np.exp(-zero_rate * years)
    if compounding_frequency == CompoundingFrequency.SIMPLE:
        return  1.0 / (1.0 + zero_rate * years)
    elif compounding_frequency in {
            CompoundingFrequency.DAILY, 
            CompoundingFrequency.MONTHLY,
            CompoundingFrequency.QUARTERLY,
            CompoundingFrequency.SEMIANNUAL,
            CompoundingFrequency.ANNUAL}:
        periods_per_year = compounding_frequency.periods_per_year        
        return 1.0 / (1.0 + zero_rate / periods_per_year) ** (periods_per_year * years)
    else:
        raise ValueError(f"Invalid compounding_frequency {compounding_frequency}")
        
        
def forward_rate(discount_factor_period_start,
                 discount_factor_period_end,
                 period_length_years,
                 compounding_frequency: CompoundingFrequency):

    DF_t1 = discount_factor_period_start 
    DF_t2 = discount_factor_period_end
    Δt = period_length_years
    
    # https://en.wikipedia.org/wiki/Forward_rate
    if compounding_frequency == CompoundingFrequency.SIMPLE:
        return (1 / Δt) * (DF_t1 / DF_t2 - 1)
    elif compounding_frequency == CompoundingFrequency.CONTINUOUS:
        return (1 / Δt) * (np.log(DF_t1) - np.log(DF_t2))

    elif compounding_frequency == CompoundingFrequency.ANNUAL:
        return (DF_t1 / DF_t2) ** (1 / Δt)  - 1 
    else:
        raise ValueError(f"Invalid compounding_frequency {compounding_frequency}")

            



# For importing the test cases defined in excel
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

#%%


            
            
            
            
            
            
            
            
            
            
            