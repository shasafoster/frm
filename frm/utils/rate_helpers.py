# -*- coding: utf-8 -*-

import numpy as np
from frm.utils.enums import CompoundingFrequency, DayCountBasis


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
        return CompoundingFrequency.periods_per_year * ((1.0 / discount_factor) ** (1.0 / (CompoundingFrequency.periods_per_year * years)) - 1.0)
    else:
        raise ValueError
    

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
            return 1.0 / (1.0 + zero_rate / CompoundingFrequency.periods_per_year) ** (CompoundingFrequency.periods_per_year * years)
    else:
        raise ValueError
        
        
def forward_rate(discount_factor_period_start,
                 discount_factor_period_end,
                 period_length_years,
                 compounding_frequency: CompoundingFrequency):

    DF_T1 = discount_factor_period_start 
    DF_T2 = discount_factor_period_end
    T2_MINUS_T1 = period_length_years
    
    # https://en.wikipedia.org/wiki/Forward_rate
    if compounding_frequency == CompoundingFrequency.CONTINUOUS:
        return (1 / T2_MINUS_T1) * (np.log(DF_T1) - np.log(DF_T2))
    elif compounding_frequency == CompoundingFrequency.SIMPLE:
        return (1 / T2_MINUS_T1) * (DF_T1 / DF_T2 - 1)
    elif compounding_frequency == CompoundingFrequency.ANNUAL:
        return (DF_T1 / DF_T2) ** (1 / T2_MINUS_T1)  - 1 
    else:
        raise ValueError

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            