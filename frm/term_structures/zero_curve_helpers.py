# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import pandas as pd
from frm.utils.enums import CompoundingFrequency, DayCountBasis


class OISCouponCalcMethod(Enum):
    DailyCompounded = 'dailycompounded'
    WeightedAverage = 'weightedaverage'
    SimpleAverage = 'simpleaverage'


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

            
def calc_ois_historical_coupon(accrual_period_start_date: pd.Timestamp,
                               accrual_period_end_date: pd.Timestamp,
                               ois_fixings: pd.DataFrame,
                               calc_method: OISCouponCalcMethod,
                               days_per_year):

    observation_start = accrual_period_start_date
    observation_end = accrual_period_end_date - pd.DateOffset(days=1)
    
    mask = np.logical_and(ois_fixings['observation_date'] >= observation_start, 
                          ois_fixings['observation_date'] <= observation_end)    
    
    ois_fixings = ois_fixings.loc[mask,:]
    ois_fixings = ois_fixings.sort_values('observation_date', ascending=True).reset_index(drop=True)
    
    if calc_method == OISCouponCalcMethod.SimpleAverage:
        cpn_rate = ois_fixings['fixing'].mean()
    else:
        date_range = pd.date_range(start=observation_start, end=observation_end, freq='D')
        date_df = pd.DataFrame({'observation_date': date_range})
        df = pd.merge_asof(date_df, ois_fixings, on='observation_date', direction='backward')
        df = df.sort_values('observation_date', ascending=False).reset_index(drop=True)

        match calc_method:
            case OISCouponCalcMethod.DailyCompounded:
                df['daily_interest'] = 1.0 + df['fixing'] / days_per_year
                cpn_rate = (df['daily_interest'].prod() - 1.0) *  days_per_year / len(date_range)
            case OISCouponCalcMethod.WeightedAverage:
                cpn_rate = df['fixing'].mean()
            case _:
                raise ValueError
    
    return cpn_rate
            
            
            
            
            
            
            
            
            