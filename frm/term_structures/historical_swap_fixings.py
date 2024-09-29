# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from dataclasses import dataclass, field, InitVar
from enum import Enum
import numpy as np
import pandas as pd

from frm.term_structures.zero_curve import ZeroCurve
from frm.utils.enums import DayCountBasis, OISCouponCalcMethod, SwapType
from frm.term_structures.zero_curve import ZeroCurve

@dataclass
class HistoricalSwapRateFixings:
    swap_type: SwapType
    fixings: pd.DataFrame
    ois_coupon_calc_method: OISCouponCalcMethod=None
    day_count_basis: DayCountBasis=None

    def __post_init__(self):
        if self.swap_type == SwapType.TERM:
            assert self.ois_coupon_calc_method is None, f"'ois_coupon_calc_method' must be None for {self.swap_type}"
            
            # Check for the required columns in fixings DataFrame
            required_columns = {'date', 'fixing'}
            assert set(self.fixings.columns) == required_columns, f"Columns in fixings must be {required_columns}. Found: {set(self.fixings.columns)}"
            
            # At a late date need to do functionality for short stub where coupon rate a calc over 1M and 3M bank bills. 
            # # Check for the required columns in fixings DataFrame
            # required_columns = {'date'}
            # missing_columns = required_columns - set(self.fixings.columns)
            # assert not missing_columns, f"Missing columns in fixings: {missing_columns}"
                          
            # # Term fixings must be specified as 'fixing_{tenor}'
            # # E.g: 'fixing_90d', 'fixing_3m', 'fixing_1y'
            # # Can be multiple columns i.e 'fixing_30d', 'fixing_60d', 'fixing_90d'            
            # # Ensure all other columns follow the 'fixing_{tenor}' format
            # invalid_columns = [col for col in self.fixings.columns if col not in required_columns and not col.startswith('fixing_')]
            # assert not invalid_columns, f"Invalid columns found in fixings: {invalid_columns}"            
                        
            # fixing_columns = [col for col in self.fixings.columns if col.startswith('fixing_')]
            # tenors = [column_name.split('_')[-1] for column_name in fixing_columns]
            
            
        elif self.swap_type == SwapType.OIS:
            assert self.ois_coupon_calc_method is not None, f"'ois_coupon_calc_method' is required for {self.swap_type}"
            assert self.day_count_basis is not None, f"'day_count_basis' is required for {self.swap_type}"
            assert type(self.ois_coupon_calc_method) == OISCouponCalcMethod
            assert type(self.day_count_basis) == DayCountBasis
        
            # Check for the required columns in fixings DataFrame
            required_columns = {'date', 'fixing'}
            assert set(self.fixings.columns) == required_columns, f"Columns in fixings must be {required_columns}. Found: {set(self.fixings.columns)}"
              
    def index_term_fixings(self, dates: pd.DatetimeIndex):
        # Create a DataFrame from the input dates and store the original index
        date_df = pd.DataFrame({'date': dates})
        
        # Sort the date DataFrame for the merge_asof operation
        date_df_sorted = date_df.sort_values('date').reset_index(drop=False)
        
        # Perform merge_asof with the sorted dates
        df = pd.merge_asof(date_df_sorted, self.fixings, on='date', direction='backward')
        
        # Restore the original order of the input dates
        df_sorted_back = df.sort_values('index').set_index('index')
        
        # Return the 'fixing' column in the order of the original input dates
        return df_sorted_back['fixing'].values

        
        
    def calc_OIS_historical_coupon_rate(
            self,
            accrual_period_start_date: pd.DatetimeIndex,
            accrual_period_end_date: pd.DatetimeIndex,
            value_date: pd.Timestamp,
            day_count_basis: DayCountBasis,
            calc_method: OISCouponCalcMethod=None):
        
        # Set default calc_method if not provided
        if calc_method is None:
            calc_method = self.ois_coupon_calc_method        
            
        observations_start = accrual_period_start_date
        observations_end = accrual_period_end_date - pd.DateOffset(days=1)

        mask = np.logical_and(self.fixings['date'] >= observations_start.min(), 
                              self.fixings['date'] <= observations_end.max())    
        applicable_fixings = self.fixings.loc[mask,:]
        applicable_fixings = applicable_fixings.sort_values('date', ascending=True).reset_index(drop=True)
        
        if observations_end.max() > applicable_fixings['date'].max() \
            or observations_end.min() < applicable_fixings['date'].min():
            raise ValueError
        
        historical_cpn_rates = np.empty(shape=len(observations_start))
        for i, (observation_start, observation_end) in enumerate(zip(observations_start, observations_end)):
            if observation_end >= value_date:
                # Coupon has a forward component
                historical_cpn_rates[i] = np.nan
            else:
                mask = np.logical_and(applicable_fixings['date'] >= observation_start,
                                      applicable_fixings['date'] <= observation_end)
                
                periods_ois_fixings = applicable_fixings.loc[mask,:]
                periods_ois_fixings = periods_ois_fixings.sort_values('date', ascending=True).reset_index(drop=True)
            
                if calc_method == OISCouponCalcMethod.SIMPLE_AVERAGE:
                    cpn_rate = periods_ois_fixings['fixing'].mean()
                else:
                    date_range = pd.date_range(start=observation_start, end=observation_end, freq='D')
                    date_df = pd.DataFrame({'date': date_range})
                    df = pd.merge_asof(date_df, periods_ois_fixings, on='date', direction='backward')
                    df = df.sort_values('date', ascending=True).reset_index(drop=True)
            
                    match calc_method:
                        case OISCouponCalcMethod.DAILY_COMPOUNDED:
                            df['daily_interest'] = 1.0 + df['fixing'] / day_count_basis.days_per_year
                            cpn_rate = (df['daily_interest'].prod() - 1.0) *  day_count_basis.days_per_year / len(date_range)
                        case OISCouponCalcMethod.WEIGHTED_AVERAGE:
                            cpn_rate = df['fixing'].mean()
                        case _:
                            raise ValueError
            
                historical_cpn_rates[i] = cpn_rate
            
        return historical_cpn_rates
    
    
#%%
    
    
    