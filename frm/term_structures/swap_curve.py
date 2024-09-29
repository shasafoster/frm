# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from dataclasses import dataclass, field, InitVar
from enum import Enum
import numpy as np
import pandas as pd

from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.zero_curve_helpers import OISCouponCalcMethod
from frm.utils.enums import DayCountBasis, OISCouponCalcMethod, SwapType
from frm.term_structures.zero_curve import ZeroCurve


@dataclass
class HistoricalSwapRateFixings:
    swap_type: SwapType
    fixings: pd.DataFrame
    ois_coupon_calc_method: OISCouponCalcMethod = None

    def __post_init__(self):
        if self.swap_type == SwapType.TERM:
            assert self.ois_coupon_calc_method is None, f"'ois_coupon_calc_method' must be None for {self.swap_type}"
        elif self.swap_type == SwapType.OIS:
            assert self.ois_coupon_calc_method is not None, f"'ois_coupon_calc_method' is required for {self.swap_type}"
        
        # Check for the required columns in fixings DataFrame
        required_columns = {'date', 'fixing'}
        missing_columns = required_columns - set(self.fixings.columns)
        assert not missing_columns, f"Missing columns in fixings: {missing_columns}"
              
        
    # def index_term_fixings(self, dates: pd.DatetimeIndex):
        
    #     mask = self.fixings['date'].isin([dates])
        
    #     fixings = 
        
        
        
        
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
        applicable_fixings = self.applicable_fixings.sort_values('date', ascending=True).reset_index(drop=True)
        
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
            
                if calc_method == OISCouponCalcMethod.SimpleAverage:
                    cpn_rate = periods_ois_fixings['fixing'].mean()
                else:
                    date_range = pd.date_range(start=observation_start, end=observation_end, freq='D')
                    date_df = pd.DataFrame({'date': date_range})
                    df = pd.merge_asof(date_df, periods_ois_fixings, on='date', direction='backward')
                    df = df.sort_values('date', ascending=False).reset_index(drop=True)
            
                    match calc_method:
                        case OISCouponCalcMethod.DailyCompounded:
                            df['daily_interest'] = 1.0 + df['fixing'] / day_count_basis.days_per_year
                            cpn_rate = (df['daily_interest'].prod() - 1.0) *  day_count_basis.days_per_year / len(date_range)
                        case OISCouponCalcMethod.WeightedAverage:
                            cpn_rate = df['fixing'].mean()
                        case _:
                            raise ValueError
            
                historical_cpn_rates[i] = cpn_rate
            
        return historical_cpn_rates
    
    
#%%
    
    
    