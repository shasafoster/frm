# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from abc import ABC
from dataclasses import dataclass
import numpy as np
import pandas as pd

from frm.enums.utils import DayCountBasis
from frm.enums.term_structures import RFRFixingCalcMethod


@dataclass
class HistoricalSwapIndexFixings(ABC):
    fixings: pd.DataFrame
    day_count_basis: DayCountBasis

    def __post_init__(self):
        required_columns = {'date', 'fixing'}
        assert set(self.fixings.columns) == required_columns, \
            f"Columns in fixings must be {required_columns}. Found: {set(self.fixings.columns)}"
        assert isinstance(self.day_count_basis, DayCountBasis), \
            "'day_count_basis' must be an instance of DayCountBasis"


@dataclass
class RFRFixings(HistoricalSwapIndexFixings):

    def get_coupon_rates(
        self,
        period_start: pd.DatetimeIndex,
        period_end: pd.DatetimeIndex,
        cpn_calc_method: RFRFixingCalcMethod
    ):
        observations_start = period_start
        observations_end = period_end - pd.DateOffset(days=1)

        mask = np.logical_and(self.fixings['date'] >= observations_start.min(),
                              self.fixings['date'] <= observations_end.max())
        applicable_fixings = self.fixings.loc[mask,:].sort_values('date', ascending=True).reset_index(drop=True)


        if observations_end.max() > applicable_fixings['date'].max() \
            or observations_end.min() < applicable_fixings['date'].min():
            raise ValueError('Observation period extends beyond available fixing data.')

        coupon_rates = [self._calculate_coupon_rate(s, e, applicable_fixings, cpn_calc_method)
                                   for s, e in zip(observations_start, observations_end)]
        return np.array(coupon_rates)

    def _calculate_coupon_rate(
            self,
            observation_start: pd.Timestamp,
            observation_end: pd.Timestamp,
            fixings: pd.DataFrame,
            method: RFRFixingCalcMethod
    ):
        mask = (fixings['date'] >= observation_start) & (fixings['date'] <= observation_end)
        periods_ois_fixings = fixings.loc[mask]

        if method == RFRFixingCalcMethod.SIMPLE_AVERAGE:
            return periods_ois_fixings['fixing'].mean()

        date_range = pd.date_range(start=observation_start, end=observation_end, freq='D')
        df = pd.merge_asof(pd.DataFrame({'date': date_range}), periods_ois_fixings, on='date', direction='backward')

        if method == RFRFixingCalcMethod.DAILY_COMPOUNDED:
            df['daily_interest'] = 1 + df['fixing'] / self.day_count_basis.days_per_year
            return (df['daily_interest'].prod() - 1) * self.day_count_basis.days_per_year / len(date_range)
        elif method == RFRFixingCalcMethod.WEIGHTED_AVERAGE:
            return df['fixing'].mean()

        raise ValueError(f"Invalid RFRFixingCalcMethod: {method}")



@dataclass
class TermFixings(HistoricalSwapIndexFixings):

    def index_historical_fixings(
            self,
            fixing_dates: pd.DatetimeIndex
    ):
        # Create a DataFrame from the input dates and store the original index
        date_df = pd.DataFrame({'date': fixing_dates})

        # Sort the date DataFrame for the merge_asof operation
        date_df_sorted = date_df.sort_values('date').reset_index(drop=False)

        # Perform merge_asof with the sorted dates
        df = pd.merge_asof(date_df_sorted, self.fixings, on='date', direction='backward')

        df_sorted_back = df.sort_values('index').set_index('index') # Restore the original order of the input dates

        return df_sorted_back['fixing'].values

    # Additional Term-specific initialization if needed

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