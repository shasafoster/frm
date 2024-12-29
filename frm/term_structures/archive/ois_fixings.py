# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from dataclasses import dataclass
import numpy as np
import pandas as pd

from frm.enums import DayCountBasis
from frm.enums import RFRFixingCalcMethod


@dataclass
class OISFixings:
    fixings: pd.DataFrame
    day_count_basis: DayCountBasis
    cpn_calc_method: RFRFixingCalcMethod=None
    name: str = None

    def __post_init__(self):
        required_columns = {'date', 'fixing'}
        assert set(self.fixings.columns) == required_columns, \
            f"Columns in fixings must be {required_columns}. Found: {set(self.fixings.columns)}"
        assert isinstance(self.day_count_basis, DayCountBasis), \
            "'day_count_basis' must be an instance of DayCountBasis"
        if self.cpn_calc_method is not None:
            assert isinstance(self.cpn_calc_method, RFRFixingCalcMethod), \
                "'cpn_calc_method' must be an instance of RFRFixingCalcMethod"


    def get_coupon_rates(
        self,
        period_start: pd.DatetimeIndex,
        period_end: pd.DatetimeIndex,
        cpn_calc_method: RFRFixingCalcMethod = None,
    ):
        cpn_calc_method = cpn_calc_method or self.cpn_calc_method

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
class TermFixings:
    fixings: pd.DataFrame
    name: str = None

    def __post_init__(self):
        required_columns = {'date', 'fixing'}
        assert set(self.fixings.columns) == required_columns, \
            f"Columns in fixings must be {required_columns}. Found: {set(self.fixings.columns)}"

    def get_coupon_rates(
        self,
        period_start: pd.DatetimeIndex,
        period_end: pd.DatetimeIndex,
        cpn_calc_method: RFRFixingCalcMethod = None,
    ):
        cpn_calc_method = cpn_calc_method or self.cpn_calc_method

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
        periods_rfr_fixings = fixings.loc[mask]

        if method == RFRFixingCalcMethod.SIMPLE_AVERAGE:
            return periods_rfr_fixings['fixing'].mean()

        date_range = pd.date_range(start=observation_start, end=observation_end, freq='D')
        df = pd.merge_asof(pd.DataFrame({'date': date_range}), periods_rfr_fixings, on='date', direction='backward')

        if method == RFRFixingCalcMethod.DAILY_COMPOUNDED:
            df['daily_interest'] = 1 + df['fixing'] / self.day_count_basis.days_per_year
            return (df['daily_interest'].prod() - 1) * self.day_count_basis.days_per_year / len(date_range)
        elif method == RFRFixingCalcMethod.WEIGHTED_AVERAGE:
            return df['fixing'].mean()

        raise ValueError(f"Invalid RFRFixingCalcMethod: {method}")