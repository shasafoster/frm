# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from dataclasses import dataclass
import numpy as np
import pandas as pd

from frm.enums.utils import CompoundingFreq, PeriodFreq
from frm.enums.term_structures import RFRFixingCalcMethod, TermRate
from frm.utils.daycount import year_frac
from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.historical_swap_index_fixings import RFRFixings, TermFixings



@dataclass
class TermSwapCurve:
    zero_curve: ZeroCurve
    historical_fixings: TermFixings

    # TODO: Term swap curve must require a fixing frequency.


    def __post_init__(self):
        assert self.zero_curve.day_count_basis == self.historical_fixings.day_count_basis

    def get_fixings(self,
                    period_start: pd.DatetimeIndex,
                    period_end: pd.DatetimeIndex,
                    fixing_freq: PeriodFreq, # TODO think about this and the enum
                    cpn_calc_method: CompoundingFreq=TermRate.SIMPLE):

        fixings = np.full(period_start.shape, np.nan)

        # TODO Choose the zero curve based on the fixing frequency
        #   Would be nice to have public API to call for data whenever needed for examples.
        # Get forward fixings
        mask_future = period_start >= self.zero_curve.curve_date
        forward_fixings = self.zero_curve.get_forward_rates(
            period_start=period_start[mask_future],
            period_end=period_end[mask_future],
            forward_rate_type=cpn_calc_method)
        fixings[mask_future] = forward_fixings

        # Get historical fixings
        mask_historical = period_start >= self.zero_curve.curve_date
        historical_fixings = self.historical_fixings.index_historical_fixings(
            fixing_dates=period_start[mask_historical])
        fixings[mask_historical] = historical_fixings

        return fixings


@dataclass
class RFRSwapCurve:
    zero_curve: ZeroCurve
    historical_fixings: RFRFixings

    def __post_init__(self):
        assert self.zero_curve.day_count_basis == self.historical_fixings.day_count_basis

    def get_discount_factors(self, dates):
        return self.zero_curve.get_discount_factors(dates=dates)

    def get_fixings(self,
                    period_start: pd.DatetimeIndex,
                    period_end: pd.DatetimeIndex,
                    cpn_calc_method: RFRFixingCalcMethod=RFRFixingCalcMethod.DAILY_COMPOUNDED):

        fixings = np.full(period_start.shape, np.nan)

        # Get forward fixings
        mask_future = period_start >= self.zero_curve.curve_date
        if mask_future.any():
            forward_fixings = self.zero_curve.get_forward_rates(
                period_start=period_start[mask_future],
                period_end=period_end[mask_future],
                forward_rate_type=cpn_calc_method)
            fixings[mask_future] = forward_fixings

        # Get historical fixings
        mask_historical = period_end <= self.zero_curve.curve_date
        if mask_historical.any():
            historical_fixings = self.historical_fixings.get_coupon_rates(
                period_start=period_start[mask_historical],
                period_end=period_end[mask_historical],
                cpn_calc_method=cpn_calc_method)
            fixings[mask_historical] = historical_fixings

        # Calculate the current/cross-over fixing
        mask_cross_over = np.logical_and(period_start < self.zero_curve.curve_date, period_end > self.zero_curve.curve_date)
        if mask_cross_over.any():
            fixings[mask_cross_over] = self._get_crossover_coupon_rate(
                crossover_period_start=period_start[mask_cross_over],
                crossover_period_end=period_end[mask_cross_over],
                cpn_calc_method=cpn_calc_method)

        return fixings


    def _get_crossover_coupon_rate(self,
                                  crossover_period_start: pd.DatetimeIndex,
                                  crossover_period_end: pd.DatetimeIndex,
                                  cpn_calc_method: RFRFixingCalcMethod):
        historical_year_frac = year_frac(crossover_period_start, self.zero_curve.curve_date, self.zero_curve.day_count_basis)
        future_year_frac = year_frac(self.zero_curve.curve_date, crossover_period_end, self.zero_curve.day_count_basis)

        curve_datetime_index = pd.DatetimeIndex([self.zero_curve.curve_date for _ in range(len(crossover_period_start))])

        historical_component = self.historical_fixings.get_coupon_rates(
            period_start=crossover_period_start,
            period_end=curve_datetime_index,
            cpn_calc_method=cpn_calc_method)
        future_component = self.zero_curve.get_forward_rates(
            period_start=curve_datetime_index,
            period_end=crossover_period_end,
            forward_rate_type=cpn_calc_method)

        crossover_rate = (historical_year_frac * historical_component
                          + (1 + historical_year_frac * historical_component) * future_year_frac * future_component) \
                         / (historical_year_frac + future_year_frac)

        return crossover_rate








