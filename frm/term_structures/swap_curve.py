# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from frm.enums.utils import ForwardRate, OISCouponCalcMethod
from frm.utils.daycount import year_fraction
from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.historical_swap_index_fixings import HistoricalSwapIndexFixings


class SwapType(Enum):
    TERM = 'term'
    OIS = 'ois'


@dataclass
class SwapCurve:
    swap_type: SwapType
    zero_curve: ZeroCurve
    historical_swap_index_fixings: HistoricalSwapIndexFixings
    forward_rate_type: ForwardRate


    def __post_init__(self):
        # Check that the day count basis and business day calendar are the same for the zero curve and historical swap index fixings

        assert self.zero_curve.day_count_basis == self.historical_swap_index_fixings.day_count_basis, 'Day count basis must be the same for zero curve and historical swap index fixings.'

        # Check that the forward rate type is a valid ForwardRate
        match self.swap_type:
            case SwapType.TERM:
                assert self.forward_rate_type in [ForwardRate.DAILY_COMPOUNDED,
                                                  ForwardRate.SIMPLE], \
                    'Forward rate type must be either ForwardRate.DAILY_COMPOUNDED or ForwardRate.SIMPLE for term swaps.'
            case SwapType.OIS:
                assert self.forward_rate_type in [ForwardRate.DAILY_COMPOUNDED,
                                                  ForwardRate.WEIGHTED_AVERAGE,
                                                  ForwardRate.SIMPLE_AVERAGE], \
                    'Forward rate type must be either OISCouponCalcMethod.DAILY_COMPOUNDED, OISCouponCalcMethod.WEIGHTED_AVERAGE, or OISCouponCalcMethod.SIMPLE_AVERAGE for OIS swaps.'

        self.curve_date = self.zero_curve.curve_date
        self.day_count_basis = self.zero_curve.day_count_basis



    def get_fixings(self, period_start, period_end, forward_rate_type=None):

        if forward_rate_type is None:
            forward_rate_type = self.forward_rate_type

        fixings = np.full(period_start.shape, np.nan)

        # Get forward fixings
        mask_future = period_start >= self.curve_date
        forward_fixings = self.zero_curve.forward_rate(
            period_start=period_start[mask_future],
            period_end=period_end[mask_future],
            forward_rate_type=forward_rate_type)
        fixings[mask_future] = forward_fixings

        match self.swap_type:
            case SwapType.OIS:
                # Get historical fixings
                mask_historical = period_end <= self.curve_date
                historical_fixings = self.historical_swap_index_fixings.calc_historical_ois_coupon_rate(
                    period_start=period_start[mask_historical],
                    period_end=period_end[mask_historical],
                    ois_coupon_calc_method=forward_rate_type.value)
                fixings[mask_historical] = historical_fixings

                # Calculate the current/cross-over fixing
                mask_cross_over = np.logical_and(period_start < self.curve_date, period_end > self.curve_date)
                historical_year_frac = year_fraction(period_start[mask_cross_over], self.curve_date, self.day_count_basis)
                future_year_frac = year_fraction(self.curve_date, period_end[mask_cross_over], self.day_count_basis)

                curve_datetimeindex = pd.DatetimeIndex(sum(mask_cross_over) * [self.curve_date])
                historical_component = self.historical_swap_index_fixings.calc_historical_ois_coupon_rate(
                    period_start=period_start[mask_cross_over],
                    period_end=curve_datetimeindex)
                future_component = self.zero_curve.forward_rate(
                    period_start=curve_datetimeindex,
                    period_end=period_end[mask_cross_over],
                    forward_rate_type=forward_rate_type)

                crossover_rate = (historical_year_frac * historical_component
                                  + (1 + historical_year_frac * historical_component) * future_year_frac * future_component) \
                                 / (historical_year_frac + future_year_frac)
                fixings[mask_cross_over] = crossover_rate


            case SwapType.TERM:

                mask_historical = period_start >= self.curve_date
                historical_fixings = self.historical_swap_index_fixings.index_historical_fixings(
                    fixing_dates=period_start[mask_historical])

                fixings[mask_historical] = historical_fixings

        return fixings











