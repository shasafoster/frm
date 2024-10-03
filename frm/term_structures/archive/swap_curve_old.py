# # -*- coding: utf-8 -*-
# import os
# if __name__ == "__main__":
#     os.chdir(os.environ.get('PROJECT_DIR_FRM'))
#
# from dataclasses import dataclass
# from enum import Enum
# import numpy as np
# import pandas as pd
#
# from frm.enums.utils import ForwardRate, OISCouponCalcMethod
# from frm.utils.daycount import year_fraction
# from frm.term_structures.zero_curve import ZeroCurve
# from frm.term_structures.historical_swap_index_fixings import HistoricalSwapIndexFixings
#
#
# class SwapType(Enum):
#     TERM = 'term'
#     OIS = 'ois'
#
#
# @dataclass
# class SwapCurve:
#     swap_type: SwapType
#     _zero_curve: ZeroCurve
#     _historical_swap_index_fixings: pd.DataFrame
#     forward_rate_type: ForwardRate
#
#
#     def __post_init__(self):
#         # Check that the day count basis and business day calendar are the same for the zero curve and historical swap index fixings
#
#         assert self.zero_curve.day_count_basis == self.historical_swap_index_fixings.day_count_basis, 'Day count basis must be the same for zero curve and historical swap index fixings.'
#
#         # Check that the forward rate type is a valid ForwardRate
#         match self.swap_type:
#             case SwapType.TERM:
#                 assert self.forward_rate_type in [ForwardRate.DAILY_COMPOUNDED,
#                                                   ForwardRate.SIMPLE], \
#                     'Forward rate type must be either ForwardRate.DAILY_COMPOUNDED or ForwardRate.SIMPLE for term swaps.'
#             case SwapType.OIS:
#                 assert self.forward_rate_type in [ForwardRate.DAILY_COMPOUNDED,
#                                                   ForwardRate.WEIGHTED_AVERAGE,
#                                                   ForwardRate.SIMPLE_AVERAGE], \
#                     'Forward rate type must be either OISCouponCalcMethod.DAILY_COMPOUNDED, OISCouponCalcMethod.WEIGHTED_AVERAGE, or OISCouponCalcMethod.SIMPLE_AVERAGE for OIS swaps.'
#
#         self.curve_date = self.zero_curve.curve_date
#         self.day_count_basis = self.zero_curve.day_count_basis
#
#
#     def get_discount_factors(self, dates):
#         return self.zero_curve.discount_factor(dates)
#
#
#     def get_fixings(self, period_start, period_end=None, forward_rate_type=None):
#
#         if forward_rate_type is None:
#             forward_rate_type = self.forward_rate_type
#
#         fixings = np.full(period_start.shape, np.nan)
#
#         # Get forward fixings
#         mask_future = period_start >= self.curve_date
#         forward_fixings = self.zero_curve.forward_rate(
#             period_start=period_start[mask_future],
#             period_end=period_end[mask_future],
#             forward_rate_type=forward_rate_type)
#         fixings[mask_future] = forward_fixings
#
#         match self.swap_type:
#             case SwapType.OIS:
#                 # Get historical fixings
#                 mask_historical = period_end <= self.curve_date
#                 historical_fixings = self.historical_swap_index_fixings.calc_historical_ois_coupon_rate(
#                     period_start=period_start[mask_historical],
#                     period_end=period_end[mask_historical],
#                     ois_coupon_calc_method=forward_rate_type.value)
#                 fixings[mask_historical] = historical_fixings
#
#                 # Calculate the current/cross-over fixing
#                 mask_cross_over = np.logical_and(period_start < self.curve_date, period_end > self.curve_date)
#                 historical_year_frac = year_fraction(period_start[mask_cross_over], self.curve_date, self.day_count_basis)
#                 future_year_frac = year_fraction(self.curve_date, period_end[mask_cross_over], self.day_count_basis)
#
#                 curve_datetimeindex = pd.DatetimeIndex(sum(mask_cross_over) * [self.curve_date])
#                 historical_component = self.historical_swap_index_fixings.calc_historical_ois_coupon_rate(
#                     period_start=period_start[mask_cross_over],
#                     period_end=curve_datetimeindex)
#                 future_component = self.zero_curve.forward_rate(
#                     period_start=curve_datetimeindex,
#                     period_end=period_end[mask_cross_over],
#                     forward_rate_type=forward_rate_type)
#
#                 crossover_rate = (historical_year_frac * historical_component
#                                   + (1 + historical_year_frac * historical_component) * future_year_frac * future_component) \
#                                  / (historical_year_frac + future_year_frac)
#                 fixings[mask_cross_over] = crossover_rate
#
#
#             case SwapType.TERM:
#
#                 mask_historical = period_start >= self.curve_date
#                 historical_fixings = self.historical_swap_index_fixings.index_historical_fixings(
#                     fixing_dates=period_start[mask_historical])
#
#                 fixings[mask_historical] = historical_fixings
#
#         return fixings
#
#
#     def index_historical_fixings(self, fixing_dates: pd.DatetimeIndex):
#
#         # Additional Term-specific initialization if needed
#
#         # At a late date need to do functionality for short stub where coupon rate a calc over 1M and 3M bank bills.
#         # # Check for the required columns in fixings DataFrame
#         # required_columns = {'date'}
#         # missing_columns = required_columns - set(self.fixings.columns)
#         # assert not missing_columns, f"Missing columns in fixings: {missing_columns}"
#
#         # # Term fixings must be specified as 'fixing_{tenor}'
#         # # E.g: 'fixing_90d', 'fixing_3m', 'fixing_1y'
#         # # Can be multiple columns i.e 'fixing_30d', 'fixing_60d', 'fixing_90d'
#         # # Ensure all other columns follow the 'fixing_{tenor}' format
#         # invalid_columns = [col for col in self.fixings.columns if col not in required_columns and not col.startswith('fixing_')]
#         # assert not invalid_columns, f"Invalid columns found in fixings: {invalid_columns}"
#
#         # fixing_columns = [col for col in self.fixings.columns if col.startswith('fixing_')]
#         # tenors = [column_name.split('_')[-1] for column_name in fixing_columns]
#
#         # Create a DataFrame from the input dates and store the original index
#         date_df = pd.DataFrame({'date': fixing_dates})
#
#         # Sort the date DataFrame for the merge_asof operation
#         date_df_sorted = date_df.sort_values('date').reset_index(drop=False)
#
#         # Perform merge_asof with the sorted dates
#         df = pd.merge_asof(date_df_sorted, self.fixings, on='date', direction='backward')
#
#         # Restore the original order of the input dates
#         df_sorted_back = df.sort_values('index').set_index('index')
#
#         # Return the 'fixing' column in the order of the original input dates
#         return df_sorted_back['fixing'].values
#
#
#     def calc_historical_ois_coupon_rate(
#             self,
#             period_start: pd.DatetimeIndex,
#             period_end: pd.DatetimeIndex,
#             ois_coupon_calc_method: OISCouponCalcMethod = None,
#     ):
#
#         if ois_coupon_calc_method is None:
#             ois_coupon_calc_method = self.ois_coupon_calc_method
#
#         observations_start = period_start
#         observations_end = period_end - pd.DateOffset(days=1)
#
#         mask = np.logical_and(self.fixings['date'] >= observations_start.min(),
#                               self.fixings['date'] <= observations_end.max())
#         applicable_fixings = self.fixings.loc[mask, :]
#         applicable_fixings = applicable_fixings.sort_values('date', ascending=True).reset_index(drop=True)
#
#         if observations_end.max() > applicable_fixings['date'].max() \
#                 or observations_end.min() < applicable_fixings['date'].min():
#             raise ValueError('Observation period extends beyond available fixing data.')
#
#         historical_coupon_rates = np.empty(shape=len(observations_start))
#         for i, (observation_start, observation_end) in enumerate(zip(observations_start, observations_end)):
#             mask = np.logical_and(applicable_fixings['date'] >= observation_start,
#                                   applicable_fixings['date'] <= observation_end)
#
#             periods_ois_fixings = applicable_fixings.loc[mask, :]
#             periods_ois_fixings = periods_ois_fixings.sort_values('date', ascending=True).reset_index(drop=True)
#
#             if ois_coupon_calc_method == OISCouponCalcMethod.SIMPLE_AVERAGE:
#                 cpn_rate = periods_ois_fixings['fixing'].mean()
#             else:
#                 date_range = pd.date_range(start=observation_start, end=observation_end, freq='D')
#                 date_df = pd.DataFrame({'date': date_range})
#                 df = pd.merge_asof(date_df, periods_ois_fixings, on='date', direction='backward')
#                 df = df.sort_values('date', ascending=True).reset_index(drop=True)
#
#                 match ois_coupon_calc_method:
#                     case OISCouponCalcMethod.DAILY_COMPOUNDED:
#                         df['daily_interest'] = 1.0 + df['fixing'] / self.day_count_basis.days_per_year
#                         cpn_rate = (df['daily_interest'].prod() - 1.0) * self.day_count_basis.days_per_year / len(
#                             date_range)
#                     case OISCouponCalcMethod.WEIGHTED_AVERAGE:
#                         cpn_rate = df['fixing'].mean()
#                     case _:
#                         raise ValueError(f"Invalid OISCouponCalcMethod: {ois_coupon_calc_method}")
#
#             historical_coupon_rates[i] = cpn_rate
#
#         return historical_coupon_rates
#
#
#
#
#
#
#
#
#
