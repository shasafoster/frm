# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
from typing import Optional
import warnings

import frm.utils
from frm.term_structures.swap_curve import TermSwapCurve, RFRSwapCurve
from frm.utils import CouponSchedule, year_frac, MarketDataNotAvailableError
from frm.enums import CompoundingFreq, TermRate, RFRFixingCalcMethod, DayCountBasis, PayRcv
from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.zero_curve_helpers import discount_factor_from_zero_rate
from scipy.optimize import root_scalar


if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

# I am writing a (python) class to support the valuation of generic swap legs and bonds.

# THe key methods for a 'leg' are
# (i) coupon date schedule → schedule with forward coupon cashflow amount (will be different for each leg type)
# (ii) discount cashflows & get PV (this will be identical for all leg types)
# (iii) solve parameter (coupon rate, spread) in order to match leg to given price/value.
# (iv) clean mv, dirty mv, accrued interest, cashflow buckets, current, non-current, etc.
# (v) DV01, +/- 100 bps shift to par rates

# Leg types:
# Fixed, Zerocoupon
# FloatTerm, FloatRFR, InflationZC, InflationYoY
# Each leg type will have a different method for calculating the forward coupon cashflow amount.



@dataclass
class Leg(ABC):
    "Generic class to cover pricing of all types of swap legs and bonds"

    schedule: CouponSchedule
    discount_curve: Optional[ZeroCurve]=None
    pay_rcv: PayRcv=PayRcv.PAY
    day_count_basis: DayCountBasis=DayCountBasis.ACT_365
    notional_ccy: str=None
    settlement_ccy: str=None

    def __post_init__(self):
        self.schedule.df.loc[:,['notional', 'notional_payment']] = self.schedule.df[['notional', 'notional_payment']] * self.pay_rcv.multiplier
        self.schedule.add_period_daycount(day_count_basis=self.day_count_basis)
        self.schedule.add_period_yearfrac(day_count_basis=self.day_count_basis)

    @abstractmethod
    def calc_coupon_payment(self, coupon_override=None):
        pass

    @abstractmethod
    def _calc_accrued_interest(self):
        pass

    @abstractmethod
    def _get_solved_field_name(self):
        pass

    @abstractmethod
    def _get_legtype_specific_schedule_cols(self):
        pass

    def _add_contractual_coupon_param(self, coupon_param):
        """Validate and apply the coupon_contractual_component to the notional schedule."""
        coupon_param = np.atleast_1d(coupon_param)
        index, valid_shapes = self.schedule.determine_valid_shapes_for_coupon_param()
        coupon_param_name = self._get_solved_field_name()

        if coupon_param.shape not in valid_shapes:
            raise ValueError(
                f"Invalid {coupon_param_name} shape: {coupon_param.shape}. "
                f"Expected one of: {valid_shapes}"
            )

        index_coupon_payment_date = self.schedule.df.columns.get_loc('coupon_payment_date')
        self.schedule.df.insert(index_coupon_payment_date, coupon_param_name, np.nan)
        if coupon_param.shape == (1,):
            self.schedule.df.loc[index, coupon_param_name] = coupon_param[0]
        else:
            self.schedule.df.loc[index, coupon_param_name] = coupon_param


    def _get_contractual_schedule_cols(self):

        abstract_columns = ['period_start', 'period_end', 'notional_payment_date', 'coupon_payment_date']
        legtype_specific_columns = self._get_legtype_specific_schedule_cols()

        cols = []
        for col in self.schedule.df.columns:
            if col in abstract_columns or col in legtype_specific_columns:
                cols.append(col)
        return cols

    def target_solve(self, target_value):
        solved_value = self._solver_helper(solved_fieldname=self._get_solved_field_name(), target_value=target_value)
        return solved_value


    def par_solve(self):
        if 'notional_payment_date' not in self.schedule.df.columns:
            par_value = 0
        else:
            if frm.utils.INCLUDE_PAYMENTS_ON_VALUE_DATE_IN_NPV:
                mask = self.schedule.df['notional_payment_date'] >= frm.utils.VALUE_DATE
            else:
                mask = self.schedule.df['notional_payment_date'] > frm.utils.VALUE_DATE
            # Par value is the outstanding notional payments.
            par_value = self.schedule.df['notional_payment'][mask].sum()

        solved_value = self._solver_helper(solved_fieldname=self._get_solved_field_name(), target_value=par_value)
        return solved_value

    def _solver_helper(self, solved_fieldname, target_value):
        field_original_value = self.schedule.df[solved_fieldname].values.copy()

        def objective_function(overide):
            return self.calc_dirty_pv(coupon_override=overide) - target_value

        solved_value = np.nan
        try:
            bracket = [-0.1, 3] # -10% to 300%
            solution = root_scalar(objective_function, bracket=bracket, method='brentq')
            if solution.converged:
                solved_value = solution.root
        finally:
            # Ensure reset, even if the solver fails
            self.schedule.df[solved_fieldname] = field_original_value

            return solved_value

    def calc_accrued_interest(self):
        # Leg specific accrued interest calculation
        accrued_interest = self._calc_accrued_interest()

        # Calculate the unsettled cashflows
        if frm.utils.INCLUDE_PAYMENTS_ON_VALUE_DATE_IN_NPV:
            mask = (self.schedule.df['coupon_payment_date'] >= frm.utils.VALUE_DATE) & (self.schedule.df['period_end'] <= frm.utils.VALUE_DATE)
        else:
            mask = (self.schedule.df['coupon_payment_date'] > frm.utils.VALUE_DATE) & (self.schedule.df['period_end'] <= frm.utils.VALUE_DATE)
        if mask.sum() > 1:
            warnings.warn('Multiple periods found for unsettled cashflow calculation.')
        unsettled_cashflows = self.schedule.df['coupon_payment'].sum()

        if frm.utils.LIMIT_ACCRUED_INTEREST_TO_UNSETTLED_CASHFLOW:
            if unsettled_cashflows > 0:
                return unsettled_cashflows
            else:
                return accrued_interest
        else:
            return accrued_interest + unsettled_cashflows

    def _discount_payments(self, prefix=''):
        if self.discount_curve is None:
            raise MarketDataNotAvailableError('Discount curve not available.')

        payment_date_col = prefix + 'payment_date'
        discount_factor_col = prefix + 'discount_factor'

        if payment_date_col in self.schedule.df.columns:
            dates = self.schedule.df[payment_date_col]
            self.schedule.df[discount_factor_col] = self.discount_curve.get_discount_factors(self.schedule.df[payment_date_col])
            self.schedule.df.loc[pd.isna(dates),discount_factor_col] = 0.0

        if frm.utils.INCLUDE_PAYMENTS_ON_VALUE_DATE_IN_NPV:
            mask = self.schedule.df[payment_date_col] < frm.utils.VALUE_DATE
        else:
            mask = self.schedule.df[payment_date_col] <= frm.utils.VALUE_DATE
        self.schedule.df.loc[mask, discount_factor_col] = 0.0

    def calc_dirty_pv(self, coupon_override=None):
        self.calc_coupon_payment(coupon_override=coupon_override)
        self.schedule.df['total_payment'] = self.schedule.df['notional_payment'] + self.schedule.df['coupon_payment']
        self._discount_payments(prefix='notional_')
        self._discount_payments(prefix='coupon_')
        self.schedule.df['notional_payment_pv'] = (self.schedule.df['notional_payment'] *
                                                               self.schedule.df['notional_discount_factor'])
        self.schedule.df['coupon_payment_pv'] = (self.schedule.df['coupon_payment'] *
                                                            self.schedule.df['coupon_discount_factor'])
        self.schedule.df['total_payment_pv'] = (self.schedule.df['notional_payment_pv'] +
                                                           self.schedule.df['coupon_payment_pv'])
        return self.schedule.df['total_payment_pv'].sum()

    def calc_clean_pv(self):
        return self.calc_dirty_pv() - self.calc_accrued_interest()

    def get_current_non_current_split(self, calc_npv=True):
        if calc_npv:
            self.calc_dirty_pv()

        current, non_current = 0, 0
        value_date_plus_one_year = frm.utils.VALUE_DATE + pd.DateOffset(years=1)

        if frm.utils.INCLUDE_PAYMENTS_ON_VALUE_DATE_IN_NPV:
            current_period_idxs = self.schedule.df['notional_payment_date'] <= value_date_plus_one_year
        else:
            current_period_idxs = self.schedule.df['notional_payment_date'] < value_date_plus_one_year
        current += self.schedule.df['notional_payment_pv'][current_period_idxs].sum()
        non_current += self.schedule.df['notional_payment_pv'][~current_period_idxs].sum()

        if frm.utils.INCLUDE_PAYMENTS_ON_VALUE_DATE_IN_NPV:
            current_period_idxs = self.schedule.df['coupon_payment_date'] <= value_date_plus_one_year
        else:
            current_period_idxs = self.schedule.df['coupon_payment_date'] < value_date_plus_one_year
        current += self.schedule.df['coupon_payment_pv'][current_period_idxs].sum()
        non_current += self.schedule.df['notional_payment_pv'][~current_period_idxs].sum()

        return current, non_current

    def _bucket_payments_helper(self, buckets, prefix):
        payment_col = prefix + 'payment'
        payment_date_col = prefix + 'payment_date'

        rows = []
        if frm.utils.INCLUDE_PAYMENTS_ON_VALUE_DATE_IN_NPV:
            mask_lower = self.schedule.df[payment_date_col] >= frm.utils.VALUE_DATE
        else:
            mask_lower = self.schedule.df[payment_date_col] > frm.utils.VALUE_DATE

        for i,v in enumerate(buckets):
            upper_pillar = frm.utils.VALUE_DATE + pd.DateOffset(years=v)
            mask_upper = self.schedule.df[payment_date_col] < (upper_pillar)
            rows.append(self.schedule.df[payment_col][mask_lower & mask_upper].sum())
            mask_lower = self.schedule.df[payment_date_col] >= upper_pillar

        mask_lower = self.schedule.df[payment_date_col] >= upper_pillar
        rows.append(self.schedule.df[payment_col][mask_lower].sum())
        index = buckets + (np.inf,)

        return pd.DataFrame(rows, index=index, columns=[payment_col])

    def bucket_payments(self, calc_npv=True, buckets=(1,2,3,5)):
        if calc_npv:
            self.calc_dirty_pv()
        coupon_payments_bucketed = self._bucket_payments_helper(buckets, prefix='coupon_')
        notional_payments_bucketed = self._bucket_payments_helper(buckets, prefix='notional_')
        combined = pd.concat([coupon_payments_bucketed, notional_payments_bucketed], axis=1)
        combined['total_payment'] = combined['coupon_payment'] + combined['notional_payment']
        return combined


    def calc_dv01(self):
        # Shared DV01 calculation with +/- 100 bps shift
        pass



@dataclass
class FixedLeg(Leg):
    fixed_rate: float | np.ndarray = np.nan

    def __post_init__(self):
        super().__post_init__()
        self._add_contractual_coupon_param(self.fixed_rate)
        #renamed_columns = {'coupon_contractual_component': self._get_solved_field_name()}
        #self.schedule.df = self.schedule.df.rename(columns=renamed_columns)
        #self.schedule.df[self._get_solved_field_name()] = self.fixed_rate

    def calc_coupon_payment(self, coupon_override=None):
        if coupon_override is not None:
            self.schedule.df['fixed_rate'] = coupon_override
        self.schedule.df['coupon_payment'] = self.pay_rcv.multiplier * self.schedule.df['notional'] * self.schedule.df['fixed_rate'] * self.schedule.df['period_yearfrac']
        mask = pd.isna(self.schedule.df['coupon_payment_date'])
        self.schedule.df.loc[mask,'coupon_payment'] = 0.0

    def _calc_accrued_interest(self):
        current_period_idxs = (self.schedule.df['period_start'] <= frm.utils.VALUE_DATE) & (self.schedule.df['period_end'] > frm.utils.VALUE_DATE)
        if current_period_idxs.sum() > 1:
            warnings.warn('Multiple periods found for accrual interest calculation.')
        accrued_period_yearfrac = year_frac(self.schedule.df['period_start'][current_period_idxs].iloc[0],
                                         frm.utils.VALUE_DATE,
                                         day_count_basis=self.day_count_basis)
        accrued_interest = (self.schedule.df['notional'][current_period_idxs]
                            * self.schedule.df['fixed_rate'][current_period_idxs]
                            * accrued_period_yearfrac).sum()
        return accrued_interest

    def _get_solved_field_name(self):
        return 'fixed_rate'

    def _get_legtype_specific_schedule_cols(self):
        return ['fixed_rate']


@dataclass
class FloatTermLeg(Leg):
    spread: float | np.ndarray = np.nan
    forward_rate_type: TermRate=TermRate.SIMPLE
    term_swap_curve: Optional[TermSwapCurve]=None

    def __post_init__(self):
        super().__post_init__()
        self._add_contractual_coupon_param(self.spread)
        # self.schedule.df[self._get_legtype_specific_schedule_cols()] = np.nan
        # renamed_columns = {'coupon_contractual_component': self._get_solved_field_name()}
        # self.schedule.df = self.schedule.df.rename(columns=renamed_columns)
        # self.schedule.df[self._get_solved_field_name()] = self.spread
        #self.calc_notional_schedule(coupon_fields_to_set_to_zero=['fixing','spread','coupon_rate'])

    def calc_coupon_payment(self, coupon_override=None):
        if self.term_swap_curve is None:
            raise MarketDataNotAvailableError('Term Swap curve not available.')

        self.schedule.df['fixing'] = np.nan
        if coupon_override is not None:
            self.schedule.df['spread'] = coupon_override
        self.schedule.df['coupon_rate'] = self.schedule.df['fixing'] + self.schedule.df['spread']
        self.schedule.df['coupon_payment'] = self.schedule.df['notional'] * self.schedule.df['coupon_rate'] * self.schedule.df['period_yearfrac']
        mask = pd.isna(self.schedule.df['coupon_payment_date'])
        self.schedule.df.loc[mask,'coupon_payment'] = 0.0

    def _calc_accrued_interest(self):
        current_period_idxs = (self.schedule.df['period_start'] <= frm.utils.VALUE_DATE) & (self.schedule.df['period_end'] > frm.utils.VALUE_DATE)
        if current_period_idxs.sum() > 1:
            warnings.warn('Multiple periods found for accrual interest calculation.')
        accrued_period_yearfrac = year_frac(self.schedule.df['period_start'][current_period_idxs].iloc[0], frm.utils.VALUE_DATE, day_count_basis=self.day_count_basis)
        accrued_interest = (self.schedule.df['notional'][current_period_idxs]
                            * self.schedule.df['spread'][current_period_idxs]
                            * accrued_period_yearfrac).sum()
        return accrued_interest

    def _get_solved_field_name(self):
        return 'spread'

    def _get_legtype_specific_schedule_cols(self):
        return ['fixing_date','fixing','spread','coupon_rate']


@dataclass
class FloatRFRLeg(Leg):
    spread: float | np.ndarray = np.nan
    forward_rate_type: RFRFixingCalcMethod=RFRFixingCalcMethod.DAILY_COMPOUNDED
    compound_spread: bool=False # TODO
    rfr_swap_curve: Optional[RFRSwapCurve]=None

    def __post_init__(self):
        super().__post_init__()
        self._add_contractual_coupon_param(self.spread)
        # self.schedule.df[self._get_legtype_specific_schedule_cols()] = np.nan
        # renamed_columns = {'coupon_contractual_component': self._get_solved_field_name()}
        # self.schedule.df = self.schedule.df.rename(columns=renamed_columns)
        # self.schedule.df[self._get_solved_field_name()] = self.spread
        #self.calc_notional_schedule(coupon_fields_to_set_to_zero=['fixing','spread','coupon_rate'])

    def calc_coupon_payment(self, coupon_override=None):
        if self.rfr_swap_curve is None:
            raise MarketDataNotAvailableError('OIS curve not available.')

        self.schedule.df['fixing'] = np.nan # TODO
        if coupon_override is not None:
            self.schedule.df['spread'] = coupon_override
        self.schedule.df['coupon_rate'] = self.schedule.df['fixing'] + self.schedule.df['spread']
        self.schedule.df['coupon_payment'] = self.schedule.df['notional'] * self.schedule.df['coupon_rate'] * self.schedule.df['period_yearfrac']
        mask = pd.isna(self.schedule.df['coupon_payment_date'])
        self.schedule.df.loc[mask,'coupon_payment'] = 0.0

    def _calc_accrued_interest(self):
        current_period_idxs = (self.schedule.df['period_start'] <= frm.utils.VALUE_DATE) & (self.schedule.df['period_end'] > frm.utils.VALUE_DATE)
        if current_period_idxs.sum() > 1:
            warnings.warn('Multiple periods found for accrual interest calculation.')
        accrued_period_yearfrac = year_frac(self.schedule.df['period_start'][current_period_idxs].iloc[0], frm.utils.VALUE_DATE, day_count_basis=self.day_count_basis)
        accrued_interest = (self.pay_rcv.multiplier * self.schedule.df['notional'][current_period_idxs]
                            * self.schedule.df['coupon_rate'][current_period_idxs]
                            * accrued_period_yearfrac).sum()
        return accrued_interest

    def _get_solved_field_name(self):
        return 'spread'

    def _get_legtype_specific_schedule_cols(self):
        return ['fixing','spread','coupon_rate']


@dataclass
class ZerocouponLeg(Leg):
    zero_rate: float=np.nan
    compounding_freq: CompoundingFreq=CompoundingFreq.ANNUAL

    def __post_init__(self):
        super().__post_init__()
        self._add_contractual_coupon_param(self.zero_rate)
        # renamed_columns = {'coupon_contractual_component': self._get_solved_field_name()}
        # self.schedule.df = self.schedule.df.rename(columns=renamed_columns)
        # self.schedule.df['zero_rate'] = self.zero_rate
        # TODO support specifying the terminal notional

    def calc_coupon_payment(self, coupon_override=None):
        if coupon_override is not None:
            self.schedule.df['zero_rate'] = coupon_override
        multiplier = (1.0 / discount_factor_from_zero_rate(
            years=self.schedule.df['period_yearfrac'],
            zero_rate=self.schedule.df['zero_rate'],
            compounding_freq=self.compounding_freq)) - 1.0
        self.schedule.df['coupon_payment'] = self.pay_rcv.multiplier * self.schedule.df['notional'] * multiplier
        mask = pd.isna(self.schedule.df['coupon_payment_date'])
        self.schedule.df.loc[mask,'coupon_payment'] = 0.0

    def _calc_accrued_interest(self):
        current_period_idxs = (self.schedule.df['period_start'] <= frm.utils.VALUE_DATE) & (
                    self.schedule.df['period_end'] > frm.utils.VALUE_DATE)
        if current_period_idxs.sum() > 1:
            warnings.warn('Multiple periods found for accrual interest calculation.')
        accrued_period_yearfrac = year_frac(self.schedule.df['period_start'][current_period_idxs].iloc[0], frm.utils.VALUE_DATE,
                                         day_count_basis=self.day_count_basis)
        multiplier = (1.0 / discount_factor_from_zero_rate(
            years=accrued_period_yearfrac,
            zero_rate=self.schedule.df['zero_rate'],
            compounding_freq=self.compounding_freq)) - 1.0
        accrued_interest =  self.pay_rcv.multiplier * self.schedule.df['notional'] * multiplier
        return accrued_interest

    def _get_solved_field_name(self):
        return 'zero_rate'

    def _get_legtype_specific_schedule_cols(self):
        return ['zero_rate']

# TBC at later date.
class InflationZCLeg(Leg):
    fixing_lag_months: int=2
    initial_fixing: float=np.nan

    def __post_init__(self):
        super().__post_init__()
        #self._add_contractual_coupon_param(self.spread)


    def calc_coupon_cashflow(self):
        pass

    def _calc_accrued_interest(self):
        pass

    def _get_solved_field_name(self):
        return 'spread' # Or final fixing?

# TBC at later date.
class InflationYoYLeg(Leg):
    fixing_lag_months: int=2
    initial_fixing: float=np.nan
    # forward_rate_type: TBC

    def __post_init__(self):
        super().__post_init__()
        #self._add_contractual_coupon_param(self.spread)

    def calc_coupon_cashflow(self):
        pass

    def _calc_accrued_interest(self):
        pass

    def _get_solved_field_name(self):
        return 'spread'


