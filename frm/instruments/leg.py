# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from enum import Enum
import numpy as np
import os
import pandas as pd
from typing import Optional, Union
import warnings
from frm.utils import Schedule, get_schedule, day_count, year_frac, get_busdaycal
from frm.enums import CompoundingFrequency, TermRate, RFRFixingCalcMethod, PeriodFrequency, DayCountBasis
from frm.term_structures.zero_curve import ZeroCurve

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))


# I am writing a (python) class to support the valution of generic swap legs and bonds.

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

# I am not sure whether to have a separate class for each leg type or to have a single class with a leg type attribute.
# If I have a leg type attribute I will have an if statement in the calculate method to determine the correct calculation.


# Swap class
# Two legs, pay leg and rcv leg.

# Multi curve class
# It is possible to use zero curves + fixings though there will be lots of fiddly bits to get right:
# - DV01 (if swap uses multiple curves of same currency)
# - the kinked AUD swap curve (for swaption term structure construction / IRS bootstrapping
# - converting a swaption vol surface from 3M to 6M using basis spreads.
# - keeping track of basis spread curves.
# - stub coupon calculation - use part of the 1M and 3M curves.
# Hence, in a multi-curve class we would:
# - have helpers for G10 swap curves
# - do global DV01 shift across AONIA, BBSW 1M/3M/6M.
# - bootstrap
# - +/- 100 bps shift to par rates or DV01 done on par rates
# We want to be able to add various rates:

# AONIA
# (a) AONIA swap rates
# (b) AONIA and BBSW 3M basis spreads

# Defining the 3M and 6M curves:
# (a) BBSW 3M swap rates for <= 3Y and BBSW 6M swap rates for > 3Y and basis spreads
# (b) BBSW 6M swap rates and 3M/6M basis spreads
# (c) BBSW 3M swap rates and 3M/6M basis spreads
# (d) BBSW 6M swap rates and BBSW 3M swap rates

# Defining the 1M curve.
# BBSW 1m/3m basis spreads

# Defining the FCB curve
# AONIA/BBSW 3M basis spread curve
# AONIA vs SOFR FCB curve, BBSW 3M vs SOFR FCB curve

# The user defines buckets of quotes:
# The quotes either (i) purely define a zero curve or (ii) define the basis between two curves.
# It is probably best to "hard code" the process.

# Method:
# Choice (a) OIS DC Stripping? Requires to have fixed rates for OIS curve if doing a iterative bootstrap solve.
# For each instrument defining a solve user needs to define
# (i) if the curve is a forward curve only solve
# (ii) or fwd and discount curve solve (i.e AONIA, or BBSW 3M under pure IBOR curve)
# (iii) or discount curve only solve (for FCB)
# If it is a forward curve solve, the user needs to the discount curve.


class ExchangeNotionals(Enum):
    START = 'start'
    END = 'end'
    BOTH = 'both'
    NEITHER = 'neither'

class PayRcv(Enum):
    PAY = 'pay'
    RCV = 'rcv'

    @property
    def multiplier(self):
        return -1 if self == PayRcv.PAY else 1


# class LegType(Enum):
#     FIXED = 'fixed'
#     ZEROCOUPON = 'zerocoupon'
#     FLOAT_TERM = 'float'
#     FLOAT_RFR = 'ois'
#     INFLATION_ZEROCOUPON = 'zerocoupon_inflation'
#     INFLATION_YOY = 'zerocoupon_yoy'
#
#     def __init__(self, value):
#         rate_names = {
#             'fixed': 'fixed_rate',
#             'float': 'spread',
#             'ois': 'spread',
#             'zerocoupon': 'zero_rate',
#             }
#         self.rate_name = rate_names[self.value]
#


@dataclass
class Leg(ABC):
    "Generic class to cover pricing of all types of swap legs and bonds"

    # Required parameters
    schedule: Schedule
    discount_curve: Optional[ZeroCurve]
    pay_rec: Optional[PayRcv]=PayRcv.PAY
    day_count_basis: DayCountBasis=DayCountBasis

    # Notional details, optional
    notional: Union[float, np.ndarray] = 100e6
    exchange_notionals: ExchangeNotionals=ExchangeNotionals.BOTH

    # If schedule has a payment delay, interest on a period starts accruing before prior period's payment.
    cap_accrued_interest_to_unsettled_period: bool=True


    @abstractmethod
    def calc_payment_schedule(self):
        pass


    def calc_notional_schedule(self):
        if np.atleast_1d(self.notional).shape == (1,):
            self.schedule.df['notional'] = self.notional
            self.schedule.df['notional_payment'] = 0 # TODO do row diff and set row 0 to 0 .

            column_order = self.schedule.df.columns

            if self.exchange_notionals == ExchangeNotionals.START or self.exchange_notionals == ExchangeNotionals.BOTH:
                # Prepend a row for the initial notional payment
                row_data = {'period_years': 0,
                            'payment_date': self.schedule.df['period_start'].iloc[0],
                            'notional': self.schedule.df['notional'].iloc[0],
                            'notional_payment': -1 * self.pay_rec.multiplier * self.notional}
                self.schedule.df = pd.concat([pd.DataFrame(row_data, index=[0]), self.schedule.df], ignore_index=True)

            if self.exchange_notionals == ExchangeNotionals.END or self.exchange_notionals == ExchangeNotionals.BOTH:
                # Append a row for the terminal notional payment
                row_data = {'period_years': 0,
                            'payment_date': self.schedule.df['period_end'].iloc[-1],
                            'notional': self.schedule.df['notional'].iloc[-1],
                            'notional_payment': self.pay_rec.multiplier * self.notional}
                self.schedule.df = pd.concat([self.schedule.df, pd.DataFrame(row_data, index=[len(self.schedule.df)])], ignore_index=True)

            self.schedule.df = self.schedule.df[column_order]
            self.schedule.df.reset_index(drop=True, inplace=True)

    def update_discounted_cashflows(self):
        self.schedule.df['discount_factor'] = self.discount_curve.get_discount_factors(self.schedule.df['payment_date'])
        self.schedule.df['payment_present_value'] = self.schedule.df['payment'] * self.schedule.df['discount_factor']


    def par_solve(self, target_value):
        # Shared parameter solver
        # Solve adjustment to to the rate.
        # Add this adj, to the fixed rate or spread.

        pass


    def target_solve(self, target_price, target_value):
        # Shared parameter solver
        pass


    def calc_accrued_interest(self):
        # Need to check how is done for ZC - as expected.
        pass


    def calc_dirty_pv(self):
        self.update_discounted_cashflows()
        return self.schedule.df['payment_present_value'].sum()


    def calc_dv01(self):
        # Shared DV01 calculation with +/- 100 bps shift
        pass


    def __post_init__(self):
        self.schedule.add_period_length_to_schedule(day_count_basis=self.day_count_basis)
        self.calc_notional_schedule()
        #self.calc_cashflow_schedule()
        #self.schedule.df['total_payment'] = np.nan
        #self.schedule.df['discount_factor'] = np.nan
        #self.schedule.df['total_payment_present_value'] = np.nan


@dataclass
class FixedLeg(Leg):
    fixed_rate: float=np.nan

    def __post_init__(self):
        super().__post_init__()

    def calc_payment_schedule(self, solver_override=None):
        self.schedule.df['fixed_rate'] = self.fixed_rate if solver_override is None else solver_override
        self.schedule.df['cashflow'] = self.schedule.df['notional'] * self.schedule.df['fixed_rate'] * self.schedule.df['period_years']
        self.schedule.df['total_payment'] = self.schedule.df['notional_payment'] + self.schedule.df['cashflow']

    def calc_accrued_interest(self):

        value_date = frm.utils.value_date

        # Calculate the unsettled cashflows
        if frm.utils.include_payments_on_value_date_in_npv:
            mask = (self.schedule.df['payment_date'] >= value_date) & (self.schedule.df['period_end'] <= value_date)
        else:
            mask = (self.schedule.df['payment_date'] > value_date) & (self.schedule.df['period_end'] <= value_date)
        if mask.sum() > 1:
            warnings.warn('Multiple periods found for unsettled cashflow calculation.')
        unsettled_cashflows = self.schedule.df['cashflow'].sum()

        # Calculate the accrued interest
        current_period_idxs = (self.schedule.df['period_start'] <= value_date) & (self.schedule.df['period_end'] > value_date)
        if current_period_idxs.sum() > 1:
            warnings.warn('Multiple periods found for accrual interest calculation.')
        accrued_period_years = year_frac(self.schedule.df['period_start'][current_period_idxs].iloc[0], value_date, day_count_basis=self.day_count_basis)
        accrued_interest = (self.schedule.df['notional'][current_period_idxs]
                            * self.schedule.df['fixed_rate'][current_period_idxs]
                            * accrued_period_years).sum()

        if frm.utils.limit_accrued_interest_to_unsettled_cashflow:
            if unsettled_cashflows > 0:
                return unsettled_cashflows
            else:
                return accrued_interest
        else:
            return accrued_interest + unsettled_cashflows


class ZerocouponLeg(Leg):
    zero_rate: float=np.nan
    compounding_frequency: CompoundingFrequency=CompoundingFrequency.ANNUAL
    # Ending notional?

    def __post_init__(self):
        super().__post_init__()

    def calc_cashflow_schedule(self):
        self.schedule.df['zero_rate'] = self.zero_rate


class FloatTermLeg(Leg):
    spread: float=np.nan
    forward_rate_type: TermRate=TermRate.SIMPLE
    forward_curve: Optional[ZeroCurve]=None

    def __post_init__(self):
        super().__post_init__()

        if self.forward_curve is None:
            self.forward_curve = self.discount_curve

    def calc_payment_schedule(self):
        self.schedule.df['fixing'] = np.nan
        self.schedule.df['coupon_rate'] = self.schedule.df['fixing'] + self.spread
        self.schedule.df['cashflow'] = self.schedule.df['notional'] * self.schedule.df['coupon_rate'] * self.schedule.df['period_years']
        self.schedule.df['total_payment'] = self.schedule.df['notional_payment'] + self.schedule.df['cashflow']


class FloatRFRLeg(Leg):
    spread: float=np.nan
    forward_rate_type: RFRFixingCalcMethod=RFRFixingCalcMethod.DAILY_COMPOUNDED
    compound_spread: bool=False
    forward_curve: Optional[ZeroCurve]=None

    def __post_init__(self):
        super().__post_init__()

        if self.forward_curve is None:
            self.forward_curve = self.discount_curve

    def calc_payment_schedule(self):
        self.schedule.df['fixing'] = np.nan
        self.schedule.df['coupon_rate'] = self.schedule.df['fixing'] + self.spread
        self.schedule.df['cashflow'] = self.schedule.df['notional'] * self.schedule.df['coupon_rate'] * self.schedule.df['period_years']
        self.schedule.df['total_payment'] = self.schedule.df['cashflow'] + self.schedule.df['notional_payment']


class InflationZCLeg(Leg):
    fixing_lag_months: int=2
    initial_fixing: float=np.nan

    def __post_init__(self):
        super().__post_init__()

    def calc_payment_schedule(self):
        # Specific implementation for InflationZC leg
        pass


class InflationYoYLeg(Leg):
    fixing_lag_months: int=2
    initial_fixing: float=np.nan
    # forward_rate_type: TBC

    def __post_init__(self):
        super().__post_init__()

    def calc_payment_schedule(self):
        # Specific implementation for InflationYoY leg
        pass


#%%



fp = 'C:/Users/shasa/Documents/frm_private/tests_private/test_optionlet_support_20240628.xlsm'
curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')
day_count_basis = DayCountBasis.ACT_365

zero_curve = ZeroCurve(curve_date=curve_date,
                       data=pd.read_excel(io=fp, sheet_name='DF_3M'),
                       day_count_basis=day_count_basis,
                       busdaycal=busdaycal,
                       interpolation_method='cubic_spline_on_zero_rates')

# Schedule parameters
schedule = Schedule(start_date=pd.Timestamp('2024-07-01'), end_date=pd.Timestamp('2025-07-01'), frequency=PeriodFrequency.QUARTERLY)

fixed_leg = FixedLeg(schedule=schedule, discount_curve=zero_curve, fixed_rate=0.05, day_count_basis=day_count_basis)
               