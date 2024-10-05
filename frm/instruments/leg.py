# -*- coding: utf-8 -*-
import os


if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

from frm.utils.daycount import day_count, year_fraction
from frm.utils.schedule import get_schedule, get_payment_dates, get_fixing_dates
from frm.enums.utils import DayCountBasis, DayRoll, PeriodFrequency, StubType, RollConvention, TimingConvention
from frm.term_structures.swap_curve import SwapCurve, OISCurve, TermSwapCurve

from enum import Enum
import numpy as np
import pandas as pd

from dataclasses import dataclass, field, InitVar 
from typing import Optional


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

class SwapLegType(Enum):
    FIXED = 'fixed'
    FLOAT = 'float'
    OIS = 'ois'
    ZEROCOUPON = 'zerocoupon'

    def __init__(self, value):
        rate_names = {
            'fixed': 'fixed_rate',
            'float': 'spread',
            'ois': 'spread',
            'zerocoupon': 'zero_rate',
            }
        self.rate_name = rate_names[self.value]



@dataclass
class SwapLeg:
    "Generic class to cover pricing of all types of swap legs and bonds"
    
    # Tenor definitions
    # (i) Specifically define the effective/maturity date of the instrument

    # Required parameters
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    payment_frequency: PeriodFrequency
    day_count_basis: DayCountBasis
    leg_type: SwapLegType

    pay_rec: Optional[PayRcv]=PayRcv.PAY
    fixed_rate: Optional[float]=None
    spread: Optional[float]=None

    # Schedule parameters. Initialisation parameters only, not stored in the class.
    first_stub: InitVar[StubType]=StubType.DEFAULT
    last_stub: InitVar[StubType]=StubType.DEFAULT
    first_cpn_end_date: InitVar[Optional[pd.Timestamp]]=None
    last_cpn_start_date: InitVar[Optional[pd.Timestamp]]=None
    day_roll: InitVar[DayRoll]=DayRoll.NONE
    roll_convention: InitVar[RollConvention]=RollConvention.MODIFIED_FOLLOWING
    roll_user_specified_dates: InitVar[bool] = False
    # Payment date parameters. Initialisation parameters only, not stored in the class.
    payment_delay: InitVar[int]=0
    payment_timing: InitVar[TimingConvention] = TimingConvention.IN_ARREARS
    # Fixing date parameters. Applicable to LEGTYPE.FLOAT only. Initialisation parameters only, not stored in the class.
    fixing_days_ahead: InitVar[int]=2
    fixing_timing: InitVar[TimingConvention] = TimingConvention.IN_ADVANCE
    # The same business day calendar is used for schedule, payment and fixing dates.
    busdaycal: InitVar[np.busdaycalendar] = np.busdaycalendar()

    # Notional and currencies definitions
    notional: Optional[float]=100e6
    #notional_currency: Optional[str]=None # Should this be in the leg class?
    #settlement_currency: Optional[str]=None # Should this be in the leg class?
    exchange_notionals: ExchangeNotionals=ExchangeNotionals.BOTH

    # Pricing curves
    #forward_curve: Optional[ZeroCurve]=None
    #discount_curve: Optional[ZeroCurve]=None

    # Attributes created in post_init
    schedule: pd.DataFrame=field(init=False)

    
    def __post_init__(self,
                      # Schedule parameters
                      first_stub,
                      last_stub,
                      first_cpn_end_date,
                      last_cpn_start_date,
                      day_roll,
                      roll_convention,
                      roll_user_specified_dates,
                      # Payment date parameters
                      payment_delay,
                      payment_timing,
                      # Fixing date parameters
                      fixing_days_ahead,
                      fixing_timing,
                      # Business day calendar
                      busdaycal):

        if self.leg_type == SwapLegType.FIXED:
            assert self.fixed_rate is not None
        if self.leg_type == SwapLegType.FLOAT or self.leg_type == SwapLegType.OIS:
            assert self.spread is not None


        schedule = get_schedule(
            start_date=self.start_date,
            end_date=self.end_date,
            frequency=self.payment_frequency,
            roll_convention=roll_convention,
            day_roll=day_roll,
            first_cpn_end_date=first_cpn_end_date,
            last_cpn_start_date=last_cpn_start_date,
            first_stub_type=first_stub,
            last_stub_type=last_stub,
            busdaycal=busdaycal,
            roll_user_specified_dates=roll_user_specified_dates,
        )

        schedule['days'] = day_count(schedule['period_start'], schedule['period_end'],self.day_count_basis)
        schedule['years'] = year_fraction(schedule['period_start'],schedule['period_end'],self.day_count_basis)

        match self.leg_type:

            case SwapLegType.FIXED:
                schedule['fixed_rate'] = self.fixed_rate

            case SwapLegType.FLOAT:
                fixing_dates = get_fixing_dates(
                    schedule=schedule,
                    fixing_days_ahead=fixing_days_ahead,
                    roll_convention=RollConvention.PRECEDING, # Hardcoded in Leg class for now
                    fixing_timing=TimingConvention.IN_ADVANCE, # Hardcoded in Leg class for now
                    busdaycal=busdaycal)
                schedule.insert(loc=0, column='fixing_date', value=fixing_dates)
                schedule['fixing'] = np.nan
                schedule['spread'] = self.spread
                schedule['coupon_rate'] = np.nan

            case SwapLegType.OIS:
                # Hardcoded in Leg class for now
                schedule.insert(loc=0, column='observation_start', value=schedule['period_start'])
                schedule.insert(loc=1, column='observation_end', value=schedule['period_end'])
                schedule['fixing'] = np.nan
                schedule['spread'] = self.spread
                schedule['coupon_rate'] = np.nan

        schedule['notional'] = self.notional
        schedule['payment_date'] = get_payment_dates(
            schedule=schedule,
            roll_convention=roll_convention,
            payment_timing=payment_timing,
            payment_delay=payment_delay,
            busdaycal=busdaycal)
        schedule['payment'] = np.nan

        if self.exchange_notionals in {ExchangeNotionals.BOTH, ExchangeNotionals.START}:
            row = pd.DataFrame(data={'notional': self.notional,
                                     'payment_date': self.start_date,
                                     'payment': -1*self.notional}, index=[0])
            row = row.reindex(columns=schedule.columns)
            schedule = pd.concat(objs=[row, schedule], ignore_index=True)

        if self.exchange_notionals in {ExchangeNotionals.BOTH, ExchangeNotionals.END}:
            row = pd.DataFrame(data={'notional':self.notional,
                                     'payment_date':self.end_date,
                                     'payment': self.notional}, index=[len(schedule)])
            row = row.reindex(columns=schedule.columns)
            schedule = pd.concat(objs=[schedule, row], ignore_index=True)

        schedule['discount_factor'] = np.nan
        schedule['present_value'] = np.nan
        schedule.reset_index(drop=True, inplace=True)
        schedule = self.calculate(schedule)

        self.schedule = schedule


    def calculate(self, schedule: pd.DataFrame) -> pd.DataFrame:
        mask_coupon = schedule['days'] > 0
        if self.leg_type == SwapLegType.FIXED:
            schedule.loc[mask_coupon, 'payment'] = (schedule['notional'] * schedule['years'] * schedule['fixed_rate'])[
                mask_coupon]
        if self.leg_type == SwapLegType.FLOAT or self.leg_type == SwapLegType.OIS:
            schedule.loc[mask_coupon, 'coupon_rate'] = schedule['fixing'] + schedule['spread']
            schedule.loc[mask_coupon, 'payment'] = \
            (schedule['notional'] * schedule['years'] * (schedule['fixing'] + schedule['spread']))[mask_coupon]
        schedule['present_value'] = schedule['payment'] * schedule['discount_factor']
        return schedule

    def price(self,
              forward_curve: SwapCurve,
              discount_curve: SwapCurve,
              rate_overide: Optional[float]=None,
              return_schedule: bool=False) -> tuple:

        schedule = self.schedule.copy()

        if rate_overide is not None:
            mask_coupons = schedule['days'] > 0
            schedule.loc[mask_coupons, self.leg_type.rate_name] = rate_overide

        if self.leg_type == SwapLegType.FLOAT:
            schedule['fixing'] = forward_curve.get_fixings(schedule['fixing_date'])
        elif self.leg_type == SwapLegType.OIS:
            schedule['fixing'] = forward_curve.get_fixings(schedule['observation_start'], schedule['observation_end'])

        mask_future = schedule['payment_date'] >= forward_curve.curve_date
        schedule.loc[mask_future,'discount_factor'] = discount_curve.get_discount_factors(schedule.loc[mask_future,'payment_date'])
        schedule.loc[~mask_future, 'discount_factor'] = 0.0

        schedule = self.calculate(schedule)
        present_value = schedule['present_value'].sum() * self.pay_rec.multiplier

        if return_schedule:
            return present_value, schedule
        else:
            return present_value


leg = SwapLeg(start_date=pd.Timestamp('2020-03-18'),
          end_date=pd.Timestamp('2031-09-18'),
          payment_frequency=PeriodFrequency.QUARTERLY,
          day_count_basis=DayCountBasis._30_360,
          leg_type=SwapLegType.OIS,
          spread=0)

print(leg.schedule)
print(leg.schedule.columns.to_list())


#%%%







    # def solver(self,
    #            solve_price: Optional[float]=None) -> (float, str):
    #
    #     if solve_price is None:
    #         if self.pay_rec == 'pay':
    #             solve_price = -self.notional
    #         elif self.pay_rec == 'rec':
    #             solve_price = self.notional
    #
    #     if self.leg_type == LegType.FIXED:
    #         initial_rate = self.fixed_rate
    #         def solver_helper(fixed_rate: [float], leg:Leg, solve_price: float) -> float:
    #             return leg.price(fixed_rate_overide=fixed_rate, calc_PV01=False)['price'] - solve_price
    #     elif self.leg_type == LegType.FLOAT or self.leg_type == LegType.OIS:
    #         initial_rate = self.float_spread
    #         def solver_helper(float_spread: [float], leg:Leg, solve_price: float) -> float:
    #             return leg.price(float_spread_overide=float_spread, calc_PV01=False)['price'] - solve_price
    #
    #     x, infodict, ier, msg = fsolve(solver_helper, [initial_rate], args=(self, solve_price), xtol=1, full_output=True)
    #     return x[0], msg
    #
    #
    # def set_discount_curve(self, zero_crv):
    #     self.dsc_crv = zero_crv
    #
    # def set_forward_curve(self, zero_crv):
    #     self.fwd_crv = zero_crv





        
        
               