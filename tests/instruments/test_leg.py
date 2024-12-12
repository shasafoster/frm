# -*- coding: utf-8 -*-
import os
import pandas as pd




if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from frm.utils import CouponSchedule
from frm.instruments.leg import FixedLeg, FloatTermLeg, FloatRFRLeg, ZerocouponLeg
from frm.enums import PeriodFreq, ExchangeNotionals, DayCountBasis



def test_fixed_leg():

    schedule = CouponSchedule(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2021-12-31'),
        freq=PeriodFreq.QUARTERLY,
        notional_amount=100e6,
        exchange_notionals=ExchangeNotionals.BOTH)

    fixed_leg = FixedLeg( # noqa F841
        schedule=schedule,
        fixed_rate=0.01,
        day_count_basis=DayCountBasis.ACT_365,
        discount_curve=None)


def test_float_term_leg():

    schedule = CouponSchedule(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2021-12-31'),
        freq=PeriodFreq.QUARTERLY,
        notional_amount=100e6,
        exchange_notionals=ExchangeNotionals.BOTH)

    float_leg = FloatTermLeg( # noqa F841
        schedule=schedule,
        spread=0.01,
        day_count_basis=DayCountBasis.ACT_365,
        discount_curve=None,
        term_swap_curve=None)

def test_float_rfr_leg():

    schedule = CouponSchedule(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2021-12-31'),
        freq=PeriodFreq.QUARTERLY,
        notional_amount=100e6,
        exchange_notionals=ExchangeNotionals.BOTH)

    float_rfr_leg = FloatRFRLeg( # noqa F841
        schedule=schedule,
        spread=0.01,
        day_count_basis=DayCountBasis.ACT_365,
        discount_curve=None,
        rfr_swap_curve=None)


def test_zero_coupon_leg():

    schedule = CouponSchedule(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2021-12-31'),
        freq=PeriodFreq.QUARTERLY,
        notional_amount=100e6,
        exchange_notionals=ExchangeNotionals.BOTH)

    zero_coupon_leg = ZerocouponLeg( # noqa F841
        schedule=schedule,
        zero_rate=0.01,
        day_count_basis=DayCountBasis.ACT_365,
        discount_curve=None)


