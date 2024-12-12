# -*- coding: utf-8 -*-
import os
import pandas as pd

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from frm.utils import CouponSchedule
from frm.instruments.leg import FixedLeg, FloatTermLeg
from frm.enums import PeriodFreq, ExchangeNotionals, DayCountBasis



def test_swap():

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

