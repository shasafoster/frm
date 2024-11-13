# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from enum import Enum
import numpy as np
import os
import pandas as pd
from typing import Optional, Union
import warnings

import frm.utils
from frm.utils import Schedule, get_schedule, day_count, year_frac, get_busdaycal
from frm.enums import CompoundingFreq, TermRate, RFRFixingCalcMethod, PeriodFreq, DayCountBasis, ExchangeNotionals
from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.zero_curve_helpers import discount_factor_from_zero_rate
from scipy.optimize import root_scalar


from frm.instruments.leg import FixedLeg, FloatTermLeg, FloatRFRLeg, ZerocouponLeg

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

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
# (ii) or fwd and discount curve solve (i.e. AONIA, or BBSW 3M under pure IBOR curve)
# (iii) or discount curve only solve (for FCB)
# If it is a forward curve solve, the user needs to the discount curve.



@dataclass
class Swap(ABC):
    "Generic class to cover pricing of all types of swap legs and bonds"

    # Required parameters
    leg1: Union[FixedLeg, FloatTermLeg, FloatRFRLeg, ZerocouponLeg]
    leg2: Union[FixedLeg, FloatTermLeg, FloatRFRLeg, ZerocouponLeg]

    # Optional parameters. Potentially move into an "Instrument" class.
    trade_date: Optional[pd.Timestamp] = None
    trade_price: Optional[float] = 0
    trade_id: Optional[str] = None
    trade_comment: Optional[str] = None
    # own_cds_curve: Optional[CDSCurve] = None
    # counterparty_cds_curve: Optional[CDSCurve] = None

    def calc_dirty_pv(self):
        leg1_pv = self.leg1.calc_dirty_pv()
        leg2_pv = self.leg2.calc_dirty_pv()
        return leg1_pv + leg2_pv

    def calc_accrued_interest(self):
        leg1_accrued_interest = self.leg1.calc_accrued_interest()
        leg2_accrued_interest = self.leg2.calc_accrued_interest()
        return leg1_accrued_interest + leg2_accrued_interest

    def calc_clean_pv(self):
        return self.calc_dirty_pv() - self.calc_accrued_interest()

    def par_solve_each_leg(self):
        leg1_par = self.leg1.par_solve()
        leg2_par = self.leg2.par_solve()
        return leg1_par, leg2_par

    def target_solve_swap(self, target_pv: float, leg_number: int):
        if leg_number == 1:
            return self.leg1.target_solve(target_pv)
        elif leg_number == 2:
            return self.leg2.target_solve(target_pv)
        else:
            raise ValueError('leg_number must be 1 or 2')

    def __post_init__(self):
        assert self.leg1.pay_rcv != self.leg2.pay_rcv, 'Legs must have opposite pay/receive perspectives'


# Potentially, create bunch of classes for different types of swaps that fill in default values (i.e ZC swap, Fixed/Float swap, etc)
# Then, for common quote types, can define all the "non-quote" parameters in the class and then pass in the quote parameters.