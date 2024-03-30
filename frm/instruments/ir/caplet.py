# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS
from frm.frm.schedule.schedule import payment_schedule, VALID_DAY_ROLL, VALID_PAYMENT_TYPE, VALID_PAYMENT_FREQUENCY, VALID_STUB, VALID_STUB_GENERAL, VALID_ROLL_CONVENTION
from frm.frm.schedule.business_day_calendar import get_calendar, VALID_CITY_HOLIDAYS, VALID_CURRENCY_HOLIDAYS
from frm.frm.market_data.iban_ccys import VALID_CCYS

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from dataclasses import dataclass, field, InitVar 
from typing import Literal, Optional



@dataclass
class CapFloor:
    "Generic class to cover pricing of all types of swap legs and bonds"
    
    # Tenor definitions
    # (i) Specifically define the effective/maturity date of the instrument
    effective_date: Optional[pd.Timestamp]=None
    maturity_date: Optional[pd.Timestamp]=None
    # (ii) Alternatively define the tenor and transcation date/trade date/pricing date 
    #      and the effective_date and maturity_date will be computed
    tenor: Optional[pd.DateOffset]=None, 
    transaction_date: Optional[pd.Timestamp]=None
    forward_starting: int=0 # number of days after the transaction date the instrument starts
    
    transaction_price: Optional[float]=100.0 # As a % of par. 100.0 is the par pricing
    
    # Coupon definitions
    pay_rec: Optional[Literal['pay','rec']]=None
    cap: Optional[float]=None
    floor: Optional[float]=None
    day_count_basis: InitVar[VALID_DAY_COUNT_BASIS] = 'ACT/ACT'
        
    # Notional and currencies definitions
    notional: float=100e6
    
    # Pricing curves
    dsc_crv: Optional['ZeroCurve']=None
    fwd_crv: Optional['ZeroCurve']=None
    
    # Non initialisation arguments
    daycounter: DayCounter = field(init=False)
    
    
    def __post_init__(self, day_count_basis, holiday_calendar):
             
                
        if self.forward_starting is None: self.forward_starting = 0
        if self.roll_convention is None: self.roll_convention='modifiedfollowing'
        if self.stub is None: self.stub = 'first_short'
        if self.payment_type is None: self.payment_type ='in_arrears'
        if self.payment_delay is None: self.payment_delay=0
        if self.fixing_days_ahead is None: self.fixing_days_ahead=2
        if self.notional is None: self.notional=100e6
        if self.exchange_notionals is None: self.exchange_notionals = 'both'
        
        assert bool(self.effective_date is not None and self.maturity_date is not None) \
            != bool(self.tenor is not None and self.transaction_date is not None)
        assert self.fixed_rate is None or self.float_spread is None
        assert self.payment_frequency is not None
        assert self.notional_currency is not None
        assert self.notional > 0
        
        if self.settlement_currency is None: self.settlement_currency = self.notional_currency
        if self.currency_holidays is None: self.currency_holidays = [self.notional_currency, self.settlement_currency]
        if holiday_calendar is None: 
            holiday_calendar = get_calendar(self.currency_holidays, self.city_holidays) 
        if self.effective_date is None: 
            self.effective_date = np.busday_offset(pd.DatetimeIndex([self.transaction_date + pd.DateOffset(days=self.forward_starting)]).values.astype('datetime64[D]'), offsets=0, roll=self.roll_convention, busdaycal=holiday_calendar)[0]
        if self.maturity_date is None: 
            self.maturity_date = np.busday_offset(pd.DatetimeIndex([self.effective_date + self.tenor]).values.astype('datetime64[D]'), offsets=0, roll='following', busdaycal=holiday_calendar)[0]

        self.daycounter = DayCounter(day_count_basis)
                
        if self.fixed_rate is not None: self.leg_type = 'fixed'
        if self.float_spread is not None: self.leg_type = 'float'