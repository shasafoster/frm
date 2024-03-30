# -*- coding: utf-8 -*-


import os
import pathlib

os.chdir(pathlib.Path(__file__).parent.parent.resolve()) 

from schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS
from schedule.schedule import payment_schedule, VALID_DAY_ROLL, VALID_PAYMENT_TYPE, VALID_PAYMENT_FREQUENCY, VALID_STUB, VALID_STUB_GENERAL, VALID_ROLL_CONVENTION
from calendars.calendar import get_calendar, VALID_CITY_HOLIDAYS, VALID_CURRENCY_HOLIDAYS
from market_data.iban_currencies import VALID_CURRENCIES
from instruments.leg import Leg

import datetime as dt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from dataclasses import dataclass, InitVar
from typing import Optional, Literal

                      
# MARKET_ISSUE = Literal['DOMESTIC''EURO_MTN']
# VALID_DEBT_CLASSES = Literal['first_lien', 'senior_secured', 'subordinated']


@dataclass
class Bond():
    
    # ************************** Pricing attributes ***************************
    # The pricing attributes are used by the Leg class
    
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
    pay_rec: Optional[Literal['pay','rec']]='rec'
    leg_type: Optional[Literal['float','fixed']]=None
    fixed_rate: Optional[float]=None
    float_spread: Optional[float]=None
    cpn_cap: Optional[float]=None
    cpn_floor: Optional[float]=None
    day_count_basis: [VALID_DAY_COUNT_BASIS] = '30/360'
    
    # Payment schedule definitions
    payment_frequency: VALID_PAYMENT_FREQUENCY='semiannually'
    roll_convention: VALID_ROLL_CONVENTION='modifiedfollowing'
    day_roll: Optional[VALID_DAY_ROLL]=None
    stub: VALID_STUB_GENERAL='first_short'
    first_stub: VALID_STUB='short'
    last_stub: Optional[VALID_STUB]=None
    first_cpn_end_date: Optional[pd.Timestamp]=None
    last_cpn_start_date: Optional[pd.Timestamp]=None
    payment_type: VALID_PAYMENT_TYPE='in_arrears'
    payment_delay: int=0
    fixing_delay: int=0
    currency_holidays: Optional[VALID_CURRENCY_HOLIDAYS]=None
    city_holidays: Optional[VALID_CITY_HOLIDAYS]=None
    holiday_calendar: [Optional[np.busdaycalendar]]=None
    
    # Notional and currencies definitions
    notional: float=100e6
    currency: Optional[VALID_CURRENCIES]=None
    
    # Accrued interest calculation
    calculation_type: Optional[int]=None
    settlement_delay: Optional[int]=None # T+1 T+2 T+3 ...
    ex_dividend_days: Optional[int]=None # number of business days prior to the coupon payment date you must own the bond to get the bond
        
    # Pricing curves
    dsc_crv: Optional['ZeroCurve']=None
    fwd_crv: Optional['ZeroCurve']=None
    
    # ************************ Description attributes *************************
    # amount_issued: Optional[float]=None
    # min_piece: Optional[float]=None
    # min_increment: Optional[float]=None

    # issuance_country: Optional[str]=None
    # market_issue: Optional[MARKET_ISSUE]=None
    # debt_class: Optional[VALID_DEBT_CLASSES]=None
    # bond_ratings: Optional[dict]=None
    
    # # Identifiers
    # ISIN: Optional[str]=None
    # CUSIP: Optional[str]=None
    # SEDOL: Optional[str]=None
    # FIGI: Optional[str]=None
    

    def __post_init__(self):
    
        if self.fixed_rate is not None and self.float_spread is None: self.leg_type = 'fixed'
        if self.fixed_rate is None and self.float_spread is not None: self.leg_type = 'float'
    
        self.leg = Leg(
            # Tenor definitions
            effective_date=self.effective_date,
            maturity_date=self.maturity_date, 
            tenor=self.tenor,
            transaction_date=self.transaction_date,
            forward_starting=self.forward_starting,
            transaction_price=self.transaction_price,
            # Payment schedule definitions
            payment_frequency=self.payment_frequency,
            roll_convention=self.roll_convention,
            day_roll=self.day_roll,
            stub=self.stub,
            first_stub=self.first_stub,
            last_stub=self.last_stub,
            first_cpn_end_date=self.first_cpn_end_date,
            last_cpn_start_date=self.last_cpn_start_date,
            payment_type=self.payment_type, 
            payment_delay=self.payment_delay,
            fixing_delay=self.fixing_delay,
            currency_holidays=self.currency_holidays,
            city_holidays=self.city_holidays,
            holiday_calendar=self.holiday_calendar,
            # Coupon definitions
            pay_rec='pay' if self.buy_sell == 'sell' else 'rec',
            leg_type=self.leg_type,
            fixed_rate=self.fixed_rate,
            float_spread=self.float_spread,
            cpn_cap=self.cpn_cap,
            cpn_floor=self.cpn_floor,
            day_count_basis=self.day_count_basis,
            # Notional and currencies definitions
            notional=self.notional,
            notional_currency=self.currency,
            exchange_notionals='end',
            # Pricing curves
            dsc_crv=self.dsc_crv,
            fwd_crv=self.fwd_crv)

    def price(self, 
              calc_PV01: bool=False, 
              fixed_rate_overide: Optional[float]=None, 
              float_spread_overide: Optional[float]=None) -> dict:
                
        pricing = self.leg.price(calc_PV01, fixed_rate_overide, float_spread_overide) 
        
        return pricing
                 

    def solver(self,
               solve_price: Optional[float]=None) -> (float, str):
        
        x, msg = self.leg.solver(solve_price)
        
        return x, msg
           
        