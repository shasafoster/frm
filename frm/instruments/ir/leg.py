# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS
from frm.frm.schedule.schedule import payment_schedule, VALID_DAY_ROLL, VALID_PAYMENT_TYPE, VALID_PAYMENT_FREQUENCY, VALID_STUB, VALID_STUB_GENERAL, VALID_ROLL_CONVENTION
from frm.frm.schedule.business_day_calendar import get_calendar
from frm.frm.schedule.tenor import calc_tenor_date
from frm.frm.market_data.iban_ccys import VALID_CCYS

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

from dataclasses import dataclass, field, InitVar 
from typing import Literal, Optional

VALID_EXCHANGE_NOTIONALS = Literal['start','end','both','neither']

@dataclass
class Leg:
    "Generic class to cover pricing of all types of swap legs and bonds"
    
    # Tenor definitions
    # (i) Specifically define the effective/maturity date of the instrument
    effective_date: Optional[pd.Timestamp]=None
    maturity_date: Optional[pd.Timestamp]=None
    # (ii) Alternatively define the tenor and transcation date/trade date/pricing date 
    #      and the effective_date and maturity_date will be computed
    tenor: Optional[pd.DateOffset]=None, 
    transaction_date: Optional[pd.Timestamp]=None
    forward_starting: int=2 # number of business days after the transaction date the instrument starts
    
    transaction_price: Optional[float]=100.0 # As a % of par. 100.0 is the par pricing
    
    # Coupon definitions
    pay_rec: Optional[Literal['pay','rec']]=None
    leg_type: Optional[Literal['float','fixed']]=None
    fixed_rate: Optional[float]=None
    float_spread: Optional[float]=None
    cpn_cap: Optional[float]=None
    cpn_floor: Optional[float]=None
    day_count_basis: InitVar[VALID_DAY_COUNT_BASIS] = 'ACT/ACT'
    
    # Payment schedule definitions
    payment_frequency: VALID_PAYMENT_FREQUENCY='quarterly'
    roll_convention: VALID_ROLL_CONVENTION='modifiedfollowing'
    day_roll: Optional[VALID_DAY_ROLL]=None
    stub: VALID_STUB_GENERAL='first_short'
    first_stub: VALID_STUB='short'
    last_stub: Optional[VALID_STUB]=None
    first_cpn_end_date: Optional[pd.Timestamp]=None
    last_cpn_start_date: Optional[pd.Timestamp]=None
    payment_type: VALID_PAYMENT_TYPE='in_arrears'
    payment_delay: int=0
    fixing_days_ahead: int=0
    currency_holidays=None
    city_holidays=None
    holiday_calendar: InitVar[Optional[np.busdaycalendar]]=None
    
    # Notional and currencies definitions
    notional: float=100e6
    notional_currency: Optional[str]=None
    settlement_currency: Optional[str]=None
    exchange_notionals: VALID_EXCHANGE_NOTIONALS='both'
    MTM_notional_currency: Optional[str]=None
    
    # Pricing curves
    dsc_crv: Optional['ZeroCurve']=None
    fwd_crv: Optional['ZeroCurve']=None
    
    # Non initialisation arguments
    daycounter: DayCounter = field(init=False)
    
    
    def __post_init__(self, day_count_basis, holiday_calendar):
             
                
        if self.forward_starting is None: self.forward_starting = 2
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
            tenor_date, _, _ = calc_tenor_date(self.transaction_date, self.tenor, curve_ccy=self.notional_currency, holiday_calendar=holiday_calendar)
            self.maturity_date = tenor_date
            
        
            #self.maturity_date = np.busday_offset(pd.DatetimeIndex([self.effective_date + self.tenor]).values.astype('datetime64[D]'), offsets=0, roll='following', busdaycal=holiday_calendar)[0]

        self.daycounter = DayCounter(day_count_basis)
                
        if self.fixed_rate is not None: self.leg_type = 'fixed'
        if self.float_spread is not None: self.leg_type = 'float'

        self.schedule = payment_schedule(start_date=self.effective_date,
                                         end_date=self.maturity_date,
                                         payment_freq=self.payment_frequency,
                                         roll_convention=self.roll_convention,
                                         day_roll=self.day_roll,
                                         stub=self.stub,
                                         first_stub=self.first_stub,
                                         last_stub=self.last_stub,
                                         first_cpn_end_date=self.first_cpn_end_date,
                                         last_cpn_start_date=self.last_cpn_start_date,
                                         payment_type=self.payment_type,
                                         payment_delay=self.payment_delay,
                                         add_fixing_dates=self.leg_type == 'float',
                                         add_initial_exchange_period=self.leg_type in {'both','start'},
                                         fixing_days_ahead=self.fixing_days_ahead,
                                         currency_holidays=self.currency_holidays,
                                         city_holidays=self.city_holidays,
                                         holiday_cal=self.holiday_calendar)
        self.schedule['period_length'] = self.daycounter.year_fraction(self.schedule['period_start'], self.schedule['period_end'])        

    def __price_helper(self, 
                       schedule: pd.DataFrame,
                       dsc_crv=None,
                       fwd_crv=None) -> (float, pd.DataFrame):
        
        if dsc_crv is None: dsc_crv = self.dsc_crv
        if fwd_crv is None: fwd_crv = self.fwd_crv
        if schedule is None: schedule = self.schedule
        
        if self.leg_type == 'float':
            schedule['forward_reset'] = fwd_crv.forward_rate(schedule['fixing_period_start'],schedule['fixing_period_end'])
            schedule['coupon_rate'] = schedule['forward_reset'] + schedule['spread']
        schedule['coupon_cashflow_amount'] = schedule['notional'] * schedule['coupon_rate'] * schedule['period_length']            
        schedule['net_cashflow_amount'] = schedule['principal_cashflow_amount'] + schedule['coupon_cashflow_amount']
        
        if self.notional_currency == self.settlement_currency:
            schedule['notional_per_settle_fx_forward_rate'] = 1.0
        else:
            raise ValueError('to be built')     
        
        schedule['net_settlement_amount'] = schedule['net_cashflow_amount'] / schedule['notional_per_settle_fx_forward_rate']
        schedule['discount_factor'] = dsc_crv.discount_factor(schedule['payment_date'])            
        schedule['net_settlement_value'] = schedule['net_settlement_amount'] * schedule['discount_factor']
        
        schedule['settlement_currency'] = self.settlement_currency
        price = schedule['net_settlement_value'].sum()
        if self.pay_rec == 'pay':
            price = -price
        
        return price, schedule

           
    def price(self, 
              calc_PV01: bool=False, 
              fixed_rate_overide: Optional[float]=None, 
              float_spread_overide: Optional[float]=None) -> dict:
                
        # Set the notional amounts 
        # If CCIRS was MTM calculate the MTM amounts here
        schedule = self.schedule.copy()
        schedule['notional'] = self.notional
        schedule['notional_currency'] = self.notional_currency

        # Specify notional cashflow payments      
        schedule['principal_cashflow_amount'] = 0.0
        if self.exchange_notionals in {'both','start'}:
            schedule.loc[0,'principal_cashflow_amount'] = - self.notional
        if self.exchange_notionals in {'both','end'}:
            schedule.loc[len(schedule)-1,'principal_cashflow_amount'] = self.notional
        
        if self.leg_type == 'fixed':
            if fixed_rate_overide is None: 
                schedule['coupon_rate'] = self.fixed_rate
            else:
                schedule['coupon_rate'] = fixed_rate_overide[0]
        if self.leg_type == 'float':
            if float_spread_overide is None: 
                schedule['spread'] = self.float_spread
            else:
                schedule['spread'] = float_spread_overide[0]
        
        pricing = {}
        pricing['price'], pricing['cashflows'] = self.__price_helper(schedule=schedule)
        
        if calc_PV01: 
            pricing['price_discount_shift'], _ = self.__price_helper(schedule=schedule, dsc_crv=self.dsc_crv.flat_shift())
            pricing['price_forward_shift'], _ = self.__price_helper(schedule=schedule, fwd_crv=self.fwd_crv.flat_shift())
            pricing['price_discount_and_forward_shift'], _ = self.__price_helper(schedule=schedule,
                                                                                         dsc_crv=self.dsc_crv.flat_shift(), 
                                                                                         fwd_crv=self.fwd_crv.flat_shift())
        return pricing
                 

    def solver(self,
               solve_price: Optional[float]=None) -> (float, str):
        
        if solve_price is None:
            if self.pay_rec == 'pay':
                solve_price = -self.notional
            elif self.pay_rec == 'rec':
                solve_price = self.notional
            
        if self.leg_type == 'fixed':
            initial_rate = self.fixed_rate
            def solver_helper(fixed_rate: [float], leg:Leg, solve_price: float) -> float:
                return leg.price(fixed_rate_overide=fixed_rate, calc_PV01=False)['price'] - solve_price
        elif self.leg_type == 'float':
            initial_rate = self.float_spread
            def solver_helper(float_spread: [float], leg:Leg, solve_price: float) -> float:
                return leg.price(float_spread_overide=float_spread, calc_PV01=False)['price'] - solve_price
                
        x, infodict, ier, msg = fsolve(solver_helper, [initial_rate], args=(self, solve_price), xtol=1, full_output=True)
        return x[0], msg
        
                
    def set_discount_curve(self, zero_crv):   
        self.dsc_crv = zero_crv

    def set_forward_curve(self, zero_crv):   
        self.fwd_crv = zero_crv
             
    def price_cap_floor(self, d1, d2, K, σ):
        
        T = self.daycounter(d1,d2)
        F = self.fwd_crv.forward_rate(d1,d2) 
        
        d1 = (np.log(F/K) + (0.5 * σ**2 * T)) / (σ*np.sqrt(T))
        d2 = d1 - σ*np.sqrt(T)
        
        bought_cap = F * norm.cdf(d1) - K * norm.cdf(d2)
        bought_put = K * norm.cdf(d2) - F * norm.cdf(d1)
        
        
        
        
        
                
                
                
                
        
        
               