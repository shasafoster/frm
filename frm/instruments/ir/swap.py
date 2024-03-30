# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
from frm.frm.schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS
from frm.frm.schedule.schedule import payment_schedule, VALID_DAY_ROLL, VALID_LAST_STUB, VALID_PAYMENT_TYPE, VALID_PAYMENT_FREQUENCY, VALID_STUB, VALID_ROLL_CONVENTION
from frm.frm.schedule.business_day_calendar import get_calendar
from frm.frm.instruments.ir.leg import Leg

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from typing import Literal, Optional
        
                                                       
class Swap():    
    def __init__(self,
                 **kwargs # dictionary of swap term defaults 
                 ):
        
        if kwargs != {}:
            dict_default_terms = kwargs['default_terms']
            
            tmp = {}
            for k,v in dict_default_terms.items():
                if k[:4] != 'pay_' and k[:4] != 'rec_':
                    tmp['pay_'+k] = v
                    tmp['rec_'+k] = v

        dict_init = {**dict_default_terms, **tmp, **kwargs['specified_terms']}
        
        list_all_inputs = ['effective_date',
                        'maturity_date',
                        'tenor',
                        'transaction_date',
                        'forward_starting',
                        'tenor_calendar',
                        'tenor_city_holidays',
                        'tenor_currency_holidays',
                         # Pay Schedule
                        'pay_payment_frequency',
                        'pay_roll_convention',
                        'pay_day_roll',
                        'pay_stub',
                        'pay_first_stub',
                        'pay_last_stub',
                        'pay_first_cpn_end_date',
                        'pay_last_cpn_start_date',
                        'pay_payment_type',
                        'pay_payment_delay',
                        'pay_fixing_days_ahead',
                        'pay_currency_holidays',
                        'pay_city_holidays',
                        'pay_holiday_calendar',
                         # Pay Notional & Currencies
                        'pay_notional',
                        'pay_notional_currency',
                        'pay_settlement_currency',
                        'pay_exchange_notionals',
                        'pay_MTM_notional_currency',
                        'pay_MTM_notional',
                         # Pay cpn Details
                        'pay_leg_type',
                        'pay_fixed_rate',
                        'pay_float_spread',
                        'pay_cpn_cap',
                        'pay_cpn_floor',
                        'pay_day_count_basis',
                         # Pay pricing yield curves
                        'pay_discount_curve',
                        'pay_forward_curve',
                        '',
                         # Rec Schedule
                        'rec_payment_frequency',
                        'rec_roll_convention',
                        'rec_day_roll',
                        'rec_stub',
                        'rec_first_stub',
                        'rec_last_stub',
                        'rec_first_cpn_end_date',
                        'rec_last_cpn_start_date',
                        'rec_payment_type',
                        'rec_payment_delay',
                        'rec_fixing_days_ahead',
                        'rec_currency_holidays',
                        'rec_city_holidays',
                        'rec_holiday_calendar',
                         # Rec Notional & Currencies
                        'rec_notional',
                        'rec_notional_currency',
                        'rec_settlement_currency',
                        'rec_exchange_notionals',
                        'rec_MTM_notional_currency',
                        'rec_MTM_notional',
                         # Rec cpn Details
                        'rec_leg_type',
                        'rec_fixed_rate',
                        'rec_float_spread',
                        'rec_cpn_cap',
                        'rec_cpn_floor',
                        'rec_day_count_basis',
                         # Rec pricing yield curves
                        'rec_discount_curve',
                        'rec_forward_curve',
                        ]      
        
        for k in list_all_inputs:
            if k not in dict_init.keys():
                dict_init[k] = None
   
        # Swap tenor 
        effective_date = dict_init['effective_date']
        maturity_date = dict_init['maturity_date']
        tenor = dict_init['tenor']
        transaction_date = dict_init['transaction_date']
        forward_starting = dict_init['forward_starting']
        # Pay Schedule
        pay_payment_frequency = dict_init['pay_payment_frequency']
        pay_roll_convention = dict_init['pay_roll_convention']
        pay_day_roll = dict_init['pay_day_roll']
        pay_stub = dict_init['pay_stub']
        pay_first_stub = dict_init['pay_first_stub']
        pay_last_stub = dict_init['pay_last_stub']
        pay_first_cpn_end_date = dict_init['pay_first_cpn_end_date']
        pay_last_cpn_start_date = dict_init['pay_last_cpn_start_date']
        pay_payment_type = dict_init['pay_payment_type']
        pay_payment_delay = dict_init['pay_payment_delay']
        pay_fixing_days_ahead = dict_init['pay_fixing_days_ahead']
        pay_currency_holidays = dict_init['pay_currency_holidays']
        pay_city_holidays = dict_init['pay_city_holidays']
        pay_holiday_calendar = dict_init['pay_holiday_calendar']
        # Pay Notional & Currencies
        pay_notional = dict_init['pay_notional']
        pay_transaction_price = dict_init['pay_notional']
        pay_notional_currency = dict_init['pay_notional_currency']
        pay_settlement_currency = dict_init['pay_settlement_currency']
        pay_exchange_notionals = dict_init['pay_exchange_notionals']
        pay_MTM_notional_currency = dict_init['pay_MTM_notional_currency']
        # Pay cpn Details
        pay_leg_type = dict_init['pay_leg_type']
        pay_fixed_rate = dict_init['pay_fixed_rate']
        pay_float_spread = dict_init['pay_float_spread']
        pay_cpn_cap = dict_init['pay_cpn_cap']
        pay_cpn_floor = dict_init['pay_cpn_floor']
        pay_day_count_basis = dict_init['pay_day_count_basis']
        # Pay pricing yield curves
        pay_discount_curve = dict_init['pay_discount_curve']
        pay_forward_curve = dict_init['pay_forward_curve']
        # Rec Schedule
        rec_payment_frequency = dict_init['rec_payment_frequency']
        rec_roll_convention = dict_init['rec_roll_convention']
        rec_day_roll = dict_init['rec_day_roll']
        rec_stub = dict_init['rec_stub']
        rec_first_stub = dict_init['rec_first_stub']
        rec_last_stub = dict_init['rec_last_stub']
        rec_first_cpn_end_date = dict_init['rec_first_cpn_end_date']
        rec_last_cpn_start_date = dict_init['rec_last_cpn_start_date']
        rec_payment_type = dict_init['rec_payment_type']
        rec_payment_delay = dict_init['rec_payment_delay']
        rec_fixing_days_ahead = dict_init['rec_fixing_days_ahead']
        rec_currency_holidays = dict_init['rec_currency_holidays']
        rec_city_holidays = dict_init['rec_city_holidays']
        rec_holiday_calendar = dict_init['rec_holiday_calendar']
        # Rec Notional & Currencies
        rec_notional = dict_init['rec_notional']
        rec_transaction_price = dict_init['rec_notional']
        rec_notional_currency = dict_init['rec_notional_currency']
        rec_settlement_currency = dict_init['rec_settlement_currency']
        rec_exchange_notionals = dict_init['rec_exchange_notionals']
        rec_MTM_notional_currency = dict_init['rec_MTM_notional_currency']
        # Rec cpn Details
        rec_leg_type = dict_init['rec_leg_type']
        rec_fixed_rate = dict_init['rec_fixed_rate']
        rec_float_spread = dict_init['rec_float_spread']
        rec_cpn_cap = dict_init['rec_cpn_cap']
        rec_cpn_floor = dict_init['rec_cpn_floor']
        rec_day_count_basis = dict_init['rec_day_count_basis']
        # Rec pricing yield curves
        rec_discount_curve = dict_init['rec_discount_curve']
        rec_forward_curve = dict_init['rec_forward_curve']        
        # Margin period of risk details
        
        self.pay_leg = Leg(
            effective_date=effective_date,
            maturity_date=maturity_date,
            tenor=tenor,
            transaction_date=transaction_date, 
            forward_starting=forward_starting, 
            transaction_price=pay_transaction_price,
            payment_frequency=pay_payment_frequency,
            roll_convention=pay_roll_convention,
            day_roll=pay_day_roll,
            stub=pay_stub,
            first_stub=pay_first_stub,
            last_stub=pay_last_stub,
            first_cpn_end_date=pay_first_cpn_end_date,
            last_cpn_start_date=pay_last_cpn_start_date,
            payment_type=pay_payment_type, 
            payment_delay=pay_payment_delay,
            fixing_days_ahead=pay_fixing_days_ahead,
            currency_holidays=pay_currency_holidays,
            city_holidays=pay_city_holidays,
            holiday_calendar=pay_holiday_calendar,
            notional=pay_notional,
            notional_currency=pay_notional_currency,
            settlement_currency=pay_settlement_currency,
            exchange_notionals=pay_exchange_notionals,
            MTM_notional_currency=pay_MTM_notional_currency,
            pay_rec='pay',
            leg_type=pay_leg_type,
            fixed_rate=pay_fixed_rate,
            float_spread=pay_float_spread,
            cpn_cap=pay_cpn_cap,
            cpn_floor=pay_cpn_floor,
            day_count_basis=pay_day_count_basis,
            dsc_crv=pay_discount_curve,
            fwd_crv=pay_forward_curve)
            
        self.rec_leg = Leg(
            effective_date=effective_date,
            maturity_date=maturity_date,
            tenor=tenor,
            transaction_date=transaction_date, 
            transaction_price=rec_transaction_price,
            forward_starting=forward_starting, 
            payment_frequency=rec_payment_frequency,
            roll_convention=rec_roll_convention,
            day_roll=rec_day_roll,
            stub=rec_stub,
            first_stub=rec_first_stub,
            last_stub=rec_last_stub,
            first_cpn_end_date=rec_first_cpn_end_date,
            last_cpn_start_date=rec_last_cpn_start_date,
            payment_type=rec_payment_type, 
            payment_delay=rec_payment_delay,
            fixing_days_ahead=rec_fixing_days_ahead,
            currency_holidays=rec_currency_holidays,
            city_holidays=rec_city_holidays,
            holiday_calendar=rec_holiday_calendar,
            notional=rec_notional,
            notional_currency=rec_notional_currency,
            settlement_currency=rec_settlement_currency,
            exchange_notionals=rec_exchange_notionals,
            MTM_notional_currency=rec_MTM_notional_currency,
            leg_type=rec_leg_type,
            pay_rec='rec',
            fixed_rate=rec_fixed_rate,
            float_spread=rec_float_spread,
            cpn_cap=rec_cpn_cap,
            cpn_floor=rec_cpn_floor,
            day_count_basis=rec_day_count_basis,
            dsc_crv=rec_discount_curve,
            fwd_crv=rec_forward_curve)
        
    def price(self, 
              calc_PV01=False):
        
        pay_pricing = self.pay_leg.price(calc_PV01=calc_PV01)
        rec_pricing = self.rec_leg.price(calc_PV01=calc_PV01)
        
        pricing = {}
        pricing['price'] = pay_pricing['price'] + rec_pricing['price'] 
        pricing['price_pay_leg'] = pay_pricing['price']
        pricing['price_rec_leg'] = rec_pricing['price']
        pricing['cashflows_pay_leg'] = pay_pricing['cashflows']
        pricing['cashflows_rec_leg'] = rec_pricing['cashflows']
        
        if calc_PV01:            
            dict_PV01 = {'PV01': (pay_pricing['price_discount_and_forward_shift'] + rec_pricing['price_discount_and_forward_shift']) - pricing['price'],
                         'PV01_discount': (pay_pricing['price_discount_shift'] + rec_pricing['price_discount_shift']) - pricing['price'],
                         'PV01_forward': (pay_pricing['price_forward_shift'] + rec_pricing['price_forward_shift']) - pricing['price'],
                         'PV01_pay_leg_discount_shift': pay_pricing['price_discount_shift'] - pay_pricing['price'],
                         'PV01_pay_leg_forward_shift': pay_pricing['price_forward_shift'] - pay_pricing['price'],
                         'PV01_pay_leg_discount_and_forward_shift': pay_pricing['price_discount_and_forward_shift'] - pay_pricing['price'], 
                         'PV01_rec_leg_discount_shift': rec_pricing['price_discount_shift'] - rec_pricing['price'],
                         'PV01_rec_leg_forward_shift': rec_pricing['price_forward_shift'] - rec_pricing['price'],
                         'PV01_rec_leg_discount_and_forward_shift': rec_pricing['price_discount_and_forward_shift'] - rec_pricing['price']} 
            pricing['PV01'] = dict_PV01
            
        return pricing
        
    def solve_to_par(self,
                     leg_to_adjust: str='pay_leg') -> (float, str):
        
        if leg_to_adjust == 'pay_leg':
            pricing_rec_leg = self.rec_leg.price()
            rate, msg = self.pay_leg.solver(solve_price=-pricing_rec_leg['price'])
        elif leg_to_adjust == 'rec_leg':
            pricing_pay_leg = self.pay_leg.price()
            rate, msg = self.rec_leg.solver(solve_price=-pricing_pay_leg['price'])
        
        return rate, msg

            
    def set_pay_leg_discount_curve(self, zero_crv):
        self.pay_leg.dsc_crv = zero_crv
        
    def set_rec_leg_discount_curve(self, zero_crv):   
        self.rec_leg.dsc_crv = zero_crv

    def set_discount_curve(self, zero_crv):   
        self.set_pay_leg_discount_curve(zero_crv)
        self.set_rec_leg_discount_curve(zero_crv)
    
    def set_pay_leg_forward_curve(self, zero_crv):
        self.pay_leg.fwd_crv = zero_crv
        
    def set_rec_leg_forward_curve(self, zero_crv):   
        self.rec_leg.fwd_crv = zero_crv

    def set_forward_curve(self, zero_crv):   
        self.set_pay_leg_forward_curve(zero_crv)
        self.set_rec_leg_forward_curve(zero_crv) 
                
            

                
                
                
                
                
                
                
                
                
        
        
               