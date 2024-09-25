# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
from frm.frm.schedule.business_day_calendar import get_calendar #, VALID_CITY_HOLIDAYS, VALID_CURRENCY_HOLIDAYS

import numpy as np
import pandas as pd
import datetime as dt
from typing import Literal, Optional


VALID_DAY_ROLL = Literal[tuple(range(1,32)),'EOM']
VALID_LAST_STUB = Literal['short','long']
VALID_PAYMENT_TYPE = Literal['in_arrears','in_advance']
VALID_PAYMENT_FREQUENCY = Literal['w','28d','m','q','s','a','z', \
                                  'weekly','28days','monthly','quarterly','semiannually','annually','zerocoupon']
VALID_PERIOD_TYPES = Literal['days','weeks','months','years']
VALID_ROLL_CONVENTION = Literal['actual','following','preceding','modifiedfollowing','modifiedpreceding']
VALID_STUB = Literal['short','long']
VALID_STUB_GENERAL = Literal['first_short','first_long','last_short','last_long']

nb_periods_period_type = {'w':   (1,'weeks'), 'weekly':   (1,'weeks'),
                          '28d': (4,'weeks'), '28days':   (4,'weeks'),
                          'm':   (1,'months'), 'monthly': (1,'months'),
                          'q':   (3,'months'), 'quarterly': (3,'months'),
                          's':   (6,'months'), 'semiannually': (6,'months'),
                          'a':   (1,'years'), 'annually': (1,'years')}
                          

def roll_dates(dates: pd.DatetimeIndex, 
               day_roll: VALID_DAY_ROLL) -> pd.DatetimeIndex:
    """Roll the days of the dates to the provided day_roll."""
    if str(day_roll).upper() == 'EOM' or day_roll == 31:
        return dates + pd.offsets.MonthEnd(0)
    elif day_roll < 29:
        return pd.DatetimeIndex([dt.datetime(d.year,d.month,int(day_roll)) for d in dates])
    else:
        rolled = []
        for d in dates:
            try:
                rolled.append(dt.datetime(d.year,d.month,int(day_roll)))
            except ValueError:
                rolled.append(d + pd.offsets.MonthEnd(0))
        return pd.DatetimeIndex(rolled)     


def date_offset(nb_periods: int,
                period_type: VALID_PERIOD_TYPES) -> pd.DateOffset:
    if period_type == 'months':
        return pd.DateOffset(months=nb_periods)
    elif period_type == 'years':
        return pd.DateOffset(years=nb_periods)
    elif period_type == 'weeks': 
        return pd.DateOffset(weeks=nb_periods)
    elif period_type == 'days':
        return pd.DateOffset(days=nb_periods)
    
    
def payment_schedule(start_date: pd.Timestamp,
                     end_date: pd.Timestamp,
                     payment_freq: VALID_PAYMENT_FREQUENCY,
                     roll_convention: Optional[VALID_ROLL_CONVENTION]='modifiedfollowing',
                     day_roll: Optional[VALID_DAY_ROLL]=None,
                     stub: Optional[VALID_STUB_GENERAL]='first_short',
                     first_stub: Optional[VALID_STUB]=None,
                     last_stub: Optional[VALID_STUB]=None,
                     first_cpn_end_date: Optional[pd.Timestamp]=None,
                     last_cpn_start_date: Optional[pd.Timestamp]=None,
                     payment_type: Optional[VALID_PAYMENT_TYPE]='in_arrears', 
                     payment_delay: int=0,
                     add_fixing_dates: bool=False,
                     add_initial_exchange_period: bool=False,
                     fixing_days_ahead: int=2,
                     currency_holidays=None,
                     city_holidays=None,
                     holiday_cal: np.busdaycalendar=None) -> pd.DataFrame:
    """
    Create a payment schedule.

    Parameters
    ----------
    start_date : pandas.Timestamp
        Specifies the effective date of the schedule
    end_date : pandas.Timestamp
        Specifies the expiration date of the schedule
    payment_freq : {'W','28D','M', 'Q', 'S', 'A'}
        Specify the payment frequency
    day_count_basis : {'ACT/ACT','ACT/360','ACT/365', '30/360', '30E/360', '30E/350 ISDA'}
    roll_convention : {'actual','following','preceding','modifiedfollowing','modifiedpreceding'},
        How to treat dates that do not fall on a valid day. The default is ‘raise’.
            'following' means to take the first valid day later in time.
            'preceding' means to take the first valid day earlier in time.
            'modifiedfollowing' means to take the first valid day later in time unless it is across a Month boundary, in which case to take the first valid day earlier in time.
            'modifiedpreceding' means to take the first valid day earlier in time unless it is across a Month boundary, in which case to take the first valid day later in time.
    stub : {'first short','first long','last short','last long'}
        Specify the type and location of stub period
    first_stub : {short, long}
        Specifies the type of the first stub
    last_stub : {short, long}
        Specifies the type of the last stub
    payment_type : {'in arrears','in advance'}
        Specifies when payments are made
    payment_delay : int
        Specifies how many days after period end the payment is made
    add_fixing_dates : bool
    fixing_days_ahead : int
        Specifies the number of days before the peiord start date the fixing date occur
    day_roll : {1,2,3,...,30,31,'EOM'}
        Specifies the day periods should start/end on, EOM = roll to the end of month
    currency_holidays : array of strings of currency code of the locale to observe
    city_holidays : array of strings of the city holidays to observe {'Amsterdam','AMS',
        Specifies the holidays of the countrys and finp.nancial centres to be observed
    holiday_cal : np.busdaycalendar 
        Specifies the holiday calendar to observe. If this parameter is provided, neither ccy or holidays may be provided
        
    Returns
    -------
    schedule : pandas.DataFrame
        Columns
            start_date 
            end_date 
            payment_date 
    """
        
    if roll_convention is None: roll_convention = 'modifiedfollowing'
    if payment_type is None: payment_type = 'in_arrears'
    if payment_delay is None: payment_delay = 0
    
    payment_freq = payment_freq.lower()
    roll_convention = roll_convention.lower()
    payment_type = payment_type.lower().replace(' ','')
    
    nb_periods, period_type = nb_periods_period_type[payment_freq.lower()]
    
    if holiday_cal is None: holiday_cal = get_calendar(currency_holidays, city_holidays) 
        
    if first_cpn_end_date is not None:
        # The expected first coupon end date if no stub
        expected_first_cpn_end_date = pd.Timestamp(np.busday_offset(dates=pd.DatetimeIndex([start_date - date_offset(nb_periods, period_type)]).astype(str), offsets=0, roll=roll_convention, busdaycal=holiday_cal)[0])
    if last_cpn_start_date is not None:
        # The expected last coupon start date if no stub
        expected_last_cpn_start_date = pd.Timestamp(np.busday_offset(dates=pd.DatetimeIndex([end_date - date_offset(nb_periods, period_type)]).astype(str), offsets=0, roll=roll_convention, busdaycal=holiday_cal)[0])

    # If the schedule terms specifiy 1 period
    if (first_cpn_end_date is not None and first_cpn_end_date == end_date) or \
        (last_cpn_start_date is not None and last_cpn_start_date == start_date) or \
        payment_freq in {'z','zc','zerocoupon'}:
        d1 = [start_date]
        d2 = [end_date]
        
    # If the schedule terms specifiy 2 periods
    elif first_cpn_end_date is not None and last_cpn_start_date is not None and (first_cpn_end_date == last_cpn_start_date):
        d1 = [start_date, first_cpn_end_date]
        d2 = [last_cpn_start_date, end_date]
        
    # Otherwise, calculate the schedule dates
    else:
        # If the trade has a maximum of one stub or if the first_cpn_end_date or first_cpn_end_date is specified, 
        # the date is where we expect it to be without a stub
        if (first_cpn_end_date is None and last_cpn_start_date is None) \
            or (first_cpn_end_date is not None and first_cpn_end_date == expected_first_cpn_end_date) \
            or (last_cpn_start_date is not None and last_cpn_start_date == expected_last_cpn_start_date):
                
            # Use a first short stub if:
            # 1) specified
            # 2) no stub parameters (stub, last_stub, first_cpn_end_date, last_cpn_start_date) where provided
            # 3) the last_cpn_start_date is in the expected position if there is no last stub
            if stub == 'first_short' or first_stub == 'short' \
               or stub is None and first_stub is None and last_stub is None and first_cpn_end_date is None and last_cpn_start_date is None \
               or (last_cpn_start_date is not None and last_cpn_start_date == expected_last_cpn_start_date):
                d1,d2 = backward_date_generation(start_date, end_date, nb_periods, period_type)       
                                                         
        
            # Use a last short stub if:
            # 1) specified
            # 2) the first_cpn_end_date is in the expected position if there is no first stub
            elif stub =='last_short' or last_stub == 'short' \
                or (first_cpn_end_date is not None and first_cpn_end_date == expected_first_cpn_end_date):
                d1,d2 = forward_date_generation(start_date, end_date, nb_periods, period_type) 
               
            
            # Use a first long stub only when specified
            elif stub == 'first_long' or first_stub == 'long':
                d1,d2 = backward_date_generation(start_date, end_date, nb_periods, period_type) 
                d1 = [d1[0]] + d1[2:]
                d2 = d2[1:]
                
            
            # Use a last long stub only when specified
            elif stub == 'last_long' or last_stub == 'long':
                d1,d2 = forward_date_generation(start_date, end_date, nb_periods, period_type) 
                d1 = d1[:-1]  
                d2 = d2[:-2] + [d2[-1]]
               
        # If there is a stub at both the start and end
        elif first_cpn_end_date is not None and last_cpn_start_date is not None:
            d1,d2 = forward_date_generation(first_cpn_end_date, last_cpn_start_date, nb_periods, period_type) 
            d1 = [start_date] + d1 + [last_cpn_start_date]
            d2 = [first_cpn_end_date] + d2 + [end_date]
   
            
        # dates will need to be rolled to the start date day
        elif first_cpn_end_date is None and last_cpn_start_date is not None:
            d1,d2 = backward_date_generation(start_date, last_cpn_start_date, nb_periods, period_type) 
            d1 = d1 + [last_cpn_start_date]
            d2 = d2 + [end_date]
          
            
        # dates will need to be rolled to the end date day
        elif first_cpn_end_date is not None and last_cpn_start_date is None:
            d1,d2 = forward_date_generation(first_cpn_end_date, end_date, nb_periods, period_type) 
            d1 = [start_date] + d1
            d2 = [first_cpn_end_date] + d2
            
              
        d1 = pd.DatetimeIndex(d1)
        d2 = pd.DatetimeIndex(d2)
              
        if day_roll is not None:
            d1 = roll_dates(d1, day_roll)
            d2 = roll_dates(d2, day_roll)
        
        if roll_convention != 'actual':
            # Roll the days of the schedule per the roll convention and business day holiday calendar
            d1 = pd.DatetimeIndex(np.busday_offset(dates=d1.values.astype('datetime64[D]'), offsets=0, roll=roll_convention, busdaycal=holiday_cal))
            d2 = pd.DatetimeIndex(np.busday_offset(dates=d2.values.astype('datetime64[D]'), offsets=0, roll=roll_convention, busdaycal=holiday_cal))

        # Use the dates specified in the function call (start_date, end_date, first_cpn_end_date, last_cpn_end_date) 
        # in the schedule, regardless if they fall on a non-business day. 
        d1 = d1.to_list()
        d2 = d2.to_list()
        if first_cpn_end_date is not None and last_cpn_start_date is not None:
            d1 = [start_date] + [first_cpn_end_date] + d1[2:-1] + [last_cpn_start_date]
            d2 = [first_cpn_end_date] + d2[1:-2] + [last_cpn_start_date] + [end_date]
        elif first_cpn_end_date is None and last_cpn_start_date is not None:
            d1 = [start_date] + d1[1:-1] + [last_cpn_start_date]
            d2 = d2[:-2] + [last_cpn_start_date] + [end_date]        
        elif not first_cpn_end_date is None and last_cpn_start_date is None:  
            d1 = [start_date] + [first_cpn_end_date] + d1[2:]
            d2 = [first_cpn_end_date] + d2[1:-1] + [end_date]
        else: 
            d1 = [start_date] + d1[1:]
            d2 = d2[:-1] + [end_date]
                
    df = pd.DataFrame({'period_start': d1, 'period_end': d2})
    df = df[df['period_start'] != df['period_end']]
    df.reset_index(drop=True,inplace=True)

    # Add the payment dates
    if payment_type == 'in_arrears':
        if payment_delay == 0:
            df['payment_date'] = df['period_end']
        else:
            df['payment_date'] = np.busday_offset(pd.DatetimeIndex(df['period_end']+pd.DateOffset(days=payment_delay)).values.astype('datetime64[D]'), offsets=0, roll='following', busdaycal=holiday_cal)
    elif payment_type == 'in_advance':
        if payment_delay == 0:
            df['payment_date'] = df['period_start'] 
        else:
            df['payment_date'] = np.busday_offset(pd.DatetimeIndex(df['period_start']+pd.DateOffset(days=payment_delay)).values.astype('datetime64[D]'), offsets=0, roll='following', busdaycal=holiday_cal)  

    # Add the fixing dates
    if add_fixing_dates:
        if fixing_days_ahead == 0:
            df['fixing_period_start'] = df['period_start'] 
            df['fixing_period_end'] = df['period_end'] 
        else:
            df['fixing_period_start'] = np.busday_offset(pd.DatetimeIndex(df['period_start']-pd.DateOffset(days=fixing_days_ahead)).values.astype('datetime64[D]'), offsets=0, roll='preceding', busdaycal=holiday_cal)
            df['fixing_period_end'] = np.busday_offset(pd.DatetimeIndex(df['period_end']-pd.DateOffset(days=fixing_days_ahead)).values.astype('datetime64[D]'), offsets=0, roll='preceding', busdaycal=holiday_cal)
        df = df[['fixing_period_start','fixing_period_end','period_start','period_end','payment_date']]

    # Add initial notional exchange date
    if add_initial_exchange_period:
        payment_date = np.busday_offset(pd.DatetimeIndex([start_date+pd.DateOffset(days=payment_delay)]).values.astype('datetime64[D]'), offsets=0)
        row = {'period_start': start_date,'period_end': start_date,'payment_date':payment_date}
        df = pd.concat([pd.DataFrame(row), df], ignore_index=True)

    return df
    

def forward_date_generation(start_date: pd.Timestamp, 
                            end_date: pd.Timestamp, 
                            nb_periods: int,
                            period_type: VALID_PERIOD_TYPES) -> (list, list):
    """
    Generates a schedule working forwards from start date to the end date

    Parameters
    ----------
    start_date : pd.Timestamp
        start date of the schedule
    end_date : pd.Timestamp
        end date of the schedule
    nb_periods : Integer
        Specifies the number of periods of length 'period_type' per time increment
    period_type : {'years','months','weeks','days'}
        Specifies the length of one period

    Returns
    -------
    tuble of lists
        start dates, end dates
    """

    d1_arr = []
    d2_arr = []
    d1 = start_date
    d2 = start_date + date_offset(nb_periods,period_type)
    i = 1

    while end_date > d2 :   
        d1_arr.append(d1)
        d2_arr.append(d2)
        d1 = start_date + date_offset(nb_periods*i,period_type) 
        d2 = start_date + date_offset(nb_periods*(i+1),period_type) 
        i = i + 1
    d1_arr.append(d1)
    d2_arr.append(end_date)

    return d1_arr,d2_arr


def backward_date_generation(start_date: pd.Timestamp, 
                             end_date: pd.Timestamp, 
                             nb_periods: int, 
                             period_type: VALID_PERIOD_TYPES) -> (list, list):
    """
    Generates a schedule working backwards from the end date to the start date

    Parameters
    ----------
    start_date : pd.Timestamp
        start date of the schedule
    end_date : pd.Timestamp
        end date of the schedule
    nb_periods : Integer
        Specifies the number of periods of length 'period_type' per time decrement
    period_type : {'years','months','weeks','days'}
        Specifies the length of one period

    Returns
    -------
    tuple of lists
        start dates, end dates
    """
    
    d1_arr = []
    d2_arr = []
    d1 = end_date - date_offset(nb_periods,period_type)
    d2 = end_date
    i = 1
    while start_date < d1:   
        d1_arr.append(d1)
        d2_arr.append(d2)
        d1 = end_date - date_offset(nb_periods*(i+1),period_type) 
        d2 = end_date - date_offset(nb_periods*i,period_type) 
        i = i + 1
    d1_arr.append(start_date)
    d2_arr.append(d2)

    return list(reversed(d1_arr)),list(reversed(d2_arr))


def create_date_grid_for_fx_exposures(curve_date: pd.Timestamp, 
                                      delivery_dates: np.array, 
                                      sampling_freq: str=None,
                                      date_grid_pillar: pd.DatetimeIndex=None,
                                      payments_on_value_date_have_value: bool=False,
                                      include_last_day_of_value: bool=False,
                                      include_day_after_last_day_of_value: bool=False) -> pd.DatetimeIndex:
    """
    Defines a date grid for calculating exposures
    Please note, for each trade, this defines the settlement date grid, hence the expiry
    date grid needs to be solved based on the delay of the market.
    For FX derivatives this is relively simple as market data inputs can be interploted from
    (i) expiry or (ii) settle dates.

    Parameters
    ----------
    curve_date : pd.Timestamp
        DESCRIPTION.
    delivery_dates : np.array
        DESCRIPTION.
    sampling_freq : str, optional
        DESCRIPTION. The default is None.
    date_grid_pillar : pd.DatetimeIndex, optional
        DESCRIPTION. The default is None.
    include_payments_on_value_date : bool, optional
        DESCRIPTION. The default is False.
    include_last_day_of_value : bool, optional
        DESCRIPTION. The default is False.
    include_day_after_last_day_of_value : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    date_grid : TYPE
        DESCRIPTION.

    """
        
    # Challenge is that exposures are defined using the rate-fixing - i.e the expiry date
    # The expiry dates are what we need to specify for our FX option valuation or FX rate simulation
    # The trade technically still has value until some time on day of delivery/settlement.
    # Hence the date grid should be constructed on a delivery/settlement basis. 
    # There is a pototo/potata view on if a cashflow has value on the settlement date. 
    # Our default view is no, it should be in accounts receivable, but this can be toggled. 
    
    delivery_dates = np.unique(delivery_dates) 
    max_settlement_date = delivery_dates.max()
    
    if sampling_freq is not None:    
        if sampling_freq == '1d':
            date_grid = forward_date_generation(curve_date, max_settlement_date, 1, 'days')
        elif sampling_freq == '1w':
            date_grid = forward_date_generation(curve_date, max_settlement_date, 1, 'weeks')
        elif sampling_freq == '1m':
            date_grid = forward_date_generation(curve_date, max_settlement_date, 1, 'months')
        elif sampling_freq == '3m':
            date_grid = forward_date_generation(curve_date, max_settlement_date, 3, 'months')    
        elif sampling_freq == '6m':
             date_grid = forward_date_generation(curve_date, max_settlement_date, 6, 'months')   
        elif sampling_freq == '12m':
             date_grid = forward_date_generation(curve_date, max_settlement_date, 12, 'months') 
             
        date_grid = pd.DatetimeIndex([curve_date]).append(pd.DatetimeIndex(date_grid[1]))
        
    elif date_grid_pillar is not None:
        assert isinstance(date_grid_pillar, pd.DatetimeIndex) 
        date_grid = date_grid_pillar
    else:
        raise ValueError("Either 'sampling_freq' or date_grid_pillar' must be specified")

    if include_last_day_of_value:
        if payments_on_value_date_have_value:
            last_day_of_value = delivery_dates
        else: 
            last_day_of_value = delivery_dates - pd.DateOffset(days=1)
        date_grid = pd.DatetimeIndex(np.concatenate(date_grid.values, last_day_of_value))

    elif include_day_after_last_day_of_value:
        if payments_on_value_date_have_value:
            last_day_of_value = delivery_dates + pd.DateOffset(days=1)
        else: 
            last_day_of_value = delivery_dates 
        date_grid = pd.DatetimeIndex(np.concatenate(date_grid.values, last_day_of_value))

    date_grid = date_grid.drop_duplicates().sort_values()
    return date_grid

