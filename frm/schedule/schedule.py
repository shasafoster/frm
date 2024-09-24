# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from typing import Literal, Optional

VALID_DAY_ROLL = list(range(1,32)) +['eom'] 


class RollConvention(Enum):
    ACTUAL = 'actual'
    FOLLOWING = 'following'
    PRECEDING = 'preceding'
    MODIFIED_FOLLOWING = 'modifiedfollowing'
    MODIFIED_PRECEDING = 'modifiedpreceding'

class PaymentType(Enum):
    IN_ARREARS = 'in_arrears'
    IN_ADVANCE = 'in_advance'
    
class StubType(Enum):
    FIRST_SHORT = 'first_short'
    FIRST_LONG = 'first_long'
    LAST_SHORT = 'last_short'
    LAST_LONG = 'last_long'


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
                          
#%%

def roll_dates(dates: pd.DatetimeIndex, 
               day_roll: Literal[VALID_DAY_ROLL]) -> pd.DatetimeIndex:
    """Roll the days of the dates to the provided day_roll."""
    
    # Input validation
    day_roll = str(day_roll).strip().lower()
    day_roll = int(day_roll) if day_roll.isdigit() else day_roll
    assert day_roll in VALID_DAY_ROLL, f'Invalid day_roll {day_roll}'
    assert isinstance(dates, pd.DatetimeIndex), f'dates must be a pd.DatetimeIndex {type(dates)} {dates}'
    
    if str(day_roll).upper() == 'eom' or day_roll == 31:
        return dates + pd.offsets.MonthEnd(0)
    
    def roll_date(d):
        try:
            return d.replace(day=day_roll)
        except ValueError:
            return d + pd.offsets.MonthEnd(0)
        
    return pd.DatetimeIndex([roll_date(d) for d in dates])


def date_offset(nb_periods: int,
                period_type: VALID_PERIOD_TYPES) -> pd.DateOffset:
    "Create a pd.DateOffset from specified nb_periods and period_type"
    match period_type:
        case 'days':
            return pd.DateOffset(days=nb_periods)
        case 'weeks': 
            return pd.DateOffset(weeks=nb_periods)
        case 'months':
            return pd.DateOffset(months=nb_periods)
        case 'years':
            return pd.DateOffset(years=nb_periods)
            
    
def payment_schedule(start_date: pd.Timestamp,
                     end_date: pd.Timestamp,
                     payment_freq: VALID_PAYMENT_FREQUENCY,
                     roll_convention: Optional[VALID_ROLL_CONVENTION]='modifiedfollowing',
                     day_roll: Optional[Literal[VALID_DAY_ROLL]]=None,
                     stub: Optional[VALID_STUB_GENERAL]='first_short',
                     first_stub: Optional[VALID_STUB]=None,
                     last_stub: Optional[VALID_STUB]=None,
                     first_cpn_end_date: Optional[pd.Timestamp]=None,
                     last_cpn_start_date: Optional[pd.Timestamp]=None,
                     payment_type: Optional[VALID_PAYMENT_TYPE]='in_arrears', 
                     payment_delay: int=0,
                     add_fixing_dates: bool=False,
                     fixing_days_ahead: int=2,
                     add_initial_exchange_period: bool=False,
                     busdaycal: np.busdaycalendar=np.busdaycalendar()) -> pd.DataFrame:
    """
    Create a payment schedule.

    Parameters
    ----------
    start_date : pandas.Timestamp
        Specifies the effective date of the schedule
    end_date : pandas.Timestamp
        Specifies the expiration date of the schedule
    payment_freq : {w','28d','m', 'q', 's', 'a', 'z'}
        Specify the payment frequency
    roll_convention : {'actual','following','preceding','modifiedfollowing','modifiedpreceding'},
        How to treat dates that do not fall on a valid day. The default is 'modifiedfollowing'.
            'following' means to take the first valid day later in time.
            'preceding' means to take the first valid day earlier in time.
            'modifiedfollowing' means to take the first valid day later in time unless it is across a month boundary, in which case to take the first valid day earlier in time.
            'modifiedpreceding' means to take the first valid day earlier in time unless it is across a month boundary, in which case to take the first valid day later in time.
    day_roll : {1,2,3,...,30,31,'eom'}
        Specifies the day periods should start/end on, 'eom' = roll to the end of month    
    stub : {'first_short','first_long','last_short','last_long'}
        Specify the type and location of stub period
    first_stub : {'short', 'long'}
        Specifies the type of the first stub
    last_stub : {'short', 'long'}
        Specifies the type of the last stub
    payment_type : {'in_arrears','in_advance'}
        Specifies when payments are made. The defaut is 'in_arrears'.
    payment_delay : int
        Specifies how many days after period start_date/end_date (if payments are in_advance/in_arrears), the payment is made. 
    add_fixing_dates : bool
        Boolean flag of whether to add a fixing date column to the schedule. Relevant if the schedule is for a floating term-rate instrument.
    fixing_days_ahead : int
        Specifies the number of days before the peiord start date the fixing date occur. The default is 2 (days ahead).
    add_initial_exchange_period : bool
        Boolean flag 
    busdaycal : np.busdaycalendar
        Specifies the business day calendar to observe. 
        
    Returns
    -------
    schedule : pandas.DataFrame
        Columns:
            - period_start
            - period_end
            - payment_date
            - fixing_period_start (optional)
            - fixing_period_end (optional)
    """
        
    # Input validation
    assert start_date < end_date, "Start date must be before end date."
    if roll_convention is None: roll_convention = RollConvention.MODIFIED_FOLLOWING.value
    if payment_type is None: payment_type = 'in_arrears'
    if payment_delay is None: payment_delay = 0
    if busdaycal is None: busdaycal = np.busdaycalendar()
    
    
    payment_freq = payment_freq.lower()
    roll_convention = roll_convention.lower()
    payment_type = payment_type.lower().replace(' ','')
    
    if payment_freq not in nb_periods_period_type:
      raise ValueError(f"Invalid payment frequency '{payment_freq}'. Must be one of {list(nb_periods_period_type.keys())}.")
    nb_periods, period_type = nb_periods_period_type[payment_freq]
    
    if first_cpn_end_date is not None:
        assert first_cpn_end_date > start_date and first_cpn_end_date <= end_date
        # The expected first coupon end date if no stub
        dt_idx = pd.DatetimeIndex([start_date - date_offset(nb_periods, period_type)]).astype(str)
        expected_first_cpn_end_date = pd.Timestamp(np.busday_offset(dates=dt_idx, offsets=0, roll=roll_convention, busdaycal=busdaycal)[0])
        
    if last_cpn_start_date is not None:
        # The expected last coupon start date if no stub
        assert last_cpn_start_date >= start_date and last_cpn_start_date < end_date
        dt_idx = pd.DatetimeIndex([end_date - date_offset(nb_periods, period_type)]).astype(str)
        expected_last_cpn_start_date = pd.Timestamp(np.busday_offset(dates=dt_idx, offsets=0, roll=roll_convention, busdaycal=busdaycal)[0])

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
        # If:
        # (a) the trade has zero or one stub (not two stubs), 
        # (b) or if the first_cpn_end_date or first_cpn_end_date is specified the date is where we expect it to be without a stub
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
                d1,d2 = date_generation(start_date, end_date, nb_periods, period_type, 'backward')       
                                                         
            # Use a last short stub if:
            # 1) specified
            # 2) the first_cpn_end_date is in the expected position if there is no first stub
            elif stub =='last_short' or last_stub == 'short' \
                or (first_cpn_end_date is not None and first_cpn_end_date == expected_first_cpn_end_date):
                d1,d2 = date_generation(start_date, end_date, nb_periods, period_type, 'forward') 
               
            # Use a first long stub only when specified
            elif stub == 'first_long' or first_stub == 'long':
                d1,d2 = date_generation(start_date, end_date, nb_periods, period_type, 'backward') 
                d1 = [d1[0]] + d1[2:]
                d2 = d2[1:]
            
            # Use a last long stub only when specified
            elif stub == 'last_long' or last_stub == 'long':
                d1,d2 = date_generation(start_date, end_date, nb_periods, period_type, 'forward') 
                d1 = d1[:-1]  
                d2 = d2[:-2] + [d2[-1]]
               
        # If there is a stub at both the start and end
        elif first_cpn_end_date is not None and last_cpn_start_date is not None:
            d1,d2 = date_generation(start_date=first_cpn_end_date, end_date=last_cpn_start_date, nb_periods=nb_periods, period_type=period_type, direction='forward') 
            d1 = [start_date] + d1 + [last_cpn_start_date]
            d2 = [first_cpn_end_date] + d2 + [end_date]
            
        # dates will need to be rolled to the start date day
        elif first_cpn_end_date is None and last_cpn_start_date is not None:
            d1,d2 = date_generation(start_date=start_date, end_date=last_cpn_start_date, nb_periods=nb_periods, period_type=period_type, direction='backward') 
            d1 = d1 + [last_cpn_start_date]
            d2 = d2 + [end_date]
          
        # dates will need to be rolled to the end date day
        elif first_cpn_end_date is not None and last_cpn_start_date is None:
            d1,d2 = date_generation(start_date=first_cpn_end_date, end_date=end_date, nb_periods=nb_periods, period_type=period_type, direction='forward') 
            d1 = [start_date] + d1
            d2 = [first_cpn_end_date] + d2
            
        d1 = pd.DatetimeIndex(d1)
        d2 = pd.DatetimeIndex(d2)
              
        if day_roll is not None:
            d1 = roll_dates(dates=d1, day_roll=day_roll)
            d2 = roll_dates(dates=d2, day_roll=day_roll)
        
        if roll_convention != RollConvention.ACTUAL.value:
            # Roll the days of the schedule per the roll convention and business day holiday calendar
            d1 = pd.DatetimeIndex(np.busday_offset(dates=d1.values.astype('datetime64[D]'), offsets=0, roll=roll_convention, busdaycal=busdaycal))
            d2 = pd.DatetimeIndex(np.busday_offset(dates=d2.values.astype('datetime64[D]'), offsets=0, roll=roll_convention, busdaycal=busdaycal))

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
    if payment_type == PaymentType.IN_ARREARS.value:
        payment_dates = pd.DatetimeIndex(df['period_end']+pd.DateOffset(days=payment_delay))
    elif payment_type == PaymentType.IN_ADVANCE.value:
        payment_dates = pd.DatetimeIndex(df['period_start']+pd.DateOffset(days=payment_delay))
    else:
        raise ValueError(f"Invalid payment type '{payment_type}'. Must be 'in_arrears' or 'in_advance'.")                                        
    df['payment_date'] = np.busday_offset(payment_dates.values.astype('datetime64[D]'), offsets=0, roll='following', busdaycal=busdaycal)
        
    # Add the fixing dates
    if add_fixing_dates:
        # Check to excel swap model
        df['fixing_period_start'] = np.busday_offset(pd.DatetimeIndex(df['period_start']-pd.DateOffset(days=fixing_days_ahead)).values.astype('datetime64[D]'), offsets=0, roll='preceding', busdaycal=busdaycal)
        df['fixing_period_end'] = np.busday_offset(pd.DatetimeIndex(df['period_end']-pd.DateOffset(days=fixing_days_ahead)).values.astype('datetime64[D]'), offsets=0, roll='preceding', busdaycal=busdaycal)
        df = df[['fixing_period_start','fixing_period_end','period_start','period_end','payment_date']]

    # Add initial notional exchange date
    # Check to excel swap model
    if add_initial_exchange_period:
        payment_date = np.busday_offset(pd.DatetimeIndex([start_date+pd.DateOffset(days=payment_delay)]).values.astype('datetime64[D]'), offsets=0)
        row = {'period_start': start_date,'period_end': start_date,'payment_date':payment_date}
        df = pd.concat([pd.DataFrame(row), df], ignore_index=True)

    return df
    

def date_generation(start_date: pd.Timestamp, 
                    end_date: pd.Timestamp, 
                    nb_periods: int, 
                    period_type: VALID_PERIOD_TYPES,
                    direction) -> (list, list):
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
    direction : {'forward','backward'}

    Returns
    -------
    tuple of lists
        start dates, end dates
    """
    
    # Input validation
    assert direction in {'forward','backward'}    
    assert start_date < end_date
    
    d1_arr = []
    d2_arr = []
    i = 0
    
    if direction == 'forward':
        current_date = start_date
        while current_date + date_offset(nb_periods, period_type) < end_date: 
            d1_arr.append(current_date)
            current_date = start_date + date_offset(nb_periods * (i + 1), period_type)
            d2_arr.append(current_date)
            i += 1
                
        d1_arr.append(current_date)
        d2_arr.append(end_date)
        
        return d1_arr, d2_arr
    
    elif direction == 'backward':
        current_date = end_date
        while current_date - date_offset(nb_periods, period_type) > start_date:   
            d2_arr.append(current_date)
            current_date = end_date - date_offset(nb_periods * (i + 1), period_type)
            d1_arr.append(current_date)
            i +=1
            
        d1_arr.append(start_date)
        d2_arr.append(current_date)
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
            date_grid = date_generation(curve_date, max_settlement_date, 1, 'days', 'forward')
        elif sampling_freq == '1w':
            date_grid = date_generation(curve_date, max_settlement_date, 1, 'weeks', 'forward')
        elif sampling_freq == '1m':
            date_grid = date_generation(curve_date, max_settlement_date, 1, 'months', 'forward')
        elif sampling_freq == '3m':
            date_grid = date_generation(curve_date, max_settlement_date, 3, 'months', 'forward')    
        elif sampling_freq == '6m':
             date_grid = date_generation(curve_date, max_settlement_date, 6, 'months', 'forward')
        elif sampling_freq == '12m':
             date_grid = date_generation(curve_date, max_settlement_date, 12, 'months', 'forward') 
             
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

