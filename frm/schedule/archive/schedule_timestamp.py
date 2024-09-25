# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from typing import Literal, Optional, Union, List, Tuple

class RollConvention(Enum):
    NONE = None
    FOLLOWING = 'following'
    PRECEDING = 'preceding'
    MODIFIED_FOLLOWING = 'modifiedfollowing'
    MODIFIED_PRECEDING = 'modifiedpreceding'

class PaymentType(Enum):
    IN_ARREARS = 'in_arrears'
    IN_ADVANCE = 'in_advance'
    
class StubType(Enum):
    SHORT = 'short'
    LONG = 'long'

class PaymentFrequency(Enum):
    WEEKLY = 'w'
    TWENTY_EIGHT_DAYS = '28d'
    MONTHLY = 'm'
    QUARTERLY = 'q'
    SEMIANNUALLY = 's'
    ANNUALLY = 'a'
    ZERO_COUPON = 'z'

VALID_DIRECTIONS = ['backward','forward']

VALID_PAYMENT_TYPE = Literal['in_arrears','in_advance']
VALID_PAYMENT_FREQUENCY = Literal['w','28d','m','q','s','a','z', \
                                  'weekly','28days','monthly','quarterly','semiannually','annually','zerocoupon']
VALID_PERIOD_TYPES = Literal['days','weeks','months','years']

nb_periods_period_type = {'w':   (1,'weeks'), 'weekly':   (1,'weeks'),
                          '28d': (4,'weeks'), '28days':   (4,'weeks'),
                          'm':   (1,'months'), 'monthly': (1,'months'),
                          'q':   (3,'months'), 'quarterly': (3,'months'),
                          's':   (6,'months'), 'semiannually': (6,'months'),
                          'a':   (1,'years'), 'annually': (1,'years')}
                          

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
                     roll_convention: str=RollConvention.MODIFIED_FOLLOWING.value,
                     day_roll: Union[int, str]=None,
                     first_cpn_end_date: pd.Timestamp=None,
                     last_cpn_start_date: pd.Timestamp=None,                     
                     first_stub_type: str=StubType.SHORT.value,
                     last_stub_type: str=None,
                     payment_type: str=PaymentType.IN_ARREARS.value,
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
    roll_convention : {'None','following','preceding','modifiedfollowing','modifiedpreceding'},
        How to treat dates that do not fall on a valid day. The default is 'modifiedfollowing'.
            'following' means to take the first valid day later in time.
            'preceding' means to take the first valid day earlier in time.
            'modifiedfollowing' means to take the first valid day later in time unless it is across a month boundary, in which case to take the first valid day earlier in time.
            'modifiedpreceding' means to take the first valid day earlier in time unless it is across a month boundary, in which case to take the first valid day later in time.
    day_roll : {1,2,3,...,30,31,'eom'}
        Specifies the day periods should start/end on, 'eom' = roll to the end of month    
    first_cpn_end_date: pandas.Timestamp
        Specifies the end date of the first coupon period. The first_cpn_end_date overides the first_stub_type field.
    last_cpn_start_date: pandas.Timestamp
        Specifies the start date of the last coupon period. The last_cpn_start_date overides the last_stub_type field.
    first_stub_type : {'short', 'long'}
        Specifies the type of the first stub. If first_cpn_end_date is specified, the first_stub_type is ignored. Defaults to 'short'.
    last_stub_type : {'short', 'long'}
        Specifies the type of the last stub. If last_cpn_start_date is specified, the last_stub_type is ignored.                 
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
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be before end_date {end_date}")
    if roll_convention is None: 
        roll_convention = RollConvention.MODIFIED_FOLLOWING.value
    if payment_type is None: 
        payment_type = 'in_arrears'
    if payment_delay is None: 
        payment_delay = 0
    if busdaycal is None: 
        busdaycal = np.busdaycalendar()
    
    payment_freq = payment_freq.lower()
    roll_convention = roll_convention.lower()
    payment_type = payment_type.lower().replace(' ','')
    
    if payment_freq not in nb_periods_period_type:
      raise ValueError(f"Invalid payment frequency '{payment_freq}'. Must be one of {list(nb_periods_period_type.keys())}.")
    nb_periods, period_type = nb_periods_period_type[payment_freq]
    
    if first_cpn_end_date is not None:
        assert first_cpn_end_date > start_date and first_cpn_end_date <= end_date
        
    if last_cpn_start_date is not None:
        assert last_cpn_start_date >= start_date and last_cpn_start_date < end_date

    if first_cpn_end_date == end_date or last_cpn_start_date == start_date or payment_freq == PaymentFrequency.ZERO_COUPON.value:
        # Function inputs explicity specify 1 period
        d1, d2 = [start_date], [end_date]
    elif first_cpn_end_date == last_cpn_start_date and first_cpn_end_date is not None and last_cpn_start_date is not None:
        # Function inputs explicity specify 2 periods
        d1 = [start_date, first_cpn_end_date]
        d2 = [last_cpn_start_date, end_date]
    else:        
        # Need to generate the schedule
        if first_cpn_end_date is None and last_cpn_start_date is None:    
            if last_stub_type is None:
                d1, d2 = generate_date_schedule(start_date=start_date, end_date=end_date, nb_periods=nb_periods, period_type=period_type, direction='backward')
                if first_stub_type == StubType.LONG.value:
                    d1 = d1[0] + [d1[2:]]
                    d2 = d2[1:]
                elif first_stub_type in [None, StubType.Short.value]:
                    pass
                else:
                    raise ValueError(f"Unexpected value for first_stub_type {first_stub_type}")
                
                d1, d2 = roll_date_schedule(d1, d2)
            
            elif first_stub_type is None and last_stub_type is not None:
                d1, d2 = generate_date_schedule(start_date=start_date, end_date=end_date, nb_periods=nb_periods, period_type=period_type, direction='forward')
                if last_stub_type == StubType.LONG.value:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
                d1, d2 = roll_date_schedule(d1, d2)
            else:
                raise ValueError(f"If both first_stub_type {first_stub_type} and last_stub_type {last_stub_type} are specified first_cpn_end_date or last_cpn_start_date must be specified")
        else:
            # Backward date generation
            d1_backward, d2_backward = generate_date_schedule(start_date=start_date, end_date=end_date, nb_periods=nb_periods, period_type=period_type, direction='backward')
            d1_backward, d2_backward = roll_date_schedule(d1_backward, d2_backward)
            # Forward date generation
            d1_forward, d2_forward = generate_date_schedule(start_date=start_date, end_date=end_date, nb_periods=nb_periods, period_type=period_type, direction='forward')
            d1_forward, d2_forward = roll_date_schedule(d1_forward, d2_forward)

            # If first_cpn_end_date is specified
            if first_cpn_end_date is not None and last_cpn_start_date is None:
                
                if first_cpn_end_date == d2_forward[0]:
                    first_stub_type = None
                    d1, d2 = d1_forward, d2_forward        
                elif first_cpn_end_date == d2_backward[0]:
                    first_stub_type = StubType.SHORT.value
                    d1, d2 = d1_backward, d2_backward      
                elif first_cpn_end_date == d2_backward[1]:
                    first_stub_type = StubType.LONG.value
                    d1 = d1_backward[0] + d1_backward[2:] 
                    d2 = d2_backward[1:]
                else:
                    d1, d2 = generate_date_schedule(start_date=first_cpn_end_date, end_date=end_date, nb_periods=nb_periods, period_type=period_type, direction='forward')    
                    d1, d2 = roll_date_schedule(d1, d2)    
                    d1 = [start_date] + d1
                    d2 = [first_cpn_end_date] + d2
                                          
            # If last_cpn_start_date is specified
            elif last_cpn_start_date is not None and first_cpn_end_date is None:
                if last_cpn_start_date == d1_backward[-1]:
                    last_stub_type = None
                    d1, d2 = d1_backward, d2_backward                       
                elif last_cpn_start_date == d1_forward[-1]:
                    last_stub_type = StubType.SHORT.value
                    d1, d2 = d1_forward, d2_forward    
                elif last_cpn_start_date == d1_forward[-2]:
                    last_stub_type = StubType.LONG.value
                    d1 = d1_forward[:-1]
                    d2 = d2_forward[:-2] + d2_forward[-1]
                else:
                    d1, d2 = generate_date_schedule(start_date=start_date, end_date=last_cpn_start_date, nb_periods=nb_periods, period_type=period_type, direction='backward')    
                    d1, d2 = roll_date_schedule(d1, d2)    
                    d1 = d1 + [last_cpn_start_date]
                    d2 = d2 + [end_date]
            
            # Both first_cpn_end_date and last_cpn_start_date are specified
            else:
                # Technically need to replicate the two prior sections to cover cases where first_cpn_end_date and last_cpn_start_date are specified 
                d1, d2 = generate_date_schedule(start_date=first_cpn_end_date, end_date=last_cpn_start_date, nb_periods=nb_periods, period_type=period_type, direction='backward')    
                d1, d2 = roll_date_schedule(d1, d2)    
                d1 = [start_date] + d1
                d2 = d2 + [end_date]
                
    df = pd.DataFrame({'period_start': d1, 'period_end': d2})
    # df = df[df['period_start'] != df['period_end']]
    df.reset_index(drop=True,inplace=True)

    # Add the payment dates
    if payment_type == PaymentType.IN_ARREARS.value:
        payment_dates = df['period_end']
    elif payment_type == PaymentType.IN_ADVANCE.value:
        payment_dates = df['period_start']
    else:
        raise ValueError(f"Invalid payment type '{payment_type}'. Must be 'in_arrears' or 'in_advance'.")                                        
    df['payment_date'] = np.busday_offset(payment_dates.values.astype('datetime64[D]'), offsets=payment_delay, roll='following', busdaycal=busdaycal)
        
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
    

def generate_date_schedule(
        start_date: np.datetime64, 
        end_date: np.datetime64, 
        nb_periods: int, 
        period_type: str,
        direction: str,
    ) -> pd.DataFrame:
    """
    Generates a schedule of start and end dates between start_date and end_date.

    Parameters
    ----------
    start_date : pd.Timestamp
        The start date of the schedule.
    end_date : pd.Timestamp
        The end date of the schedule.
    nb_periods : int
        The number of periods of length 'period_type' per increment.
    period_type : {'years', 'months', 'weeks', 'days'}
        The unit of time for each period.
    direction : {'forward', 'backward'}
        The direction in which to generate dates.
    Returns
    -------
    Tuble[List, List]
    
    Raises
    ------
    ValueError
        If any of the inputs are invalid.
    """
    
  # Input validation
    if direction not in {'forward', 'backward'}:
        raise ValueError(f"Invalid direction '{direction}'. Must be 'forward' or 'backward'.")
    if not isinstance(nb_periods, int) or nb_periods <= 0:
        raise ValueError(f"'nb_periods' must be a positive integer, got {nb_periods}.")
    if period_type not in {'years', 'months', 'weeks', 'days'}:
        raise ValueError(f"Invalid period_type '{period_type}'. Must be one of 'years', 'months', 'weeks', 'days'.")
    if not isinstance(start_date, np.datetime64) or not isinstance(end_date, np.datetime64):
        raise TypeError("'start_date' {start_date} and 'end_date' {end_date} must be numpy datetime64 objects.")
    if start_date >= end_date:
        raise ValueError("'start_date' must be earlier than 'end_date'.")    
        
    start_dates = np.array()
    end_dates = np.array()
    i = 0
    
    if direction == 'forward':
        current_date = start_date
        while current_date + date_offset(nb_periods, period_type) < end_date: 
            start_dates.append(current_date)
            current_date = start_date + date_offset(nb_periods * (i + 1), period_type) # Note 3*(date += 1m_offset) != date += 3m_offset
            end_dates.append(current_date)
            i += 1
                
        start_dates.append(current_date)
        end_dates.append(end_date)
        
    elif direction == 'backward':
        current_date = end_date
        while current_date - date_offset(nb_periods, period_type) > start_date:   
            end_dates.append(current_date)
            current_date = end_date - date_offset(nb_periods * (i + 1), period_type) # Note 3*(date += 1m_offset) != date += 3m_offset 
            start_dates.append(current_date)
            i +=1
            
        start_dates.append(start_date)
        end_dates.append(current_date)
        start_dates.reverse()
        end_dates.reverse()
        
    return start_dates, end_dates


def roll_date_schedule(
        start_dates: List,
        end_dates: List,
        day_roll: Union[int, str],
        roll_convention: str=RollConvention.MODIFIED_FOLLOWING.value, 
        busdaycal: np.busdaycalendar=None,
        roll_user_specified_dates=False
    ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Rolls a date schedule per the roll date, day roll convention, and business day calendar.
    
    Parameters
    ----------
    start_dates : List[pd.Timestamp]
        List of start dates.
    end_dates : List[pd.Timestamp]
        List of end dates.
    day_roll : Union[int, str]
        The day to roll to (1-31 or 'eom' for end of month).
    roll_convention : str, optional
        Rolling convention to apply for invalid dates, default is 'modifiedfollowing'.
    busdaycal : np.busdaycalendar, optional
        Business day calendar to use.
    roll_user_specified_dates : bool, optional
        Flag to indicate if the user-specified dates should be rolled according to the business day calendar.
        
    Returns
    -------
    Tuple[pd.DatetimeIndex, pd.DatetimeIndex]
        The rolled start and end dates.
    """
        
    def apply_roll(dates, roll_type, convention, calendar):
        return pd.DatetimeIndex(
            np.busday_offset(dates.values.astype('datetime64[D]'), offsets=0, roll=convention, busdaycal=calendar)
        )

    if day_roll is not None:
        start_dates[1:] = apply_specific_day_roll(dates=start_dates[1:], day_roll=day_roll) 
        end_dates[:-1] = apply_specific_day_roll(dates=end_dates[:-1], day_roll=day_roll)

    if roll_user_specified_dates:
        start_dates = apply_roll(start_dates, day_roll, roll_convention, busdaycal)
        end_dates = apply_roll(end_dates, day_roll, roll_convention, busdaycal)
    else:
        start_dates[1:] = apply_roll(start_dates[1:], day_roll, roll_convention, busdaycal)
        end_dates[:-1] = apply_roll(end_dates[:-1], day_roll, roll_convention, busdaycal)

    return start_dates, end_dates
    

def apply_specific_day_roll(
    dates: pd.DatetimeIndex, 
    day_roll: Union[int, str]
) -> pd.DatetimeIndex:
    """
    Roll the days of the dates to the provided day_roll.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The original dates to roll.
    day_roll : int or str
        The day to roll to (1-31 or 'eom' for end of month).

    Returns
    -------
    pd.DatetimeIndex
        The schedule rolling on the specific day roll
    """
    # Input validation
    day_roll = str(day_roll).strip().lower()
    day_roll = int(day_roll) if day_roll.isdigit() else day_roll
    
    if isinstance(day_roll, str):
        day_roll = day_roll.strip().lower()
        if day_roll != 'eom':
            raise ValueError(f"Invalid day_roll '{day_roll}'. Must be an integer between 1 and 31 or 'eom'.")
    elif isinstance(day_roll, int):
        if not 1 <= day_roll <= 31:
            raise ValueError(f"Invalid day_roll '{day_roll}'. Must be an integer between 1 and 31 or 'eom'.")
    else:
        raise TypeError(f"Invalid type for day_roll: {type(day_roll)}")
        
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(f"'dates' must be a pandas DatetimeIndex, got {type(dates)}")
    
    if str(day_roll).upper() == 'eom' or day_roll == 31:
        return dates + pd.offsets.MonthEnd(0)
    
    def roll_date(d):
        try:
            return d.replace(day=day_roll)
        except ValueError:
            return d + pd.offsets.MonthEnd(0)
        
    return pd.DatetimeIndex([roll_date(d) for d in dates])


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
            date_grid = generate_date_schedule(curve_date, max_settlement_date, 1, 'days', 'forward')
        elif sampling_freq == '1w':
            date_grid = generate_date_schedule(curve_date, max_settlement_date, 1, 'weeks', 'forward')
        elif sampling_freq == '1m':
            date_grid = generate_date_schedule(curve_date, max_settlement_date, 1, 'months', 'forward')
        elif sampling_freq == '3m':
            date_grid = generate_date_schedule(curve_date, max_settlement_date, 3, 'months', 'forward')    
        elif sampling_freq == '6m':
             date_grid = generate_date_schedule(curve_date, max_settlement_date, 6, 'months', 'forward')
        elif sampling_freq == '12m':
             date_grid = generate_date_schedule(curve_date, max_settlement_date, 12, 'months', 'forward') 
             
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

