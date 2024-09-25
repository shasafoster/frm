# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from typing import Literal, Optional, Union, List, Tuple
from frm.schedule.schedule_enums import RollConvention, PaymentType, StubType, Frequency, DayRoll


def set_default(value, default):
    if pd.isna(value) or value is None:
        return default
    return value


def payment_schedule(start_date: pd.Timestamp,
                     end_date: pd.Timestamp,
                     payment_freq: str,
                     roll_convention: str=RollConvention.MODIFIED_FOLLOWING.value,
                     day_roll: Union[int, str]=None,
                     first_cpn_end_date: pd.Timestamp=None,
                     last_cpn_start_date: pd.Timestamp=None,                     
                     first_stub_type: str=None,
                     last_stub_type: str=None,
                     payment_type: str=PaymentType.IN_ARREARS.value,
                     payment_delay: int=0,
                     busdaycal: np.busdaycalendar=np.busdaycalendar(),
                     roll_user_specified_dates=False) -> pd.DataFrame:
    """
    Create a payment schedule.

    Parameters
    ----------
    start_date : pandas.Timestamp
        Specifies the effective date of the schedule
    end_date : pandas.Timestamp
        Specifies the expiration date of the schedule
    payment_freq : {'d','w','28d','m', 'q', 's', 'a', 'z'}
        Specify the payment frequency
    roll_convention : {'None','following','preceding','modifiedfollowing','modifiedpreceding'},
        How to treat dates (period_start, period_end, paymen_date) that do not fall on a valid day. The default is 'modifiedfollowing'.
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
    busdaycal : np.busdaycalendar
        Specifies the business day calendar to observe. 
    roll_user_specified_dates : bool
        Boolean flag for whether to roll (per business day calendar and roll convention) the user specified dates (start_date, end_date, first_cpn_end_date, last_cpn_start_date)
        
    Returns
    -------
    schedule : pandas.DataFrame
        Columns:
            - period_start
            - period_end
            - payment_date
    """
            
    # Set defaults for optional parameters
    first_cpn_end_date = set_default(first_cpn_end_date, None)
    last_cpn_start_date = set_default(last_cpn_start_date, None)    
    payment_delay = set_default(payment_delay, 0)
    busdaycal = set_default(busdaycal, np.busdaycalendar())
    roll_user_specified_dates = set_default(roll_user_specified_dates, False)

    # Clean and validate function parameters
    roll_convention = RollConvention.from_value(roll_convention) 
    day_roll = DayRoll.from_value(day_roll)
    first_stub_type = StubType.from_value(first_stub_type)
    last_stub_type = StubType.from_value(last_stub_type)
    payment_type = PaymentType.from_value(payment_type)

    # Input validation
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be before end_date {end_date}")
    
    if Frequency.is_valid(payment_freq):
        freq_obj = Frequency.from_value(payment_freq)
    else:
        raise ValueError(f"Invalid 'payment_freq' {payment_freq}")     
    
    if first_cpn_end_date is not None:
        assert first_cpn_end_date > start_date and first_cpn_end_date <= end_date
        
    if last_cpn_start_date is not None:
        assert last_cpn_start_date >= start_date and last_cpn_start_date < end_date

    if first_cpn_end_date == end_date or last_cpn_start_date == start_date or freq_obj.value == Frequency.ZERO_COUPON.value:
        # Function inputs explicity specify 1 period
        d1, d2 = [start_date], [end_date]
    elif first_cpn_end_date == last_cpn_start_date and first_cpn_end_date is not None and last_cpn_start_date is not None:
        # Function inputs explicity specify 2 periods
        d1 = [start_date, first_cpn_end_date]
        d2 = [last_cpn_start_date, end_date]
    else:        
        # Need to generate the schedule
        if first_cpn_end_date is None and last_cpn_start_date is None:    
            if last_stub_type.value is None:
                d1, d2 = generate_date_schedule(start_date, end_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
                if first_stub_type.value == StubType.LONG.value:
                    d1 = [d1[0]] + d1[2:]
                    d2 = d2[1:]
                #d1, d2 = roll_date_schedule(d1, d2, day_roll, roll_convention, busdaycal, roll_user_specified_dates)
            
            elif first_stub_type.value is None and last_stub_type.value is not None:
                d1, d2 = generate_date_schedule(start_date, end_date, freq_obj, 'forward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
                if last_stub_type.value == StubType.LONG.value:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
                #d1, d2 = roll_date_schedule(d1, d2, day_roll, roll_convention, busdaycal, roll_user_specified_dates)
            elif first_stub_type.value is not None and last_stub_type.value is not None:
                raise ValueError("If a schedule has first and last stubs they must be specified via first_cpn_end_date and last_cpn_start_date")
            else:
                raise ValueError("Unexpected logic branch")
                
        else:
            # If first_cpn_end_date or last_cpn_start_date are specified we want generate the schedule 
            # via generate_date_schedule(start_date, end_date, ...) and see if the first_cpn_end_date or last_cpn_start_date align. 
            # If they align, we use this generated schedule. 
            # If they don't align, we generate the schedule:
            # (i) via backward generation from last_cpn_start_date if only last_cpn_start_date is specified
            # (ii) via forward generation from first_cpn_end_date if only first_cpn_end_date is specified
            # (iii) between first_cpn_end_date and last_cpn_start_date if both are specified
            
            # Backward date generation
            d1_backward, d2_backward = generate_date_schedule(start_date, end_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
            #d1_backward, d2_backward = roll_date_schedule(d1_backward, d2_backward, day_roll, roll_convention, busdaycal, roll_user_specified_dates)

            # Forward date generation
            d1_forward, d2_forward = generate_date_schedule(start_date, end_date, freq_obj, 'forward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
            #d1_forward, d2_forward = roll_date_schedule(d1_forward, d2_forward, day_roll, roll_convention, busdaycal, roll_user_specified_dates)

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
                    d1, d2 = generate_date_schedule(first_cpn_end_date, end_date, freq_obj, 'forward', roll_convention, day_roll, busdaycal, roll_user_specified_dates) 
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
                    d1, d2 = generate_date_schedule(start_date, last_cpn_start_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
                    d1 = d1 + [last_cpn_start_date]
                    d2 = d2 + [end_date]
            
            # Both first_cpn_end_date and last_cpn_start_date are specified
            else:
                # Technically need to replicate the two prior sections to cover cases where 
                # (i) there are no stubs but first_cpn_end_date and last_cpn_start_date are specified anyway (1-3 day errors may be caued by non-business days)
                d1, d2 = generate_date_schedule(first_cpn_end_date, last_cpn_start_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)    
                d1 = [start_date] + d1 + [last_cpn_start_date]
                d2 = [first_cpn_end_date] + d2 + [end_date]
                
    # Add the payment dates
    if payment_type.value == PaymentType.IN_ARREARS.value:
        datetime64_array = np.array(d2, dtype='datetime64[D]')
    elif payment_type.value == PaymentType.IN_ADVANCE.value:
        datetime64_array = np.array(d1, dtype='datetime64[D]')
    else:
        raise ValueError(f"Invalid payment type '{payment_type}'. Must be 'in_arrears' or 'in_advance'.")                                       
    payment_dates = pd.DatetimeIndex(np.busday_offset(datetime64_array, offsets=payment_delay, roll=roll_convention.value, busdaycal=busdaycal)).astype('datetime64[ns]')

    df = pd.DataFrame({'period_start': d1, 'period_end': d2, 'payment_date': payment_dates})    

    # # Add the fixing dates
    # if add_fixing_dates:
    #     # Check to excel swap model
    #     df['fixing_period_start'] = np.busday_offset(pd.DatetimeIndex(df['period_start']-pd.DateOffset(days=fixing_days_ahead)).values.astype('datetime64[D]'), offsets=0, roll='preceding', busdaycal=busdaycal)
    #     df['fixing_period_end'] = np.busday_offset(pd.DatetimeIndex(df['period_end']-pd.DateOffset(days=fixing_days_ahead)).values.astype('datetime64[D]'), offsets=0, roll='preceding', busdaycal=busdaycal)
    #     df = df[['fixing_period_start','fixing_period_end','period_start','period_end','payment_date']]

    # # Add initial notional exchange date
    # # Check to excel swap model
    # if add_initial_exchange_period:
    #     # payment_date = np.busday_offset(pd.DatetimeIndex([start_date+pd.DateOffset(days=payment_delay)]).values.astype('datetime64[D]'), offsets=0)
    #     row = {'period_start': start_date,'period_end': start_date,'payment_date':start_date}
    #     df = pd.concat([pd.DataFrame(row), df], ignore_index=True)
    return df
    

def generate_date_schedule(
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp, 
        frequency: Frequency,
        direction: str,
        roll_convention: RollConvention,
        day_roll: DayRoll,
        busdaycal: np.busdaycalendar,
        roll_user_specified_dates: bool
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
    roll_convention : RollConvention
        TBC
    day_roll : DayRoll
        TBC
    busdaycal : np.busdaycalendar
        TBC
    roll_user_specified_dates : bool
        
    Returns
    -------
    Tuble[List, List]
    
    Raises
    ------
    ValueError
        If any of the inputs are invalid.
    """
    
    def busday_offset_timestamp(pd_timestamp, offsets, roll, busdaycal):
        np_datetime64D = np.array([pd_timestamp]).astype('datetime64[D]')
        rolled_date_np = np.busday_offset(np_datetime64D, offsets=offsets, roll=roll, busdaycal=busdaycal)[0]
        return pd.Timestamp(rolled_date_np)
    
    def apply_specific_day_roll(pd_timestamp, specific_day_roll):
        try:
            return pd_timestamp.replace(day=specific_day_roll.value)
        except ValueError:
            return pd_timestamp + pd.offsets.MonthEnd(0)   

    # Input validation
    if direction not in {'forward', 'backward'}:
        raise ValueError(f"Invalid direction '{direction}'. Must be 'forward' or 'backward'.")
    if not isinstance(start_date, pd.Timestamp) or not isinstance(end_date, pd.Timestamp):
        raise TypeError("'start_date' {start_date} and 'end_date' {end_date} must be pandas Timestamp objects.")
    if start_date >= end_date:
        raise ValueError("'start_date' must be earlier than 'end_date'.")    
        
    # Implementation rationale:
    # - append to both start_dates and end_dates in while loop so don't have to consider edge case of list length = 1
    # - Note the 'roll day' component must be applied inside the while loop due to holidays/weekends 
    #   which could move a day past the while loop condition.

    i = 1
    start_dates = []
    end_dates = []
    if roll_user_specified_dates:
        start_date_rolled = busday_offset_timestamp(start_date, 0, roll_convention.value, busdaycal)
        end_date_rolled = busday_offset_timestamp(end_date, 0, roll_convention.value, busdaycal)
    
    if direction == 'forward':
        current_date = busday_offset_timestamp((start_date + frequency.date_offset), 0, roll_convention.value, busdaycal)
        current_date = apply_specific_day_roll(current_date, day_roll)
        
        while current_date < end_date: 
            start_dates.append(current_date)
            end_dates.append(current_date)
            current_date = start_date + frequency.multiply_date_offset(i+1)   
            current_date = apply_specific_day_roll(current_date, day_roll)
            current_date = busday_offset_timestamp(current_date, 0, roll_convention.value, busdaycal)
            i += 1
            
        if roll_user_specified_dates:
            start_dates = [start_date_rolled] + start_dates
            end_dates.append(end_date_rolled)
        else:
            start_dates = [start_date] + start_dates
            end_dates.append(end_date)
        
    elif direction == 'backward':
        current_date = busday_offset_timestamp((end_date - frequency.date_offset), 0, roll_convention.value, busdaycal)
        current_date = apply_specific_day_roll(current_date, day_roll)
        
        while current_date > start_date:   
            start_dates.append(current_date)
            end_dates.append(current_date)
            current_date = end_date - frequency.multiply_date_offset(i+1)     
            current_date = apply_specific_day_roll(current_date, day_roll)
            current_date = busday_offset_timestamp(current_date, 0, roll_convention.value, busdaycal)
            i +=1
        
        if roll_user_specified_dates:
            start_dates.append(start_date_rolled)
            end_dates = [end_date_rolled] + end_dates
        else:
            start_dates.append(start_date)
            end_dates = [end_date] + end_dates
        
        start_dates.reverse()
        end_dates.reverse()
        
    return start_dates, end_dates


def roll_date_schedule(
        start_dates: List,
        end_dates: List,
        day_roll: DayRoll,
        roll_convention: RollConvention, 
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
        Rolling convention to apply for non business day dates.
    busdaycal : np.busdaycalendar, optional
        Business day calendar to use.
    roll_user_specified_dates : bool, optional
        Flag to indicate if the user-specified dates should be rolled according to the business day calendar.
        
    Returns
    -------
    Tuple[pd.DatetimeIndex, pd.DatetimeIndex]
        The rolled start and end dates.
    """
        
    def apply_day_roll(dates, day_roll_value):
        dates = pd.DatetimeIndex(dates)
                
        # Split logic for speed 
        if isinstance(day_roll_value, int) and day_roll_value <= 28:
            return pd.DatetimeIndex([date.replace(day=day_roll_value) for date in dates])
        elif day_roll_value in {'eom', 31}:
            return dates + pd.offsets.MonthEnd(0)
        elif day_roll_value in {29, 30}:
            return pd.DatetimeIndex([date.replace(day=day_roll_value) 
                                     if date.day <= day_roll_value
                                     else date + pd.offsets.MonthEnd(0) for date in dates])
        else:
            raise ValueError
    
    def roll_non_business_days(dates, roll_convention, busdaycal):
        datetime64_array = np.array(dates, dtype='datetime64[D]')
        return pd.DatetimeIndex(
            np.busday_offset(datetime64_array, offsets=0, roll=roll_convention, busdaycal=busdaycal)
        ).astype('datetime64[ns]')

    if day_roll.value is not None and len(start_dates) > 1:
        start_dates[1:] = apply_day_roll(dates=start_dates[1:], day_roll_value=day_roll.value) 
        end_dates[:-1] = apply_day_roll(dates=end_dates[:-1], day_roll_value=day_roll.value)

    if roll_user_specified_dates:
        start_dates = roll_non_business_days(start_dates, roll_convention.value, busdaycal)
        end_dates = roll_non_business_days(end_dates, roll_convention.value, busdaycal)
    elif not roll_user_specified_dates and len(start_dates) > 1:
        start_dates[1:] = roll_non_business_days(start_dates[1:], roll_convention.value, busdaycal)
        end_dates[:-1] = roll_non_business_days(end_dates[:-1], roll_convention.value, busdaycal)

    return start_dates, end_dates
    

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

