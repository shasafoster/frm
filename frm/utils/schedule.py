# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
import numpy as np
import pandas as pd
from typing import Literal, Optional, Union, List, Tuple
from frm.enums.utils import RollConvention, PaymentType, StubType, PeriodFrequency, DayRoll


def set_default(value, default):
    if pd.isna(value) or value is None:
        return default
    return value


def add_payment_date(
        schedule: pd.DataFrame,
        roll_convention: str=RollConvention.MODIFIED_FOLLOWING.value,
        payment_type: str=PaymentType.IN_ARREARS.value,
        payment_delay: int=0,
        busdaycal: np.busdaycalendar=np.busdaycalendar()
    ) -> pd.DataFrame:
    """
    Adds payment dates to a schedule.

    Parameters
    ----------
    schedule : pd.DataFrame
        Dataframe with date columns 'period_start' and 'period_end'
    roll_convention: str, optional
        DESCRIPTION. The default is RollConvention.MODIFIED_FOLLOWING.value.
    payment_type : str, optional
        Specifies when payments are made. The default is PaymentType.IN_ARREARS.value.
    payment_delay : int, optional
        Specifies how many days after period start_date/end_date (if payments are in_advance/in_arrears), the payment is made. The default is 0.
    busdaycal : np.busdaycalendar, optional
        DESCRIPTION. The default is np.busdaycalendar().

    Returns
    -------
    schedule : pandas.DataFrame
        Columns:
            - period_start
            - period_end
            - payment_date
    """

    roll_convention = RollConvention.from_value(roll_convention) 
    payment_type = PaymentType.from_value(payment_type)
    payment_delay = set_default(payment_delay, 0)
    
    # Add the payment dates
    if payment_type == PaymentType.IN_ARREARS:
        dates = schedule['period_end'].to_numpy(dtype='datetime64[D]')
    elif payment_type == PaymentType.IN_ADVANCE:
        dates = schedule['period_start'].to_numpy(dtype='datetime64[D]')

    payment_dates_np = np.busday_offset(dates, offsets=payment_delay, roll=roll_convention.value, busdaycal=busdaycal)
    schedule['payment_date'] = pd.DatetimeIndex(payment_dates_np).astype('datetime64[ns]')
                           
    return schedule


def schedule(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        frequency: str,
        roll_convention: str=RollConvention.MODIFIED_FOLLOWING.value,
        day_roll: Union[int, str]=None,
        first_cpn_end_date: pd.Timestamp=None,
        last_cpn_start_date: pd.Timestamp=None,                     
        first_stub_type: str=None,
        last_stub_type: str=None,
        busdaycal: np.busdaycalendar=np.busdaycalendar(),
        roll_user_specified_dates=False
        ) -> pd.DataFrame:
    """
    Create a schedule. Optional detailed stub logic.

    Parameters
    ----------
    start_date : pandas.Timestamp
        Specifies the effective date of the schedule
    end_date : pandas.Timestamp
        Specifies the expiration date of the schedule
    frequency : str
        Specify the period frequency
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
    """
            
    # Set defaults for optional parameters
    first_cpn_end_date = set_default(first_cpn_end_date, None)
    last_cpn_start_date = set_default(last_cpn_start_date, None)    
    
    busdaycal = set_default(busdaycal, np.busdaycalendar())
    roll_user_specified_dates = set_default(roll_user_specified_dates, False)

    # Clean and validate function parameters
    roll_convention = RollConvention.from_value(roll_convention) 
    day_roll = DayRoll.from_value(day_roll)
    first_stub_type = StubType.from_value(first_stub_type)
    last_stub_type = StubType.from_value(last_stub_type)

    # Input validation
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be before end_date {end_date}")
    
    if PeriodFrequency.is_valid(frequency):
        freq_obj = PeriodFrequency.from_value(frequency)
    else:
        raise ValueError(f"Invalid 'frequency' {frequency}")     
    
    if first_cpn_end_date is not None:
        assert first_cpn_end_date > start_date and first_cpn_end_date <= end_date
        
    if last_cpn_start_date is not None:
        assert last_cpn_start_date >= start_date and last_cpn_start_date < end_date

    if first_cpn_end_date == end_date or last_cpn_start_date == start_date or freq_obj.value == PeriodFrequency.ZERO_COUPON.value:
        # Function inputs explicity specify 1 period
        d1, d2 = [start_date], [end_date]
    elif first_cpn_end_date == last_cpn_start_date and first_cpn_end_date is not None and last_cpn_start_date is not None:
        # Function inputs explicity specify 2 periods
        d1 = [start_date, first_cpn_end_date]
        d2 = [last_cpn_start_date, end_date]
    else:        
        # Need to generate the schedule
        if first_cpn_end_date is None and last_cpn_start_date is None:  
            assert first_stub_type != StubType.DEFINED_PER_FIRST_CPN_END_DATE
            assert last_stub_type != StubType.DEFINED_PER_LAST_CPN_START_DATE
            
            if first_stub_type == StubType.DEFAULT and last_stub_type == StubType.DEFAULT:
                # If no stub is specified, defaults to market convention on the 1st stub, no last stub.
                first_stub_type = StubType.market_convention()
                last_stub_type = StubType.NONE
            elif first_stub_type == StubType.DEFAULT:
                first_stub_type  = StubType.NONE
            elif last_stub_type == StubType.DEFAULT:
                last_stub_type = StubType.NONE
            else:
                raise ValueError("Unexpected logic branch - please raise GitHub issue")                
                
            if last_stub_type == StubType.NONE:
                d1, d2 = generate_date_schedule(start_date, end_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
                if first_stub_type == StubType.LONG:
                    d1 = [d1[0]] + d1[2:]
                    d2 = d2[1:]
            elif first_stub_type == StubType.NONE and last_stub_type != StubType.NONE:
                d1, d2 = generate_date_schedule(start_date, end_date, freq_obj, 'forward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
                if last_stub_type == StubType.LONG:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
            elif first_stub_type in [StubType.SHORT, StubType.LONG] and last_stub_type in [StubType.SHORT, StubType.LONG]:
                raise ValueError("If a schedule has first and last stubs they must be specified via first_cpn_end_date and last_cpn_start_date")
            else:
                raise ValueError("Unexpected logic branch - please raise GitHub issue")
                
        else:
            # If first_cpn_end_date or last_cpn_start_date are specified we want generate the schedule 
            # via generate_date_schedule(start_date, end_date, ...) and see if 
            # the first_cpn_end_date or last_cpn_start_date match the generated schedule.
            #
            # If they align, we use this generated schedule. 
            #
            # If they don't align, we generate the schedule:
            # (i) via backward generation from last_cpn_start_date if only last_cpn_start_date is specified
            # (ii) via forward generation from first_cpn_end_date if only first_cpn_end_date is specified
            # (iii) between first_cpn_end_date and last_cpn_start_date if both are specified
            
            # Step 1: Generate the schedule by backward and forward date generation
            d1_backward, d2_backward = generate_date_schedule(start_date, end_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
            d1_forward, d2_forward = generate_date_schedule(start_date, end_date, freq_obj, 'forward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)

            # Step 2a : Check if the first_cpn_end_date matches any generated schedules
            
            direction = []
            if first_cpn_end_date is not None:
                if first_cpn_end_date == d2_forward[0]:
                    direction.append('forward')
                    first_stub_type = StubType.NONE      
                elif first_cpn_end_date == d2_backward[0]:
                    direction.append('backward')
                    first_stub_type = StubType.SHORT 
                elif first_cpn_end_date == d2_backward[1]:
                    direction.append('backward')
                    first_stub_type = StubType.LONG
                else:
                    # Need to construct schedule using first_cpn_end_date
                    first_stub_type = StubType.DEFINED_PER_FIRST_CPN_END_DATE

            # Step 2a : Check if the last_cpn_start_date matches any generated schedules
            if last_cpn_start_date is not None:    
                if last_cpn_start_date == d1_backward[-1]:
                    direction.append('backward')
                    last_stub_type = StubType.NONE                      
                elif last_cpn_start_date == d1_forward[-1]:
                    direction.append('forward')
                    last_stub_type = StubType.SHORT  
                elif last_cpn_start_date == d1_forward[-2]:
                    direction.append('forward')
                    last_stub_type = StubType.LONG
                else:
                    # Need to construct schedule using last_cpn_start_date
                    last_stub_type = StubType.DEFINED_PER_LAST_CPN_START_DATE
            
            # Step 2c : Set default stub values to:
            #              (i) NONE if other stub is SHORT/LONG
            #              (ii) SHORT if other stub is NONE
            if first_stub_type == StubType.DEFAULT and last_stub_type != StubType.DEFINED_PER_LAST_CPN_START_DATE:
                assert last_stub_type != StubType.DEFAULT
                if last_stub_type == StubType.NONE:
                    first_stub_type = StubType.market_convention()
                else:
                    first_stub_type = StubType.NONE
            if last_stub_type == StubType.DEFAULT and first_stub_type != StubType.DEFINED_PER_FIRST_CPN_END_DATE:
                assert first_stub_type != StubType.DEFAULT
                if first_stub_type == StubType.NONE:
                    last_stub_type = StubType.market_convention()
                else:
                    last_stub_type = StubType.NONE         
                
            # Step 3: Contruct the schedules
            if first_stub_type == StubType.DEFINED_PER_FIRST_CPN_END_DATE and last_stub_type == StubType.DEFINED_PER_LAST_CPN_START_DATE:
                # Generate date schedule between first_cpn_end_date and last_cpn_start_date
                d1, d2 = generate_date_schedule(first_cpn_end_date, last_cpn_start_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)    
                d1 = [start_date] + d1 + [last_cpn_start_date]
                d2 = [first_cpn_end_date] + d2 + [end_date]
            elif first_stub_type == StubType.DEFINED_PER_FIRST_CPN_END_DATE and last_stub_type != StubType.DEFINED_PER_LAST_CPN_START_DATE:
                # Forward generation from from first_cpn_end_date and if long last stub, combine last two periods
                d1, d2 = generate_date_schedule(first_cpn_end_date, end_date, freq_obj, 'forward', roll_convention, day_roll, busdaycal, roll_user_specified_dates) 
                d1 = [start_date] + d1
                d2 = [first_cpn_end_date] + d2
                if last_stub_type == StubType.LONG:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
            elif first_stub_type != StubType.DEFINED_PER_FIRST_CPN_END_DATE and last_stub_type == StubType.DEFINED_PER_LAST_CPN_START_DATE:
                # Backward generation from last_cpn_start_date and if long first stub, combine first two periods
                d1, d2 = generate_date_schedule(start_date, last_cpn_start_date, freq_obj, 'backward', roll_convention, day_roll, busdaycal, roll_user_specified_dates)
                d1 = d1 + [last_cpn_start_date]
                d2 = d2 + [end_date]                
                if first_stub_type == StubType.LONG:
                    d1 = [d1[0]] + d1[2:]
                    d2 = d2[1:]
            else:
                # Don't need to generate any additional schedules
                assert len(set(direction)) == 1
                
                if first_stub_type == StubType.NONE:
                    # Forward date generation
                    d1, d2 = d1_forward, d2_forward
                    if last_stub_type == StubType.LONG:
                        d1 = d1[:-1]
                        d2 = d2[:-2] + [d2[-1]]
                elif last_stub_type == StubType.NONE:
                    # Backward date generation
                    d1, d2 = d1_backward, d2_backward
                    if first_stub_type == StubType.LONG:
                        d1 = [d1[0]] + d1[2:]
                        d2 = d2[1:]          
                else:
                    raise ValueError("Unexpected logic branch - please raise GitHub issue")
                
    return pd.DataFrame({'period_start': d1, 'period_end': d2})

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

    

def generate_date_schedule(
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp, 
        frequency: PeriodFrequency,
        direction: str,
        roll_convention: RollConvention=RollConvention.MODIFIED_FOLLOWING,
        day_roll: DayRoll=DayRoll.NONE,
        busdaycal: np.busdaycalendar=np.busdaycalendar(),
        roll_user_specified_dates: bool=False,
    ) -> Tuple[List, List]:
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
        How to treat dates that fall on a non business day.
    day_roll : DayRoll
        Specifies the day periods should start/end on.   
    busdaycal : np.busdaycalendar
        Specifies the business day calendar to observe. 
    roll_user_specified_dates : bool
        Boolean flag for whether to roll (per business day calendar and roll convention) the user specified dates (start_date, end_date) 
    
    Returns
    -------
    Tuple[List, List]
    
    Raises
    ------
    ValueError, TypeError
        If any of the inputs have invalid values or types
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

