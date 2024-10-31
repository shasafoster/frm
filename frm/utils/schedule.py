# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from frm.enums.utils import RollConvention, TimingConvention, StubType, PeriodFrequency, DayRoll, DayCountBasis
from frm.utils.daycount import year_fraction, day_count


def set_default(value, default):
    if pd.isna(value) or value is None:
        return default
    return value


# def get_payment_dates(
#         schedule: pd.DataFrame,
#         payment_delay: int=0,
#         roll_convention: RollConvention=RollConvention.MODIFIED_FOLLOWING,
#         payment_timing: TimingConvention=TimingConvention.IN_ARREARS,
#         busdaycal: np.busdaycalendar=np.busdaycalendar()
#     ) -> pd.DatetimeIndex:
#     """
#     Calculates payment dates based on a schedule.
#
#     Parameters
#     ----------
#     schedule : pandas.DataFrame
#         Pandas DataFrame with date columns 'period_start' and 'period_end'
#     payment_delay : int, optional
#         Specifies how many days after period start_date/end_date (if payments are in_advance/in_arrears), the payment is made. The default is 0.
#     roll_convention: RollConvention, optional
#         The default is RollConvention.MODIFIED_FOLLOWING.
#     payment_timing : TimingConvention, optional
#         Specifies when payments are made. The default is PaymentType.IN_ARREARS.
#     busdaycal : numpy.busdaycalendar, optional
#         The default is numpy.busdaycalendar().
#
#     Returns
#     -------
#     pandas.DatetimeIndex
#
#
#     """
#
#     # Add the payment dates
#     match payment_timing:
#         case TimingConvention.IN_ARREARS:
#             dates = schedule['period_end'].to_numpy(dtype='datetime64[D]')
#         case TimingConvention.IN_ADVANCE:
#             dates = schedule['period_start'].to_numpy(dtype='datetime64[D]')
#
#     dates_np = np.busday_offset(dates, offsets=payment_delay, roll=roll_convention.value, busdaycal=busdaycal)
#     return pd.DatetimeIndex(dates_np).astype('datetime64[ns]')
#
# # include_payment_dates, payment_delay, payment_type, payment_roll_convention
# # - inherit from schedule: busdaycal
# # include_fixing_dates, fixing_days_ahead, fixing_type, fixing_roll_convention
# # - inherit from schedule: busdaycal
#
# def get_fixing_dates(
#         schedule: pd.DataFrame,
#         fixing_days_ahead: int = 0,
#         roll_convention: RollConvention = RollConvention.PRECEDING,
#         fixing_timing: TimingConvention = TimingConvention.IN_ADVANCE,
#         busdaycal: np.busdaycalendar = np.busdaycalendar()
# ) -> pd.DatetimeIndex:
#     """
#     Calculates fixing dates based on a schedule.
#
#     Parameters
#     ----------
#     schedule : pandas.DataFrame
#         Pandas DataFrame with date columns 'period_start' and 'period_end'
#     fixing_days_ahead : int, optional
#         Specifies how many days prior to period start_date/end_date (if payments are in_advance/in_arrears), the fixing is made. The default is 0.
#     roll_convention: RollConvention, optional
#         The default is RollConvention.MODIFIED_FOLLOWING.
#     fixing_timing : TimingConvention, optional
#         Specifies when fixings occur. The default is TimingConvention.IN_ADVANCE.
#     busdaycal : numpy.busdaycalendar, optional
#         The default is numpy.busdaycalendar().
#
#     Returns
#     -------
#     pandas.DatetimeIndex
#
#     """
#
#     # Add the payment dates
#     match fixing_timing:
#         case TimingConvention.IN_ARREARS:
#             dates = schedule['period_end'].to_numpy(dtype='datetime64[D]')
#         case TimingConvention.IN_ADVANCE:
#             dates = schedule['period_start'].to_numpy(dtype='datetime64[D]')
#
#     dates_np = np.busday_offset(dates, offsets=-1*fixing_days_ahead, roll=roll_convention.value, busdaycal=busdaycal)
#     return pd.DatetimeIndex(dates_np).astype('datetime64[ns]')




@dataclass
class Schedule:
    # Schedule parameters
    start_date: [pd.Timestamp, np.datetime64]
    end_date: [pd.Timestamp, np.datetime64]
    frequency: PeriodFrequency
    roll_convention: RollConvention = RollConvention.MODIFIED_FOLLOWING
    day_roll: DayRoll = DayRoll.NONE
    first_cpn_end_date: Union[pd.Timestamp, np.datetime64, None] = None
    last_cpn_start_date: Union[pd.Timestamp, np.datetime64, None] = None
    first_stub_type: StubType = StubType.DEFAULT
    last_stub_type: StubType = StubType.DEFAULT
    roll_user_specified_dates: bool = False
    busdaycal: np.busdaycalendar = np.busdaycalendar()
    # Payment date parameters
    add_payment_dates: bool = True
    payment_delay: int = 0
    payment_timing: TimingConvention = TimingConvention.IN_ARREARS
    # Fixing date parameters
    add_fixing_dates: bool = False
    fixing_days_ahead: int = 0
    fixing_timing: TimingConvention = TimingConvention.IN_ADVANCE

    # The schedule DataFrame is initialized after the object is created
    df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.generate_schedule()

    def generate_schedule(self):
        # Generate the schedule DataFrame using the stored parameters
        self.df = get_schedule(
            start_date=self.start_date,
            end_date=self.end_date,
            frequency=self.frequency,
            roll_convention=self.roll_convention,
            day_roll=self.day_roll,
            first_cpn_end_date=self.first_cpn_end_date,
            last_cpn_start_date=self.last_cpn_start_date,
            first_stub_type=self.first_stub_type,
            last_stub_type=self.last_stub_type,
            roll_user_specified_dates=self.roll_user_specified_dates,
            busdaycal=self.busdaycal,
            add_payment_dates=self.add_payment_dates,
            payment_delay=self.payment_delay,
            payment_timing=self.payment_timing,
            add_fixing_dates=self.add_fixing_dates,
            fixing_days_ahead=self.fixing_days_ahead,
            fixing_timing=self.fixing_timing,
        )

    def update_and_regenerate(self, **kwargs):
        """
        Update specified parameters and regenerate the schedule.

        Parameters:
            **kwargs: Dictionary of parameters to update.
        """
        # Update the parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'Schedule' object has no attribute '{key}'")
        # Regenerate the schedule with updated parameters
        self.generate_schedule()


    def add_period_length_to_schedule(self, day_count_basis: DayCountBasis):
        self.df = add_period_length_to_schedule(self.df, day_count_basis)



def add_period_length_to_schedule(schedule_df: pd.DataFrame, day_count_basis: DayCountBasis):
    """ Add the period length in days and years to the schedule DataFrame, always replacing existing columns"""

    # Remove 'period_days' and 'period_years' if they already exist
    if 'period_days' in schedule_df.columns:
        schedule_df.pop('period_days')
    if 'period_years' in schedule_df.columns:
        schedule_df.pop('period_years')

    # Get the column index for 'period_end'
    col_index = schedule_df.columns.get_loc('period_end')

    # Insert 'period_days' and 'period_years' at the specified positions
    schedule_df.insert(loc=col_index + 1, column='period_days',
                       value=day_count(schedule_df['period_start'], schedule_df['period_end'], day_count_basis))
    schedule_df.insert(loc=col_index + 2, column='period_years',
                       value=year_fraction(schedule_df['period_start'], schedule_df['period_end'], day_count_basis))

    return schedule_df


def get_schedule(
        # Schedule parameters
        start_date: [pd.Timestamp, np.datetime64],
        end_date: [pd.Timestamp, np.datetime64],
        frequency: PeriodFrequency,
        roll_convention: RollConvention=RollConvention.MODIFIED_FOLLOWING,
        day_roll: DayRoll=DayRoll.NONE,
        first_cpn_end_date: Union[pd.Timestamp, np.datetime64, None] = None,
        last_cpn_start_date: Union[pd.Timestamp, np.datetime64, None] = None,
        first_stub_type: StubType=StubType.DEFAULT,
        last_stub_type: StubType=StubType.DEFAULT,
        roll_user_specified_dates: bool=False,
        busdaycal: np.busdaycalendar=np.busdaycalendar(),
        # Payment date parameters
        add_payment_dates: bool=True,
        payment_delay: int=0,
        payment_timing: TimingConvention = TimingConvention.IN_ARREARS,
        # Fixing date parameters
        add_fixing_dates: bool=False,
        fixing_days_ahead: int=0,
        fixing_timing: TimingConvention = TimingConvention.IN_ADVANCE,
        ) -> pd.DataFrame:
    """
    Create a schedule. Optional detailed stub logic.

    Parameters
    ----------
    start_date : pandas.Timestamp
        Specifies the effective date of the schedule
    end_date : pandas.Timestamp
        Specifies the termination date of the schedule
    frequency : str
        Specify the period frequency
    roll_convention : RollConvention
        How to treat dates that do not fall on a valid day. The default is RollConvention.MODIFIED_FOLLOWING.
    day_roll : DayRoll
        Specifies the day periods should start/end on. The default is DayRoll.NONE.
    first_cpn_end_date: pandas.Timestamp
        Specifies the end date of the first coupon period. The first_cpn_end_date overrides the first_stub_type field.
    last_cpn_start_date: pandas.Timestamp
        Specifies the start date of the last coupon period. The last_cpn_start_date overrides the last_stub_type field.
    first_stub_type : StubType
        Specifies the type of the first stub. If first_cpn_end_date is specified, the first_stub_type is ignored.
    last_stub_type : StubType
        Specifies the type of the last stub. If last_cpn_start_date is specified, the last_stub_type is ignored.
    roll_user_specified_dates : bool
        Boolean flag for whether to roll (per business day calendar and roll convention) the user specified dates (start_date, end_date, first_cpn_end_date, last_cpn_start_date)
    busdaycal : np.busdaycalendar
        Specifies the business day calendar to observe.
    add_payment_dates : bool
        Boolean flag for whether to add payment dates to the schedule. The default is True.
    payment_delay : int
        Specifies how many days after period start_date/end_date (if payments are in_advance/in_arrears), the payment is made. The default is 0.
    payment_timing : TimingConvention
        Specifies when payments are made. The default is TimingConvention.IN_ARREARS.
    add_fixing_dates : bool
        Boolean flag for whether to add fixing dates to the schedule. The default is False.
    fixing_days_ahead : int
        Specifies how many days prior to period start_date/end_date (if payments are in_advance/in_arrears), the fixing is made. The default is 0.
    fixing_timing : TimingConvention
        Specifies when fixings occur. The default is TimingConvention.IN_ADVANCE

    Returns
    -------
    schedule : pandas.DataFrame
        Columns:
            - fixing_date (if add_fixing_dates=True)
            - period_start
            - period_end
            - payment_date (if add_payment_dates=True)
    """

    # Set defaults for optional parameters
    #first_cpn_end_date = set_default(first_cpn_end_date, None)
    #last_cpn_start_date = set_default(last_cpn_start_date, None)

    #busdaycal = set_default(busdaycal, np.busdaycalendar())
    #roll_user_specified_dates = set_default(roll_user_specified_dates, False)

    # Clean and validate function parameters
    #roll_convention = RollConvention.from_value(roll_convention)
    #day_roll = DayRoll.from_value(day_roll)
    #first_stub_type = StubType.from_value(first_stub_type)
    #last_stub_type = StubType.from_value(last_stub_type)

    start_date, end_date, first_cpn_end_date, last_cpn_start_date = \
        [pd.Timestamp(d) if isinstance(d, np.datetime64) else d for d in \
         [start_date, end_date, first_cpn_end_date, last_cpn_start_date]]

    # Input validation
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be before end_date {end_date}")

    #if PeriodFrequency.is_valid(frequency):
    #    freq_obj = PeriodFrequency.from_value(frequency)
    #else:
    #    raise ValueError(f"Invalid 'frequency' {frequency}")

    if first_cpn_end_date is not None:
        assert start_date < first_cpn_end_date <= end_date

    if last_cpn_start_date is not None:
        assert start_date <= last_cpn_start_date < end_date

    if first_cpn_end_date == end_date or last_cpn_start_date == start_date or frequency.value == PeriodFrequency.ZERO_COUPON.value:
        # Function inputs explicitly specify 1 period
        d1, d2 = [start_date], [end_date]
    elif first_cpn_end_date == last_cpn_start_date and first_cpn_end_date is not None and last_cpn_start_date is not None:
        # Function inputs explicitly specify 2 periods
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
                d1, d2 = generate_date_schedule(start_date, end_date, frequency, 'backward', roll_convention, day_roll, roll_user_specified_dates, busdaycal)
                if first_stub_type == StubType.LONG:
                    d1 = [d1[0]] + d1[2:]
                    d2 = d2[1:]
            elif first_stub_type == StubType.NONE and last_stub_type != StubType.NONE:
                d1, d2 = generate_date_schedule(start_date, end_date, frequency, 'forward', roll_convention, day_roll, roll_user_specified_dates, busdaycal)
                if last_stub_type == StubType.LONG:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
            elif first_stub_type in [StubType.SHORT, StubType.LONG] and last_stub_type in [StubType.SHORT, StubType.LONG]:
                raise ValueError("If a schedule has first and last stubs they must be specified via first_cpn_end_date and last_cpn_start_date")
            else:
                raise ValueError("Unexpected logic branch - please raise GitHub issue")

        else:
            # If first_cpn_end_date or last_cpn_start_date are specified we want to generate the schedule
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
            d1_backward, d2_backward = generate_date_schedule(start_date, end_date, frequency, 'backward', roll_convention, day_roll, roll_user_specified_dates, busdaycal)
            d1_forward, d2_forward = generate_date_schedule(start_date, end_date, frequency, 'forward', roll_convention, day_roll, roll_user_specified_dates, busdaycal)

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

            # Step 3: Construct the schedules
            if first_stub_type == StubType.DEFINED_PER_FIRST_CPN_END_DATE and last_stub_type == StubType.DEFINED_PER_LAST_CPN_START_DATE:
                # Generate date schedule between first_cpn_end_date and last_cpn_start_date
                d1, d2 = generate_date_schedule(first_cpn_end_date, last_cpn_start_date, frequency, 'backward', roll_convention, day_roll, roll_user_specified_dates, busdaycal)
                d1 = [start_date] + d1 + [last_cpn_start_date]
                d2 = [first_cpn_end_date] + d2 + [end_date]
            elif first_stub_type == StubType.DEFINED_PER_FIRST_CPN_END_DATE and last_stub_type != StubType.DEFINED_PER_LAST_CPN_START_DATE:
                # Forward generation from from first_cpn_end_date and if long last stub, combine last two periods
                d1, d2 = generate_date_schedule(first_cpn_end_date, end_date, frequency, 'forward', roll_convention, day_roll, roll_user_specified_dates, busdaycal)
                d1 = [start_date] + d1
                d2 = [first_cpn_end_date] + d2
                if last_stub_type == StubType.LONG:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
            elif first_stub_type != StubType.DEFINED_PER_FIRST_CPN_END_DATE and last_stub_type == StubType.DEFINED_PER_LAST_CPN_START_DATE:
                # Backward generation from last_cpn_start_date and if long first stub, combine first two periods
                d1, d2 = generate_date_schedule(start_date, last_cpn_start_date, frequency, 'backward', roll_convention, day_roll, roll_user_specified_dates, busdaycal)
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


    schedule = pd.DataFrame({'period_start': d1, 'period_end': d2})

    if add_fixing_dates:
        match fixing_timing:
            case TimingConvention.IN_ARREARS:
                dates = schedule['period_end'].to_numpy(dtype='datetime64[D]')
            case TimingConvention.IN_ADVANCE:
                dates = schedule['period_start'].to_numpy(dtype='datetime64[D]')
            case _:
                raise ValueError(f"Invalid fixing_timing {fixing_timing}")

        dates_np = np.busday_offset(dates, offsets=-1 * fixing_days_ahead, roll=roll_convention.value, busdaycal=busdaycal)
        schedule.insert(0, 'fixing_date', pd.DatetimeIndex(dates_np).astype('datetime64[ns]'))

    if add_payment_dates:
        match payment_timing:
            case TimingConvention.IN_ARREARS:
                dates = schedule['period_end'].to_numpy(dtype='datetime64[D]')
            case TimingConvention.IN_ADVANCE:
                dates = schedule['period_start'].to_numpy(dtype='datetime64[D]')
            case _:
                raise ValueError(f"Invalid payment_timing {payment_timing}")

        dates_np = np.busday_offset(dates, offsets=payment_delay, roll=roll_convention.value, busdaycal=busdaycal)
        schedule['payment_date'] = pd.DatetimeIndex(dates_np).astype('datetime64[ns]')

    return schedule


def generate_date_schedule(
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp, 
        frequency: PeriodFrequency,
        direction: str,
        roll_convention: RollConvention=RollConvention.MODIFIED_FOLLOWING,
        day_roll: DayRoll=DayRoll.NONE,
        roll_user_specified_dates: bool=False,
        busdaycal: np.busdaycalendar=np.busdaycalendar()
    ) -> Tuple[List, List]:
    """
    Generates a schedule of start and end dates between start_date and end_date.
    
    Parameters
    ----------
    start_date : pd.Timestamp
        The start date of the schedule.
    end_date : pd.Timestamp
        The end date of the schedule.
    frequency : PeriodFrequency
        The frequency of the schedule.
    direction : {'forward', 'backward'}
        The direction in which to generate dates.
    roll_convention : RollConvention
        How to treat dates that fall on a non business day.
    day_roll : DayRoll
        Specifies the day periods should start/end on.
    roll_user_specified_dates : bool
        Boolean flag for whether to roll (per business day calendar and roll convention) the user specified dates (start_date, end_date) 
    busdaycal : np.busdaycalendar
        Specifies the business day calendar to observe.

    Returns
    -------
    Tuple[List, List]
    
    Raises
    ------
    ValueError, TypeError
        If any of the inputs have invalid values or types

    TODO: Consider options for passing param=(start_date, end_date, frequency, None, None, None, None, None, None, None, None) to function
          E.g. Wrap with fields with Optional[] and set defaults in function from enum.set_default()
    """
    
    def busday_offset_timestamp(pd_timestamp, offsets, roll, busdaycal):
        np_datetime64D = np.array([pd_timestamp]).astype('datetime64[D]')
        rolled_date_np = np.busday_offset(np_datetime64D, offsets=offsets, roll=roll, busdaycal=busdaycal)[0]
        return pd.Timestamp(rolled_date_np)
    
    def apply_specific_day_roll(pd_timestamp: pd.Timestamp,
                                specific_day_roll: DayRoll) -> pd.Timestamp:
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
        start_date_in_schedule = busday_offset_timestamp(start_date, 0, roll_convention.value, busdaycal)
        end_date_in_schedule = busday_offset_timestamp(end_date, 0, roll_convention.value, busdaycal)
    else:
        start_date_in_schedule = start_date
        end_date_in_schedule = end_date

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

        start_dates = [start_date_in_schedule] + start_dates
        end_dates.append(end_date_in_schedule)
        
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

        start_dates.append(start_date_in_schedule)
        end_dates = [end_date_in_schedule] + end_dates
        
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

