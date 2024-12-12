# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
import datetime
from frm.enums.utils import RollConv, TimingConvention, Stub, PeriodFreq, DayRoll, DayCountBasis, ExchangeNotionals
from frm.utils.daycount import year_frac, day_count


if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))


@dataclass
class BaseSchedule:
    # Base Schedule parameters
    start_date: [pd.Timestamp, np.datetime64, datetime.date, datetime.datetime]
    end_date: [pd.Timestamp, np.datetime64, datetime.date, datetime.datetime]
    freq: PeriodFreq
    roll_conv: RollConv = RollConv.MODIFIED_FOLLOWING
    day_roll: DayRoll = DayRoll.UNADJUSTED
    first_period_end: Union[pd.Timestamp, np.datetime64, datetime.date, datetime.datetime, None] = None
    last_period_start: Union[pd.Timestamp, np.datetime64, datetime.date, datetime.datetime, None] = None
    first_stub: Stub = Stub.DEFAULT
    last_stub: Stub = Stub.DEFAULT
    roll_user_specified_dates: bool = False
    cal: np.busdaycalendar = np.busdaycalendar()
    # The schedule DataFrame is initialized after the object is created
    df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # Convert to pandas Timestamp
        self.start_date = pd.Timestamp(self.start_date)
        self.end_date = pd.Timestamp(self.end_date)
        self.first_period_end = pd.Timestamp(self.first_period_end) if self.first_period_end is not None else None
        self.last_period_start = pd.Timestamp(self.last_period_start) if self.last_period_start is not None else None
        self.make_schedule()

    def make_schedule(self):
        # Generate the schedule DataFrame using the stored parameters
        self.df = make_schedule(
            start_date=self.start_date,
            end_date=self.end_date,
            freq=self.freq,
            roll_conv=self.roll_conv,
            day_roll=self.day_roll,
            first_period_end=self.first_period_end,
            last_period_start=self.last_period_start,
            first_stub=self.first_stub,
            last_stub=self.last_stub,
            roll_user_specified_dates=self.roll_user_specified_dates,
            cal=self.cal
        )

        if self.roll_user_specified_dates:
            self.start_date = self.df['period_start'].iloc[0]
            self.end_date = self.df['period_end'].iloc[-1]

    def add_payment_dates(self,
                          payment_delay: int=0,
                          payment_timing: TimingConvention = TimingConvention.IN_ARREARS,
                          col_name: str='payment_date'):
        """
        Add payment dates to the schedule DataFrame for valid periods.

        Parameters
        ----------
        payment_delay : int
            Specifies how many days after period start_date/end_date (if payments are in_advance/in_arrears), the payment is made. The default is 0.
        payment_timing : TimingConvention
            Specifies when payments are made. The default is TimingConvention.IN_ARREARS.
        col_name: str
            Column name to use for the payment dates. The default is 'payment_date'.
        """
        match payment_timing:
            case TimingConvention.IN_ARREARS:
                dates = self.df['period_end'].to_numpy(dtype='datetime64[D]')
            case TimingConvention.IN_ADVANCE:
                dates = self.df['period_start'].to_numpy(dtype='datetime64[D]')
            case _:
                raise ValueError(f"Invalid payment_timing {payment_timing}")

        # Convert NaT values to a mask
        valid_dates_mask = ~pd.isna(dates)
        valid_dates = dates[valid_dates_mask]

        # Apply np.busday_offset only to non-NaT dates
        adjusted_dates = np.empty_like(dates, dtype='datetime64[D]')
        adjusted_dates[valid_dates_mask] = np.busday_offset(valid_dates, offsets=payment_delay, roll=self.roll_conv.value, busdaycal=self.cal)
        adjusted_dates[~valid_dates_mask] = 'NaT'

        # Update DataFrame
        self.df[col_name] = pd.DatetimeIndex(adjusted_dates).astype('datetime64[ns]')

    def add_period_daycount(self, day_count_basis: DayCountBasis):
        """ Add the period length in days to the schedule DataFrame, always replacing existing columns"""

        # Remove 'period_days' and 'period_years' if they already exist
        if 'period_daycount' in self.df.columns:
            self.df.pop('period_daycount')

        # Get the column index for 'period_end'
        col_index = self.df.columns.get_loc('period_end')

        # Insert 'period_days' and 'period_years' at the specified positions
        mask = np.logical_and(self.df['period_start'].notnull(), self.df['period_end'].notnull())
        days = np.full(self.df.shape[0], np.nan)
        days[mask] = day_count(self.df['period_start'][mask], self.df['period_end'][mask], day_count_basis)
        self.df.insert(loc=col_index + 1, column='period_daycount',value=days)

    def add_period_yearfrac(self, day_count_basis: DayCountBasis):
        """ Add the period length years to the schedule DataFrame, always replacing existing columns"""

        if 'period_yearfrac' in self.df.columns:
            self.df.pop('period_yearfrac')

        # Get the column index for 'period_end'
        col_index = self.df.columns.get_loc('period_end')

        # Insert 'period_days' and 'period_years' at the specified positions
        mask = np.logical_and(self.df['period_start'].notnull(), self.df['period_end'].notnull())
        years = np.full(self.df.shape[0], np.nan)
        years[mask] = year_frac(self.df['period_start'][mask], self.df['period_end'][mask], day_count_basis)
        self.df.insert(loc=col_index + 2, column='period_yearfrac', value=years)


@dataclass
class NotionalSchedule(BaseSchedule):
    notional_amount: float | np.ndarray = 100_000_000
    exchange_notionals: ExchangeNotionals = ExchangeNotionals.NEITHER
    notional_payment_delay: int = 0
    notional_payment_timing: TimingConvention = TimingConvention.IN_ARREARS
    initial_notional_exchange_date: Union[pd.Timestamp, np.datetime64, datetime.date, datetime.datetime, None] = None

    def __post_init__(self):
        super().__post_init__()
        self.add_notional_schedule()

    def add_notional_schedule(self):
        """Adds columns 'notional', 'notional_payment' and 'notional_payment_date' to the schedule DataFrame"""

        notional_amount = np.atleast_1d(self.notional_amount)
        if notional_amount.shape == (1,):
            self.df['notional'] = notional_amount[0]
            self.df['notional_payment'] = 0
        elif notional_amount.shape == self.df.shape[0]:
            self.df['notional'] = notional_amount
            self.df['notional_payment'] = notional_amount[:-1] - notional_amount[1:]
        else:
            raise ValueError("Invalid notional_amount shape")

        self.add_payment_dates(self.notional_payment_delay, self.notional_payment_timing, col_name='notional_payment_date')

        if self.exchange_notionals == ExchangeNotionals.START or self.exchange_notionals == ExchangeNotionals.BOTH:
            column_order = self.df.columns
            if self.initial_notional_exchange_date is None:
                self.initial_notional_exchange_date = self.df['period_start'].iloc[0]

            row_data = {'notional_payment_date': self.initial_notional_exchange_date}
            self.df = pd.concat([pd.DataFrame(row_data, index=[0]), self.df], ignore_index=True)
            self.df = self.df[column_order]
            self.df.reset_index(drop=True, inplace=True)

            self.df.loc[self.df.index[0], 'notional'] = 0
            self.df.loc[self.df.index[0], 'notional_payment'] = -1 * notional_amount[0]

        if self.exchange_notionals == ExchangeNotionals.END or self.exchange_notionals == ExchangeNotionals.BOTH:
            # Add final notional exchange period
            self.df.loc[self.df.index[-1], 'notional_payment'] = notional_amount[-1]


@dataclass
class CouponSchedule(NotionalSchedule):
    coupon_payment_delay: int = 0
    coupon_payment_timing: TimingConvention = TimingConvention.IN_ARREARS

    def __post_init__(self):
        super().__post_init__()
        self.add_payment_dates(self.coupon_payment_delay, self.coupon_payment_timing, col_name='coupon_payment_date')

    def determine_valid_shapes_for_coupon_param(self):
        """Determine valid shapes for the contractual coupon param (fixed rate, spread) based on exchange_notionals."""
        row_count = self.df.shape[0]
        if self.exchange_notionals in [ExchangeNotionals.START, ExchangeNotionals.BOTH]:
            index = self.df.index[1:] # The first row is the initial exchange of notionals, with no coupon
            valid_shape = [(1,), (row_count - 1,)]
        else:
            index = self.df.index[0:]
            valid_shape = [(1,), (row_count,)]
        return index, valid_shape






def make_schedule(
        start_date: [pd.Timestamp, np.datetime64, datetime.date, datetime.datetime],
        end_date: [pd.Timestamp, np.datetime64, datetime.date, datetime.datetime],
        freq: PeriodFreq,
        roll_conv: RollConv=RollConv.MODIFIED_FOLLOWING,
        day_roll: DayRoll=DayRoll.UNADJUSTED,
        first_period_end: Union[pd.Timestamp, np.datetime64, datetime.date, datetime.datetime, None] = None,
        last_period_start: Union[pd.Timestamp, np.datetime64, datetime.date, datetime.datetime, None] = None,
        first_stub: Stub=Stub.DEFAULT,
        last_stub: Stub=Stub.DEFAULT,
        roll_user_specified_dates: bool=False,
        cal: np.busdaycalendar=np.busdaycalendar(),
        ) -> pd.DataFrame:
    """
    Create a schedule. Optional detailed stub logic.

    Parameters
    ----------
    start_date : pandas.Timestamp
        Specifies the effective date of the schedule
    end_date : pandas.Timestamp
        Specifies the termination date of the schedule
    freq : str
        Specify the period frequency
    roll_conv : RollConv
        How to treat dates that do not fall on a valid day. The default is RollConv.MODIFIED_FOLLOWING.
    day_roll : DayRoll
        Specifies the day periods should start/end on. The default is DayRoll.UNADJUSTED.
    first_period_end: pandas.Timestamp
        Specifies the end date of the first period. The first_period_end overrides the first_stub field.
    last_period_start: pandas.Timestamp
        Specifies the start date of the last period. The last_period_start overrides the last_stub field.
    first_stub : Stub
        Specifies the type of the first stub. If first_period_end is specified, the first_stub is ignored.
    last_stub : Stub
        Specifies the type of the last stub. If last_period_start is specified, the last_stub is ignored.
    roll_user_specified_dates : bool
        Boolean flag for whether to roll (per business day calendar and roll convention) the user specified dates (start_date, end_date, first_period_end, last_period_start)
    cal : np.busdaycalendar
        Specifies the business day calendar to observe.

    Returns
    -------
    schedule : pandas.DataFrame
        Columns:
            - fixing_date (if add_fixing_dates=True)
            - period_start
            - period_end
            - payment_date (if add_payment_dates=True)
    """

    start_date, end_date, first_period_end, last_period_start = [
        pd.Timestamp(d) if isinstance(d, (np.datetime64, datetime.date, datetime.datetime)) else d
        for d in [start_date, end_date, first_period_end, last_period_start]
    ]

    # Input validation
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be before end_date {end_date}")

    if first_period_end is not None:
        assert start_date < first_period_end <= end_date

    if last_period_start is not None:
        assert start_date <= last_period_start < end_date

    # if freq == PeriodFreq.CDS:
    #     pass
    # elif freq == PeriodFreq.IMM:
    #     pass

    if freq == PeriodFreq.ZERO_COUPON or first_period_end == end_date or last_period_start == start_date:
        # Function inputs explicitly specify 1 period
        d1, d2 = [start_date], [end_date]
    elif first_period_end == last_period_start and first_period_end is not None and last_period_start is not None:
        # Function inputs explicitly specify 2 periods
        d1 = [start_date, first_period_end]
        d2 = [last_period_start, end_date]
    else:
        if first_period_end is None and last_period_start is None:
            assert first_stub != Stub.DEFINED_PER_FIRST_PERIOD_END_DATE
            assert last_stub != Stub.DEFINED_PER_LAST_PERIOD_START_DATE

            if first_stub == Stub.DEFAULT and last_stub == Stub.DEFAULT:
                # If no stub is specified, defaults to market convention on the 1st stub, no last stub.
                first_stub = Stub.market_convention()
                last_stub = Stub.NONE
            elif first_stub == Stub.DEFAULT:
                first_stub  = Stub.NONE
            elif last_stub == Stub.DEFAULT:
                last_stub = Stub.NONE
            else:
                raise ValueError("Unexpected logic branch - please raise GitHub issue")

            if last_stub == Stub.NONE:
                d1, d2 = generate_date_schedule(start_date, end_date, freq, 'backward', roll_conv, day_roll, roll_user_specified_dates, cal)
                if first_stub == Stub.LONG:
                    d1 = [d1[0]] + d1[2:]
                    d2 = d2[1:]
            elif first_stub == Stub.NONE and last_stub != Stub.NONE:
                d1, d2 = generate_date_schedule(start_date, end_date, freq, 'forward', roll_conv, day_roll, roll_user_specified_dates, cal)
                if last_stub == Stub.LONG:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
            elif first_stub in [Stub.SHORT, Stub.LONG] and last_stub in [Stub.SHORT, Stub.LONG]:
                raise ValueError("If a schedule has first and last stubs they must be specified via first_period_end and last_period_start")
            else:
                raise ValueError("Unexpected logic branch - please raise GitHub issue")

        else:
            # If first_period_end or last_period_start are specified we want to generate the schedule
            # via generate_date_schedule(start_date, end_date, ...) and see if
            # the first_period_end or last_period_start match the generated schedule.
            #
            # If they align, we use this generated schedule.
            #
            # If they don't align, we generate the schedule:
            # (i) via backward generation from last_period_start if only last_period_start is specified
            # (ii) via forward generation from first_period_end if only first_period_end is specified
            # (iii) between first_period_end and last_period_start if both are specified

            # Step 1: Generate the schedule by backward and forward date generation
            d1_backward, d2_backward = generate_date_schedule(start_date, end_date, freq, 'backward', roll_conv, day_roll, roll_user_specified_dates, cal)
            d1_forward, d2_forward = generate_date_schedule(start_date, end_date, freq, 'forward', roll_conv, day_roll, roll_user_specified_dates, cal)

            # Step 2a : Check if the first_period_end matches any generated schedules

            direction = []
            if first_period_end is not None:
                if first_period_end == d2_forward[0]:
                    direction.append('forward')
                    first_stub = Stub.NONE
                elif first_period_end == d2_backward[0]:
                    direction.append('backward')
                    first_stub = Stub.SHORT
                elif first_period_end == d2_backward[1]:
                    direction.append('backward')
                    first_stub = Stub.LONG
                else:
                    # Need to construct schedule using first_period_end
                    first_stub = Stub.DEFINED_PER_FIRST_PERIOD_END_DATE

            # Step 2a : Check if the last_period_start matches any generated schedules
            if last_period_start is not None:
                if last_period_start == d1_backward[-1]:
                    direction.append('backward')
                    last_stub = Stub.NONE
                elif last_period_start == d1_forward[-1]:
                    direction.append('forward')
                    last_stub = Stub.SHORT
                elif last_period_start == d1_forward[-2]:
                    direction.append('forward')
                    last_stub = Stub.LONG
                else:
                    # Need to construct schedule using last_period_start
                    last_stub = Stub.DEFINED_PER_LAST_PERIOD_START_DATE

            # Step 2c : Set default stub values to:
            #              (i) NONE if other stub is SHORT/LONG
            #              (ii) SHORT if other stub is NONE
            if first_stub == Stub.DEFAULT and last_stub != Stub.DEFINED_PER_LAST_PERIOD_START_DATE:
                assert last_stub != Stub.DEFAULT
                if last_stub == Stub.NONE:
                    first_stub = Stub.market_convention()
                else:
                    first_stub = Stub.NONE
            if last_stub == Stub.DEFAULT and first_stub != Stub.DEFINED_PER_FIRST_PERIOD_END_DATE:
                assert first_stub != Stub.DEFAULT
                if first_stub == Stub.NONE:
                    last_stub = Stub.market_convention()
                else:
                    last_stub = Stub.NONE

            # Step 3: Construct the schedules
            if first_stub == Stub.DEFINED_PER_FIRST_PERIOD_END_DATE and last_stub == Stub.DEFINED_PER_LAST_PERIOD_START_DATE:
                # Generate date schedule between first_period_end and last_period_start
                d1, d2 = generate_date_schedule(first_period_end, last_period_start, freq, 'backward', roll_conv, day_roll, roll_user_specified_dates, cal)
                d1 = [start_date] + d1 + [last_period_start]
                d2 = [first_period_end] + d2 + [end_date]
            elif first_stub == Stub.DEFINED_PER_FIRST_PERIOD_END_DATE and last_stub != Stub.DEFINED_PER_LAST_PERIOD_START_DATE:
                # Forward generation from first_period_end and if long last stub, combine last two periods
                d1, d2 = generate_date_schedule(first_period_end, end_date, freq, 'forward', roll_conv, day_roll, roll_user_specified_dates, cal)
                d1 = [start_date] + d1
                d2 = [first_period_end] + d2
                if last_stub == Stub.LONG:
                    d1 = d1[:-1]
                    d2 = d2[:-2] + [d2[-1]]
            elif first_stub != Stub.DEFINED_PER_FIRST_PERIOD_END_DATE and last_stub == Stub.DEFINED_PER_LAST_PERIOD_START_DATE:
                # Backward generation from last_period_start and if long first stub, combine first two periods
                d1, d2 = generate_date_schedule(start_date, last_period_start, freq, 'backward', roll_conv, day_roll, roll_user_specified_dates, cal)
                d1 = d1 + [last_period_start]
                d2 = d2 + [end_date]
                if first_stub == Stub.LONG:
                    d1 = [d1[0]] + d1[2:]
                    d2 = d2[1:]
            else:
                # Don't need to generate any additional schedules
                assert len(set(direction)) == 1

                if first_stub == Stub.NONE:
                    # Forward date generation
                    d1, d2 = d1_forward, d2_forward
                    if last_stub == Stub.LONG:
                        d1 = d1[:-1]
                        d2 = d2[:-2] + [d2[-1]]
                elif last_stub == Stub.NONE:
                    # Backward date generation
                    d1, d2 = d1_backward, d2_backward
                    if first_stub == Stub.LONG:
                        d1 = [d1[0]] + d1[2:]
                        d2 = d2[1:]
                else:
                    raise ValueError("Unexpected logic branch - please raise GitHub issue")

    schedule = pd.DataFrame({'period_start': d1, 'period_end': d2})
    return schedule


def generate_date_schedule(
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp, 
        freq: PeriodFreq,
        direction: str,
        roll_conv: RollConv=RollConv.MODIFIED_FOLLOWING,
        day_roll: DayRoll=DayRoll.UNADJUSTED,
        roll_user_specified_dates: bool=False,
        cal: np.busdaycalendar=np.busdaycalendar()
    ) -> Tuple[List, List]:
    """
    Generates a schedule of start and end dates between start_date and end_date.
    
    Parameters
    ----------
    start_date : pd.Timestamp
        The start date of the schedule.
    end_date : pd.Timestamp
        The end date of the schedule.
    freq : PeriodFreq
        The frequency of the schedule.
    direction : {'forward', 'backward'}
        The direction in which to generate dates.
    roll_conv : RollConv
        How to treat dates that fall on a non business day.
    day_roll : DayRoll
        Specifies the day periods should start/end on.
    roll_user_specified_dates : bool
        Boolean flag for whether to roll (per business day calendar and roll convention) the user specified dates (start_date, end_date) 
    cal : np.busdaycalendar
        Specifies the business day calendar to observe.

    Returns
    -------
    Tuple[List, List]
    
    Raises
    ------
    ValueError, TypeError
        If any of the inputs have invalid values or types

    TODO: Consider options for passing param=(start_date, end_date, freq, None, None, None, None, None, None, None, None) to function
          E.g. Wrap with fields with Optional[] and set defaults in function from enum.set_default()
    """
    
    def busday_offset_timestamp(pd_timestamp, offsets, roll_conv, cal):
        np_datetime64D = np.array([pd_timestamp]).astype('datetime64[D]')
        rolled_date_np = np.busday_offset(np_datetime64D, offsets=offsets, roll=roll_conv, busdaycal=cal)[0]
        return pd.Timestamp(rolled_date_np)
    
    def apply_specific_day_roll(pd_timestamp: pd.Timestamp,
                                specific_day_roll: DayRoll) -> pd.Timestamp:
        if specific_day_roll == DayRoll.UNADJUSTED:
            return pd_timestamp
        else:
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
        start_date_in_schedule = busday_offset_timestamp(start_date, 0, roll_conv.value, cal)
        end_date_in_schedule = busday_offset_timestamp(end_date, 0, roll_conv.value, cal)
    else:
        start_date_in_schedule = start_date
        end_date_in_schedule = end_date

    if direction == 'forward':
        current_date = apply_specific_day_roll(start_date + freq.date_offset, day_roll)
        current_date = busday_offset_timestamp(current_date, 0, roll_conv.value, cal)
        
        while current_date < end_date: 
            start_dates.append(current_date)
            end_dates.append(current_date)
            current_date = start_date + freq.multiply_date_offset(i+1)
            current_date = apply_specific_day_roll(current_date, day_roll)
            current_date = busday_offset_timestamp(current_date, 0, roll_conv.value, cal)
            i += 1

        start_dates = [start_date_in_schedule] + start_dates
        end_dates.append(end_date_in_schedule)
        
    elif direction == 'backward':
        current_date = apply_specific_day_roll(end_date - freq.date_offset, day_roll)
        current_date = busday_offset_timestamp(current_date, 0, roll_conv.value, cal)
        
        while current_date > start_date:   
            start_dates.append(current_date)
            end_dates.append(current_date)
            current_date = end_date - freq.multiply_date_offset(i+1)
            current_date = apply_specific_day_roll(current_date, day_roll)
            current_date = busday_offset_timestamp(current_date, 0, roll_conv.value, cal)
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

