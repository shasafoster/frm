# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
from enum import Enum
import pandas as pd
import numpy as np
from frm.enums.helper import  clean_enum_value, is_valid_enum_value, get_enum_member


class ZeroCurveInterpMethod(Enum):
    LINEAR_ON_LN_DISCOUNT = 'linear_on_ln_discount'
    CUBIC_SPLINE_ON_LN_DISCOUNT = 'cubic_spline_on_ln_discount'
    CUBIC_SPLINE_ON_CCZR = 'cubic_spline_on_cczr'

    @classmethod
    def default(cls):
        return cls.LINEAR_ON_LN_DISCOUNT # Return the default enum value

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""

        def specific_cleaning(value):
            return value

        cleaned_value = clean_enum_value(value)
        cleaned_value = specific_cleaning(cleaned_value)

        if cleaned_value is None:
            return cls.default()
        for enum_member in cls:
            if enum_member.value == cleaned_value \
                    or enum_member.value.replace('/', '') == cleaned_value:
                return enum_member

        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")

    @property
    def display_name(self):
        dict_ = {
            'LINEAR_ON_LN_DISCOUNT': 'Linear on log of discount factors',
            'CUBIC_SPLINE_ON_LN_DISCOUNT': 'Cubic spline on log of discount factors',
            'CUBIC_SPLINE_ON_CCZR': 'Cubic spline on continuously compounded zero rates'
        }
        return dict_[self.name]


class ZeroCurveExtrapMethod(Enum):
    NONE = 'none'
    FLAT = 'flat'

    @classmethod
    def default(cls):
        return cls.NONE # Return the default enum value

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""

        def specific_cleaning(value):
            return value

        cleaned_value = clean_enum_value(value)
        cleaned_value = specific_cleaning(cleaned_value)

        if cleaned_value is None:
            return cls.default()
        for enum_member in cls:
            if enum_member.value == cleaned_value \
                    or enum_member.value.replace('/', '') == cleaned_value:
                return enum_member

        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")

    @property
    def display_name(self):
        return self.name.title()



class PayRcv(Enum):
    PAY = 'pay'
    RCV = 'rcv'

    @property
    def multiplier(self):
        return -1 if self == PayRcv.PAY else 1

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""

        def specific_cleaning(value):
            return value

        cleaned_value = clean_enum_value(value)
        cleaned_value = specific_cleaning(cleaned_value)

        if cleaned_value is None:
            return cls.default()
        for enum_member in cls:
            if enum_member.value == cleaned_value \
                    or enum_member.value.replace('/', '') == cleaned_value:
                return enum_member

        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")

    @property
    def display_name(self):
        return self.name.title()


class BuySell(Enum):
    BUY = 'buy'
    SELL = 'sell'

    @property
    def multiplier(self):
        return -1 if self == BuySell.SELL else 1

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""

        def specific_cleaning(value):
            return value

        cleaned_value = clean_enum_value(value)
        cleaned_value = specific_cleaning(cleaned_value)

        if cleaned_value is None:
            return cls.default()
        for enum_member in cls:
            if enum_member.value == cleaned_value \
                    or enum_member.value.replace('/', '') == cleaned_value:
                return enum_member

        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")

    @property
    def display_name(self):
        return self.name.title()


class DayCountBasis(Enum):
    # If storing instrument definitions in CDM, use 'code' to define valid fieldnames
    _30_360 = '30/360'
    _30E_360 = '30e/360'
    _30E_360_ISDA = '30e/360_isda'
    ACT_360 = 'act/360'
    ACT_365 = 'act/365'
    ACT_ACT = 'act/act'
    ACT_366 = 'act/366'

    def __init__(self, value):        
        days_per_year = {
            '30/360': 360,
            '30e/360': 360,
            '30e/360_isda': 360,
            'act/360': 360,
            'act/365': 365,
            'act/act': np.nan,
            'act/366': 366,
            }
        self.days_per_year = days_per_year[self.value]

    @classmethod
    def default(cls):
        return cls.ACT_ACT # Return the default enum value

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""
        
        def specific_cleaning(value):
            value = value.replace('actual','act')
            if value == 'act/365fixed':
                value = 'act/365'
            return value        

        cleaned_value = clean_enum_value(value)
        cleaned_value = specific_cleaning(cleaned_value)
        
        if cleaned_value is None:
            return cls.default()        
        for enum_member in cls:
            if enum_member.value == cleaned_value \
                or enum_member.value.replace('/','') == cleaned_value:
                return enum_member

        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}") 

    @property
    def display_name(self):
        return self.value.upper().replace('_', ' ').strip()


class CompoundingFreq(Enum):
    # If storing instrument definitions in CDM, use 'code' to define valid fieldnames
    SIMPLE = 'simple'
    CONTINUOUS = 'continuous'
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    SEMIANNUAL = 'semiannual'
    ANNUAL = 'annual'

    def __init__(self, value):        
        periods_per_year_map = {
            'simple': None,
            'continuous': None,
            'daily': 365,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'semiannual': 2,
            'annual': 1
            }
        self.periods_per_year = periods_per_year_map[self.value]

    @classmethod
    def is_valid(cls, value):
        return is_valid_enum_value(cls, value)

    @classmethod
    def from_value(cls, value):
        return get_enum_member(cls, value)
        # """Create an enum member from the given value, if valid."""
        # cleaned_value= clean_enum_value(value)
        # for enum_member in cls:
        #     if enum_member.value == cleaned_value:
        #         return enum_member
        # # List all valid codes in case of an error
        # valid_values = [enum_member.value for enum_member in cls]
        # raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")

    @property
    def display_name(self):
        return self.name.title().replace('_', ' ').strip()


class PeriodFreq(Enum):
    # If storing instrument definitions in CDM, use 'code' to define valid fieldnames
    DAILY = 'daily'
    WEEKLY = 'weekly'
    _28_DAYS = '28days'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    SEMIANNUAL = 'semiannual'
    ANNUAL = 'annual'
    ZERO_COUPON = 'zerocoupon'
    #CDS = 'cds' # 20th of Mar/Jun/Sep/Dec
    #IMM = 'imm' # 3rd Wednesday of Mar/Jun/Sep/Dec

    def __init__(self, value):        
        date_offset_map = {
            'daily': pd.DateOffset(days=1),
            'weekly': pd.DateOffset(weeks=1),
            '28days': pd.DateOffset(days=28),
            'monthly': pd.DateOffset(months=1),
            'quarterly': pd.DateOffset(months=3),
            'semiannual': pd.DateOffset(months=6),
            'annual': pd.DateOffset(years=1),
            'zerocoupon': None, # No date offset for zero coupon
            #'cds': None, # Custom logic
            #'imm': None, # Custom logic
            }
        self.date_offset = date_offset_map[self.value]

    def multiply_date_offset(self, factor):
        """Multiply the current date_offset by the given factor."""
        # The reason for this function is x * (date += offset) != date += (x * offset).
        # Proof:
        # pd.Timestamp('2023-08-31') + 3 * pd.DateOffset(months=3) == pd.Timestamp('2024-05-29')
        # pd.Timestamp('2023-08-31') + pd.DateOffset(months=3*3) == pd.Timestamp('2024-05-31')
        # The 1st calculation regresses to the shortest end-of-month (Feb)

        if self.date_offset is None:
            return None  # Handle None for ZERO_COUPON case

        return pd.DateOffset(
            days=self.date_offset.kwds.get('days', 0) * factor,
            weeks=self.date_offset.kwds.get('weeks', 0) * factor,
            months=self.date_offset.kwds.get('months', 0) * factor,
            years=self.date_offset.kwds.get('years', 0) * factor
        )

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value, cls._transform_value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        return get_enum_member(cls, value, cls._transform_value)

    @staticmethod
    def _transform_value(value: str) -> str:
        value = value.replace('/', '').replace('_', '').replace('per', '').lower().strip()

        period_frequency_aliases = {
            'daily': ['d', 'day', 'days', '1d', '1day', '1days'],
            'weekly': ['w', 'week', 'weeks', 'wk', '1w', '1wk', '1week', '1weeks'],
            '28days': ['28d', '28day', '28days'],
            'monthly': ['m', 'month', 'months', 'mon', '1m', '1mo', '1month', '1months'],
            'quarterly': ['q', 'quarter', 'quarters', 'quart', '3m', '3mo', '3month', '3months'],
            'semiannual': ['s','sa', 'semiannual', 'semiannually', 'semi','6m', '6mo', '6month', '6months'],
            'annual': ['a', 'year', 'years', 'yrs', 'yr', 'annually', '1y', '1yr', '1year', '1years'],
            'zerocoupon': ['z', 'zc', 'zerocoupon', 'zero']
        }

        for key, aliases in period_frequency_aliases.items():
            if value == key or value in aliases:
                return key
        return value

    @property
    def display_name(self):
        return self.name.title().replace('_', ' ').strip()


class DayRoll(Enum):
    UNADJUSTED = 'unadjusted'
    _1 = 1
    _2 = 2
    _3 = 3
    _4 = 4
    _5 = 5
    _6 = 6
    _7 = 7
    _8 = 8
    _9 = 9
    _10 = 10
    _11 = 11
    _12 = 12
    _13 = 13
    _14 = 14
    _15 = 15
    _16 = 16
    _17 = 17
    _18 = 18
    _19 = 19
    _20 = 20
    _21 = 21
    _22 = 22
    _23 = 23
    _24 = 24
    _25 = 25
    _26 = 26
    _27 = 27
    _28 = 28
    _29 = 29
    _30 = 30
    _31 = 31
    EOM = 'eom' # End of month

    @classmethod
    def default(cls):
        return cls.UNADJUSTED # Return the default enum value

    @classmethod
    def is_valid(cls, value):
        return is_valid_enum_value(cls, value)

    @classmethod
    def from_value(cls, value):
        return get_enum_member(cls, value)

    @property
    def display_name(self):
        return self.name.title().replace('_', ' ').strip()
        
        
    
class RollConv(Enum):
    UNADJUSTED = 'unadjusted'
    FOLLOWING = 'following'
    PRECEDING = 'preceding'
    MODIFIED_FOLLOWING = 'modifiedfollowing'
    MODIFIED_PRECEDING = 'modifiedpreceding'

    @classmethod
    def is_valid(cls, value):
        return is_valid_enum_value(cls, value)

    @classmethod
    def from_value(cls, value):
        return get_enum_member(cls, value)

    @classmethod
    def default(cls):
        return cls.MODIFIED_FOLLOWING # Return the default enum value

    @property
    def display_name(self):
        return self.name.replace('_', ' ').title()


class TimingConvention(Enum):
    IN_ARREARS = 'in_arrears'
    IN_ADVANCE = 'in_advance'

    @classmethod
    def is_valid(cls, value):
        return is_valid_enum_value(cls, value)

    @classmethod
    def from_value(cls, value):
        return get_enum_member(cls, value)
        
    @property
    def display_name(self):
        return self.name.title().replace('_', ' ')

    @classmethod
    def fixing_default(cls):
        return cls.IN_ADVANCE # Return the default enum value

    @classmethod
    def payment_default(cls):
        return cls.IN_ARREARS # Return the default enum value

class Stub(Enum):
    NONE = 'none'
    SHORT = 'short'
    LONG = 'long'
    DEFAULT = 'default'
    DEFINED_PER_FIRST_PERIOD_END_DATE = 'defined_per_first_period_end_date'
    DEFINED_PER_LAST_PERIOD_START_DATE = 'defined_per_last_period_start_date'

    @classmethod
    def is_valid(cls, value):
        return is_valid_enum_value(cls, value)

    @classmethod
    def from_value(cls, value):
        return get_enum_member(cls, value)

    @classmethod
    def default(cls):
        return cls.DEFAULT  # Return the default enum value

    @classmethod
    def market_convention(cls):
        return cls.SHORT  # Market convention is a short stub

    @property
    def display_name(self):
        return self.name.title().replace('_', ' ')
        

class ExchangeNotionals(Enum):
    START = 'start'
    END = 'end'
    BOTH = 'both'
    NEITHER = 'neither'

    @classmethod
    def is_valid(cls, value):
        return is_valid_enum_value(cls, value)

    @classmethod
    def from_value(cls, value):
        return get_enum_member(cls, value)

    @property
    def display_name(self):
        return self.name.title()

    @classmethod
    def default(cls):
        return cls.NEITHER # Return the default enum value


