# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
from enum import Enum
import pandas as pd



def clean_enum_value(value):
    if isinstance(value, str):
        value = value.lower().strip().replace(' ','_')
        if value.isdigit():
            value = int(value)
    elif pd.isna(value) or value is None:
        value = None
    return value
        


class Frequency(Enum):
    # If storing instrument definitions in CDM, use 'code' to define valid fieldnames
    DAILY = 'd'
    WEEKLY = 'w'
    FORTNIGHTLY = '2w'
    TWENTY_EIGHT_DAYS = '28d'
    MONTHLY = 'm'
    QUARTERLY = 'q'
    SEMIANNUAL = 's'
    ANNUAL = 'a'
    ZERO_COUPON = 'z'

    def __init__(self, value):        
        date_offset_map = {
            'd' : pd.DateOffset(days=1),
            'w' : pd.DateOffset(weeks=1),
            '2w': pd.DateOffset(weeks=2),
            '28d': pd.DateOffset(days=28),
            'm': pd.DateOffset(months=1),
            'q': pd.DateOffset(months=3),
            's': pd.DateOffset(months=6),
            'a': pd.DateOffset(years=1),
            'z': None
            }
        self.date_offset = date_offset_map[self.value]

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""
        cleaned_value= clean_enum_value(value)
        for enum_member in cls:
            if enum_member.value == cleaned_value:
                return enum_member
        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")

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



class DayRoll(Enum):
    NONE = None
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
        return cls.NONE # Return the default enum value

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""
        cleaned_value= clean_enum_value(value)
        if cleaned_value is None:
            return cls.default()        
        for enum_member in cls:
            if enum_member.value == cleaned_value:
                return enum_member

        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}") 
        
        
    
    
class RollConvention(Enum):
    NO_ROLL = 'no_roll'
    FOLLOWING = 'following'
    PRECEDING = 'preceding'
    MODIFIED_FOLLOWING = 'modifiedfollowing' # 
    MODIFIED_PRECEDING = 'modifiedpreceding'
     
    @classmethod
    def default(cls):
        return cls.MODIFIED_FOLLOWING # Return the default enum value

    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}
 
    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""
        cleaned_value= clean_enum_value(value)
        for enum_member in cls:
            if enum_member.value == cleaned_value:
                return enum_member
        if cleaned_value is None:
            return cls.default()
            
        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")  



class PaymentType(Enum):
    IN_ARREARS = 'in_arrears'
    IN_ADVANCE = 'in_advance'
    
    @classmethod
    def default(cls):
        return cls.IN_ARREARS # Return the default enum value    
    
    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls} 
    
    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""
        cleaned_value= clean_enum_value(value)
        for enum_member in cls:
            if enum_member.value == cleaned_value:
                return enum_member
        if cleaned_value is None:
            return cls.default()
            
        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}") 
        
    
    
class StubType(Enum):
    NONE = 'none'
    SHORT = 'short'
    LONG = 'long'
    DEFAULT = 'default'
    DEFINED_PER_FIRST_CPN_END_DATE = 'defined_per_first_cpn_end_date'
    DEFINED_PER_LAST_CPN_START_DATE = 'defined_per_last_cpn_start_date'
        
    
    @classmethod
    def default(cls):
        return cls.DEFAULT # Return the default enum value    
    
    @classmethod
    def market_convention(cls):
        return cls.SHORT # Market convention is a short stub 
    
    @classmethod
    def is_valid(cls, value):
        value = clean_enum_value(value)
        return value in {enum_member.value for enum_member in cls}

    @classmethod
    def from_value(cls, value):
        """Create an enum member from the given value, if valid."""
        cleaned_value= clean_enum_value(value)
        if cleaned_value is None:
            return cls.default()
        for enum_member in cls:
            if enum_member.value == cleaned_value:
                return enum_member
            
        # List all valid codes in case of an error
        valid_values = [enum_member.value for enum_member in cls]
        raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")    
        
        


