# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import pandas as pd

def clean_enum_value(value, transform_fn=None):
    if isinstance(value, str):
        value = value.lower().strip().replace(' ','_')
        if transform_fn:
            value = transform_fn(value)
        if value.isdigit():
            value = int(value)
    elif pd.isna(value) or value is None:
        value = None
    return value
        
    
def is_valid_enum_value(enum_class, value):
    value = clean_enum_value(value)
    return value in {enum_member.value for enum_member in enum_class}


def get_enum_member(enum_class, value, transform_fn=None):
    cleaned_value= clean_enum_value(value, transform_fn)
    if cleaned_value is None:
        return enum_class.default()
    for enum_member in enum_class:
        if enum_member.value == cleaned_value:
            return enum_member

    # List all valid codes in case of an error
    valid_values = [enum_member.value for enum_member in enum_class]
    raise ValueError(f"Invalid value: {value}. Valid codes are: {valid_values}")
        
        


