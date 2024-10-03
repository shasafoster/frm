# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
from enum import Enum
import pandas as pd
import numpy as np
from frm.enums.helper import  clean_enum_value, is_valid_enum_value, get_enum_member


class OISCouponCalcMethod(Enum):
    DAILY_COMPOUNDED = 'dailycompounded'
    WEIGHTED_AVERAGE = 'weightedaverage'
    SIMPLE_AVERAGE = 'simpleaverage'

class TermRate(Enum):
    SIMPLE = 'simple'
    CONTINUOUS = 'continuous'
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    SEMIANNUAL = 'semiannual'
    ANNUAL = 'annual'
