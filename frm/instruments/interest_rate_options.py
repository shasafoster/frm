# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

from frm.utils.daycount import day_count, year_fraction
from frm.utils.schedule import get_schedule, get_payment_dates, get_fixing_dates
from frm.enums.utils import DayCountBasis, DayRoll, PeriodFrequency, StubType, RollConvention, TimingConvention
#from frm.term_structures.swap_curve import TermSwapCurve, OISCurve

from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, InitVar 
from typing import Optional



# Cap/Floor = List of Caplets/Floorlets
# Can simplify to just have methods loop over rows in the schedule. Each row is a caplet/floorlet.
# Valuation + greeks = sum of caplet/floorlet values
# Cap/Floor schedule is the same as the underlying leg schedule
# Inputs - same as inputs to "float leg" in swap, plus Strike
# Caplet/floorlet valuation + greeks is simply black76 formula
# Valuation requires zero_curve object (forward and discount curves) and ir_vol_curve object.

# IR vol curve
# Pandas dataframe:
# (i) SABR smile for each expiry
# (ii) interpolation of SABR smile for every day




        
        
               