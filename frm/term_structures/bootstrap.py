# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from dataclasses import dataclass
import numpy as np
import pandas as pd

from frm.enums.utils import CompoundingFrequency, PeriodFrequency
from frm.enums.term_structures import OISCouponCalcMethod, TermRate
from frm.utils.daycount import year_fraction
from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.historical_swap_index_fixings import OISFixings, TermFixings


# Cash rates and futures define their own forward and discount curves.
# DF of cash rate definitions to → ZeroCurve
# DF of futures quotes → Zero Curve


# Swap curve bootstrappers
# Boostrap
# (i) forward curve
# (ii) discount curve
# (iii) forward and discount curves
# Based on fixed rates or basis spreads


# Inputs
# Set of quote instruments, quote leg vs reference leg
# Zero curve (to be solved)
# Defined zero curves (if any).
# Adjust pillar points to solve instruments to nil.
