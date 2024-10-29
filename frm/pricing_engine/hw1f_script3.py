# -*- coding: utf-8 -*-
import os

from frm.enums.term_structures import TermRate

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.hull_white_1_factor_class import HullWhite1Factor
from frm.enums.utils import DayCountBasis, CompoundingFrequency
from frm.utils.business_day_calendar import get_busdaycal
from prettytable import PrettyTable
from scipy.stats import norm

fp = 'C:/Users/shasa/Documents/frm_private/tests_private/term_structures/test_optionlet_support_20240628.xlsm'
curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')

zero_curve = ZeroCurve(curve_date=curve_date,
                       data=pd.read_excel(io=fp, sheet_name='DF_3M'),
                       day_count_basis=DayCountBasis.ACT_365,
                       busdaycal=busdaycal,
                       interpolation_method='cubic_spline_on_zero_rates')


hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=0.001, vol=0.2)
hw1f.setup_theta()
#hw1f.calc_error_for_theta_fit(print_results=True)


t1 = 0.7589
t2 =  1.00822

d1 = pd.Timestamp('2025-04-01')
d2 = pd.Timestamp('2025-07-01')

#print("ZeroCurve F:",hw1f.zero_curve.get_forward_rates(period_start=d1, period_end=d2, forward_rate_type=TermRate.SIMPLE)[0])
#print("HW1F F:",hw1f.get_forward_rate(t1, t2))

K = 0.04
annuity_factor = 0.2383
cp = 1

px = 100e6 * hw1f.price_zero_coupon_bond_option(T=t1, S=t2, K=K, cp=cp)
print('optionlet px: {:,.0f}'.format(px))

#%%%price_zero_coupon_bond_option(self, T, S, K, cp, t=0)


