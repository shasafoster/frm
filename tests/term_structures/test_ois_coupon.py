# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))



# from frm.instruments.ir.swap import Swap
# from frm.instruments.ir.swap_defaults import USD_LIBOR_SWAP_ABOVE_1Y, USD_LIBOR_SWAP_1Y


import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.zero_curve_helpers import calc_ois_historical_coupon, OISCouponCalcMethod


# For importing the test cases defined in excel
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)



def test_calc_ois_historical_coupon():

    ois_fixings = pd.read_excel(current_dir + '\\ois_publications.xlsx', sheet_name='sofr')
    ois_fixings = ois_fixings.sort_values('observation_date', ascending=True).reset_index(drop=True)
    ois_fixings['fixing'] = ois_fixings['fixing'] / 100.0
    
    days_per_year = 360
        
    d1 = pd.Timestamp(2024,9,10) # Coupon Accrual Start Date
    d2 = pd.Timestamp(2024,9,28) # Coupon Accrual End Date
    
    daily_compounded = calc_ois_historical_coupon(d1, d2, ois_fixings, OISCouponCalcMethod.DailyCompounded, days_per_year)
    weighted_average = calc_ois_historical_coupon(d1, d2, ois_fixings, OISCouponCalcMethod.WeightedAverage, days_per_year)
    simple_average = calc_ois_historical_coupon(d1, d2, ois_fixings, OISCouponCalcMethod.SimpleAverage, days_per_year)
    
    
    assert abs(daily_compounded - 5.091666 * 0.01) < 1e-8
    assert abs(weighted_average - 5.085556 * 0.01) < 1e-8
    assert abs(simple_average   - 5.106923 * 0.01) < 1e-8













