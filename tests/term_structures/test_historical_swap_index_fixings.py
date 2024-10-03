# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from frm.term_structures.historical_swap_index_fixings import HistoricalSwapIndexFixings
from frm.enums.utils import DayCountBasis, OISCouponCalcMethod, ForwardRate


# For importing the test cases defined in excel
#import sys
#current_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(current_dir)


def test_HistoricalSwapIndexFixings():

    fixings_fp = './tests/term_structures/historical_swap_index_fixings.xlsx'

    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    fixings_df = pd.read_excel(fixings_fp, sheet_name='usd_sofr')
    fixings_df = fixings_df.sort_values('date', ascending=True).reset_index(drop=True)
    fixings_df['fixing'] = fixings_df['fixing'] / 100.0
    
    epsilon = 1e-8
    value_date = pd.Timestamp(2024,6,28)

    historical_fixings = HistoricalSwapIndexFixings(fixings=fixings_df, ois_coupon_calc_method=OISCouponCalcMethod.DAILY_COMPOUNDED, day_count_basis=DayCountBasis.ACT_360)
    
    # Test single
    d1 = pd.DatetimeIndex([pd.Timestamp(2022,10,4)]) 
    d2 = pd.DatetimeIndex([pd.Timestamp(2023,10,3)]) 
    daily_compounded = historical_fixings.calc_historical_ois_coupon_rate(d1, d2, OISCouponCalcMethod.DAILY_COMPOUNDED)
    weighted_average = historical_fixings.calc_historical_ois_coupon_rate(d1, d2, OISCouponCalcMethod.WEIGHTED_AVERAGE)
    simple_average = historical_fixings.calc_historical_ois_coupon_rate(d1, d2, OISCouponCalcMethod.SIMPLE_AVERAGE)
    
    assert abs(daily_compounded - 0.047019224) < epsilon
    assert abs(weighted_average - 0.045938736) < epsilon
    assert abs(simple_average   - 0.045956855) < epsilon
    
    # Test vector
    data = {
        'fixing_start': ['Wed 18-Mar-2020', 'Thu 18-Jun-2020', 'Fri 18-Sep-2020', 'Fri 18-Dec-2020', 'Thu 18-Mar-2021',
                         'Mon 21-Jun-2021', 'Mon 20-Sep-2021', 'Mon 20-Dec-2021', 'Fri 18-Mar-2022', 'Tue 21-Jun-2022',
                         'Mon 19-Sep-2022', 'Mon 19-Dec-2022', 'Mon 20-Mar-2023', 'Tue 20-Jun-2023', 'Mon 18-Sep-2023',
                         'Mon 18-Dec-2023', 'Mon 18-Mar-2024'],
        'fixing_end': ['Thu 18-Jun-2020', 'Fri 18-Sep-2020', 'Fri 18-Dec-2020', 'Thu 18-Mar-2021', 'Mon 21-Jun-2021',
                       'Mon 20-Sep-2021', 'Mon 20-Dec-2021', 'Fri 18-Mar-2022', 'Tue 21-Jun-2022', 'Mon 19-Sep-2022',
                       'Mon 19-Dec-2022', 'Mon 20-Mar-2023', 'Tue 20-Jun-2023', 'Mon 18-Sep-2023', 'Mon 18-Dec-2023',
                       'Mon 18-Mar-2024', 'Tue 18-Jun-2024'],
        'daily_compounded': ['0.03989330680%', '0.09348930352%', '0.08473424620%', '0.05233670687%', '0.01168438786%',
                             '0.05000312513%', '0.04923379875%', '0.05216236850%', '0.56409784646%', '1.97134170167%',
                             '3.42778709990%', '4.45295400110%', '4.95897401737%', '5.23547963755%', '5.35263821776%',
                             '5.35386318206%', '5.35383702507%'],
        'weighted_average': ['0.03989130435%', '0.09347826087%', '0.08472527473%', '0.05233333333%', '0.01168421053%',
                             '0.05000000000%', '0.04923076923%', '0.05215909091%', '0.56368421053%', '1.96655555556%',
                             '3.41318681319%', '4.42835164835%', '4.92815217391%', '5.20188888889%', '5.31714285714%',
                             '5.31835164835%', '5.31793478261%'],
        'simple_average': ['0.04031250000%', '0.09375000000%', '0.08419354839%', '0.05216666667%', '0.01123076923%',
                           '0.05000000000%', '0.04887096774%', '0.05311475410%', '0.54328125000%', '1.96096774194%',
                           '3.37903225806%', '4.42967213115%', '4.91920634921%', '5.19935483871%', '5.31650793651%',
                           '5.31836065574%', '5.31734375000%']
    }
    
    df = pd.DataFrame(data)
    df['fixing_start'] = pd.to_datetime(df['fixing_start'], format='%a %d-%b-%Y')
    df['fixing_end'] = pd.to_datetime(df['fixing_end'], format='%a %d-%b-%Y')
    for v in ['daily_compounded','weighted_average','simple_average']:
        df[v] = df[v].str.rstrip('%').astype(float) / 100.0
    
    d1 = df['fixing_start']
    d2 = df['fixing_end']
    
    daily_compounded = historical_fixings.calc_historical_ois_coupon_rate(d1, d2, OISCouponCalcMethod.DAILY_COMPOUNDED)
    weighted_average = historical_fixings.calc_historical_ois_coupon_rate(d1, d2, OISCouponCalcMethod.WEIGHTED_AVERAGE)
    simple_average = historical_fixings.calc_historical_ois_coupon_rate(d1, d2, OISCouponCalcMethod.SIMPLE_AVERAGE)
    
    assert (abs(daily_compounded - df['daily_compounded']) < epsilon).all()
    assert (abs(weighted_average - df['weighted_average']) < epsilon).all()
    assert (abs(simple_average - df['simple_average']) < epsilon).all()    

    # Test term fixings

    fixings_df = pd.read_excel(fixings_fp, sheet_name='nzd_bkbm_3m')
    fixings_df = fixings_df.sort_values('date', ascending=True).reset_index(drop=True)
    fixings_df['fixing'] = fixings_df['fixing'] / 100.0
    
    historical_fixings = HistoricalSwapIndexFixings(fixings_df, day_count_basis=DayCountBasis.ACT_365)
    dates = pd.DatetimeIndex([pd.Timestamp(2024,9,25), pd.Timestamp(2024,6,25), pd.Timestamp(2024,9,25), pd.Timestamp(2024,6,25)])
    fixings = historical_fixings.index_historical_fixings(dates)
    
    assert (fixings == np.array([0.0492, 0.0561, 0.0492, 0.0561])).any()


if __name__ == "__main__":    
    test_HistoricalSwapIndexFixings()













