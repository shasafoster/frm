# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from frm.term_structures.historical_swap_fixings import HistoricalSwapRateFixings
from frm.utils.enums import DayCountBasis, OISCouponCalcMethod, SwapType


# For importing the test cases defined in excel
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def test_HistoricalSwapRateFixings():

    historical_fixings_df = pd.read_excel('C:/Users/shasa/Documents/frm_project/tests/term_structures/historical_fixings.xlsx', sheet_name='usd_sofr')
    historical_fixings_df = historical_fixings_df.sort_values('date', ascending=True).reset_index(drop=True)
    historical_fixings_df['fixing'] = historical_fixings_df['fixing'] / 100.0
    
    epsilon = 1e-8
    value_date = pd.Timestamp(2024,6,28)
    day_count_basis = DayCountBasis.from_value('act/360')
    
    historical_fixings = HistoricalSwapRateFixings(SwapType.OIS, historical_fixings_df, OISCouponCalcMethod.DAILY_COMPOUNDED, DayCountBasis.ACT_360)
    
    # Test single
    d1 = pd.DatetimeIndex([pd.Timestamp(2022,10,4)]) # accrual_period_start_date
    d2 = pd.DatetimeIndex([pd.Timestamp(2023,10,3)]) # accrual_period_end_date
    daily_compounded = historical_fixings.calc_OIS_historical_coupon_rate(d1, d2, value_date, day_count_basis, OISCouponCalcMethod.DAILY_COMPOUNDED)
    weighted_average = historical_fixings.calc_OIS_historical_coupon_rate(d1, d2, value_date, day_count_basis, OISCouponCalcMethod.WEIGHTED_AVERAGE)
    simple_average = historical_fixings.calc_OIS_historical_coupon_rate(d1, d2, value_date, day_count_basis, OISCouponCalcMethod.SIMPLE_AVERAGE)
    
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
    
    daily_compounded = historical_fixings.calc_OIS_historical_coupon_rate(d1, d2, value_date, day_count_basis, OISCouponCalcMethod.DAILY_COMPOUNDED)
    weighted_average = historical_fixings.calc_OIS_historical_coupon_rate(d1, d2, value_date, day_count_basis, OISCouponCalcMethod.WEIGHTED_AVERAGE)
    simple_average = historical_fixings.calc_OIS_historical_coupon_rate(d1, d2, value_date, day_count_basis, OISCouponCalcMethod.SIMPLE_AVERAGE)
    
    assert (abs(daily_compounded - df['daily_compounded']) < epsilon).all()
    assert (abs(weighted_average - df['weighted_average']) < epsilon).all()
    assert (abs(simple_average - df['simple_average']) < epsilon).all()    



    # Test term fixings

    historical_fixings_df = pd.read_excel('C:/Users/shasa/Documents/frm_project/tests/term_structures/historical_fixings.xlsx', sheet_name='nzd_bkbm_3m')
    historical_fixings_df = historical_fixings_df.sort_values('date', ascending=True).reset_index(drop=True)
    historical_fixings_df['fixing'] = historical_fixings_df['fixing'] / 100.0
    
    epsilon = 1e-8
    value_date = pd.Timestamp(2024,6,28)
    
    historical_fixings = HistoricalSwapRateFixings(SwapType.TERM, historical_fixings_df)
    dates = pd.DatetimeIndex([pd.Timestamp(2024,9,25), pd.Timestamp(2024,6,25), pd.Timestamp(2024,9,25), pd.Timestamp(2024,6,25)])
    fixings = historical_fixings.index_term_fixings(dates)
    
    assert (fixings == np.array([0.0492, 0.0561, 0.0492, 0.0561])).any()


if __name__ == "__main__":    
    test_HistoricalSwapRateFixings()
    




#%%






#%%





data = {
    'date': ['30-Jun-2024', '9-Jul-2024', '16-Jul-2024', '23-Jul-2024', '2-Aug-2024', '3-Sep-2024', '2-Oct-2024', '4-Nov-2024', 
             '2-Dec-2024', '2-Jan-2025', '3-Feb-2025', '3-Mar-2025', '2-Apr-2025', '2-May-2025', '2-Jun-2025', '2-Jul-2025', 
             '2-Oct-2025', '2-Jan-2026', '2-Apr-2026', '2-Jul-2026', '2-Jul-2027', '3-Jul-2028', '2-Jul-2029', '2-Jul-2030', 
             '2-Jul-2031', '2-Jul-2032', '5-Jul-2033', '3-Jul-2034', '2-Jul-2035', '2-Jul-2036', '5-Jul-2039', '5-Jul-2044', 
             '2-Jul-2049', '2-Jul-2054', '2-Jul-2064', '2-Jul-2074'],
    'discount_factor': [1.00000, 0.99837, 0.99734, 0.99631, 0.99484, 0.99017, 0.98600, 0.98134, 0.97746, 0.97326, 
                        0.96902, 0.96543, 0.96164, 0.95795, 0.95422, 0.95071, 0.94036, 0.93052, 0.92138, 0.91239, 
                        0.87852, 0.84665, 0.81616, 0.78615, 0.75693, 0.72859, 0.70089, 0.67432, 0.64824, 0.62304, 
                        0.55286, 0.45724, 0.38931, 0.33641, 0.26723, 0.22828]
}

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')

curve_date = pd.Timestamp(2024,6,28)

zero_curve = ZeroCurve(curve_date=curve_date, zero_data=df, historical_fixings=historical_fixings)

data = {
    'fixing_start': ['18-Mar-2020', '18-Jun-2020', '18-Sep-2020', '18-Dec-2020', '18-Mar-2021',
                     '21-Jun-2021', '20-Sep-2021', '20-Dec-2021', '18-Mar-2022', '21-Jun-2022',
                     '19-Sep-2022', '19-Dec-2022', '20-Mar-2023', '20-Jun-2023', '18-Sep-2023',
                     '18-Dec-2023', '18-Mar-2024', '18-Jun-2024', '18-Sep-2024', '18-Dec-2024',
                     '18-Mar-2025', '18-Jun-2025', '18-Sep-2025', '18-Dec-2025', '18-Mar-2026',
                     '18-Jun-2026', '18-Sep-2026', '18-Dec-2026', '18-Mar-2027', '21-Jun-2027',
                     '20-Sep-2027', '20-Dec-2027', '20-Mar-2028', '20-Jun-2028', '18-Sep-2028',
                     '18-Dec-2028', '19-Mar-2029', '18-Jun-2029', '18-Sep-2029', '18-Dec-2029',
                     '18-Mar-2030', '18-Jun-2030', '18-Sep-2030', '18-Dec-2030', '18-Mar-2031',
                     '18-Jun-2031'],
    'fixing_end': ['18-Jun-2020', '18-Sep-2020', '18-Dec-2020', '18-Mar-2021', '21-Jun-2021',
                   '20-Sep-2021', '20-Dec-2021', '18-Mar-2022', '21-Jun-2022', '19-Sep-2022',
                   '19-Dec-2022', '20-Mar-2023', '20-Jun-2023', '18-Sep-2023', '18-Dec-2023',
                   '18-Mar-2024', '18-Jun-2024', '18-Sep-2024', '18-Dec-2024', '18-Mar-2025',
                   '18-Jun-2025', '18-Sep-2025', '18-Dec-2025', '18-Mar-2026', '18-Jun-2026',
                   '18-Sep-2026', '18-Dec-2026', '18-Mar-2027', '21-Jun-2027', '20-Sep-2027',
                   '20-Dec-2027', '20-Mar-2028', '20-Jun-2028', '18-Sep-2028', '18-Dec-2028',
                   '19-Mar-2029', '18-Jun-2029', '18-Sep-2029', '18-Dec-2029', '18-Mar-2030',
                   '18-Jun-2030', '18-Sep-2030', '18-Dec-2030', '18-Mar-2031', '18-Jun-2031',
                   '18-Sep-2031'],
    'cpn_daily_compounded': [0.0003989331, 0.0009348930, 0.0008473425, 0.0005233671, 0.0001168439,
                             0.0005000313, 0.0004923380, 0.0005216237, 0.0056409785, 0.0197134170,
                             0.0342778710, 0.0445295400, 0.0495897402, 0.0523547964, 0.0535263822,
                             0.0535386318, 0.0535383703, 0.0545176793, 0.0515994888, 0.0488073021,
                             0.0459646033, 0.0432822870, 0.0416368219, 0.0399620679, 0.0390964133,
                             0.0377162365, 0.0374871518, 0.0374852062, 0.0374949355, 0.0365428742,
                             0.0364130535, 0.0364130535, 0.0364148894, 0.0364349026, 0.0364407400,
                             0.0364407400, 0.0364407400, 0.0370208178, 0.0371227051, 0.0371207971,
                             0.0371246133, 0.0374741343, 0.0375349215, 0.0375329709, 0.0375368721,
                             0.0376875902]
}

df = pd.DataFrame(data)
df['fixing_start'] = pd.to_datetime(df['fixing_start'], format='%d-%b-%Y')
df['fixing_end'] = pd.to_datetime(df['fixing_end'], format='%d-%b-%Y')

%%

value_date = pd.Timestamp(2024,6,28)
mask_historical = df['fixing_end'] < value_date
mask_crossover = np.logical_and(df['fixing_end'] >= value_date, df['fixing_start'] < value_date)
mask_forward = df['fixing_start'] >= value_date



#%%
DF_t1 = zero_curve.discount_factor(df['fixing_start'])
DF_t2 = zero_curve.discount_factor(df['fixing_end'])




%%




start_date = pd.Timestamp(2023,1,17)
curve_date = pd.Timestamp(2023,6,30)
end_date = pd.Timestamp(2024,1,17)

hist_component_cpn_rate = 4.820573 * 0.01
fwd_component_cpn_rate = 5.381977 * 0.01    

hist_component_year_frac = year_fraction(start_date, curve_date, day_count_basis)
fwd_component_year_frac = year_fraction(curve_date, end_date, day_count_basis)

cpn_multiplier = hist_component_year_frac * hist_component_cpn_rate \
                 + (1 + hist_component_year_frac * hist_component_cpn_rate) * fwd_component_year_frac * fwd_component_cpn_rate
            
cpn_rate = cpn_multiplier / year_fraction(start_date, end_date, day_count_basis)
           



ois_fixings = pd.read_excel('C:/Users/shasa/Documents/frm_project/tests/term_structures/ois_publications.xlsx', sheet_name='sofr')
ois_fixings = ois_fixings.sort_values('observation_date', ascending=True).reset_index(drop=True)
ois_fixings['fixing'] = ois_fixings['fixing'] / 100.0

day_count_basis = DayCountBasis.from_value('act/360')
busdaycal = get_busdaycal('USD')
calc_method = OISCouponCalcMethod.DailyCompounded

fixing_schedule = schedule(start_date=pd.Timestamp(2020,3,18),
                           end_date=pd.Timestamp(2031,9,18),
                           frequency='quarterly',
                           roll_convention='modifiedfollowing',
                           busdaycal=busdaycal)

result = calc_ois_historical_coupon(fixing_schedule['period_start'], fixing_schedule['period_end'], ois_fixings, calc_method, day_count_basis)











