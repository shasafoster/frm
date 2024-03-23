# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

from frm.schedule.tenor import calc_tenor_date
from frm.schedule.calendar import get_calendar
import pandas as pd
import numpy as np

# def test_tenor_to_date_offset():
    
#     # Test scalar inputs
#     assert tenor_to_date_offset('SPOT') == np.array(DateOffset(days=0))   

#     assert tenor_to_date_offset('1D') == np.array(DateOffset(days=1))   
#     assert tenor_to_date_offset('30D') == np.array(DateOffset(days=30))   
#     assert tenor_to_date_offset('360D') == np.array(DateOffset(days=360))   
    
#     assert tenor_to_date_offset('1W') == np.array(DateOffset(weeks=1))   
#     assert tenor_to_date_offset('52W') == np.array(DateOffset(weeks=52))   
#     assert tenor_to_date_offset('104W') == np.array(DateOffset(weeks=104))
    
#     assert tenor_to_date_offset('1M') == np.array(DateOffset(months=1)) 
#     assert tenor_to_date_offset('12M') == np.array(DateOffset(months=12))   
#     assert tenor_to_date_offset('120M') == np.array(DateOffset(months=120))
    
#     assert tenor_to_date_offset('1Y') == np.array(DateOffset(years=1))   
#     assert tenor_to_date_offset('10Y') == np.array(DateOffset(years=10))   
#     assert tenor_to_date_offset('100Y') == np.array(DateOffset(years=100))
    
#     assert tenor_to_date_offset('1Y3M') == np.array(DateOffset(months=(1*12+3)))   
#     assert tenor_to_date_offset('10Y6M') == np.array(DateOffset(months=(10*12+6)))   
#     assert tenor_to_date_offset('100Y1M') == np.array(DateOffset(months=(100*12+1)))
    
#     # Test array inputs
#     arr = np.array(['1Y','2Y'])
#     assert (tenor_to_date_offset(arr) == np.array([DateOffset(years=1),DateOffset(years=2)])).all()   



def test_calc_tenor_date():

    curve_date = pd.Timestamp(2023,6,30)
    tenor_name = '1 year'
    curve_ccy = 'audusd'
    holiday_calendar = get_calendar(['usd','aud'])
    tenor_date, tenor_name_cleaned, spot_date = calc_tenor_date(curve_date, tenor_name, curve_ccy, holiday_calendar=holiday_calendar)
    assert tenor_date == pd.Timestamp(2024,7,5) # np.datetime64('2024-07-05')
    assert tenor_name_cleaned == '1y'
    assert spot_date == pd.Timestamp(2023,7,5) # np.datetime64('2023-07-05')

    curve_date = pd.Timestamp(2023,6,30)
    tenor_name = ['1 year','2 year']
    curve_ccy = 'audusd'
    cal = get_calendar(['usd','aud'])
    tenor_date, tenor_name_cleaned, spot_date = calc_tenor_date(curve_date, tenor_name, curve_ccy, holiday_calendar=holiday_calendar)
    # arr = np.array([np.datetime64('2024-07-05'), np.datetime64('2025-07-07')])
    arr = np.array([pd.Timestamp(2024,7,5), pd.Timestamp(2025,7,7)])
    assert (tenor_date == arr).all()
    assert (tenor_name_cleaned == ['1y','2y']).all()
    assert spot_date == pd.Timestamp(2023,7,5)

    
if __name__ == '__main__':
    test_calc_tenor_date()



