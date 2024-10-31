# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

from frm.utils.tenor import tenor_to_date_offset # get_tenor_settlement_date
from frm.utils.business_day_calendar import get_busdaycal
import pandas as pd
import numpy as np

#%%

def test_tenor_name_to_date_offset():
    # Test scalar inputs
    assert tenor_to_date_offset('sp') == pd.DateOffset(days=0)

    assert tenor_to_date_offset('1d') == pd.DateOffset(days=1)
    assert tenor_to_date_offset('30d') == pd.DateOffset(days=30)
    assert tenor_to_date_offset('360d') == pd.DateOffset(days=360)

    assert tenor_to_date_offset('1w') == pd.DateOffset(weeks=1)
    assert tenor_to_date_offset('52w') == pd.DateOffset(weeks=52)
    assert tenor_to_date_offset('104w') == pd.DateOffset(weeks=104)

    assert tenor_to_date_offset('1m') == pd.DateOffset(months=1)
    assert tenor_to_date_offset('12m') == pd.DateOffset(months=12)
    assert tenor_to_date_offset('120m') == pd.DateOffset(months=120)

    assert tenor_to_date_offset('1y') == pd.DateOffset(years=1)
    assert tenor_to_date_offset('10y') == pd.DateOffset(years=10)
    assert tenor_to_date_offset('100y') == pd.DateOffset(years=100)

    assert tenor_to_date_offset('1y3m') == pd.DateOffset(months=(1 * 12 + 3))
    assert tenor_to_date_offset('10y6m') == pd.DateOffset(months=(10 * 12 + 6))
    assert tenor_to_date_offset('100y1m') == pd.DateOffset(months=(100 * 12 + 1))
    



# def test_get_tenor_settlement_date():
#
#     curve_date = pd.Timestamp(2023,6,30)
#     tenor_name = '1 year'
#     curve_ccy = 'audusd'
#     busdaycal = get_busdaycal(['usd','aud'])
#     tenor_date, tenor_name_cleaned, spot_date = get_tenor_settlement_date(curve_date, tenor_name, curve_ccy, busdaycal=busdaycal)
#     assert tenor_date == pd.Timestamp(2024,7,5)
#     assert tenor_name_cleaned == '1y'
#     assert spot_date == pd.Timestamp(2023,7,5)
#
#     curve_date = pd.Timestamp(2023,6,30)
#     tenor_name = ['1 year','2 year']
#     curve_ccy = 'audusd'
#     busdaycal = get_busdaycal(['usd','aud'])
#     tenor_date, tenor_name_cleaned, spot_date = get_tenor_settlement_date(curve_date, tenor_name, curve_ccy, busdaycal=busdaycal)
#     arr = np.array([pd.Timestamp(2024,7,5), pd.Timestamp(2025,7,7)])
#     assert (tenor_date == arr).all()
#     assert (tenor_name_cleaned == ['1y','2y']).all()
#     assert spot_date == pd.Timestamp(2023,7,5)

    
if __name__ == '__main__':
    #test_get_tenor_settlement_date()
    test_tenor_name_to_date_offset()



