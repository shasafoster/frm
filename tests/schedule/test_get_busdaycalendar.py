# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import numpy as np
import datetime as dt
from frm.schedule.business_day_calendar import busdaycal


def test_get_busdaycal():
    # First Matariki 
    holiday_calendar = busdaycal(ccys='NZD')
    assert (holiday_calendar.weekmask == np.array([ True,  True,  True,  True,  True, False, False])).all()
    assert dt.date(2022,6,24) in holiday_calendar.holidays 

    # NSW Bank Holiday on the first Monday of August (of every year)
    holiday_calendar = busdaycal(ccys='AUD')
    assert (holiday_calendar.weekmask == np.array([ True,  True,  True,  True,  True, False, False])).all()
    assert dt.date(2024,8,5) in holiday_calendar.holidays 
    
    # Friday is a non-business day for Israel
    holiday_calendar = busdaycal(ccys=['ILS','usd'])
    assert (holiday_calendar.weekmask == np.array([ True,  True,  True,  True,  False, False, False])).all()  
    
    
