# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import numpy as np
import datetime as dt
from frm.schedule.business_day_calendar import get_busdaycalendar



def test_get_busdaycalendar():

    holiday_calendar = get_busdaycalendar(ccys='NZD')
    assert (holiday_calendar.weekmask == np.array([ True,  True,  True,  True,  True, False, False])).all()
    assert dt.date(2022,6,24) in holiday_calendar.holidays # First Matariki

    holiday_calendar = get_busdaycalendar(ccys='AUD')
    assert (holiday_calendar.weekmask == np.array([ True,  True,  True,  True,  True, False, False])).all()
    assert dt.date(2024,8,5) in holiday_calendar.holidays # NSW Bank Holiday
    
    holiday_calendar = get_busdaycalendar(ccys=['ILS','usd'])
    assert (holiday_calendar.weekmask == np.array([ True,  True,  True,  True,  False, False, False])).all()  
    
    
