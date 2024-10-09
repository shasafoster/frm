# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import numpy as np
import datetime as dt
from frm.utils.business_day_calendar import get_busdaycal


def test_busdaycal():
    # First Matariki 
    busdaycal = get_busdaycal(keys='NZD')
    assert (busdaycal.weekmask == np.array([ True,  True,  True,  True,  True, False, False])).all()
    assert dt.date(2022,6,24) in busdaycal.holidays 

    # NSW Bank Holiday on the first Monday of August (of every year)
    busdaycal = get_busdaycal(keys='AUD')
    assert (busdaycal.weekmask == np.array([ True,  True,  True,  True,  True, False, False])).all()
    assert dt.date(2024,8,5) in busdaycal.holidays 
    
    # Friday is a non-business day for Israel
    busdaycal = get_busdaycal(keys=['ILS','usd'])
    assert (busdaycal.weekmask == np.array([ True,  True,  True,  True,  False, False, False])).all()  
    
    

if __name__ == "__main__":
    busdaycal = get_busdaycal(keys='USD')



