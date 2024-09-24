# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import datetime as dt
import pandas as pd
import numpy as np
import math

from frm.schedule.daycounter import DayCounter, to_datetimeindex

def isclose_custom(a, b, abs_tol=1e-10):
    return math.isclose(a, b, abs_tol=abs_tol)


def test_to_datetimeindex():

    datetimeidx = pd.DatetimeIndex(['2000-01-01'])
    
    date_obj = pd.DatetimeIndex(data=[dt.date(2000,1,1)])
    assert to_datetimeindex(date_obj) == datetimeidx
    
    date_obj = dt.date(2000,1,1)
    assert to_datetimeindex(date_obj) == datetimeidx
    
    date_obj = dt.datetime(2000,1,1)
    assert to_datetimeindex(date_obj) == datetimeidx
    
    date_obj = pd.Timestamp('2000-01-01') 
    assert to_datetimeindex(date_obj) == datetimeidx
    
    date_obj = np.datetime64('2000-01-01')
    assert to_datetimeindex(date_obj) == datetimeidx
    
    date_obj = pd.Series(pd.to_datetime('2000-01-01'))
    assert to_datetimeindex(date_obj) == datetimeidx  


def test_multiple_type():  
    daycounter_obj = DayCounter('act/360')
    
    days1 = daycounter_obj.day_count(dt.date(2020,12,31),dt.date(2021,12,31))
    days2 = daycounter_obj.day_count(np.datetime64('2020-12-31'),np.datetime64('2021-12-31'))
    days3 = daycounter_obj.day_count(pd.Timestamp('2020-12-31'),pd.Timestamp('2021-12-31'))
    assert days1 == days2 and days1 == days3   
    
    years1 = daycounter_obj.year_fraction(dt.date(2020,12,31),dt.date(2021,12,31))
    years2 = daycounter_obj.year_fraction(np.datetime64('2020-12-31'),np.datetime64('2021-12-31'))
    years3 = daycounter_obj.year_fraction(pd.Timestamp('2020-12-31'),pd.Timestamp('2021-12-31'))
    assert years1 == years2 and years1 == years3  


def test_30360_Bond_Basis():
    # Test cases taken from the excel file "30-360-2006ISDADefs" sourced from https://www.isda.org/2008/12/22/30-360-day-count-conventions/
    # Webpage saved to  WayBackMachine on 23 September 2024, https://web.archive.org/web/20240923055727/https://www.isda.org/2008/12/22/30-360-day-count-conventions/

    daycounter_obj = DayCounter('30/360')

    # Example 1: End dates do not involve last day of February
    start_dates = pd.DatetimeIndex(['08/20/06', '02/20/07', '08/20/07', '02/20/08', '08/20/08', '02/20/09'])
    end_dates = pd.DatetimeIndex(['02/20/07', '08/20/07', '02/20/08', '08/20/08', '02/20/09', '08/20/09'])
    
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [180, 180, 180, 180, 180, 180]).all()
    
    # Example 2: End dates include some end-February dates
    start_dates = pd.DatetimeIndex(['08/31/06', '02/28/07', '08/31/07', '02/29/08', '08/31/08', '02/28/09'])
    end_dates = pd.DatetimeIndex(['02/28/07', '08/31/07', '02/29/08', '08/31/08', '02/28/09', '08/31/09'])
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [178, 183, 179, 182, 178, 183]).all()
    
    # Example 3: Miscellaneous calculations
    start_dates = pd.DatetimeIndex([
        '01/31/06', '01/30/06', '02/28/06', '02/14/06', '09/30/06', '10/31/06', 
        '08/31/07', '02/28/08', '02/28/08', '02/28/08', '02/26/07', '02/26/07', 
        '02/29/08', '02/28/08', '02/28/08'
    ])
    end_dates = pd.DatetimeIndex([
        '02/28/06', '02/28/06', '03/03/06', '02/28/06', '10/31/06', '11/28/06', 
        '02/28/08', '08/28/08', '08/30/08', '08/31/08', '02/28/08', '02/29/08', 
        '02/28/09', '03/30/08', '03/31/08'
    ])
    daycounter_obj = DayCounter('30/360')
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [28, 28, 5, 14, 30, 28, 178, 180, 182, 183, 362, 363, 359, 32, 33]).all()

    # Additional tests copied from https://github.com/miradulo/isda_daycounters
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2012, 1, 13)
    assert isclose_custom(daycounter_obj.day_count(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.day_count(start_date, end_date), 720)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, end_date), 2.0)


def test_30360E_Euro_Bond_Basis():
    # Test cases taken from the excel file "30-360-2006ISDADefs" sourced from https://www.isda.org/2008/12/22/30-360-day-count-conventions/
    # Webpage saved to  WayBackMachine on 23 September 2024, https://web.archive.org/web/20240923055727/https://www.isda.org/2008/12/22/30-360-day-count-conventions/

    daycounter_obj = DayCounter('30e/360')

    # Example 1: End dates do not involve last day of February
    start_dates = pd.DatetimeIndex(['08/20/06', '02/20/07', '08/20/07', '02/20/08', '08/20/08', '02/20/09'])
    end_dates = pd.DatetimeIndex(['02/20/07', '08/20/07', '02/20/08', '08/20/08', '02/20/09', '08/20/09'])
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [180, 180, 180, 180, 180, 180]).all()
    
    # Example 2: End dates include some end-February dates
    start_dates = pd.DatetimeIndex([
        '02/28/06', '08/31/06', '02/28/07', '08/31/07', '02/29/08', '08/31/08', 
        '02/28/09', '08/31/09', '02/28/10', '08/31/10', '02/28/11', '08/31/11'
    ]) 
    end_dates = pd.DatetimeIndex([
        '08/31/06', '02/28/07', '08/31/07', '02/29/08', '08/31/08', '02/28/09', 
        '08/31/09', '02/28/10', '08/31/10', '02/28/11', '08/31/11', '02/29/12'
    ])
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [182, 178, 182, 179, 181, 178, 182, 178, 182, 178, 182, 179]).all()

    # Example 3: Miscellaneous calculations
    start_dates = pd.DatetimeIndex([
        '01/31/06', '01/30/06', '02/28/06', '02/14/06', '09/30/06', '10/31/06', 
        '08/31/07', '02/28/08', '02/28/08', '02/28/08', '02/26/07', '02/26/07', 
        '02/29/08', '02/28/08', '02/28/08'
    ])
    
    end_dates = pd.DatetimeIndex([
        '02/28/06', '02/28/06', '03/03/06', '02/28/06', '10/31/06', '11/28/06', 
        '02/28/08', '08/28/08', '08/30/08', '08/31/08', '02/28/08', '02/29/08', 
        '02/28/09', '03/30/08', '03/31/08'
    ])
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [28, 28, 5, 14, 30, 28, 178, 180, 182, 182, 362, 363, 359, 32, 32]).all()

    # Additional tests copied from https://github.com/miradulo/isda_daycounters
    start_date = dt.date(2010, 8, 31)
    end_date = dt.date(2011, 2, 28)
    assert isclose_custom(daycounter_obj.day_count(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.day_count(start_date, end_date), 178.0)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, end_date), 178/360.)


def test_30360E_ISDA():
    # Test cases taken from the excel file "30-360-2006ISDADefs" sourced from https://www.isda.org/2008/12/22/30-360-day-count-conventions/
    # Webpage saved to  WayBackMachine on 23 September 2024, https://web.archive.org/web/20240923055727/https://www.isda.org/2008/12/22/30-360-day-count-conventions/

    daycounter_obj = DayCounter('30e/360_isda')

    start_dates = pd.DatetimeIndex(['08/20/06', '02/20/07', '08/20/07', '02/20/08', '08/20/08', '02/20/09'])    
    end_dates = pd.DatetimeIndex(['02/20/07', '08/20/07', '02/20/08', '08/20/08', '02/20/09', '08/20/09'])
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [180, 180, 180, 180, 180, 180]).all()

    start_dates = pd.DatetimeIndex([
        '02/28/06', '08/31/06', '02/28/07', '08/31/07', '02/29/08', '08/31/08', 
        '02/28/09', '08/31/09', '02/28/10', '08/31/10', '02/28/11', '08/31/11'
    ])
    end_dates = pd.DatetimeIndex([
        '08/31/06', '02/28/07', '08/31/07', '02/29/08', '08/31/08', '02/28/09', 
        '08/31/09', '02/28/10', '08/31/10', '02/28/11', '08/31/11', '02/29/12'
    ])
    days = daycounter_obj.day_count(start_dates, end_dates)
    assert (days == [180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 179]).all()


    start_dates = pd.DatetimeIndex([
        '01/31/06', '01/30/06', '02/28/06', '02/14/06', '09/30/06', '10/31/06',
        '08/31/07', '02/28/08', '02/28/08', '02/28/08', '02/28/07', '02/28/07',
        '02/29/08', '02/29/08', '02/29/08'
    ])
    
    end_dates = pd.DatetimeIndex([
        '02/28/06', '02/28/06', '03/03/06', '02/28/06', '10/31/06', '11/28/06',
        '02/28/08', '08/28/08', '08/30/08', '08/31/08', '02/28/08', '02/29/08',
        '02/28/09', '03/30/08', '03/31/08'
    ])
    days = daycounter_obj.day_count(start_dates, end_dates)
    correct_result = [30, 30, 3, 16, 30, 28, 178, 180, 182, 182, 358, 360, 360, 30, 30]
    assert (days == correct_result).all()

    # Additional tests copied from https://github.com/miradulo/isda_daycounters
    start_date = dt.date(2011, 8, 31)
    end_date = dt.date(2012, 2, 29)
    assert isclose_custom(daycounter_obj.day_count(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.day_count(start_date, end_date, False), 180)
    assert isclose_custom(daycounter_obj.day_count(start_date, end_date), 179)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, end_date), 179/360.)


def test_act360():
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2012, 1, 3)
    daycounter_obj = DayCounter('act/360')

    # Copied from https://github.com/miradulo/isda_daycounters
    assert isclose_custom(daycounter_obj.day_count(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.day_count(start_date, end_date), 720)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, end_date), 720 / 360)

def test_act365():
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2012, 1, 13)
    daycounter_obj = DayCounter('act/365')

    # Copied from https://github.com/miradulo/isda_daycounters
    assert isclose_custom(daycounter_obj.day_count(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.day_count(start_date, end_date), 730)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, end_date), 730 / 365 )


def test_actact():
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2014, 1, 13)
    daycounter_obj = DayCounter('act/act')

    assert isclose_custom(daycounter_obj.day_count(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.day_count(start_date, end_date), 1461)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, start_date), 0.0)
    assert isclose_custom(daycounter_obj.year_fraction(start_date, end_date), 1461 / ((365*3 + 366) / 4))


if __name__ == "__main__":
    test_to_datetimeindex()
    test_multiple_type()
    test_30360_Bond_Basis()
    test_30360E_Euro_Bond_Basis()
    test_30360E_ISDA()
    test_act360()
    test_act365()
    test_actact()
    
    
    




