# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

from frm.schedule.daycounter import DayCounter
import datetime as dt
import pandas as pd
import numpy as np

from unittest import TestCase
import os


#%% Datetime

x = DayCounter(None)

#%%


start_dates = pd.DatetimeIndex([dt.date(2020,12,31),
                                dt.date(2021,3,31),
                                dt.date(2021,6,30),
                                dt.date(2021,9,30)])

end_dates = pd.DatetimeIndex([dt.date(2021,3,31),
                              dt.date(2021,6,30),
                              dt.date(2021,9,30),
                              dt.date(2021,12,31)])

DayCount = DayCounter('act/360')

days = DayCount.day_count(start_dates,end_dates)

days1 = DayCount.day_count(dt.date(2020,12,31),dt.date(2021,12,31))
days2 = DayCount.day_count(np.datetime64('2020-12-31'),np.datetime64('2021-12-31'))
days3 = DayCount.day_count(pd.Timestamp('2020-12-31'),pd.Timestamp('2021-12-31'))


years = DayCount.year_fraction(start_dates,end_dates)

years1 = DayCount.year_fraction(dt.date(2020,12,31),dt.date(2021,12,31))
years2 = DayCount.year_fraction(np.datetime64('2020-12-31'),np.datetime64('2021-12-31'))
years3 = DayCount.year_fraction(pd.Timestamp('2020-12-31'),pd.Timestamp('2021-12-31'))



#%% 

class DatetimeIndexTest(TestCase):

    start_dates = pd.DatetimeIndex([dt.date(2020,12,31),
                                    dt.date(2021,3,31),
                                    dt.date(2021,6,30),
                                    dt.date(2021,9,30)])
    
    end_dates = pd.DatetimeIndex([dt.date(2021,3,31),
                                  dt.date(2021,6,30),
                                  dt.date(2021,9,30),
                                  dt.date(2021,12,31)])
    
    DayCount = DayCounter('act/360')
    DayCount.day_count(start_dates,end_dates)
    

    def test_day_count(self):
        """Test the day_count function"""

        # Case 1: test day count for same date
        self.assertEqual(
            self.DayCounter.day_count(self.start_dates, self.start_dates).to_list(),
            [0.0, 0.0, 0.0, 0.0]
        )

        # Case 2: test day count for arbitrary date difference
        self.assertEqual(
            self.DayCounter.day_count(self.start_dates, self.end_dates).to_list(),
            [90, 91, 92, 92]
        )

        # Case 3: test day count for negative date difference
        self.assertEqual(
            self.DayCounter.day_count(self.end_dates, self.start_dates).to_list(),
            -720
        )

    def test_year_fraction(self):
        """Test the year_fraction function"""

        # Case 1: test year fraction for same date
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_dates, self.start_dates).to_list(),
            [0.0, 0.0, 0.0, 0.0]
        )

        # Case 2: test year fraction for arbitrary date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_dates, self.end_dates).to_list(),
            2.0
        )

        # Case 3: test year fraction for negative date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.end_dates, self.start_dates).to_list(),
            -2.0
        )

#%%


class actual360Test(TestCase):

    DayCounter = DayCounter('act/360')
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2012, 1, 3)

    def test_day_count(self):
        """Test the day_count function"""

        # Case 1: test day count for same date
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test day count for arbitrary date difference
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.end_date),
            720
        )

        # Case 3: test day count for negative date difference
        self.assertEqual(
            self.DayCounter.day_count(self.end_date, self.start_date),
            -720
        )

    def test_year_fraction(self):
        """Test the year_fraction function"""

        # Case 1: test year fraction for same date
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test year fraction for arbitrary date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.end_date),
            2.0
        )

        # Case 3: test year fraction for negative date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.end_date, self.start_date),
            -2.0
        )


class actual365Test(TestCase):

    DayCounter = DayCounter('act/365')
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2012, 1, 13)

    def test_day_count(self):
        """Test the day_count function"""

        # Case 1: test day count for same date
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test day count for arbitrary date difference
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.end_date),
            730
        )

        # Case 3: test day count for negative date difference
        self.assertEqual(
            self.DayCounter.day_count(self.end_date, self.start_date),
            -730
        )

    def test_year_fraction(self):
        """Test the year_fraction function"""

        # Case 1: test year fraction for same date
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test year fraction for arbitrary date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.end_date),
            2.0
        )

        # Case 3: test year fraction for negative date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.end_date, self.start_date),
            -2.0
        )


class actualactualTest(TestCase):

    DayCounter = DayCounter('act/act')
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2014, 1, 13)

    def test_day_count(self):
        """Test the day_count function"""

        # Case 1: test day count for same date
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test day count for arbitrary date difference
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.end_date),
            1461
        )

        # Case 3: test day count for negative date difference
        self.assertEqual(
            self.DayCounter.day_count(self.end_date, self.start_date),
            -1461
        )

    def test_year_fraction(self):
        """Test the year_fraction function"""

        # Case 1: test year fraction for same date
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test year fraction for arbitrary date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.end_date),
            4.0
        )

        # Case 3: test year fraction for negative date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.end_date, self.start_date),
            -4.0
        )


class Thirty360Test(TestCase):

    DayCounter = DayCounter('30/360')
    start_date = dt.date(2010, 1, 13)
    end_date = dt.date(2012, 1, 13)

    def test_day_count(self):
        """Test the day_count function"""

        # Case 1: test day count for same date
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test day count for arbitrary date difference
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.end_date),
            720
        )

        # Case 3: test day count for negative date difference
        self.assertEqual(
            self.DayCounter.day_count(self.end_date, self.start_date),
            -720
        )

    def test_year_fraction(self):
        """Test the year_fraction function"""

        # Case 1: test year fraction for same date
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test year fraction for arbitrary date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.end_date),
            2.0
        )

        # Case 3: test year fraction for negative date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.end_date, self.start_date),
            -2.0
        )

class ThirtyE360Test(TestCase):

    DayCounter = DayCounter('30e/360')
    start_date = dt.date(2010, 8, 31)
    end_date = dt.date(2011, 2, 28)

    def test_day_count(self):
        """Test the day_count function"""

        # Case 1: test day count for same date
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test day count for arbitrary date difference
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.end_date),
            178.0
        )

        # Case 3: test day count for negative date difference
        self.assertEqual(
            self.DayCounter.day_count(self.end_date, self.start_date),
            -178.0
        )

    def test_year_fraction(self):
        """Test the year_fraction function"""

        # Case 1: test year fraction for same date
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test year fraction for arbitrary date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.end_date),
            178/360.
        )

        # Case 3: test year fraction for negative date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.end_date, self.start_date),
            -178/360.
        )

class ThirtyE360ISDATest(TestCase):

    DayCounter = DayCounter('30e/360_isda') 
    start_date = dt.date(2011, 8, 31)
    end_date = dt.date(2012, 2, 29)

    def test_day_count(self):
        """Test the day_count function"""

        # Case 1: test day count for same date
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test day count for arbitrary date difference
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.end_date),
            180
        )

        # Case 3: test day count for negative date difference
        self.assertEqual(
            self.DayCounter.day_count(self.end_date, self.start_date),
            -180
        )


        # Case 4: test day count for arbitrary date difference and termination date true
        self.assertEqual(
            self.DayCounter.day_count(self.start_date, self.end_date, is_end_date_on_termination = True),
            179
        )


    def test_year_fraction(self):
        """Test the year_fraction function"""

        # Case 1: test year fraction for same date
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.start_date),
            0.0
        )

        # Case 2: test year fraction for arbitrary date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.start_date, self.end_date),
            180/360.
        )

        # Case 3: test year fraction for negative date difference
        self.assertEqual(
            self.DayCounter.year_fraction(self.end_date, self.start_date),
            -180/360.
        )
