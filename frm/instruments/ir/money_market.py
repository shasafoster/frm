# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

from frm.frm.schedule.tenor import calc_tenor_date
from frm.frm.schedule.business_day_calendar import get_calendar
from frm.frm.schedule.daycounter import DayCounter

import numpy as np
import pandas as pd    


class Deposit():
        
    def __init__(self,  effective_date: pd.Timestamp, 
                        contractual_interest_rate_naac: float,
                        maturity_date: pd.Timestamp=None, 
                        tenor_name: str=np.nan,
                        day_count_basis: str='act/act',
                        local_currency_holidays=None,
                        city_holidays=None,
                        holiday_calendar=None
                        ):
    
        if holiday_calendar is None:
            holiday_calendar = get_calendar(local_currency_holidays, city_holidays)     
    
        if pd.isnull(maturity_date):
            maturity_date, tenor_name, _ = calc_tenor_date(effective_date, tenor_name, holiday_calendar=holiday_calendar, curve_ccy=local_currency_holidays)
        
        assert effective_date < maturity_date
        
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.daycounter = DayCounter(day_count_basis)
        self.contractual_interest_rate_naac = contractual_interest_rate_naac
        self.tenor_name = tenor_name
        
    def implied_discount_factor(self):
        yrs = self.daycounter.year_fraction(self.effective_date, self.maturity_date)
        return 1 / (1 + self.interest_rate*yrs)
    
    def implied_cczr(self):
        yrs = self.daycounter.year_fraction(self.effective_date, self.maturity_date)
        return -np.log(1 / (1 + self.interest_rate*yrs)) / yrs
    





