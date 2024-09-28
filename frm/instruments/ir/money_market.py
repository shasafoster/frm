# # -*- coding: utf-8 -*-
# import os
# if __name__ == "__main__":
#     os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

# from frm.utils.tenor import get_tenor_settlement_date
# from frm.utils.business_day_calendar import get_busdaycal
# from frm.utils.daycounter import DayCounter

# import numpy as np
# import pandas as pd    


# class Deposit():
        
#     def __init__(self,  effective_date: pd.Timestamp, 
#                         contractual_interest_rate_naac: float,
#                         maturity_date: pd.Timestamp=None, 
#                         tenor_name: str=np.nan,
#                         day_count_basis: str='act/act',
#                         local_currency_holidays=None,
#                         city_holidays=None,
#                         holiday_calendar=None
#                         ):
    
#         if holiday_calendar is None:
#             holiday_calendar = get_busdaycal(keys) 
    
#         if pd.isnull(maturity_date):
#             maturity_date, tenor_name, _ = get_tenor_settlement_date(effective_date, tenor_name, holiday_calendar=holiday_calendar, curve_ccy=local_currency_holidays)
        
#         assert effective_date < maturity_date
        
#         self.effective_date = effective_date
#         self.maturity_date = maturity_date
#         self.daycounter = DayCounter(day_count_basis)
#         self.contractual_interest_rate_naac = contractual_interest_rate_naac
#         self.tenor_name = tenor_name
        
#     def implied_discount_factor(self):
#         yrs = self.daycounter.year_fraction(self.effective_date, self.maturity_date)
#         return 1 / (1 + self.interest_rate*yrs)
    
#     def implied_cczr(self):
#         yrs = self.daycounter.year_fraction(self.effective_date, self.maturity_date)
#         return -np.log(1 / (1 + self.interest_rate*yrs)) / yrs
    





