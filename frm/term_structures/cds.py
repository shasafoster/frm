# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import frm
from frm.utils import clean_tenor, tenor_to_date_offset, day_count, year_frac, Schedule, get_busdaycal
from frm.enums import DayCountBasis, PeriodFrequency, ExchangeNotionals, DayRoll
from frm.term_structures.zero_curve import ZeroCurve

from scipy.optimize import root_scalar
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Literal
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
import math
import time

@dataclass
class HazardCurve:
    # Required inputs
    curve_date: pd.Timestamp
    hazard_data: pd.DataFrame

    # Optional init inputs
    recovery_rate: float = 0.4
    day_count_basis: DayCountBasis = DayCountBasis.ACT_360
    busdaycal: np.busdaycalendar = np.busdaycalendar()  # used in simple-average forward rate calculation
    #interpolation_method: str = 'cubic_spline_on_zero_rates'
    #extrapolation_method: str = 'none'

    # Set in __post_init__
    loss_given_default: float = field(init=False)

    def __post_init__(self):

        assert 0 <= self.recovery_rate <= 1, 'Recovery rate must be between 0 and 1'
        self.loss_given_default = 1.0 - self.recovery_rate

df = pd.DataFrame({
    'tenor': ['3M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y'],
    'quoted_spread': [0.01] * 12,
    'piecewise_hazard_rate': [0.0166343430, 0.0166316570, 0.0166316557, 0.0166315349,
                              0.0166315426, 0.0166314646, 0.0166315574, 0.0166315574,
                              0.0166315182, 0.0166315301, 0.0166315131, 0.0166315364],
    'flat_hazard_rate': [0.0166343430, 0.0166329564, 0.0166323032, 0.0166319263,
                         0.0166318048, 0.0166317265, 0.0166316963, 0.0166316622,
                         0.0166316278, 0.0166316048, 0.0166315914, 0.0166315825]
})

frm.utils.VALUE_DATE = pd.Timestamp('2024-06-28')
fp = 'C:/Users/shasa/Documents/frm_private/tests_private/test_optionlet_support_20240628.xlsm'
curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')
day_count_basis = DayCountBasis.ACT_365

zero_curve = ZeroCurve(curve_date=curve_date,
                       data=pd.read_excel(io=fp, sheet_name='DF_3M'),
                       day_count_basis=day_count_basis,
                       busdaycal=busdaycal,
                       interpolation_method='linear_on_log_of_discount_factors')

curve_date = pd.Timestamp('2024-06-28')
CDS_DAY_ROLL = DayRoll._20
day_count_basis = DayCountBasis.ACT_360
loss_given_default = 0.6

#def get_cds_effective_period_from_tenor():
curve_date_month = curve_date.month
final_month_of_current_calendar_quarter = math.ceil(curve_date_month / 3) * 3
final_month_of_current_prior_quarter = 12 if final_month_of_current_calendar_quarter == 3 else final_month_of_current_calendar_quarter - 3
start_month = final_month_of_current_calendar_quarter if curve_date.day > CDS_DAY_ROLL.value else final_month_of_current_prior_quarter
start_year = curve_date.year - 1 if start_month > curve_date_month else curve_date.year
start_date = pd.Timestamp(dt.datetime(start_year, start_month, CDS_DAY_ROLL.value))
tenor_offset_start_date = start_date if start_date.month in [6,12] else start_date + pd.DateOffset(months=3)
df['tenor'] = df['tenor'].apply(clean_tenor)
df['effective_date'] = start_date
df['termination_date'] = tenor_offset_start_date + df['tenor'].apply(tenor_to_date_offset)


t1 = time.time()

df['hazard_rate'] = np.nan

for i,row in df.iterrows():
    if i >= 0:
        sch = Schedule(start_date=row['effective_date'],
                            end_date=row['termination_date'],
                            frequency=PeriodFrequency.QUARTERLY,
                            day_roll=CDS_DAY_ROLL,
                            add_coupon_payment_dates=True,
                            add_notional_schedule=True,
                            exchange_notionals=ExchangeNotionals.NEITHER)

        accrued_year_frac = year_frac(row['effective_date'], curve_date, day_count_basis)
        sch.df.loc[0,'period_start'] = max(sch.df.loc[0,'period_start'], curve_date)
        sch.df['discount_factor'] = zero_curve.get_discount_factors(sch.df['coupon_payment_date'])
        sch.add_period_length_to_schedule(day_count_basis=day_count_basis)
        sch.df.drop(columns=['notional_payment', 'notional_payment_date'], inplace=True)

        sch.df['premium_rate'] = row['quoted_spread']
        sch.df['premium_payment'] = sch.df['notional'] * sch.df['premium_rate'] * sch.df['period_years']
        accrued_interest = (sch.df['notional'] * accrued_year_frac * sch.df['premium_rate']).iloc[0]
        sch.df['premium_pv'] = np.nan # Placeholder

        sch.df['hazard_rate'] = np.nan
        for j,row_ in df[df['hazard_rate'].notna()].iterrows():
            if j == 0:
                mask = sch.df['period_end'] <= row_['termination_date']
            else:
                mask = (row_['effective_date'] >= sch.df['period_start']) & (row_['termination_date'] <= sch.df['period_end'])
            sch.df.loc[mask,'hazard_rate'] = row_['hazard_rate']

        mask_to_solve = sch.df['hazard_rate'].isna()

        def objective_function(terminal_hazard_rate):
            sch.df.loc[mask_to_solve,'hazard_rate'] = terminal_hazard_rate

            sch.df['period_conditional_survival_prob'] = np.exp(-sch.df['hazard_rate'] * sch.df['period_years'])
            sch.df['cumulative_survival_prob'] = sch.df['period_conditional_survival_prob'].cumprod()
            sch.df['cumulative_prob_default'] = 1 - sch.df['cumulative_survival_prob']
            sch.df['period_prob_default'] = sch.df['cumulative_prob_default'].diff()
            sch.df.loc[sch.df.index[0],'period_prob_default'] = sch.df.loc[sch.df.index[0],'cumulative_prob_default']

            sch.df['premium_pv'] = sch.df['premium_payment'] * sch.df['cumulative_survival_prob'] * sch.df['discount_factor']
            sch.df['protection_pv'] = sch.df['notional'] * loss_given_default * sch.df['period_prob_default'] * sch.df['discount_factor']

            premium_pv = sch.df['premium_pv'].sum()
            protection_pv = sch.df['protection_pv'].sum()
            return premium_pv - protection_pv

        # Define an initial guess and range for the hazard rate to search for the solution
        initial_guess = row['quoted_spread']  # Example initial guess
        lower_bound, upper_bound = 0.00001, 1.0  # Adjust bounds as needed

        # Apply root finding to solve the objective function for terminal hazard rate
        result = root_scalar(objective_function, bracket=[lower_bound, upper_bound], method='brentq')

        # Check if the solution converged and retrieve the terminal hazard rate
        if result.converged:
            terminal_hazard_rate = result.root
            sch.df.loc[mask_to_solve, 'hazard_rate'] = terminal_hazard_rate
            print(f'Terminal hazard rate solved: {terminal_hazard_rate}')
        else:
            raise ValueError('Failed to converge on a solution for the terminal hazard rate')

        df.loc[i,'hazard_rate'] = terminal_hazard_rate


df.to_clipboard()
sch.df.to_clipboard()
#
# print('Premium PV: {:,.0f}'.format(premium_pv))
# print('Protection PV: {:,.0f}'.format(protection_pv))
# print('CDS NPV: {:,.0f}'.format(premium_pv - protection_pv))

t2 = time.time()

print(t2-t1)