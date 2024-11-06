# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import frm
from frm.utils import clean_tenor, tenor_to_date_offset, day_count, year_frac, Schedule
from frm.enums import DayCountBasis, PeriodFrequency, ExchangeNotionals, DayRoll
from copy import deepcopy

import scipy 
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Literal
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
import math

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
    'piecewise_hazard_rate': [0.016635110702829, 0.016631656971491, 0.016631465481214, 0.016631557272586,
                              0.016631535298202, 0.016631543419796, 0.016631464702924, 0.016631557505000,
                              0.016631521451674, 0.016631532048776, 0.016631534218126, 0.016631526005033],
    'flat_hazard_rate': [0.016635110702829, 0.016633297692564, 0.016632362365548, 0.016631965936734,
                         0.016631829353973, 0.016631763238987, 0.016631709452895, 0.016631671533193,
                         0.016631634435844, 0.016631608910795, 0.016631596990494, 0.016631584145977]
})


#%%

curve_date = pd.Timestamp('2023-06-30')

CDS_DAY_ROLL = DayRoll._20
day_count_basis = DayCountBasis.ACT_360


# This is a quote helper function
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





row = df.iloc[5]

sch = Schedule(start_date=row['effective_date'],
                    end_date=row['termination_date'],
                    frequency=PeriodFrequency.QUARTERLY,
                    day_roll=CDS_DAY_ROLL,
                    add_coupon_payment_dates=True,
                    add_notional_schedule=True,
                    exchange_notionals=ExchangeNotionals.NEITHER)


hazard_rate = 1.66315572725858 / 100
loss_given_default = 0.6

sch.add_period_length_to_schedule(day_count_basis=day_count_basis)
sch.df['premium_rate'] = row['quoted_spread']
sch.df['premium_payment'] = sch.df['notional'] * sch.df['premium_rate'] * sch.df['period_years']
sch.df['premium_pv'] = np.nan # Placeholder

sch.df['credit_risk_period_start'] = sch.df['period_start']
sch.df['credit_risk_period_end'] = sch.df['period_end']
sch.df.loc[0,'credit_risk_period_start'] = max(sch.df.loc[0,'credit_risk_period_start'], curve_date)
sch.df['credit_risk_period_years'] = year_frac(sch.df['credit_risk_period_start'], sch.df['credit_risk_period_end'], frm.utils.DATE2YEARFRAC_DAY_COUNT_BASIS)

for i,row in sch.df.iterrows():
    mask = row['period_end'] <= df['termination_date']
    sch.df.loc[i,'hazard_rate'] = df.loc[mask,'piecewise_hazard_rate'].values[-1]

sch.df['hazard_rate'] = hazard_rate
sch.df['period_conditional_survival_prob'] = np.exp(-hazard_rate * sch.df['period_years'])
sch.df['cumulative_survival_prob'] = sch.df['period_conditional_survival_prob'].cumprod()
sch.df['cumulative_prob_default'] = 1 - sch.df['cumulative_survival_prob']
sch.df['period_prob_default'] = sch.df['cumulative_prob_default'].diff()
sch.df.loc[sch.df.index[0],'period_prob_default'] = sch.df.loc[sch.df.index[0],'cumulative_prob_default']

sch.df['premium_pv'] = sch.df['premium_payment'] * sch.df['cumulative_survival_prob']
sch.df['protection_pv'] = sch.df['notional'] * loss_given_default * sch.df['period_prob_default']

#%%
