# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 
        
from frm.utils.daycount import day_count, year_frac
from frm.utils.tenor import clean_tenor, tenor_to_date_offset
from frm.utils.utilities import convert_column_to_consistent_data_type
from frm.enums import DayCountBasis, CompoundingFreq, RFRFixingCalcMethod, TermRate
from frm.term_structures.zero_curve_helpers import zero_rate_from_discount_factor, discount_factor_from_zero_rate

from scipy.interpolate import splrep, splev
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Literal
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta

VALID_INTERPOLATION_METHOD = Literal['linear_on_ln_discount','cubic_spline_on_ln_discount','cubic_spline_on_cczr']
VALID_EXTRAPOLATION_METHOD = Literal['none','flat']


@dataclass
class ZeroCurve: 
    # Required inputs
    curve_date: pd.Timestamp
    pillar_df: pd.DataFrame
    
    # Used in the __post_init__ but not set as attributes
    compounding_freq: InitVar[CompoundingFreq]=None # defines the zero_rate compounding frequency in 'data'
    
    # Optional init inputs
    day_count_basis: DayCountBasis=DayCountBasis.ACT_ACT
    cal: np.busdaycalendar=np.busdaycalendar() # used in simple-average forward rate calculation
    interp_method: str='linear_on_ln_discount'
    extrap_method: str='none'


    # Attributes set in __post_init__
    cubic_spline_definition: tuple=field(init=False)
    
    def __post_init__(self, compounding_freq):
        
        pillar_df = self.pillar_df
        if compounding_freq is None and 'zero_rate' in pillar_df.columns:
            raise ValueError("'compounding_freq' must be specified zero rates specified in 'pillar_df'")
            
        self._process_pillar_df(pillar_df, compounding_freq)
        self._setup_daily_df()
    
    def _process_pillar_df(self, pillar_df, compounding_freq):

        only_one_of_columns_X = ['tenor', 'date', 'days', 'years']
        only_one_of_columns_Y = ['zero_rate','discount_factor']
        
        if len(pillar_df.columns.to_list()) != 2:
            raise ValueError('Exactly two columns must be specified: \n'
                             '(i) One of: ' + ', '.join(only_one_of_columns_X) + '\n' 
                             '(ii) One of: ' + ', '.join(only_one_of_columns_Y))            
        
        X_columns = [col for col in only_one_of_columns_X if col in pillar_df.columns]
        if len(X_columns) != 1:
            raise ValueError('Exactly one of the following columns must be specified: ' + ', '.join(only_one_of_columns_X))       
        X_column_name = X_columns[0]
        
        Y_columns  = [col for col in only_one_of_columns_Y if col in pillar_df.columns]
        if len(Y_columns) != 1:
            raise ValueError('Exactly one of the following columns must be specified: ' + ', '.join(only_one_of_columns_Y))
        Y_column_name = Y_columns[0]

        def __calculate_days(): 
            pillar_df['days'] = (pillar_df['date'] - self.curve_date).dt.days
        def __calculate_years(): 
            pillar_df['years'] = year_frac(self.curve_date, pillar_df['date'], self.day_count_basis)
        
        match X_column_name:
            case 'tenor':
                pillar_df.loc[:,'tenor'] = pillar_df['tenor'].apply(clean_tenor)
                date_offset = pillar_df['tenor'].apply(tenor_to_date_offset)
                dates = self.curve_date + date_offset
                pillar_df['date'] = np.busday_offset(dates.values.astype('datetime64[D]'), offsets=0, roll='following', busdaycal=self.cal)
                __calculate_days()
                __calculate_years()
            case 'date':
                __calculate_days()
                __calculate_years()
            case 'days':
                pillar_df['date'] =  pillar_df['days'].apply(lambda days: self.curve_date + dt.timedelta(days=days))
                __calculate_years()
            case 'years':
                pass
            
        pillar_df = pillar_df.sort_values(by='years', ascending=True).reset_index(drop=True)
        pillar_df = convert_column_to_consistent_data_type(pillar_df)
            
        if Y_column_name == 'zero_rate':
            pillar_df['discount_factor'] = discount_factor_from_zero_rate(
                years=pillar_df['years'],
                zero_rate=pillar_df['zero_rate'],
                compounding_freq=compounding_freq)
            pillar_df.drop(columns=['zero_rate'], inplace=True)

        # Add continuously compounded interest rate for internal use
        pillar_df['cczr'] = -1 * np.log(pillar_df['discount_factor']) / pillar_df['years']

        # Add pillar points at t=0.
        # This is a dependency to the HW1F class which requires a short rate at t=0
        first_row = {'years': 0.0, 'discount_factor': 1.0}
        if 'tenor' in pillar_df.columns:
            first_row['tenor'] = ''
        if 'date' in pillar_df.columns:
            first_row['date'] = self.curve_date
        if 'days' in pillar_df.columns:
            first_row['days'] = 0.0

        match self.interp_method:
            case 'linear_on_ln_discount':
                first_row['cczr'] = pillar_df['cczr'].iloc[0]
                self.cubic_spline_definition = None
            case 'cubic_spline_on_ln_discount':
                x = [0.0] + pillar_df['years'].to_list()
                y = [0.0] + np.log(pillar_df['discount_factor']).to_list()
                self.cubic_spline_definition = splrep(x=x, y=y, k=3)
                dt = 1e-6
                first_row['cczr'] = -1 * splev(dt, self.cubic_spline_definition, der=0) / dt
            case 'cubic_spline_on_cczr':
                x = pillar_df['years'].to_list()
                y = pillar_df['cczr'].to_list()
                self.cubic_spline_definition = splrep(x=x, y=y, k=3)
                first_row['cczr'] = splev(0, self.cubic_spline_definition, der=0)
            case _:
                raise ValueError(f"Invalid interpolation method {self.interp_method}")

        first_row_df = pd.DataFrame([first_row], columns=pillar_df.columns)
        pillar_df = pd.concat([first_row_df, pillar_df])
        pillar_df.reset_index(inplace=True, drop=True)

        # Reorder the columns into a consistent format
        column_order = ['tenor','date','days','years','cczr','discount_factor']
        pillar_df = pillar_df[[col for col in column_order if col in pillar_df.columns]]
        
        self.pillar_df = pillar_df
        
        
    def _setup_daily_df(self):
        
        if 'date' in self.pillar_df.columns:
            max_date = max(self.pillar_df['date'])
        else:
            max_years = int(max(self.pillar_df['years']))
            max_date = self.curve_date + relativedelta(years=max_years)
            while year_frac(self.curve_date, max_date, self.day_count_basis) < max_years:
                max_date += dt.timedelta(days=1)
                
        date_range = pd.date_range(self.curve_date,max_date,freq='d')
        days = day_count(self.curve_date, date_range, self.day_count_basis)
        years = year_frac(self.curve_date,date_range, self.day_count_basis)

        if self.interp_method == 'linear_on_ln_discount':
            ln_df_interp = np.interp(x=years, xp=self.pillar_df['years'], fp=np.log(self.pillar_df['discount_factor']))
            cczr = -1 * ln_df_interp / years
            self.daily_df = pd.DataFrame({'date': date_range.to_list(),
                                          'days': days,
                                          'years': years,
                                          'cczr': cczr,
                                          'discount_factor': np.exp(ln_df_interp)})

        elif self.interp_method == 'cubic_spline_on_ln_discount':
            ln_df_interp = splev(x=years, tck=self.cubic_spline_definition, der=0)
            cczr = -1 * ln_df_interp / years
            self.daily_df = pd.DataFrame({'date': date_range.to_list(),
                                          'days': days,
                                          'years': years,
                                          'cczr': cczr,
                                          'discount_factor': np.exp(ln_df_interp)})

        elif self.interp_method == 'cubic_spline_on_cczr':
                zero_rate_interp = splev(years, self.cubic_spline_definition, der=0)
                self.daily_df = pd.DataFrame({'date': date_range.to_list(),
                                              'days': days,
                                              'years': years,
                                              'cczr': zero_rate_interp,
                                              'discount_factor': np.exp(-1 * zero_rate_interp * years)})
        else:
            raise ValueError(f"Invalid interpolation method {self.interp_method}")

       
    def flat_shift(self, 
                   basis_points: float=1) -> 'ZeroCurve':
        dates = list(self.pillar_df.keys())
        years = year_frac(self.curve_date, dates, self.day_count_basis)
        shifted = - np.log(list(self.pillar_df.values)) / years + basis_points / 10000
        shifted = pd.DataFrame({'date': dates,
                                'years': years,
                                'discount_factor':np.exp(-shifted * years)})
        shifted.loc[0,'discount_factor'] = 1.0 # set the discount factor on the curve date to be 1.0
        return ZeroCurve(curve_date=self.curve_date, 
                         pillar_df=shifted,
                         day_count_basis=self.day_count_basis)
                   
    def get_forward_rates(self,
                          period_start: Union[pd.Timestamp, np.datetime64, dt.datetime, dt.date, pd.Series, pd.DatetimeIndex],
                          period_end: Union[pd.Timestamp, np.datetime64, dt.datetime, dt.date, pd.Series, pd.DatetimeIndex],
                          forward_rate_type: [TermRate, RFRFixingCalcMethod]) -> np.array:

        period_start = pd.to_datetime(period_start)
        period_end = pd.to_datetime(period_end)

        if isinstance(period_start, (pd.Series, pd.DatetimeIndex)):
            period_start = pd.DatetimeIndex(period_start)
        else:
            period_start = pd.DatetimeIndex([period_start])

        if isinstance(period_end, (pd.Series, pd.DatetimeIndex)):
            period_end = pd.DatetimeIndex(period_end)
        else:
            period_end = pd.DatetimeIndex([period_end])


        assert len(period_start) == len(period_end)
        assert (period_start >= self.curve_date).all()
        assert (period_end >= period_start).all()

        if forward_rate_type in {RFRFixingCalcMethod.WEIGHTED_AVERAGE, RFRFixingCalcMethod.SIMPLE_AVERAGE}:
            
            dates = pd.date_range(start=period_start.min(), end=period_end.max(), freq='D')
            discount_factors = self.get_discount_factors(dates=dates)
            daily_interest_multiplier = discount_factors[:-1] / discount_factors[1:]
            daily_simple_interest_rate = (daily_interest_multiplier - 1) * self.day_count_basis.days_per_year
            dates_np = dates.to_numpy(dtype='datetime64[D]')
            busday_flag = pd.Series(np.is_busday(dates_np, busdaycal=self.cal), index=dates)
            
            helper_data = {
                    'date': dates[:-1],
                    'daily_interest_multiplier': daily_interest_multiplier,
                    'simple_daily_rate': daily_simple_interest_rate,
                    'business_day_flag': busday_flag.values[:-1]
                }
            helper_df = pd.DataFrame(helper_data)
            
            result = np.full(period_start.shape, np.nan)

            for i,(start_date,end_date) in enumerate(zip(period_start, period_end)):
                mask = np.logical_and(helper_df['date'] >= start_date,
                                      helper_df['date'] < end_date)
                match forward_rate_type:
                    case RFRFixingCalcMethod.WEIGHTED_AVERAGE:
                        result[i] = helper_df.loc[mask,'simple_daily_rate'].mean()
                    case RFRFixingCalcMethod.SIMPLE_AVERAGE:
                        mask = np.logical_and(mask, helper_df['business_day_flag'])
                        result[i] = helper_df.loc[mask,'simple_daily_rate'].mean()
                        
                    # Returns same result as ForwardRate.SIMPLE formulae
                    # case ForwardRate.DAILY_COMPOUNDED:
                    #     year_frac_ = year_frac(start_date, end_date, self.day_count_basis)
                    #     result[i] = (helper_df.loc[mask,'daily_interest_multiplier'].product() - 1) / year_frac_
                        
            return result
        
        else:
            Δt = pd.Series(year_frac(period_start, period_end, self.day_count_basis))
            DF_t1 = self.get_discount_factors(dates=period_start)
            DF_t2 = self.get_discount_factors(dates=period_end)
        
            # https://en.wikipedia.org/wiki/Forward_rate
            if forward_rate_type in {TermRate.SIMPLE, RFRFixingCalcMethod.DAILY_COMPOUNDED}:
                result = (1.0 / Δt) * (DF_t1 / DF_t2 - 1.0)
            elif forward_rate_type == TermRate.CONTINUOUS:
                result = (1.0 / Δt) * (np.log(DF_t1) - np.log(DF_t2))
            elif forward_rate_type == TermRate.ANNUAL:
                result = (DF_t1 / DF_t2) ** (1.0 / Δt)  - 1.0
            else:
                raise ValueError(f"Invalid forward_rate_type {forward_rate_type}")        

            return result.values
        
    def get_instantaneous_forward_rate(self, years):
        if self.interp_method == 'cubic_spline_on_cczr':
            zero_rate = self.get_zero_rates(years=years, compounding_freq=CompoundingFreq.CONTINUOUS)
            zero_rate_1st_deriv = splev(x=years, tck=self.cubic_spline_definition, der=1)
            return zero_rate + years * zero_rate_1st_deriv
        elif self.interp_method == 'cubic_spline_on_ln_discount':
            ln_discount_1st_deriv = splev(x=years, tck=self.cubic_spline_definition, der=1)
            return -ln_discount_1st_deriv


    def get_discount_factors(self,
                            dates: Optional[Union[pd.Timestamp, np.datetime64, dt.datetime, dt.date, pd.Series, pd.DatetimeIndex]]=None,
                            days: Optional[Union[int, pd.Series]]=None,
                            years: Optional[Union[float, pd.Series]]=None) -> pd.Series:

        df = self.index_daily_data(dates, days, years)
        return df['discount_factor'].values
           
        
    def get_zero_rates(self,
                      compounding_freq: CompoundingFreq,
                      dates: Optional[Union[pd.Timestamp, np.datetime64, dt.datetime, dt.date, pd.Series, pd.DatetimeIndex]]=None,
                      days:  Optional[Union[int, pd.Series]]=None,
                      years: Optional[Union[float, pd.Series]]=None
                      ) -> pd.Series:
                
        df = self.index_daily_data(dates, days, years)

        if compounding_freq == CompoundingFreq.CONTINUOUS:
            return df['cczr'].values
        else:
            zero_rate = zero_rate_from_discount_factor(years=df['years'].values,
                                                       discount_factor=df['discount_factor'].values,
                                                       compounding_freq=compounding_freq)
            return zero_rate
        
        
    def index_daily_data(self, 
                  dates: Optional[Union[pd.Timestamp, np.datetime64, dt.datetime, dt.date, pd.Series, pd.DatetimeIndex]]=None,
                  days: Optional[Union[int, pd.Series]]=None,
                  years: Optional[Union[float, pd.Series]]=None) -> pd.DataFrame:
        
        inputs = {'dates': dates, 'days': days, 'years': years}
        if sum(x is not None for x in inputs.values()) != 1:
            raise ValueError('Only one input among days, date, or years is allowed.')            
        
        if years is not None:
            if self.interp_method == 'linear_on_ln_discount':
                ln_df_interp = np.interp(x=years, xp=self.pillar_df['years'], fp= np.log(self.pillar_df['discount_factor']))
                cczr = -1 * ln_df_interp / years
                discount_factor = np.exp(ln_df_interp)
            elif self.interp_method == 'cubic_spline_on_ln_discount':
                ln_df_interp = splev(years, self.cubic_spline_definition, der=0)
                cczr = -1 * ln_df_interp / years
                discount_factor = np.exp(ln_df_interp)
            elif self.interp_method == 'cubic_spline_on_cczr':
                cczr = splev(years, self.cubic_spline_definition, der=0)
                discount_factor = np.exp(-1 * cczr * years)
            else:
                raise ValueError

            years, cczr, discount_factor = map(np.atleast_1d, (years, cczr, discount_factor))
            return pd.DataFrame({'years': years, 'cczr': cczr, 'discount_factor': discount_factor})
        else:
            if days is not None:
                dates = self.curve_date + dt.timedelta(days=days)

            dates = pd.to_datetime(dates)
            if not(isinstance(dates, pd.DatetimeIndex)):
                if isinstance(dates, pd.Series):
                    dates = pd.DatetimeIndex(dates)
                else:
                    dates = pd.DatetimeIndex([dates])

            # Message suffix if any dates are outside the available data
            if self.extrap_method == 'none':
                msg = f". NaN will be returned as 'extrapolation_method' is {self.extrap_method}"
            elif self.extrap_method == 'flat':
                msg = f". Flat extrapolation will be applied as 'extrap_method' is {self.extrap_method}"
            
            # Check if any of the dates are outside the available data
            min_date = self.daily_df['date'].min()
            below_range = dates < min_date
            dates_below = []
            if any(below_range):
                bound_date = self.daily_df['date'].min()
                for i,date in enumerate(dates[below_range]):
                    print('Date', date.strftime('%Y-%m-%d'), 'is below the min available data of ' + bound_date.strftime('%Y-%m-%d') + msg)
                    if self.extrap_method == 'none':
                        dates_below.append(np.nan)
                    elif self.extrap_method == 'flat':
                        dates_below.append(bound_date)
            
            max_date = self.daily_df['date'].max()
            above_range = dates > max_date
            dates_above = []
            if any(above_range):
                bound_date = self.daily_df['date'].max()
                for i,date in enumerate(dates[above_range]):
                    print('Date', date.strftime('%Y-%m-%d'), 'is above the max available data of ' + bound_date.strftime('%Y-%m-%d') + msg)
                    if self.extrap_method == 'none':
                        dates_above.append(np.nan)
                    elif self.extrap_method == 'flat':
                        dates_above.append(bound_date)
          
            in_range = np.logical_and(np.logical_not(below_range),np.logical_not(above_range))
            cleaned_dates = dates_below + list(dates[in_range].values) + dates_above
            df = pd.merge(pd.DataFrame(data=cleaned_dates, columns=['date']), self.daily_df, left_on='date',right_on='date', how='left')
            return df    

       
    def plot(self, forward_rate_term=pd.DateOffset(months=3)):
            
        # Zero rates
        min_date = self.daily_df['date'].iloc[1]
        max_date = self.daily_df['date'].iloc[-1]
        date_range = pd.date_range(min_date,max_date,freq='d')
        years_zr = year_frac(self.curve_date, date_range, self.day_count_basis)
        zero_rates = pd.Series(self.get_zero_rates(compounding_freq=CompoundingFreq.CONTINUOUS, dates=date_range)) * 100
        
        fig, ax = plt.subplots()

        # Zero Rate
        ax.plot(years_zr, zero_rates, label='zero rate')  
  
        # Forward rates
        d1 = pd.date_range(min_date,max_date - forward_rate_term,freq='d')
        d2 = pd.date_range(min_date + forward_rate_term,max_date ,freq='d')
        years_fwd = year_frac(self.curve_date, d1, self.day_count_basis)
        fwd_rates = pd.Series(self.get_forward_rates(d1, d2, TermRate.SIMPLE)) * 100
        ax.plot(years_fwd, fwd_rates)
        
        ax.set(xlabel='years', ylabel='interest rate (%)')
        
        ax.grid()
        ax.legend()
        plt.xlim([min(years_zr), 1.05*max(years_zr)])
        plt.show()



        
        

        
        
    