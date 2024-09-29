# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 
        
#from frm.instruments.ir.swap import Swap
#from frm.instruments.ir.leg import Leg
from frm.utils.daycount import day_count, year_fraction
from frm.utils.tenor import get_tenor_settlement_date
from frm.utils.utilities import convert_column_to_consistent_data_type
from frm.utils.enums import DayCountBasis, CompoundingFrequency
from frm.term_structures.zero_curve_helpers import zero_rate_from_discount_factor, discount_factor_from_zero_rate, forward_rate

from enum import Enum
import scipy 
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Literal
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta

VALID_INTERPOLATION_METHOD = Literal['linear_on_log_of_discount_factors','cubic_spline_on_zero_rates']
VALID_EXTRAPOLATION_METHOD = Literal['none','flat']



@dataclass
class ZeroCurve: 
    # Required inputs
    curve_date: pd.Timestamp
    data: pd.DataFrame
    
    # Used in the __post_init__ but not set as attributes
    compounding_frequency: InitVar[str]=None
    busdaycal: InitVar[np.busdaycalendar]=None
    
    # Optional inputs
    day_count_basis: DayCountBasis=DayCountBasis.ACT_ACT
    interpolation_method: str='cubic_spline_on_zero_rates'
    extrapolation_method: str='none'
    
    # Attributes set in __post_init__
    cubic_spline_definition: str=field(init=False)
    
    def __post_init__(self, compounding_frequency, busdaycal):
        busdaycal = busdaycal if busdaycal is not None else np.busdaycalendar()
        
        data = self.data
        if compounding_frequency is None and 'zero_rate' in data.columns:
            raise ValueError("'compounding_frequency' must be specified zero rates specified in data")        
            
        self.__process_input_data(data, compounding_frequency, busdaycal)
        self.__df_of_daily_data_setup()

    

    def __process_input_data(self, data, compounding_frequency, busdaycal):
        
        only_one_of_columns_X = ['days', 'tenor', 'years', 'date']
        only_one_of_columns_Y = ['zero_rate','discount_factor']
        
        if len(data.columns.to_list()) != 2:
            raise ValueError('Exactly two columns must be specified: \n'
                             '(i) One of: ' + ', '.join(only_one_of_columns_X) + '\n' 
                             '(ii) One of: ' + ', '.join(only_one_of_columns_Y))            
        
        X_columns = [col for col in only_one_of_columns_X if col in data.columns]
        if len(X_columns) != 1:
            raise ValueError('Exactly one of the following columns must be specified: ' + ', '.join(only_one_of_columns_X))       
        X_column_name = X_columns[0]
        
        Y_columns  = [col for col in only_one_of_columns_Y if col in data.columns]
        if len(Y_columns) != 1:
            raise ValueError('Exactly one of the following columns must be specified: ' + ', '.join(only_one_of_columns_Y))
        Y_column_name = Y_columns[0]

        def __calculate_days(): 
            data['days'] = (data['date'] - self.curve_date).dt.days
        def __calculate_years(): 
            data['years'] = year_fraction(self.curve_date, data['date'], self.day_count_basis)
        
        match X_column_name:
            case 'tenor':
                date, tenor, spot_date = get_tenor_settlement_date(self.curve_date, data['tenor'].values, self.curve_ccy, busdaycal)
                data['tenor'] = tenor
                data['date'] = date
                __calculate_days()
                __calculate_years()
            case 'date':
                __calculate_days()
                __calculate_years()
            case 'days':
                data['date'] =  data['days'].apply(lambda days: self.curve_date + dt.timedelta(days=days))
                __calculate_years()
            case 'years':
                pass
            
        data = data.sort_values(by='years', ascending=True)       
        data = data.reset_index(drop=True)
        data = convert_column_to_consistent_data_type(data)
            
        match Y_column_name:
            case 'zero_rate':
                data['discount_factor'] = discount_factor_from_zero_rate(
                    years=data['years'],
                    zero_rate=data['zero_rate'],
                    compounding_frequency=compounding_frequency)
            case 'discount_factor':
                pass
            
        # Add nominal annual continuously compounded interest rate for internal use
        data['nacc'] = -1 * np.log(data['discount_factor']) / data['years']
        data = data.dropna(subset=['nacc']) 
        
        if 'zero_rate' in data.columns:
            data.drop(columns=['zero_rate'], inplace=True)
        
        # Add pillar point at t=0 to improve interpolation if linear interpolation is applied
        # If cubic splines interpolation is applied this method can cause unstable splines
        # at the short end that affect the hull-white 1 factor theta, hence it is not applied. 
        # If, it is desired for completeness, the nacc, was set at t=0, based on the spline excluding this value.
        # This has not been tested but may work. 
        if self.interpolation_method == 'linear_on_log_of_discount_factors':
            first_row = {'years': 0.0, 'discount_factor': 1.0}
            first_row['nacc'] = data['nacc'].iloc[0]    
            if 'tenor' in data.columns:
                first_row['tenor'] = ''
            if 'date' in data.columns:
                first_row['date'] = self.curve_date                
            if 'days' in data.columns:
                first_row['days'] = 0.0
            first_row_df = pd.DataFrame([first_row], columns=data.columns)
            data = pd.concat([first_row_df, data])
            data.reset_index(inplace=True, drop=True)            
        
        # Reorder the columns into a consistent format
        column_order = ['tenor','date','days','years','nacc','discount_factor']
        data = data[[col for col in column_order if col in data.columns]]                         
        
        self.data = data        
        
        
    def __df_of_daily_data_setup(self):    
        
        if 'date' in self.data.columns:
            max_date = max(self.data['date'])
        else:
            max_years = int(max(self.data['years']))
            max_date = self.curve_date + relativedelta(years=max_years)
            while year_fraction(self.curve_date, max_date, self.day_count_basis) < max_years:
                max_date += dt.timedelta(days=1)
                
        date_range = pd.date_range(self.curve_date,max_date,freq='d')
        days = day_count(self.curve_date, date_range, self.day_count_basis)
        years = year_fraction(self.curve_date,date_range, self.day_count_basis)
        
        if self.interpolation_method == 'cubic_spline_on_zero_rates':
            self.cubic_spline_definition = scipy.interpolate.splrep(x=self.data['years'].to_list(),
                                                                    y=self.data['nacc'].to_list(), k=3)  
            zero_rate_interpolated = scipy.interpolate.splev(years, self.cubic_spline_definition, der=0)
            self.data_daily = pd.DataFrame({'date': date_range.to_list(), 
                                                 'days': days,
                                                 'years': years, 
                                                 'nacc': zero_rate_interpolated,
                                                 'discount_factor': np.exp(-1 * zero_rate_interpolated * years)})
            
        elif self.interpolation_method == 'linear_on_log_of_discount_factors':            
            ln_df_interpolated = np.interp(x=years, xp=self.data['years'], fp= np.log(self.data['discount_factor']))            
            self.data_daily = pd.DataFrame({'date': date_range.to_list(),  
                                                 'days': days,
                                                 'years': years,
                                                 'nacc': -1 * ln_df_interpolated / years,
                                                 'discount_factor': np.exp(ln_df_interpolated)})        
       
    def flat_shift(self, 
                   basis_points: float=1) -> 'ZeroCurve':
        dates = list(self.data.keys())
        years = year_fraction(self.curve_date, dates, self.day_count_basis)
        shifted = - np.log(list(self.data.values())) / years + basis_points / 10000    
        df_shifted_data = pd.DataFrame({'date': dates, 
                                        'years': years, 
                                        'discount_factor':np.exp(-shifted * years)})
        df_shifted_data.loc[0,'discount_factor'] = 1.0 # set the discount factor on the curve date to be 1.0
        return ZeroCurve(curve_date=self.curve_date, 
                         data = df_shifted_data,
                         day_count_basis = self.day_count_basis)
                   
    def forward_rate(self,
                     period_start: Union[pd.Timestamp, pd.Series],
                     period_end: Union[pd.Timestamp, pd.Series],
                     compounding_frequency: CompoundingFrequency=CompoundingFrequency.SIMPLE) -> pd.Series:

        assert len(period_start) == len(period_end)
        assert type(period_start) == type(period_end)
        assert (period_start >= self.curve_date).all()
                    
        Δt = pd.Series(year_fraction(period_start, period_end, self.day_count_basis))            
        DF_T1 = self.discount_factor(period_start)
        DF_T2 = self.discount_factor(period_end)
        forward_fixings = forward_rate(DF_T1, DF_T2, Δt, compounding_frequency)
        
        return pd.Series(forward_fixings.to_list())
       
    def instantaneous_forward_rate(self, years):        
        if self.interpolation_method == 'cubic_spline_on_zero_rates':
            zero_rate = self.zero_rate(years=years)
            zero_rate_1st_deriv = scipy.interpolate.splev(x=years, tck=self.cubic_spline_definition, der=1) 
            return zero_rate + years * zero_rate_1st_deriv
        else:
            raise ValueError('only supported for cubic spline interpolation method(s)')
        
        
    def discount_factor(self, 
                  dates: Optional[Union[pd.Timestamp, pd.Series]]=None,
                  days: Optional[Union[int, pd.Series]]=None,
                  years: Optional[Union[float, pd.Series]]=None,) -> pd.Series:
        
            df = self.index_daily_data(dates, days, years)
            return df['discount_factor'].values
           
        
    def zero_rate(self, 
                  compounding_frequency: CompoundingFrequency,
                  dates: Union[pd.Timestamp, pd.Series] = None,
                  days: Union[int, pd.Series] = None,
                  years: Union[float, pd.Series] = None,
                  ) -> pd.Series:
                
            df = self.index_daily_data(dates, days, years)
            
            if compounding_frequency == CompoundingFrequency.CONTINUOUS:
                return df['nacc'].values
            else: 
                zero_rate = zero_rate_from_discount_factor(years=df['years'],
                                                           discount_factor=df['discount_factor'], 
                                                           compounding_frequency=compounding_frequency)
                return zero_rate        
        
        
    def index_daily_data(self, 
                  dates: Optional[Union[pd.Timestamp, pd.Series]]=None,
                  days: Optional[Union[int, pd.Series]]=None,
                  years: Optional[Union[float, pd.Series]]=None,) -> pd.DataFrame:
        
        inputs = {'dates': dates, 'days': days, 'years': years}
        if sum(x is not None for x in inputs.values()) != 1:
            raise ValueError('Only one input among days, date, or years is allowed.')            
        
        if years is not None:
            if self.interpolation_method == 'cubic_spline_on_zero_rates':
                nacc = scipy.interpolate.splev(years, self.cubic_spline_definition, der=0)
                discount_factor = np.exp(-1 * nacc * years)
                df = pd.DataFrame({'years': years,'nacc': nacc,'discount_factor': discount_factor})
                return df
                
            elif self.interpolation_method == 'linear_on_log_of_discount_factors':
                ln_df_interpolated = np.interp(x=years, xp=self.data['years'], fp= np.log(self.data['discount_factor']))
                nacc = -1 * ln_df_interpolated / years
                discount_factor = np.exp(ln_df_interpolated)
                df = pd.DataFrame({'years': years,'nacc': nacc,'discount_factor': discount_factor})
                return df
        else:
            if days is not None:
                dates = self.curve_date + dt.timedelta(days=days)
            if type(dates) is pd.Timestamp: 
                dates = pd.Series([dates])
                
            # Message suffix if any dates are outside the available data
            if self.extrapolation_method == 'none':    
                msg = f". NaN will be returned as 'extrapolation_method' is {self.extrapolation_method}"
            elif self.extrapolation_method == 'flat':
                msg = f". Flat extropolation will be applied as 'extrapolation_method' is {self.extrapolation_method}"          
            
            # Check if any of the dates are outside the available data
            min_date = self.data_daily['date'].min()
            below_range = dates < min_date
            dates_below = []
            if any(below_range):
                bound_date = self.data_daily['date'].min()
                for i,date in enumerate(dates[below_range]):
                    print('Date', date.strftime('%Y-%m-%d'), 'is below the min available data of ' + bound_date.strftime('%Y-%m-%d') + msg)
                    if self.extrapolation_method == 'none':    
                        dates_below.append(np.nan)
                    elif self.extrapolation_method == 'flat':    
                        dates_below.append(bound_date)
            
            max_date = self.data_daily['date'].max()
            above_range = dates > max_date
            dates_above = []
            if any(above_range):
                bound_date = self.data_daily['date'].max()
                for i,date in enumerate(dates[above_range]):
                    print('Date', date.strftime('%Y-%m-%d'), 'is above the max available data of ' + bound_date.strftime('%Y-%m-%d') + msg)
                    if self.extrapolation_method == 'none':    
                        dates_above.append(np.nan)
                    elif self.extrapolation_method == 'flat':    
                        dates_above.append(bound_date)
          
            in_range = np.logical_and(np.logical_not(below_range),np.logical_not(above_range))
            cleaned_dates = dates_below + list(dates[in_range].values) + dates_above
            df = pd.merge(pd.DataFrame(data=cleaned_dates, columns=['date']), self.data_daily, left_on='date',right_on='date', how='left')
            return df    

       
    def plot(self, forward_rate_terms=[90]):
            
        # Zero rates
        min_date = self.data_daily['date'].iloc[1]
        max_date = self.data_daily['date'].iloc[-1]
        date_range = pd.date_range(min_date,max_date,freq='d')
        years_zr = year_fraction(self.curve_date, date_range, self.day_count_basis)
        zero_rates = pd.Series(self.zero_rate(compounding_frequency=CompoundingFrequency.CONTINUOUS, dates=date_range)) * 100
        
        fig, ax = plt.subplots()
        
        ax.plot(years_zr, zero_rates, label='zero rate')  
  
        # Forward rates
        for term in forward_rate_terms:
            d1 = pd.date_range(min_date,max_date - pd.DateOffset(days=term),freq='d')
            d2 = pd.date_range(min_date + pd.DateOffset(days=term),max_date ,freq='d')
        
            years_fwd = year_fraction(self.curve_date, d1, self.day_count_basis)
            fwd_rates = pd.Series(self.forward_rate(d1, d2, CompoundingFrequency.Simple)) * 100       
        
            ax.plot(years_fwd, fwd_rates, label=str(term)+' day forward rate') 
        
        ax.set(xlabel='years', ylabel='interest rate (%)')
        
        ax.grid()
        ax.legend()
        plt.xlim([min(years_zr), 1.05*max(years_zr)])
        plt.show()



        
        

        
        
    