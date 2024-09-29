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
    """
    Generic class to support 
        (i) Term and OIS swap curves
        (ii) Bond curves
    """

    curve_date: pd.Timestamp
    
    # Curve settings
    day_count_basis: DayCountBasis=DayCountBasis.ACT_ACT
    interpolation_method: VALID_INTERPOLATION_METHOD = 'cubic_spline_on_zero_rates'
    extrapolation_method: VALID_EXTRAPOLATION_METHOD = 'none'
    
    # Optional initialisation parameters
    historical_fixings: InitVar[Optional[pd.DataFrame]]=None
    zero_data: InitVar[Optional[pd.DataFrame]] = None
    instruments: InitVar[Optional[dict]] = None
    dsc_crv: InitVar[Optional['ZeroCurve']] = None
    fwd_crv: InitVar[Optional['ZeroCurve']] = None
    cubic_spline_definition: InitVar[Optional] = None
    compounding_frequency: InitVar[Optional[CompoundingFrequency]] = None    
    
    # Non initialisation arguments
    daily_data: pd.DataFrame = field(init=False, repr=False)
    
    def __post_init__(self, historical_fixings, zero_data, instruments, dsc_crv, fwd_crv, cubic_spline_definition, compounding_frequency, busdaycal=None):
                           
        busdaycal = busdaycal if busdaycal is not None else np.busdaycalendar()
        
        if compounding_frequency is None and 'zero_rate' in zero_data.columns:
            raise ValueError('compounding_frequency must be specified')        
        
        if historical_fixings is not None:
            assert 'date' in historical_fixings.columns and 'fixing' in historical_fixings.columns 
            self.historical_fixings = dict(zip(historical_fixings.date, historical_fixings.fixing)) 
            
        if zero_data is not None:
            # Zero curve is defined by specified zero rates/discount factors
            self.__process_input_zero_data(zero_data, compounding_frequency, busdaycal)
            self.__df_of_daily_zero_data_setup()
        else:
            # Bootstrap zero curve from par instruments
            self.__bootstrap(instruments=instruments, dsc_crv=dsc_crv, fwd_crv=fwd_crv)
    

    def __process_input_zero_data(self, zero_data, compounding_frequency, busdaycal):
        
        only_one_of_columns_X = ['days', 'tenor', 'years', 'date']
        only_one_of_columns_Y = ['zero_rate','discount_factor']
        
        if len(zero_data.columns.to_list()) != 2:
            raise ValueError('Exactly two columns must be specified: \n'
                             '(i) One of: ' + ', '.join(only_one_of_columns_X) + '\n' 
                             '(ii) One of: ' + ', '.join(only_one_of_columns_Y))            
        
        X_columns = [col for col in only_one_of_columns_X if col in zero_data.columns]
        if len(X_columns) != 1:
            raise ValueError('Exactly one of the following columns must be specified: ' + ', '.join(only_one_of_columns_X))       
        X_column_name = X_columns[0]
        
        Y_columns  = [col for col in only_one_of_columns_Y if col in zero_data.columns]
        if len(Y_columns) != 1:
            raise ValueError('Exactly one of the following columns must be specified: ' + ', '.join(only_one_of_columns_Y))
        Y_column_name = Y_columns[0]

        def __calculate_days(): 
            zero_data['days'] = (zero_data['date'] - self.curve_date).dt.days
        def __calculate_years(): 
            zero_data['years'] = year_fraction(self.curve_date, zero_data['date'], self.day_count_basis)
        
        match X_column_name:
            case 'tenor':
                date, tenor, spot_date = get_tenor_settlement_date(self.curve_date, zero_data['tenor'].values, self.curve_ccy, busdaycal)
                zero_data['tenor'] = tenor
                zero_data['date'] = date
                __calculate_days()
                __calculate_years()
            case 'date':
                __calculate_days()
                __calculate_years()
            case 'days':
                zero_data['date'] =  zero_data['days'].apply(lambda days: self.curve_date + dt.timedelta(days=days))
                __calculate_years()
            case 'years':
                pass
            
        zero_data = zero_data.sort_values(by='years', ascending=True)       
        zero_data = zero_data.reset_index(drop=True)
        zero_data = convert_column_to_consistent_data_type(zero_data)
            
        match Y_column_name:
            case 'zero_rate':
                zero_data['discount_factor'] = discount_factor_from_zero_rate(
                    years=zero_data['years'],
                    zero_rate=zero_data['zero_rate'],
                    compounding_frequency=compounding_frequency)
            case 'discount_factor':
                pass
            
        # Add nominal annual continuously compounded interest rate for internal use
        zero_data['nacc'] = -1 * np.log(zero_data['discount_factor']) / zero_data['years']
        zero_data = zero_data.dropna(subset=['nacc']) 
        
        if 'zero_rate' in zero_data.columns:
            zero_data.drop(columns=['zero_rate'], inplace=True)
        
        # Add pillar point at t=0 to improve interpolation if linear interpolation is applied
        # If cubic splines interpolation is applied this method can cause unstable splines
        # at the short end that affect the hull-white 1 factor theta, hence it is not applied. 
        # If, it is desired for completeness, the nacc, was set at t=0, based on the spline excluding this value.
        # This has not been tested but may work. 
        if self.interpolation_method == 'linear_on_log_of_discount_factors':
            first_row = {'years': 0.0, 'discount_factor': 1.0}
            first_row['nacc'] = zero_data['nacc'].iloc[0]    
            if 'tenor' in zero_data.columns:
                first_row['tenor'] = ''
            if 'date' in zero_data.columns:
                first_row['date'] = self.curve_date                
            if 'days' in zero_data.columns:
                first_row['days'] = 0.0
            first_row_df = pd.DataFrame([first_row], columns=zero_data.columns)
            zero_data = pd.concat([first_row_df, zero_data])
            zero_data.reset_index(inplace=True, drop=True)            
        
        # Reorder the columns into a consistent format
        column_order = ['tenor','date','days','years','nacc','discount_factor']
        zero_data = zero_data[[col for col in column_order if col in zero_data.columns]]                         
        
        self.zero_data = zero_data        
        
        
    def __df_of_daily_zero_data_setup(self):    
        
        if 'date' in self.zero_data.columns:
            max_date = max(self.zero_data['date'])
        else:
            max_years = int(max(self.zero_data['years']))
            max_date = self.curve_date + relativedelta(years=max_years)
            while year_fraction(self.curve_date, max_date, self.day_count_basis) < max_years:
                max_date += dt.timedelta(days=1)
                
        date_range = pd.date_range(self.curve_date,max_date,freq='d')
        days = day_count(self.curve_date, date_range, self.day_count_basis)
        years = year_fraction(self.curve_date,date_range, self.day_count_basis)
        
        if self.interpolation_method == 'cubic_spline_on_zero_rates':
            self.cubic_spline_definition = scipy.interpolate.splrep(x=self.zero_data['years'].to_list(),
                                                                    y=self.zero_data['nacc'].to_list(), k=3)  
            zero_rate_interpolated = scipy.interpolate.splev(years, self.cubic_spline_definition, der=0)
            self.zero_data_daily = pd.DataFrame({'date': date_range.to_list(), 
                                                 'days': days,
                                                 'years': years, 
                                                 'nacc': zero_rate_interpolated,
                                                 'discount_factor': np.exp(-1 * zero_rate_interpolated * years)})
            
        elif self.interpolation_method == 'linear_on_log_of_discount_factors':            
            ln_df_interpolated = np.interp(x=years, xp=self.zero_data['years'], fp= np.log(self.zero_data['discount_factor']))            
            self.zero_data_daily = pd.DataFrame({'date': date_range.to_list(),  
                                                 'days': days,
                                                 'years': years,
                                                 'nacc': -1 * ln_df_interpolated / years,
                                                 'discount_factor': np.exp(ln_df_interpolated)})        
       
    def __bootstrap(self, 
                  instruments: dict,
                  dsc_crv: Optional['ZeroCurve']=None,
                  fwd_crv: Optional['ZeroCurve']=None):
        
        pass
        
        # When dsc_crv is specified, we apply the dsc_crv to all relevent instruments and solve the fwd_crv that gives them a par value 
        # Example: 
        # Solving for the USD LIBOR 3M forward curve from USD LIBOR 3M fix-flt par swaps quoted from LCH (i.e that are fully collatarised). 
        # As these instruments are collatarised the discount curve does not equal the forward curve. 
        # We must solve this curve prior (i.e a USD SOFR curve or a cheapest-to-deliver OIS curve) and input it as the dsc_crv.
        
        # When fwd_crv is specified, we apply the fwd_crv to all relevent instruments and solve the dsc_crv that gives them a par value
        # Example:
        # Solving for the discount curve from USD floating rate bonds that reference USD LIBOR 1M
        # We must solve the USD LIBOR 1M curve prior and input this as the fwd_crv. 
        # As these instruments are collatarised the discount curve does not equal the forward curve. 
        # We must solve this discount curve prior (i.e a USD SOFR curve or a cheapest-to-deliver OIS curve)
        
        # This functionality will cover most most applications.  
        # This fuctionality does NOT cover the case you want to boostrap with 
        # a set of instruments reference this curve as the forward curve
        # with another set of instruments that reference this curve as the discount curve. 
        
        # self.zero_data = {self.curve_date: 1.0}
        # self.zero_data_daily = pd.DataFrame({'date': [self.curve_date],
        #                              'years_to_date': [0.0], 
        #                              'discount_factor': [1.0]})

        # if 'deposits' in instruments.keys():
        #     deposits = instruments['deposits']
        #     deposits.sort(key=lambda x: x.maturity_date, reverse=False) 
        #     for deposit in instruments['deposits']:
        #         assert deposit.effective_date == self.curve_date, f"deposit.effective_date: {deposit.effective_date}, self.curve_date: {self.curve_date}"
        #         self.zero_data[deposit.maturity_date] = deposit.implied_discount_factor()
        #     self.__interpolate_zero_data()

        # if 'futures' in instruments.keys():
        #     futures = instruments['futures']
        #     futures.sort(key=lambda x: x.maturity_date, reverse=False) 
        #     # Bootstrapping using futures/FRAs requires the futures to have overlap over the end and start dates of two successive futures
        #     # For example, a future ends on 15 June 2022, the next future must start on 15 June 2022 or earlier            
        #     for i,future in enumerate(futures):
        #         self.zero_data[future.maturity_date] = self.discount_factor(pd.Timestamp(future.effective_date)).at[0] \
        #             / (future.forward_interest_rate * future.daycounter.year_fraction(future.effective_date,future.maturity_date) + 1)                                        
        #         self.__interpolate_zero_data()
        
        # if 'FRAs' in instruments.keys():
        #     FRAs = instruments['FRAs']
        #     FRAs.sort(key=lambda x: x.maturity_date, reverse=False) 
        #     # Bootstrapping using futures/FRAs requires the futures to have overlap over the end and start dates of two successive futures
        #     # For example, a future ends on 15 June 2022, the next future must start on 15 June 2022 or earlier            
        #     for i,fra in enumerate(FRAs):
        #         self.zero_data[fra.maturity_date] = self.discount_factor([pd.Timestamp(future.effective_date)]).at[0] \
        #             / (fra.forward_interest_rate * fra.daycounter.year_fraction(fra.effective_date,fra.maturity_date) + 1)                                        
        #         self.__interpolate_zero_data()
             
        # if 'swaps' in instruments.keys():
        #     swaps = instruments['swaps']
        #     swaps.sort(key=lambda x: x.pay_leg.schedule['payment_date'].iloc[-1], reverse=False) 
            
        #     for i,swap in enumerate(swaps):
        #         final_payment_date = swap.pay_leg.schedule['payment_date'].iloc[-1]

        #         if final_payment_date > self.zero_data_daily['date'].iloc[-1]:
                    
        #             # Set the initial guess to be the swaps par coupon rate
        #             zero_rate = swap.pay_leg.fixed_rate if swap.pay_leg.leg_type == 'fixed' else swap.rec_leg.fixed_rate   
        #             self.zero_data[final_payment_date] = np.exp(-zero_rate * year_fraction(self.curve_date, final_payment_date, self.day_count_basis))
                    
        #             if dsc_crv is not None:
        #                 swap.set_discount_curve(dsc_crv)
        #             if fwd_crv is not None:
        #                 swap.set_forward_curve(fwd_crv)
                    
        #             def solve_to_par(zero_cpn_rate: [float],
        #                              swap: Swap,
        #                              zero_curve: ZeroCurve,
        #                              final_payment_date: pd.Timestamp,
        #                              solve_fwd_crv: bool,
        #                              solve_dsc_crv: bool):
                        
        #                 zero_curve.zero_data[final_payment_date] = \
        #                     np.exp(-zero_cpn_rate[0] * zero_curve.daycounter.year_fraction(zero_curve.curve_date, final_payment_date))
        #                 zero_curve.__interpolate_zero_data()
              
        #                 if solve_fwd_crv:
        #                     swap.set_forward_curve(zero_curve)
        #                 if solve_dsc_crv:
        #                     swap.set_discount_curve(zero_curve)

        #                 pricing = swap.price()
        #                 return pricing['price']
                    
        #             solve_fwd_crv = fwd_crv is None
        #             solve_dsc_crv = dsc_crv is None    
        #             x, infodict, ier, msg = fsolve(solve_to_par, [
        #                                            zero_rate], 
        #                                            args=(swap, self, final_payment_date, solve_fwd_crv, solve_dsc_crv), 
        #                                            full_output=True)

        #             self.zero_data[final_payment_date] = np.exp(-x[0] * year_fraction(self.curve_date, final_payment_date, self.day_count_basis))
        #             self.__interpolate_zero_data()
                    
        #             if ier != 1:
        #                 print('Error bootstrapping swap with maturity', swap.pay_leg.maturity_date.strftime('%Y-%m-%d'), msg)                        
        #         else:        
        #             print('Swap with maturity', swap.pay_leg.maturity_date.strftime('%Y-%m-%d'), \
        #                   'excluded from bootstrapping as prior instruments have covered this period.')

        # if 'legs' in instruments.keys():
        #     legs = instruments['legs']
        #     legs.sort(key=lambda x: x.schedule['payment_date'].iloc[-1], reverse=False) 
            
        #     for i,leg in enumerate(legs):
        #         final_payment_date = leg.schedule['payment_date'].iloc[-1]

        #         if final_payment_date > self.zero_data_daily['date'].iloc[-1]:
                    
        #             # Set the initial guess to be the swaps par coupon rate
        #             zero_rate = leg.fixed_rate if leg.leg_type == 'fixed' else leg.fixed_rate   
        #             self.zero_data[final_payment_date] = np.exp(-zero_rate * year_fraction(self.curve_date, final_payment_date, self.day_count_basis))
                    
        #             if dsc_crv is not None:
        #                 leg.set_discount_curve(dsc_crv)
        #             if fwd_crv is not None:
        #                 leg.set_forward_curve(fwd_crv)
                    
        #             def solve_to_par(zero_cpn_rate: [float],
        #                              leg: Leg,
        #                              zero_curve: ZeroCurve,
        #                              final_payment_date: pd.Timestamp,
        #                              solve_fwd_crv: bool,
        #                              solve_dsc_crv: bool):
                        
        #                 zero_curve.zero_data[final_payment_date] = \
        #                     np.exp(-zero_cpn_rate[0] * zero_curve.daycounter.year_fraction(zero_curve.curve_date, final_payment_date))
        #                 zero_curve.__interpolate_zero_data()
              
        #                 if solve_fwd_crv:
        #                     leg.set_forward_curve(zero_curve)
        #                 if solve_dsc_crv:
        #                     leg.set_discount_curve(zero_curve)

                        
        #                 pricing = leg.price()
        #                 return pricing['price'] - leg.notional * (leg.transaction_price / 100.0)
                    
        #             solve_fwd_crv = fwd_crv is None
        #             solve_dsc_crv = dsc_crv is None    
        #             x, infodict, ier, msg = fsolve(solve_to_par, [
        #                                            zero_rate], 
        #                                            args=(leg, self, final_payment_date, solve_fwd_crv, solve_dsc_crv), 
        #                                            full_output=True)

        #             self.zero_data[final_payment_date] = np.exp(-x[0] * year_fraction(self.curve_date, final_payment_date, self.day_count_basis))
        #             self.__interpolate_zero_data()
                    
        #             if ier != 1:
        #                 print('Error bootstrapping leg with maturity', leg.maturity_date.strftime('%Y-%m-%d'), msg)                        
        #         else:        
        #             print('Leg with maturity', leg.maturity_date.strftime('%Y-%m-%d'), \
        #                   'excluded from bootstrapping as prior instruments have covered this period.')
    
    def flat_shift(self, 
                   basis_points: float=1) -> 'ZeroCurve':
        tenor_dates = list(self.zero_data.keys())
        years_to_date = year_fraction(self.curve_date, tenor_dates, self.day_count_basis)
        shifted = - np.log(list(self.zero_data.values())) / years_to_date + basis_points / 10000    
        df_shifted_data = pd.DataFrame({'date': tenor_dates, 
                                        'years_to_date': years_to_date, 
                                        'discount_factor':np.exp(-shifted * years_to_date)})
        df_shifted_data.loc[0,'discount_factor'] = 1.0 # set the discount factor on the curve date to be 1.0
        return ZeroCurve(curve_date=self.curve_date, 
                         zero_data = df_shifted_data,
                         day_count_basis = self.day_count_basis)
                   
    def forward_rate(self,
                     d1: Union[pd.Timestamp, pd.Series],
                     d2: Union[pd.Timestamp, pd.Series],
                     compounding_frequency: CompoundingFrequency) -> pd.Series:

        assert len(d1) == len(d2)
        assert type(d1) == type(d2)
        bool_cond = d1 < self.curve_date
        
        historical_fixings = []
        if sum(bool_cond) > 0:
            historical_fixings = pd.Series([self.historical_fixings[d] if d in self.historical_fixings.keys() else np.nan for d in d1[bool_cond]])
            
        years = pd.Series(year_fraction(d1, d2, self.day_count_basis))            
        DF_T1 = self.discount_factor(d1[np.logical_not(bool_cond)])
        DF_T2 = self.discount_factor(d2[np.logical_not(bool_cond)])
        forward_fixings = forward_rate(DF_T1, DF_T2, years, compounding_frequency)
        
        return pd.Series(historical_fixings + forward_fixings.to_list())
       
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
        
            df = self.index_daily_zero_data(dates, days, years)
            return df['discount_factor'].values
           
        
    def zero_rate(self, 
                  compounding_frequency: CompoundingFrequency,
                  dates: Union[pd.Timestamp, pd.Series] = None,
                  days: Union[int, pd.Series] = None,
                  years: Union[float, pd.Series] = None,
                  ) -> pd.Series:
                
            df = self.index_daily_zero_data(dates, days, years)
            
            if compounding_frequency == CompoundingFrequency.CONTINUOUS:
                return df['nacc'].values
            else: 
                zero_rate = zero_rate_from_discount_factor(years=df['years'],
                                                           discount_factor=df['discount_factor'], 
                                                           compounding_frequency=compounding_frequency)
                return zero_rate        
        
        
    def index_daily_zero_data(self, 
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
                ln_df_interpolated = np.interp(x=years, xp=self.zero_data['years'], fp= np.log(self.zero_data['discount_factor']))
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
            min_date = self.zero_data_daily['date'].min()
            below_range = dates < min_date
            dates_below = []
            if any(below_range):
                bound_date = self.zero_data_daily['date'].min()
                for i,date in enumerate(dates[below_range]):
                    print('Date', date.strftime('%Y-%m-%d'), 'is below the min available data of ' + bound_date.strftime('%Y-%m-%d') + msg)
                    if self.extrapolation_method == 'none':    
                        dates_below.append(np.nan)
                    elif self.extrapolation_method == 'flat':    
                        dates_below.append(bound_date)
            
            max_date = self.zero_data_daily['date'].max()
            above_range = dates > max_date
            dates_above = []
            if any(above_range):
                bound_date = self.zero_data_daily['date'].max()
                for i,date in enumerate(dates[above_range]):
                    print('Date', date.strftime('%Y-%m-%d'), 'is above the max available data of ' + bound_date.strftime('%Y-%m-%d') + msg)
                    if self.extrapolation_method == 'none':    
                        dates_above.append(np.nan)
                    elif self.extrapolation_method == 'flat':    
                        dates_above.append(bound_date)
          
            in_range = np.logical_and(np.logical_not(below_range),np.logical_not(above_range))
            cleaned_dates = dates_below + list(dates[in_range].values) + dates_above
            df = pd.merge(pd.DataFrame(data=cleaned_dates, columns=['date']), self.zero_data_daily, left_on='date',right_on='date', how='left')
            return df    

       
    def plot(self, forward_rate_terms=[90]):
            
        # Zero rates
        min_date = self.zero_data_daily['date'].iloc[1]
        max_date = self.zero_data_daily['date'].iloc[-1]
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


    def subtract_curve(self, 
                   zero_curve: 'ZeroCurve',
                   day_count_basis: DayCountBasis=None,
                   name: Optional[str]=None,
                   task=None):
    
        # Returns a zero curve constructed from zero rates where,
        # zero rates = self.zero_rates - zero_curve.zero_rates
        
        assert self.curve_date == zero_curve.curve_date        
        if day_count_basis is None:
            day_count_basis = self.day_count_basis
            
        if name is None:
            if self.name is not None and zero_curve.name is not None:
                name = '[' + self.name + '] - [' + zero_curve.name + ']'
            
        min_date = min(self.zero_data_daily['date'].iloc[-1], zero_curve.daily_data['date'].iloc[-1])

        date_range = pd.date_range(self.curve_date, min_date,freq='d')
        date_range = date_range[1:] # exclude the zero_rate on self.data which is probably a np.nan

        zero_rates_self = self.zero_rate(dates=date_range)
        zero_rates_other = zero_curve.zero_rate(dates=date_range)
        
        zero_rate_difference = zero_rates_self - zero_rates_other
        df_diff = pd.DataFrame({'date': date_range,
                                'zero_rate': zero_rate_difference})
        
        return ZeroCurve(date=self.curve_date, 
                         name=name,
                         zero_data=df_diff,
                         day_count_basis=day_count_basis)
        
    
    def add_curve(self, 
                  zero_curve: 'ZeroCurve',
                  day_count_basis: DayCountBasis=None,
                  name: Optional[str]=None):
    
        # Returns a zero curve constructed from zero rates where,
        # zero rates = self.zero_rates + zero_curve.zero_rates
        
        assert self.curve_date == zero_curve.curve_date        
        if day_count_basis is None:
            day_count_basis = self.day_count_basis
            
        if name is None:
            if self.name is not None and zero_curve.name is not None:
                name = '[' + self.name + '] + [' + zero_curve.name + ']'
            
        min_date = min(self.zero_data_daily['date'].iloc[-1], zero_curve.daily_data['date'].iloc[-1])

        date_range = pd.date_range(self.curve_date, min_date,freq='d')
        date_range = date_range[1:] # exclude the zero_rate on self.data which is probably a np.nan

        zero_rates_self = self.zero_rate(dates=date_range)
        zero_rates_other = zero_curve.zero_rate(dates=date_range)
        
        zero_rate_difference = zero_rates_self + zero_rates_other
        df_diff = pd.DataFrame({'date': date_range,
                                'zero_rate': zero_rate_difference})
        
        return ZeroCurve(date=self.curve_date, 
                         name=name,
                         zero_data=df_diff,
                         day_count_basis=day_count_basis)
        
        

        
        
    