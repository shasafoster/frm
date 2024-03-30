# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
from frm.frm.market_data.fx_volatility_surface_helpers import fx_σ_input_helper     
from frm.frm.market_data.ir_zero_curve import ZeroCurve
from frm.frm.pricing_engine.garman_kohlhagen import gk_price, gk_solve_implied_σ, gk_solve_strike
from frm.frm.pricing_engine.heston_gk import heston_fit_vanilla_fx_smile, heston1993_price_fx_vanilla_european, heston_carr_madan_fx_vanilla_european

from frm.frm.schedule.tenor import calc_tenor_date, get_spot_offset
from frm.frm.schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS_TYPES
from frm.frm.schedule.business_day_calendar import get_calendar        
from frm.frm.utilities.utilities import convert_column_type    

import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import fsolve, root_scalar
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline 
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Literal
import warnings
import re
import matplotlib.pyplot as plt

VALID_DELTA_CONVENTIONS = ['regular_spot_delta','regular_forward_delta','premium_adjusted_spot_delta','premium_adjusted_forward_delta'] 
VALID_FX_SMILE_INTERPOLATION_METHOD = Literal['univariate_spline',
                                              'cubic_spline',
                                              'heston_analytical_1993',
                                              'heston_carr_madan_gauss_kronrod_quadrature',
                                              'heston_carr_madan_fft_w_simpsons',
                                              'vanna_volga',
                                              'sabr',
                                              'svi_sabr']





@dataclass
class FXVolatilitySurface:
    curve_date: pd.Timestamp
    curve_ccy: str
    fx_forward_curve: pd.DataFrame
    domestic_zero_curve: ZeroCurve
    foreign_zero_curve: ZeroCurve

    # Non initialisation arguments
    daycounter: DayCounter = field(init=False)
    spot_date: pd.Timestamp
    fx_spot_rate: float = field(init=False)
    domestic_ccy: str = field(init=False)
    foreign_ccy: str = field(init=False)
    Δ_σ_daily: pd.DataFrame=None
    σ_pillar: dict=None
    K_σ_daily_smile_func: dict=None
    
    # volatility definitions
    σ_pillar: pd.DataFrame=None
    smile_interpolation_method: Optional[VALID_FX_SMILE_INTERPOLATION_METHOD] = 'cubic_spline'
    spot_date: Optional[pd.Timestamp] = None
    
    #atmΔ: pd.DataFrame=None
    #atmF: pd.DataFrame=None
    #bf25Δ: pd.DataFrame=None
    #rr25Δ: pd.DataFrame=None
    #bf10Δ: pd.DataFrame=None
    #bf10Δ: pd.DataFrame=None
    
    day_count_basis: InitVar[Optional[VALID_DAY_COUNT_BASIS_TYPES]] = 'act/act'
    
    

    def __post_init__(self, day_count_basis):
         
        self.daycounter = DayCounter(day_count_basis)
        self.K_σ_daily_smile_func = dict()

        holiday_calendar = get_calendar(ccys=[self.curve_ccy[:3],self.curve_ccy[3:]])

        if self.spot_date is None:
            result = calc_tenor_date(self.curve_date, 'sp', self.curve_ccy, holiday_calendar=holiday_calendar, spot_offset=True)
            holiday_rolled_offset_date, cleaned_tenor_name, spot_date = result
            assert holiday_rolled_offset_date == spot_date
            self.spot_date = spot_date

        # curve_ccy validation
        self.curve_ccy = self.curve_ccy.lower().strip()
        assert len(self.curve_ccy) == 6, self.curve_ccy
        self.foreign_ccy = self.curve_ccy[:3]
        self.domestic_ccy = self.curve_ccy[3:]
        if self.foreign_ccy == 'usd' and self.domestic_ccy in {'aud','eur','gbp','nzd'}:
            warnings.warn("non conventional fx market, typically 'usd' is the domestic currency for 'audusd', 'eurusd', 'gbpusd' and 'nzdusd' pairs")
        elif self.domestic_ccy == 'usd' and self.foreign_ccy not in {'aud','eur','gbp','nzd'}:
            warnings.warn("non conventional fx market, typically 'usd' is the foreign currency  except for 'audusd', 'eurusd', 'gbpusd' and 'nzdusd' pairs")
 
        # fx_forward_curve validation 
        assert 'fx_forward_rate' in self.fx_forward_curve.columns 
        self.fx_forward_curve = convert_column_type(self.fx_forward_curve)
        assert ('tenor_name' in self.fx_forward_curve.columns) or ('expiry_date' in self.fx_forward_curve.columns and 'delivery_date' in self.fx_forward_curve.columns)
        if 'expiry_date' not in self.fx_forward_curve.columns and 'delivery_date' not in self.fx_forward_curve.columns:
            
            result = calc_tenor_date(self.curve_date, self.fx_forward_curve['tenor_name'], self.curve_ccy, holiday_calendar=holiday_calendar, spot_offset=True)
            holiday_rolled_offset_date, cleaned_tenor_name, spot_date = result
            self.fx_forward_curve['tenor_name'] = cleaned_tenor_name
            self.fx_forward_curve['delivery_date'] = holiday_rolled_offset_date
            self.fx_forward_curve['years_to_delivery'] = self.daycounter.year_fraction(self.curve_date, self.fx_forward_curve['tenor_years_delivery'])            

            spot_offset = -1 * get_spot_offset(curve_ccy=self.curve_ccy)
            holiday_rolled_offset_date = np.busday_offset(holiday_rolled_offset_date, offsets=spot_offset, roll='preceeding', busdaycal=holiday_calendar)
                
            result = calc_tenor_date(self.curve_date, self.fx_forward_curve['tenor_name'], self.curve_ccy, holiday_calendar=holiday_calendar, spot_offset=True)
            holiday_rolled_offset_date, cleaned_tenor_name, spot_date = result
            
            self.fx_forward_curve['expiry_date'] = holiday_rolled_offset_date
            self.fx_forward_curve['yrs_to_expiry'] = self.daycounter.year_fraction(self.curve_date, self.fx_forward_curve['tenor_years_observation'])
                                    
        self.fx_forward_curve.set_index('expiry_date', inplace=True,drop=True)
            
        if self.curve_date not in self.fx_forward_curve.index:
            raise ValueError("The curve date (" + self.curve_date.strftime('%Y-%m-%d') + ') fx rate (i.e the fx spot rate) is in missing in fx_forward_curve')
        
        self.fx_spot_rate = self.fx_forward_curve.loc[self.curve_date,'fx_forward_rate']
        
        # σ_pillar validation
        if self.σ_pillar is not None:
            
            σ_pillar = self.σ_pillar
            
            σ_pillar['curve_date'] = self.curve_date
            σ_pillar['curve_ccy'] = self.curve_ccy
            
            
            results = fx_σ_input_helper(σ_pillar)
            
            
            # Convert columns to type float (can be set to type object from splicing or NaNs)
            for col in self.σ_pillar.columns:
                if self.σ_pillar[col].apply(isinstance, args=(float,)).all():
                    self.σ_pillar[col] = pd.to_numeric(self.σ_pillar[col])
            if 'tenor_date' not in self.σ_pillar.columns:
                result = calc_tenor_date(self.curve_date, self.σ_pillar['tenor_name'], self.curve_ccy, holiday_calendar=holiday_calendar)
                holiday_rolled_offset_date, cleaned_tenor_name, spot_date = result            
                self.σ_pillar['tenor_date'] = holiday_rolled_offset_date
                self.σ_pillar['tenor_name'] = cleaned_tenor_name
            self.σ_pillar.set_index('tenor_date', inplace=True,drop=True)       
            self.σ_pillar.insert(loc=0, column='tenor_years', value=self.daycounter.year_fraction(self.curve_date, self.σ_pillar.index).values)                    

            cols = [v for v in self.σ_pillar.columns if v not in ['tenor_date','tenor_name','tenor_years','Δ_convention']]
            for i,v in enumerate(cols):
                pattern1 = r'^σ_(\d{1,2})Δ(call|put)$'
                pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
                atm_column_names = ['σ_atmΔneutral','σ_atmf']
                if (re.match(pattern1, v) and 1 <= int(re.match(pattern1, v).group(1)) <= 99) \
                    or (re.match(pattern2, v) and 1 <= int(re.match(pattern2, v).group(1)) <= 99) \
                    or v in atm_column_names:    
                    pass
                else:
                    msg = 'user added column' + "'" + v + "'" + ' does not ' \
                        + 'match regex pattern ' + "'" + pattern1 + "', or pattern " + "'" + pattern2 + "'," \
                        + ' and is not in the allowed list (' + ', '.join(atm_column_names) + ')'
                    raise ValueError(msg)
            self.σ_pillar = self.__interp_daily_FXF_and_IR_rates(self.σ_pillar.copy())
            
            
            if True:                
                ###################################################################
                # 1. Calculate the delta-strike surface from the delta-volatility surface           
                
                self.σ_pillar_input_Δ_convention = self.σ_pillar.copy()
                Δ_K_pillar = self.σ_pillar.copy()
                Δ_K_pillar.loc[:, Δ_K_pillar.columns.str.contains('σ')] = np.nan
                Δ_K_pillar.rename(columns=lambda x: x.replace('σ', 'k'), inplace=True)
    
                helper_cols = ['tenor_date','tenor_name','tenor_years','fx_forward_rate','foreign_ccy_continuously_compounded_zero_rate','domestic_ccy_continuously_compounded_zero_rate','Δ_convention']
                dict_K_σ = {v.replace('σ', 'k'): v for v in self.σ_pillar.columns if v not in helper_cols}
                for date, row in Δ_K_pillar.iterrows():
                    
                    
                    
                    # Solve strikes 
                    
                    # Scalars
                    S = self.fx_spot_rate
                    r_f=row['foreign_ccy_continuously_compounded_zero_rate']
                    r_d=row['domestic_ccy_continuously_compounded_zero_rate']
                    tau=row['tenor_years']                
                    Δ_convention=row['Δ_convention']
                    F=row['fx_forward_rate']  
                    # Arrays
                    cp = [1 if v[-4:] == 'call' else -1 if v[-3:] == 'put' else None for v in dict_K_σ.keys()]
                    Δ = [0.5 if v == 'k_atmΔneutral' else cp[i] * float(v.split('_')[1].split('Δ')[0]) / 100 for i,v in enumerate(dict_K_σ.keys())]
                    σ = self.σ_pillar.loc[date, dict_K_σ.values()].values
                                        
                    K = gk_solve_strike(S=S,tau=tau,r_f=r_f,r_d=r_d,σ=σ,Δ=Δ,Δ_convention=Δ_convention,F=F)
                    Δ_K_pillar.loc[date,dict_K_σ.keys()] = K
                                                
  
                self.Δ_K_pillar = Δ_K_pillar

            ###################################################################
            # Interpolate the term structure of volatility to get a daily volatility smile            
            dates = pd.date_range(min(self.σ_pillar.index),max(self.σ_pillar.index),freq='d')
            tenor_years = self.daycounter.year_fraction(self.curve_date,dates)
            df_interp = pd.DataFrame({'tenor_date':dates, 'tenor_years': tenor_years})
            df_pillar = self.σ_pillar.copy()

            # Merge to find closest smaller and larger tenors for each target tenor
            df_lower_pillar = pd.merge_asof(df_interp.sort_values('tenor_years'), df_pillar, on='tenor_date', direction='backward', suffixes=('_interp', '_pillar'))
            df_upper_pillar = pd.merge_asof(df_interp.sort_values('tenor_years'), df_pillar, on='tenor_date', direction='forward', suffixes=('_interp', '_pillar'))
                    
            # Convert to numpy for efficient calculations
            t1 = df_lower_pillar['tenor_years_pillar'].to_numpy()
            t2 = df_upper_pillar['tenor_years_pillar'].to_numpy()
            t = df_interp['tenor_years'].to_numpy()
            t1 = t1[:, np.newaxis]
            t2 = t2[:, np.newaxis]
            t = t[:, np.newaxis]
            σ_t1 = df_lower_pillar[cols].to_numpy()
            σ_t2 = df_upper_pillar[cols].to_numpy()             
            
            # Interpolation logic
            mask = t1 != t2
            σ_t = np.where(mask, self.flat_forward_interp(t1, σ_t1, t2, σ_t2, t), σ_t1)
            df_interp[cols] = σ_t            
            df_interp['Δ_convention'] = 'regular_forward_Δ' # need to add a section to calculate forward delta from spot delta
            df_interp.set_index('tenor_date', inplace=True,drop=True)    
            self.Δ_σ_daily = self.__interp_daily_FXF_and_IR_rates(df_interp)


    def __interp_daily_FXF_and_IR_rates(self, df):
        offset = 1 if 'tenor_name' in df.columns else 0
        
        df.insert(loc=1+offset, column='foreign_ccy_continuously_compounded_zero_rate', value=np.nan)
        df.insert(loc=2+offset, column='domestic_ccy_continuously_compounded_zero_rate', value=np.nan)
        df.insert(loc=3+offset, column='fx_forward_rate', value=np.nan)
    
        df['foreign_ccy_continuously_compounded_zero_rate'] = self.foreign_zero_curve.zero_rate(dates=df.index,compounding_frequency='continuously').values       
        df['domestic_ccy_continuously_compounded_zero_rate'] = self.domestic_zero_curve.zero_rate(dates=df.index,compounding_frequency='continuously').values
        df['fx_forward_rate'] = self.interp_fx_forward_curve(expiry_dates=df.index, flat_extrapolation=True)
        return df


    def interp_fx_forward_curve(self, 
                                expiry_dates: pd.DatetimeIndex,
                                flat_extrapolation: bool=True):
        """
        Interpolate the FX forward curve to match given expiry_dates.
        Please note expiry date is 2 business days prior to the spot date
        """
        
        unique_dates = expiry_dates.drop_duplicates()
        combined_index = self.fx_forward_curve.index.union(unique_dates)
        result = self.fx_forward_curve.reindex(combined_index)
        start_date, end_date = self.fx_forward_curve.index.min(), self.fx_forward_curve.index.max()
        
        if flat_extrapolation:
            result['fx_forward_rate'] = result['fx_forward_rate'].interpolate(method='time', limit_area='inside').ffill().bfill()
            # Find out of range dates and warn
            out_of_range_dates = unique_dates[(unique_dates < start_date) | (unique_dates > end_date)]
            for date in out_of_range_dates:
                warnings.warn(f"Date {date} is outside the range {start_date} - {end_date}, flat extrapolation applied.")
        else:
            result['fx_forward_rate'] = result['fx_forward_rate'].interpolate(method='time', limit_area='inside')
        
        result = result.reindex(expiry_dates)
        
        return result['fx_forward_rate']
        
        
                
    def forward_volatility(self, 
                           t1: Union[float, np.array], 
                           σ_t1: Union[float, np.array], 
                           t2: Union[float, np.array], 
                           σ_t2: Union[float, np.array]) -> Union[float, np.array]:
        """
        Calculate the at-the-money forward volatility from time t1 to t2.
        The forward volatility is based on the consistency condition:
        σ_t1**2 * t1 + σ_t1_t2**2 * (t2- t1) = σ_t1**2 * t2    
    
        Parameters:
        - t1 (float): Time to first maturity
        - σ_t1 (float): At-the-money volatility at time t1
        - t2 (float): Time to second maturity
        - σ_t2 (float): At-the-money volatility at time t2
    
        Returns:
        - np.array: Forward volatility from time t1 to t2
        """
        tau = t2 - t1
        if np.any(tau == 0):
            warnings.warn("t2 and t1 are equal. NaN will be returned.")
    
        result = (σ_t2**2 * t2 - σ_t1**2 * t1) / tau
        if np.any(result < 0):
            raise ValueError("Negative value encountered under square root.")
    
        return np.sqrt(result)

    def flat_forward_interp(self, 
                            t1: Union[float, np.array], 
                            σ_t1: Union[float, np.array], 
                            t2: Union[float, np.array], 
                            σ_t2: Union[float, np.array], 
                            t: Union[float, np.array]) -> Union[float, np.array]:
        """
        Interpolate the at-the-money volatility at a given time 't' using flat forward interpolation.
    
        Parameters:
        - t1 (Union[float, array]): Time to first maturity
        - σ_t1 (Union[float, array]): At-the-money volatility at time t1
        - t2 (Union[float, array]): Time to second maturity
        - σ_t2 (Union[float, array]): At-the-money volatility at time t2
        - t (Union[float, array]): Time at which to interpolate the volatility
    
        Returns:
        - array: Interpolated volatility at time 't'
        """
        σ_t12 = self.forward_volatility(t1, σ_t1, t2, σ_t2)
        return np.sqrt((σ_t1**2 * t1 + σ_t12**2 * (t - t1)) / t)



    def fit_and_plot_smile(self):
        
        def convert_colunm_name_to_delta_for_plt(input_str):
            if 'put' in input_str:
                return int(input_str.split('Δ')[0][2:])
            elif 'call' in input_str:
                return 100 - int(input_str.split('Δ')[0][2:])
            elif 'neutral' in input_str:
                return 50
        
        def convert_colunm_name_to_delta(input_str):
            if 'put' in input_str:
                return -1 * int(input_str.split('Δ')[0][2:]) / 100
            elif 'call' in input_str:
                return int(input_str.split('Δ')[0][2:]) / 100
            elif 'neutral' in input_str:
                return 0.5           
            
        i = 0
        for date, row in self.σ_pillar.iterrows():
            
            # Scalars
            tenor_name = row['tenor_name']
            Δ_convention=row['Δ_convention']
            S = self.fx_spot_rate
            r_f=row['foreign_ccy_continuously_compounded_zero_rate']
            r_d=row['domestic_ccy_continuously_compounded_zero_rate']
            tau=row['tenor_years']                
            F=row['fx_forward_rate'] 
            
            # Arrays
            Δ = [convert_colunm_name_to_delta(c) for c in self.σ_pillar.columns if c[0] == 'σ']
            Δ_for_plt = [convert_colunm_name_to_delta_for_plt(c) for c in self.σ_pillar.columns if c[0] == 'σ']
            cp = np.sign(Δ)
            σ = row[[c for c in self.σ_pillar.columns if c[0] == 'σ']].values             
                     
            if self.smile_interpolation_method[:6] == 'heston':
                v0, vv, kappa, theta, rho, lambda_, IV, SSE = \
                    heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ, S, r_f, r_d, tau, cp, pricing_method=self.smile_interpolation_method)
            
                # Displaying output
                print(f'=== {tenor_name} calibration results ===')
                print(f'v0, vv, kappa, theta, rho: {v0, vv, kappa, theta, rho}')
                print(f'SSE {SSE*100}')
                
                # Plotting
                i+=1
                plt.figure(i+1)
                plt.plot(Δ_for_plt, σ * 100, 'ko-', linewidth=1)
                plt.plot(Δ_for_plt, IV * 100, 'rs--', linewidth=1)
                plt.legend([f'{tenor_name} smile', 'Heston fit'], loc='upper right')
                plt.xlabel('Delta [%]')
                plt.ylabel('Implied volatility [%]')
                plt.xticks(Δ_for_plt)
                plt.title(self.smile_interpolation_method)
                plt.show()   
                
            else:
                raise ValueError
                  

    def interp_σ_surface(self, dates: pd.DatetimeIndex, K: np.array, cp):
        """
        Private method to interpolate the σ surface for given dates, updating the 
        `K_σ_daily_smile_func` attribute with appropriate interpolation functions or Heston parameters.
    
        Parameters:
        dates (pd.DatetimeIndex): Array of dates to perform the σ surface interpolation for.
    
        Raises:
        ValueError: If the sum of squared errors (SSE) from the Heston fit exceeds a threshold, indicating a poor fit.
    
        Note:
        Updates the `K_σ_daily_smile_func` attribute with either a spline interpolation function or Heston parameters for each date in `dates`.
        """        
        
        K = np.atleast_1d(K).astype(float)
        cp = np.atleast_1d(cp).astype(float)
        
        assert dates.shape == K.shape
        assert dates.shape == cp.shape
        
        df = self.Δ_σ_daily.loc[pd.DatetimeIndex(dates),:].copy()
        cols_to_drop = [col for col in df.columns if 'σ' in col]
        df.drop(columns=cols_to_drop, inplace=True)
        df['K'] = K
        df['call_put'] = cp

        df_unique_dates = self.Δ_σ_daily.loc[pd.DatetimeIndex(set(dates)),:].copy()
        df_unique_dates.loc[:, df_unique_dates.columns.str.contains('σ')] = np.nan
        df_unique_dates.rename(columns=lambda x: x.replace('σ', 'k'), inplace=True)

        helper_cols = ['tenor_date','tenor_name','tenor_years','fx_forward_rate','foreign_ccy_continuously_compounded_zero_rate','domestic_ccy_continuously_compounded_zero_rate','Δ_convention']
        dict_K_σ = {v.replace('σ', 'k'): v for v in self.Δ_σ_daily.columns if v not in helper_cols}
                
        result = []
    
        # Interpolate volatility surface for each date in df
        for date, row in df_unique_dates.iterrows():
            
            if date not in self.K_σ_daily_smile_func.keys(): 
                
                # Scalars
                S = self.fx_spot_rate
                r_f=row['foreign_ccy_continuously_compounded_zero_rate']
                r_d=row['domestic_ccy_continuously_compounded_zero_rate']
                tau=row['tenor_years']                
                Δ_convention=row['Δ_convention']
                F=row['fx_forward_rate']     
                
                # Arrays
                cp = [1 if v[-4:] == 'call' else -1 if v[-3:] == 'put' else 1 for v in dict_K_σ.keys()]
                Δ = [0.5 if v == 'k_atmΔneutral' else cp[i] * float(v.split('_')[1].split('Δ')[0]) / 100 for i,v in enumerate(dict_K_σ.keys())]
                σ = self.Δ_σ_daily.loc[date, dict_K_σ.values()].values                
                            
                if self.smile_interpolation_method == 'univariate_spline':
                    K = gk_solve_strike(S=S,tau=tau,r_f=r_f,r_d=r_d,σ=σ,Δ=Δ,Δ_convention=Δ_convention,F=F)
                    df_unique_dates.loc[date,dict_K_σ.keys()] = K 
                    self.K_σ_daily_smile_func[date] = InterpolatedUnivariateSpline(x=K, y=σ)
                
                elif self.smile_interpolation_method == 'cubic_spline':
                    K = gk_solve_strike(S=S,tau=tau,r_f=r_f,r_d=r_d,σ=σ,Δ=Δ,Δ_convention=Δ_convention,F=F)
                    df_unique_dates.loc[date,dict_K_σ.keys()] = K 
                    self.K_σ_daily_smile_func[date] = CubicSpline(x=K, y=σ)
                    
                elif self.smile_interpolation_method[:6] == 'heston':
                    v0, vv, kappa, theta, rho, lambda_, IV, SSE = \
                        heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ, S, r_f, r_d, tau, cp, pricing_method=self.smile_interpolation_method)
                    if SSE < 0.001:
                        self.K_σ_daily_smile_func[date] = v0, vv, kappa, theta, rho, lambda_, IV, SSE
                    else:
                        raise ValueError('SSE is a large value, ', round(SSE,4), ' heston fit at ', date,' is likely poor')
            
        # Get the implied vol for each strike and date combination
        for date,row in df.iterrows():
            K_target = row['K']
            
            if self.smile_interpolation_method in {'univariate_spline', 'cubic_spline'}:
                σ = self.K_σ_daily_smile_func[date](K_target)
                result.append(σ)
            elif self.smile_interpolation_method[:6] == 'heston':
                S = self.fx_spot_rate
                r_f=row['foreign_ccy_continuously_compounded_zero_rate']
                r_d=row['domestic_ccy_continuously_compounded_zero_rate']
                tau=row['tenor_years']                
                F=row['fx_forward_rate']               
                cp = row['call_put']

                v0, vv, kappa, theta, rho, lambda_, IV, SSE = self.K_σ_daily_smile_func[date]
                
                if self.smile_interpolation_method == 'heston_analytical_1993':
                    X = heston1993_price_fx_vanilla_european(S, tau, r_f, r_d, cp, K_target, v0, vv, kappa, theta, rho, lambda_)
                elif self.smile_interpolation_method == 'heston_carr_madan_gauss_kronrod_quadrature':     
                    X = heston_carr_madan_fx_vanilla_european(S, tau, r_f, r_d, cp, K_target, v0, vv, kappa, theta, rho, integration_method=0)
                elif self.smile_interpolation_method == 'heston_carr_madan_fft_w_simpsons':
                    X = heston_carr_madan_fx_vanilla_european(S, tau, r_f, r_d, cp, K_target, v0, vv, kappa, theta, rho, integration_method=1)
                
                implied_σ = gk_solve_implied_σ(S=S, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K_target, X=X, σ_guess=v0) 
                result.append(implied_σ)            
                        
        return np.array(result)
    
    def price_fx_vanilla_european(self, 
                                  expiry_datetimeindex,        
                                  K, 
                                  cp,
                                  analytical_greeks_flag):
                      
        K = np.atleast_1d(K)
        cp = np.atleast_1d(cp)

        assert expiry_datetimeindex.shape == K.shape
        assert expiry_datetimeindex.shape == cp.shape

        σ = self.interp_σ_surface(expiry_datetimeindex, K, cp)
        r_f = self.foreign_zero_curve.zero_rate(expiry_datetimeindex,compounding_frequency='continuously').values
        r_d = self.domestic_zero_curve.zero_rate(expiry_datetimeindex,compounding_frequency='continuously').values
        F = self.interp_fx_forward_curve(expiry_dates=expiry_datetimeindex, flat_extrapolation=True)

        result = gk_price(
            S=self.fx_spot_rate,
            tau=self.daycounter.year_fraction(self.curve_date, expiry_datetimeindex),
            r_f=r_f,
            r_d=r_d,
            cp=cp,
            K=K,
            σ=σ,
            F=F,
            analytical_greeks_flag=analytical_greeks_flag
        )    

        if analytical_greeks_flag:
            return result[0], result[1], pd.DataFrame({'σ': σ, 'r_f':r_f, 'r_d':r_d, 'F':F})

        return result
        




