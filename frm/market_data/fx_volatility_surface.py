# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
from frm.frm.market_data.fx_volatility_surface_helpers import fx_σ_input_helper, VALID_DELTA_CONVENTIONS, interp_fx_forward_curve
from frm.frm.market_data.ir_zero_curve import ZeroCurve
from frm.frm.pricing_engine.garman_kohlhagen import gk_price, gk_solve_implied_σ, gk_solve_strike
from frm.frm.pricing_engine.heston_garman_kohlhagen import heston_fit_vanilla_fx_smile, heston1993_price_fx_vanilla_european, heston_carr_madan_price_fx_vanilla_european, heston_cosine_price_fx_vanilla_european
from frm.frm.pricing_engine.monte_carlo_generic import generate_rand_nbs
from frm.frm.pricing_engine.geometric_brownian_motion import simulate_gbm_path
from frm.frm.pricing_engine.heston import simulate_heston

from frm.frm.schedule.tenor import calc_tenor_date, get_spot_offset
from frm.frm.schedule.daycounter import DayCounter, VALID_DAY_COUNT_BASIS_TYPES
from frm.frm.schedule.business_day_calendar import get_calendar        
from frm.frm.utilities.utilities import convert_column_to_consistent_data_type, generic_market_data_input_cleanup_and_validation    

#%%

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


# Models TBC: sabr, svi_sabr, vanna_volga, sv_local_vol, jump_diffusion
VALID_FX_SMILE_INTERPOLATION_METHOD = Literal['univariate_spline',
                                              'cubic_spline',
                                              'heston_analytical_1993',
                                              'heston_carr_madan_gauss_kronrod_quadrature',
                                              'heston_carr_madan_fft_w_simpsons',
                                              'heston_cosine']


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
    smile_interpolation_method: Optional[VALID_FX_SMILE_INTERPOLATION_METHOD] = 'heston_cosine'
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
        self.fx_forward_curve = convert_column_to_consistent_data_type(self.fx_forward_curve)
        assert ('tenor_name' in self.fx_forward_curve.columns) or ('expiry_date' in self.fx_forward_curve.columns and 'delivery_date' in self.fx_forward_curve.columns)
        if 'expiry_date' not in self.fx_forward_curve.columns and 'delivery_date' not in self.fx_forward_curve.columns:
            
            # Calculate and set the delivery date
            result = calc_tenor_date(self.curve_date, self.fx_forward_curve['tenor_name'], self.curve_ccy, holiday_calendar=holiday_calendar, spot_offset=True)
            delivery_date, cleaned_tenor_name, spot_date = result
            self.fx_forward_curve['tenor_name'] = cleaned_tenor_name
            self.fx_forward_curve['delivery_date'] = delivery_date
            self.fx_forward_curve['years_to_delivery'] = self.daycounter.year_fraction(self.curve_date, self.fx_forward_curve['delivery_date'])            

            # Calculate and set the expiry date
            spot_offset = -1 * get_spot_offset(curve_ccy=self.curve_ccy)
            expiry_date = np.busday_offset(delivery_date.values.astype('datetime64[D]'), offsets=spot_offset, roll='preceding', busdaycal=holiday_calendar)
            self.fx_forward_curve['expiry_date'] = expiry_date
            self.fx_forward_curve['years_to_expiry'] = self.daycounter.year_fraction(self.curve_date, self.fx_forward_curve['expiry_date'])

        if self.curve_date not in self.fx_forward_curve['expiry_date'].values:
            raise ValueError("The curve date (" + self.curve_date.strftime('%Y-%m-%d') + ') fx rate (i.e the fx spot rate) is in missing in fx_forward_curve')
        else:
            mask = self.fx_forward_curve['expiry_date'] == self.curve_date
            self.fx_spot_rate = self.fx_forward_curve.loc[mask,'fx_forward_rate'].iloc[0]
        
        # σ_pillar validation
        if self.σ_pillar is not None:
            σ_pillar = self.σ_pillar
            σ_pillar['curve_date'] = self.curve_date
            σ_pillar['curve_ccy'] = self.curve_ccy
            σ_pillar['day_count_basis'] = self.daycounter.day_count_basis
            
            σ_pillar = generic_market_data_input_cleanup_and_validation(σ_pillar, spot_offset=False)
            σ_pillar = fx_σ_input_helper(σ_pillar)
            σ_pillar.set_index('tenor_date', drop=True, inplace=True)
                    
            σ_pillar = self.__interp_daily_FXF_and_IR_rates(σ_pillar)   
                 
            self.σ_pillar = σ_pillar            
            
            if 'σ_atmf' in self.σ_pillar.keys():
                self.K_pillar = σ_pillar['fx_forward_rate']
            else:
                ###################################################################
                # Calculate the delta-strike surface from the delta-volatility surface           
                K_pillar = self.σ_pillar.copy()
                K_pillar.loc[:, K_pillar.columns.str.contains('σ')] = np.nan
                K_pillar.rename(columns=lambda x: x.replace('σ', 'k'), inplace=True)
    
                dict_K_σ = {v.replace('σ', 'k'): v for v in self.σ_pillar.columns if 'σ' in v}
                for date, row in K_pillar.iterrows():
                    # Scalars
                    S0 = self.fx_spot_rate
                    r_f=row['foreign_ccy_continuously_compounded_zero_rate']
                    r_d=row['domestic_ccy_continuously_compounded_zero_rate']
                    tau=row['tenor_years']                
                    Δ_convention=row['Δ_convention']
                    F=row['fx_forward_rate']  
                    # Arrays                 
                    cp = [1 if v[-4:] == 'call' else -1 if v[-3:] == 'put' else None for v in dict_K_σ.keys()]
                    Δ = [0.5 if v == 'k_atmΔneutral' else cp[i] * float(v.split('_')[1].split('Δ')[0]) / 100 for i,v in enumerate(dict_K_σ.keys())]
                    σ = self.σ_pillar.loc[date, dict_K_σ.values()].values
                                        
                    K = gk_solve_strike(S0=S0,tau=tau,r_f=r_f,r_d=r_d,σ=σ,Δ=Δ,Δ_convention=Δ_convention,F=F)
                    K_pillar.loc[date,dict_K_σ.keys()] = K
                                                
                self.K_pillar = K_pillar

            ###################################################################
            # Interpolate the term structure of volatility to get a daily volatility smile            
            dates = pd.date_range(min(self.σ_pillar.index),max(self.σ_pillar.index),freq='d')
            tenor_years = self.daycounter.year_fraction(self.curve_date,dates)
            df_interp = pd.DataFrame({'tenor_date':dates, 'tenor_years': tenor_years})
            df_pillar = self.σ_pillar.copy()

            # Merge to find closest smaller and larger tenors for each target tenor
            df_lower_pillar = pd.merge_asof(df_interp.sort_values('tenor_years'), df_pillar, on='tenor_date', direction='backward', suffixes=('_interp', '_pillar'))
            df_upper_pillar = pd.merge_asof(df_interp.sort_values('tenor_years'), df_pillar, on='tenor_date', direction='forward', suffixes=('_interp', '_pillar'))
                    
            # cols equals the volatility smile to be interpolated
            cols = [v for v in df_pillar.columns.to_list() if 'σ' in v] 
            
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
        # to do - need to rework so index is 0,1,2,3... and
        # 'expiry_date' and 'delivery_date' are two new columns.

        df['foreign_ccy_continuously_compounded_zero_rate'] = np.nan
        df['domestic_ccy_continuously_compounded_zero_rate'] = np.nan
        df['fx_forward_rate'] = np.nan
    
        df['foreign_ccy_continuously_compounded_zero_rate'] = self.foreign_zero_curve.zero_rate(dates=df.index,compounding_frequency='continuously').values       
        df['domestic_ccy_continuously_compounded_zero_rate'] = self.domestic_zero_curve.zero_rate(dates=df.index,compounding_frequency='continuously').values
        df['fx_forward_rate'] = interp_fx_forward_curve(self.fx_forward_curve, dates=df.index, date_type='expiry_date', flat_extrapolation=True).values
        return df


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
        - σ_t1 (float): At-the-money volatility at time (in years) to expiry date 1
        - t2 (float): Time to second maturity
        - σ_t2 (float): At-the-money volatility at time (in years) to expiry date 2
    
        Returns:
        - np.array: Forward volatility from time t1 to t2
        """
        tau = t2 - t1
        if np.any(tau == 0):
            warnings.warn("t2 and t1 are equal. NaN will be returned.")
        elif np.any(tau < 0):
            raise ValueError("t2 is less than t1. Please swap ")
    
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
                var0, vv, kappa, theta, rho, lambda_, IV, SSE = \
                    heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ, S, r_f, r_d, tau, cp, pricing_method=self.smile_interpolation_method)
            
                # Displaying output
                print(f'=== {tenor_name} calibration results ===')
                print(f'var0, vv, kappa, theta, rho: {var0, vv, kappa, theta, rho}')
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


    def interp_σ_smile_from_pillar_points(self, 
                         expiry_dates: pd.DatetimeIndex):
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
                
        df_unique_dates = self.Δ_σ_daily.loc[pd.DatetimeIndex(set(expiry_dates)),:].copy()
        df_unique_dates.loc[:, df_unique_dates.columns.str.contains('σ')] = np.nan
        df_unique_dates.rename(columns=lambda x: x.replace('σ', 'k'), inplace=True)

        helper_cols = ['tenor_date','tenor_name','tenor_years','fx_forward_rate','foreign_ccy_continuously_compounded_zero_rate','domestic_ccy_continuously_compounded_zero_rate','Δ_convention']
        dict_K_σ = {v.replace('σ', 'k'): v for v in self.Δ_σ_daily.columns if v not in helper_cols}
                
        # Interpolate volatility surface for each date in df
        for expiry_date, row in df_unique_dates.iterrows():
            
            if expiry_date not in self.K_σ_daily_smile_func.keys(): 
                
                # Scalars
                S0 = self.fx_spot_rate
                r_f=row['foreign_ccy_continuously_compounded_zero_rate']
                r_d=row['domestic_ccy_continuously_compounded_zero_rate']
                tau=row['tenor_years']                
                Δ_convention=row['Δ_convention']
                F=row['fx_forward_rate']     
                
                # Arrays
                cp = [1 if v[-4:] == 'call' else -1 if v[-3:] == 'put' else 1 for v in dict_K_σ.keys()]
                Δ = [0.5 if v == 'k_atmΔneutral' else cp[i] * float(v.split('_')[1].split('Δ')[0]) / 100 for i,v in enumerate(dict_K_σ.keys())]
                σ = self.Δ_σ_daily.loc[expiry_date, dict_K_σ.values()].values                
                            
                if self.smile_interpolation_method in ['univariate_spline','cubic_spline']:
                    K = gk_solve_strike(S0=S0,tau=tau,r_f=r_f,r_d=r_d,σ=σ,Δ=Δ,Δ_convention=Δ_convention,F=F)
                    df_unique_dates.loc[expiry_date,dict_K_σ.keys()] = K 
                    if self.smile_interpolation_method == 'univariate_spline':
                        self.K_σ_daily_smile_func[expiry_date] = InterpolatedUnivariateSpline(x=K, y=σ)
                    elif self.smile_interpolation_method == 'cubic_spline':
                        self.K_σ_daily_smile_func[expiry_date] = CubicSpline(x=K, y=σ)
                elif self.smile_interpolation_method[:6] == 'heston':
                    var0, vv, kappa, theta, rho, lambda_, IV, SSE = \
                        heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ, S0, r_f, r_d, tau, cp, pricing_method=self.smile_interpolation_method)
                    if SSE < 0.001:
                        result = {
                            'var0': var0,
                            'vv': vv,
                            'kappa': kappa,
                            'theta': theta,
                            'rho': rho,
                            'lambda_': lambda_,
                            'IV': IV,
                            'SSE': SSE
                            }
                        
                        self.K_σ_daily_smile_func[expiry_date] = result
                    else:
                        raise ValueError('SSE is a large value, ', round(SSE,4), ' heston fit at ', expiry_date,' is likely poor')
            
            
    def interp_σ_surface(self, 
                          expiry_dates: pd.DatetimeIndex, 
                          K: np.array, 
                          cp: np.array):
            
        K = np.atleast_1d(K).astype(float)
        cp = np.atleast_1d(cp).astype(float)
        
        assert expiry_dates.shape == K.shape
        assert expiry_dates.shape == cp.shape
    
        # Interpolate the volatility smile for the given expiries
        self.interp_σ_smile_from_pillar_points(expiry_dates)
    
    
        df = self.Δ_σ_daily.loc[expiry_dates,:].copy()
        
        cols_to_drop = [col for col in df.columns if 'σ' in col]
        df.drop(columns=cols_to_drop, inplace=True)
        df['K'] = K
        df['call_put'] = cp

        helper_cols = ['tenor_date','tenor_name','tenor_years','fx_forward_rate','foreign_ccy_continuously_compounded_zero_rate','domestic_ccy_continuously_compounded_zero_rate','Δ_convention']
        dict_K_σ = {v.replace('σ', 'k'): v for v in self.Δ_σ_daily.columns if v not in helper_cols}
                
        result = []
    
        # Get the implied vol for each strike and date combination
        # At some point need to make this over unique date, K
        for expiry_date,row in df.iterrows():
            
            K_target = row['K']
            
            if self.smile_interpolation_method in {'univariate_spline', 'cubic_spline'}:
                
                σ = self.K_σ_daily_smile_func[expiry_date](K_target)
                result.append(σ)
                
            elif self.smile_interpolation_method[:6] == 'heston':
                
                S0 = self.fx_spot_rate
                r_f = row['foreign_ccy_continuously_compounded_zero_rate']
                r_d = row['domestic_ccy_continuously_compounded_zero_rate']
                tau = row['tenor_years']                
                F = row['fx_forward_rate']               
                cp = row['call_put']

                var0 = self.K_σ_daily_smile_func[expiry_date]['var0']
                vv = self.K_σ_daily_smile_func[expiry_date]['vv']
                kappa = self.K_σ_daily_smile_func[expiry_date]['kappa']
                theta = self.K_σ_daily_smile_func[expiry_date]['theta']
                rho = self.K_σ_daily_smile_func[expiry_date]['rho']
                lambda_ = self.K_σ_daily_smile_func[expiry_date]['lambda_']
                
                if self.smile_interpolation_method == 'heston_analytical_1993':
                    X = heston1993_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K_target, var0, vv, kappa, theta, rho, lambda_)
                elif self.smile_interpolation_method == 'heston_carr_madan_gauss_kronrod_quadrature':     
                    X = heston_carr_madan_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K_target, var0, vv, kappa, theta, rho, integration_method=0)
                elif self.smile_interpolation_method == 'heston_carr_madan_fft_w_simpsons':
                    X = heston_carr_madan_price_fx_vanilla_european(S0, tau, r_f, r_d, cp, K_target, var0, vv, kappa, theta, rho, integration_method=1)
                elif self.smile_interpolation_method == 'heston_cosine':
                    X = heston_cosine_price_fx_vanilla_european(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K_target, var0=var0, vv=vv, kappa=kappa, theta=theta, rho=rho)
                
                implied_σ = gk_solve_implied_σ(S=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K_target, X=X, σ_guess=var0**2) 
                result.append(implied_σ)            
                        
        return np.array(result)

    
    def price_fx_vanilla_european(self, 
                                  expiry_datetimeindex,        
                                  K, 
                                  cp,
                                  analytical_greeks_flag,
                                  intrinsic_time_split_flag):
                      
        K = np.atleast_1d(K)
        cp = np.atleast_1d(cp)

        assert expiry_datetimeindex.shape == K.shape
        assert expiry_datetimeindex.shape == cp.shape

        mask = np.logical_and.reduce([expiry_datetimeindex >= self.σ_pillar.index.min(), expiry_datetimeindex <= self.σ_pillar.index.max()])
        σ = np.full(expiry_datetimeindex.shape, np.nan)
        
        if mask.any():
            self.interp_σ_smile_from_pillar_points(expiry_datetimeindex[mask])
            σ[mask] = self.interp_σ_surface(expiry_datetimeindex[mask], K, cp)
            
        r_f = self.foreign_zero_curve.zero_rate(expiry_datetimeindex,compounding_frequency='continuously').values
        r_d = self.domestic_zero_curve.zero_rate(expiry_datetimeindex,compounding_frequency='continuously').values
        F = interp_fx_forward_curve(self.fx_forward_curve, dates=expiry_datetimeindex, date_type='expiry_date', flat_extrapolation=True)

        results = gk_price(
            S0=self.fx_spot_rate,
            tau=self.daycounter.year_fraction(self.curve_date, expiry_datetimeindex),
            r_f=r_f,
            r_d=r_d,
            cp=cp,
            K=K,
            σ=σ,
            F=F,
            analytical_greeks_flag=analytical_greeks_flag,
            intrinsic_time_split_flag=intrinsic_time_split_flag
        )    
        
        results['market_data_inputs'] = pd.DataFrame({'σ': σ, 'r_f':r_f, 'r_d':r_d, 'F':F})

        return results
    
    
    def simulate_fx_rate_path(self,
                              date_grid=None,
                              nb_simulations=None,
                              flag_apply_antithetic_variates=None,
                              method: str='geometric_brownian_motion'):
        
        results = dict()
        
        if date_grid is None:
            date_grid = self.σ_pillar.index # create the date_grid based on the volatility input tenors
        
        if method == 'geometric_brownian_motion':
            
            if self.curve_date not in date_grid:
                date_grid = pd.DatetimeIndex([self.curve_date]).append(date_grid)
            
            date_grid = date_grid.unique().sort_values(ascending=True)
            tau = self.daycounter.year_fraction(self.curve_date,date_grid).values # this should be based on expiry_dates
            dt = tau[1:] - tau[:-1] 
            dt = np.insert(dt, 0, 0)
            
            fx_forward_rates_t2 = interp_fx_forward_curve(self.fx_forward_curve, dates=date_grid, date_type='delivery_date').values
            fx_forward_rates_t1 = np.zeros(shape=fx_forward_rates_t2.shape)
            fx_forward_rates_t1[0] = self.fx_spot_rate
            fx_forward_rates_t1[1:] = fx_forward_rates_t2[:-1].copy()
            period_drift = np.log(fx_forward_rates_t2 / fx_forward_rates_t1) / dt
            period_drift[0] = np.nan

            # tk need to update this to do with delivery dates no index's
            mask = np.logical_and.reduce([date_grid >= self.σ_pillar.index.min(), date_grid <= self.σ_pillar.index.max()])
            σ_atm = np.full(date_grid.shape, np.nan)
            if mask.any():
                σ_atm[mask] = self.Δ_σ_daily['σ_atmΔneutral'][date_grid[mask]]
                
            # Apply flat extrapolation for dates outside the volatility range
            σ_atm[date_grid < self.σ_pillar.index.min()] = self.Δ_σ_daily['σ_atmΔneutral'][date_grid[mask]][0]
            σ_atm[date_grid > self.σ_pillar.index.max()] = self.Δ_σ_daily['σ_atmΔneutral'][date_grid[mask]][-1]

            t1 = tau[:-1]
            t2 = tau[1:]
            σ_atm_t1 = σ_atm[:-1].copy()
            σ_atm_t2 = σ_atm[1:].copy()
            σ_forward = self.forward_volatility(t1, σ_atm_t1, t2, σ_atm_t2)

            # Setup data frame with the key market data inputs for easier review
            df_gbm_monte_carlo_market_data_inputs = pd.DataFrame()
            df_gbm_monte_carlo_market_data_inputs['dates'] = date_grid.values
            df_gbm_monte_carlo_market_data_inputs['tau'] = tau
            df_gbm_monte_carlo_market_data_inputs['dt'] = dt
            df_gbm_monte_carlo_market_data_inputs['fx_forward_rates'] = fx_forward_rates_t2
            df_gbm_monte_carlo_market_data_inputs['drift'] = period_drift
            df_gbm_monte_carlo_market_data_inputs['atm_volatility'] = σ_atm
            df_gbm_monte_carlo_market_data_inputs['forward_atm_volatility'] = np.insert(σ_forward, 0, np.nan)        
            results['gbm_monte_carlo_market_data_inputs'] = df_gbm_monte_carlo_market_data_inputs

            rand_nbs = generate_rand_nbs(nb_timesteps=len(date_grid)-1,
                                         nb_rand_vars=1,
                                         nb_simulations=nb_simulations,
                                         flag_apply_antithetic_variates=flag_apply_antithetic_variates)
        
            fx_rate_simulation_paths = simulate_gbm_path(initial_px=self.fx_spot_rate,
                                                         drift=period_drift[1:],
                                                         forward_volatility=σ_forward,
                                                         timestep_length=dt[1:],
                                                         rand_nbs=rand_nbs)
            results['fx_rate_simulation_paths'] = fx_rate_simulation_paths
            
            return results
        
        
        elif method == 'heston':
            
            # Interpolate the volatility smile for the given expiries 
            self.interp_σ_smile_from_pillar_points(expiry_dates=date_grid) # this function should be able to be called by delivery and expiry

            # Date grid is defined as the delivery date.
            # Need to update entire volatility surface with two indexe's, expiry and delivery date
            
            # It doesn't really make sense to simulate a rate path with just a Heston model. 
            # The standard heston model, is best for for pricing exotics / options where the only input is the termiminal fx rate
            # Need the Heston model to be linked to the term structure - i.e Local Stochastic volatility model
            
            results = {}
            
            for i,delivery_date in enumerate(date_grid):
                
                print(i, delivery_date)
                    
                tau = self.daycounter.year_fraction(self.curve_date,delivery_date)
                fx_forward_rate = interp_fx_forward_curve(self.fx_forward_curve, dates=pd.DatetimeIndex([delivery_date]), date_type='delivery_date').values
                mu = np.log(fx_forward_rate/self.fx_spot_rate) / tau
            
                var0 = self.K_σ_daily_smile_func[delivery_date]['var0']
                vv = self.K_σ_daily_smile_func[delivery_date]['vv']
                kappa = self.K_σ_daily_smile_func[delivery_date]['kappa']
                theta = self.K_σ_daily_smile_func[delivery_date]['theta']
                rho = self.K_σ_daily_smile_func[delivery_date]['rho']
                lambda_ = self.K_σ_daily_smile_func[delivery_date]['lambda_']  
                
                rand_nbs = generate_rand_nbs(nb_timesteps=100,
                                             nb_rand_vars=2,
                                             nb_simulations=10*1000,
                                             flag_apply_antithetic_variates=False)
                
                sim_results = simulate_heston(s0=self.fx_spot_rate,
                                mu=mu,
                                var0=var0,
                                vv=vv,
                                kappa=kappa,
                                theta=theta,
                                rho=rho,
                                tau=tau,
                                rand_nbs=rand_nbs,
                                method='quadratic_exponential')
                
                results[delivery_date] = sim_results
                
            return results



        
                          
        

