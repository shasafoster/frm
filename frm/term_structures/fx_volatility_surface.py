# -*- coding: utf-8 -*-
import os

from frm.utils import workday

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

from frm.enums import FXSmileInterpolationMethod, DayCountBasis, CompoundingFreq
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.monte_carlo_generic import generate_rand_nbs
from frm.pricing_engine.heston import heston_calibrate_vanilla_smile, heston_price_vanilla_european, heston_simulate
from frm.pricing_engine.garman_kohlhagen import garman_kohlhagen_price, garman_kohlhagen_solve_strike_from_delta, garman_kohlhagen_solve_implied_vol
from frm.pricing_engine.geometric_brownian_motion import simulate_gbm_path
from frm.pricing_engine.volatility_generic import forward_volatility, flat_forward_interp
from frm.term_structures.fx_helpers import (clean_vol_quotes_column_names,
                                            get_delta_smile_quote_details,
                                            check_delta_convention,
                                            fx_term_structure_helper,
                                            fx_forward_curve_helper,
                                            interp_fx_forward_curve_df,
                                            validate_ccy_pair,
                                            solve_call_put_quotes_from_strategy_quotes)

from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from frm.utils import year_frac, get_busdaycal, resolve_fx_curve_dates

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, InitVar
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


@dataclass
class FXVolatilitySurface:
    # Mandatory initialisation attributes
    domestic_ccy: str
    foreign_ccy: str
    fx_forward_curve_df: pd.DataFrame # Quotes in perspective of # of domestic currency units per foreign currency unit
    domestic_zero_curve: ZeroCurve
    foreign_zero_curve: ZeroCurve
    vol_quotes: InitVar[pd.DataFrame]

    #### Optional initialisation attributes

    # One of curve_date or spot_date must be provided. If spot_date is not provided, it will be calculated from curve_date and spot_offset or set per market convention
    curve_date: Optional[pd.Timestamp] = None
    spot_offset: Optional[int] = None
    spot_date: Optional[pd.Timestamp] = None

    cal: np.busdaycalendar = None
    day_count_basis: DayCountBasis = DayCountBasis.ACT_ACT
    smile_interpolation_method: FXSmileInterpolationMethod = FXSmileInterpolationMethod.UNIVARIATE_SPLINE

    # Non initialisation attributes set in __post_init__
    fx_spot_rate: float = field(init=False)
    vol_smile_pillar_df: pd.DataFrame = field(init=False)
    vol_smile_daily_df: pd.DataFrame = field(init=False)
    vol_smile_daily_func: dict = field(init=False)

    quotes_column_names: np.ndarray = field(init=False)
    quotes_signed_delta: np.ndarray = field(init=False)
    quotes_call_put_flag: np.ndarray = field(init=False)


    def __post_init__(self, vol_quotes):

        self.vol_smile_daily_func = {}

        self.ccy_pair, self.domestic_ccy, self.foreign_ccy =  validate_ccy_pair(self.foreign_ccy + self.domestic_ccy)

        if self.cal is None:
            self.cal = get_busdaycal([self.domestic_ccy, self.foreign_ccy])

        self.curve_date, self.spot_offset, self.spot_date = resolve_fx_curve_dates(self.ccy_pair,
                                                                                   self.cal,
                                                                                   self.curve_date,
                                                                                   self.spot_offset,
                                                                                   self.spot_date)
        assert self.curve_date == self.domestic_zero_curve.curve_date
        assert self.curve_date == self.foreign_zero_curve.curve_date

        self.fx_forward_curve_df = fx_forward_curve_helper(fx_forward_curve_df=self.fx_forward_curve_df,
                                                           curve_date=self.curve_date,
                                                           spot_offset=self.spot_offset,
                                                           cal=self.cal)
        mask = self.fx_forward_curve_df['tenor'] == 'sp'
        self.fx_spot_rate = self.fx_forward_curve_df.loc[mask, 'fx_forward_rate'].values[0]

        # Process volatility input and setup pillar dataframe and daily helper dataframe
        vol_quotes = clean_vol_quotes_column_names(vol_quotes)
        vol_quotes_call_put = solve_call_put_quotes_from_strategy_quotes(vol_quotes)
        delta_smile_quote_details = get_delta_smile_quote_details(vol_quotes_call_put)
        self.quotes_column_names = np.array(delta_smile_quote_details['quotes_column_names'].to_list())
        self.quotes_signed_delta = np.array(delta_smile_quote_details['quotes_signed_delta'].to_list())
        self.quotes_call_put_flag = np.array(delta_smile_quote_details['quotes_call_put_flag'].to_list())
        self.vol_smile_pillar_df = self._setup_vol_smile_pillar_df(vol_quotes_call_put)
        self.vol_smile_daily_df = self._setup_vol_smile_daily_df()
        self.strike_pillar_df = self._setup_strike_pillar()


    def _add_fx_ir_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['fx_forward_rate'] = interp_fx_forward_curve_df(fx_forward_curve_df=self.fx_forward_curve_df,
                                                           dates=df['expiry_date'],
                                                           date_type='fixing_date')
        df['domestic_zero_rate'] = self.domestic_zero_curve.get_zero_rates(dates=df['expiry_date'],
                                                                          compounding_freq=CompoundingFreq.CONTINUOUS)
        df['foreign_zero_rate'] = self.foreign_zero_curve.get_zero_rates(dates=df['expiry_date'],
                                                                        compounding_freq=CompoundingFreq.CONTINUOUS)
        return df


    def _setup_vol_smile_pillar_df(self, vol_smile_pillar_df):
            vol_smile_pillar_df['warnings'] = ''

            vol_smile_pillar_df = check_delta_convention(vol_smile_pillar_df, self.ccy_pair)
            vol_smile_pillar_df = fx_term_structure_helper(df=vol_smile_pillar_df,
                                                           curve_date=self.curve_date,
                                                           spot_offset=self.spot_offset,
                                                           cal=self.cal,
                                                           rate_set_date_str='expiry_date')
            vol_smile_pillar_df['expiry_years'] = year_frac(self.curve_date, vol_smile_pillar_df['expiry_date'], self.day_count_basis)
            vol_smile_pillar_df = self._add_fx_ir_to_df(vol_smile_pillar_df)
            column_order = ['warnings'] \
                           + (['tenor'] if 'tenor' in vol_smile_pillar_df.columns else []) \
                           + ['expiry_date', 'delivery_date', 'expiry_years', 'fx_forward_rate', 'domestic_zero_rate',
                              'foreign_zero_rate', 'delta_convention'] \
                           + list(self.quotes_column_names)
            return vol_smile_pillar_df[column_order]


    def _setup_vol_smile_daily_df(self):
        min_expiry = self.vol_smile_pillar_df['expiry_date'].min()
        max_expiry = self.vol_smile_pillar_df['expiry_date'].max()
        expiry_dates = pd.date_range(min_expiry, max_expiry, freq='d')
        delivery_dates = workday(expiry_dates, offset=self.spot_offset, cal=self.cal)

        expiry_years = year_frac(self.curve_date, expiry_dates, self.day_count_basis)

        vol_smile_daily_df = pd.DataFrame({'expiry_date_daily': expiry_dates,
                                           'delivery_date_daily': delivery_dates,
                                           'expiry_years_daily': expiry_years})

        vol_smile_pillar_df = self.vol_smile_pillar_df.copy()
        vol_smile_pillar_df['expiry_years'] = year_frac(self.curve_date, vol_smile_pillar_df['expiry_date'],self.day_count_basis)

        # Merge to find closest smaller and larger tenors for each target tenor
        lower_pillar = pd.merge_asof(vol_smile_daily_df, vol_smile_pillar_df, left_on='expiry_date_daily',
                                     right_on='expiry_date', direction='backward')
        upper_pillar = pd.merge_asof(vol_smile_daily_df, vol_smile_pillar_df, left_on='expiry_date_daily',
                                     right_on='expiry_date', direction='forward')

        # TODO: Adjust the interpolation to consider the delta convention where pillar points have different delta conventions
        vol_smile_daily_df['delta_convention'] = upper_pillar['delta_convention']

        # Convert to numpy for efficient calculations
        t1 = lower_pillar['expiry_years'].to_numpy()
        t2 = upper_pillar['expiry_years'].to_numpy()
        t = vol_smile_daily_df['expiry_years_daily'].to_numpy()
        t1 = t1[:, np.newaxis]
        t2 = t2[:, np.newaxis]
        t = t[:, np.newaxis]
        vol_t1 = lower_pillar[self.quotes_column_names].to_numpy()
        vol_t2 = upper_pillar[self.quotes_column_names].to_numpy()

        vol_smile_daily_df.rename(columns={'expiry_date_daily': 'expiry_date',
                                           'expiry_years_daily': 'expiry_years'}, inplace=True)

        # Interpolate the volatility smile
        vol_smile_daily_df[self.quotes_column_names] = flat_forward_interp(t1, vol_t1, t2, vol_t2, t)

        vol_smile_daily_df = self._add_fx_ir_to_df(vol_smile_daily_df)

        column_order = ['expiry_date', 'expiry_years', 'fx_forward_rate', 'domestic_zero_rate', 'foreign_zero_rate', 'delta_convention'] + list(self.quotes_column_names)

        return vol_smile_daily_df[column_order]


    def _setup_strike_pillar(self):
        strike_pillar_df = self.vol_smile_pillar_df.copy()
        strike_pillar_df.loc[:, self.quotes_column_names] = np.nan
        for i, row in strike_pillar_df.iterrows():
            strikes = garman_kohlhagen_solve_strike_from_delta(
                S0=self.fx_spot_rate,
                tau=row['expiry_years'],
                r_d=row['domestic_zero_rate'],
                r_f=row['foreign_zero_rate'],
                vol=self.vol_smile_pillar_df.loc[i, self.quotes_column_names].values,
                signed_delta=self.quotes_signed_delta,
                delta_convention=row['delta_convention'],
                F=row['fx_forward_rate'])
            strike_pillar_df.loc[i, self.quotes_column_names] = strikes
        return strike_pillar_df


    def _solve_vol_daily_smile_func(self, expiry_dates: pd.DatetimeIndex):
        """
        Internal method for solving the function definition for the volatility smile.
        The 'function' is a spline or the Heston parameters that allow for the interpolation of the volatility smile for any strike/delta.
        The function is solved for each date in `expiry_dates` and stored in the `vol_smile_daily_func` attribute.

        Parameters:
        dates (pd.DatetimeIndex): Array of dates to perform the σ surface interpolation for.

        Raises:
        ValueError: If the sum of squared errors (SSE) from the Heston fit exceeds a threshold, indicating a poor fit.
        """

        for expiry_date in expiry_dates.unique():
            if expiry_date not in self.vol_smile_daily_func.keys():

                mask = self.vol_smile_daily_df['expiry_date'] == expiry_date
                row = self.vol_smile_daily_df.loc[mask].copy()

                # Scalars
                S0 = self.fx_spot_rate
                r_f=row['foreign_zero_rate'].iloc[0]
                r_d=row['domestic_zero_rate'].iloc[0]
                tau=row['expiry_years'].iloc[0]
                delta_convention=row['delta_convention'].iloc[0]
                F=row['fx_forward_rate'].iloc[0]

                if F is not None:
                    # Use market forward rate and imply the curry basis-adjusted domestic interest rate
                    F = np.atleast_1d(F).astype(float)
                    r_d_basis_adj = np.log(F / S0) / tau + r_f  # from F = S0 * exp((r_d - r_f) * tau)
                    r = r_d_basis_adj
                    q = r_f
                else:
                    r = r_d
                    q = r_f

                # Arrays
                vol = row[self.quotes_column_names].iloc[0].values
                signed_delta = self.quotes_signed_delta
                cp = self.quotes_call_put_flag

                if self.smile_interpolation_method in [FXSmileInterpolationMethod.UNIVARIATE_SPLINE,
                                                       FXSmileInterpolationMethod.CUBIC_SPLINE]:
                    K = garman_kohlhagen_solve_strike_from_delta(
                        S0=S0, tau=tau, r_d=r_d, r_f=r_f, vol=vol, signed_delta=signed_delta, delta_convention=delta_convention, F=F)
                    if self.smile_interpolation_method == FXSmileInterpolationMethod.UNIVARIATE_SPLINE:
                        if len(self.quotes_column_names) < 3:
                            raise ValueError('Cannot fit InterpolatedUnivariateSpline with less than 3 points, please provide more points.')
                        degree = min(3, max(2, len(self.quotes_column_names) - 1))
                        self.vol_smile_daily_func[expiry_date] = InterpolatedUnivariateSpline(x=K, y=vol, k=degree)
                    elif self.smile_interpolation_method == FXSmileInterpolationMethod.CUBIC_SPLINE:
                        if len(self.quotes_column_names) < 4:
                            raise ValueError('Cannot fit CubicSpline with less than 4 points, please provide more points or use a different interpolation method')
                        self.vol_smile_daily_func[expiry_date] = CubicSpline(x=K, y=vol)
                elif self.smile_interpolation_method in [FXSmileInterpolationMethod.HESTON_1993,
                                                         FXSmileInterpolationMethod.HESTON_CARR_MADAN_GAUSS_KRONROD_QUADRATURE,
                                                         FXSmileInterpolationMethod.HESTON_CARR_MADAN_FFT_W_SIMPSONS,
                                                         FXSmileInterpolationMethod.HESTON_COSINE,
                                                         FXSmileInterpolationMethod.HESTON_LIPTON]:
                    var0, vv, kappa, theta, rho, lambda_, IV, SSE = \
                        heston_calibrate_vanilla_smile(volatility_quotes=vol,
                                                       delta_of_quotes=signed_delta,
                                                       S0=S0,
                                                       r=r,
                                                       q=q,
                                                       tau=tau,
                                                       cp=cp,
                                                       delta_convention=delta_convention,
                                                       pricing_method=self.smile_interpolation_method.value)

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
                        self.vol_smile_daily_func[expiry_date] = result
                    else:
                        raise ValueError('SSE is a large value, ', round(SSE, 4), ' heston fit at ', expiry_date,
                                         ' is likely poor')


    def interp_vol_surface(self,
                         expiry_dates: pd.DatetimeIndex,
                         K: np.ndarray,
                         cp: np.ndarray) -> pd.DataFrame:

        K = np.atleast_1d(K).astype(float)
        cp = np.atleast_1d(cp).astype(float)
        assert expiry_dates.shape == K.shape
        assert expiry_dates.shape == cp.shape

        self._solve_vol_daily_smile_func(expiry_dates)

        mask = self.vol_smile_daily_df['expiry_date'].isin(expiry_dates)
        tmp_df = self.vol_smile_daily_df.loc[mask].copy()
        tmp_df.set_index('expiry_date', inplace=True)
        interp_df = tmp_df.loc[expiry_dates].copy()
        interp_df.index.name = 'expiry_date'
        interp_df.reset_index(inplace=True, drop=False)
        interp_df['K'] = K
        interp_df['cp'] = cp
        interp_df['vol'] = np.nan

        for i,row in interp_df.iterrows():
            vol_smile_func = self.vol_smile_daily_func[row['expiry_date']]

            if self.smile_interpolation_method in [FXSmileInterpolationMethod.UNIVARIATE_SPLINE,
                                                    FXSmileInterpolationMethod.CUBIC_SPLINE]:
                interp_df.loc[i,'vol'] = vol_smile_func(K[i])
            elif self.smile_interpolation_method in [FXSmileInterpolationMethod.HESTON_1993,
                                                    FXSmileInterpolationMethod.HESTON_CARR_MADAN_GAUSS_KRONROD_QUADRATURE,
                                                    FXSmileInterpolationMethod.HESTON_CARR_MADAN_FFT_W_SIMPSONS,
                                                    FXSmileInterpolationMethod.HESTON_COSINE,
                                                    FXSmileInterpolationMethod.HESTON_LIPTON]:

                S0 = self.fx_spot_rate
                r_f = row['foreign_zero_rate']
                r_d = row['domestic_zero_rate']
                tau = row['expiry_years']
                F = row['fx_forward_rate']

                if F is not None:
                    # Use market forward rate and imply the curry basis-adjusted domestic interest rate
                    F = np.atleast_1d(F).astype(float)
                    r_d_basis_adj = np.log(F / S0) / tau + r_f  # from F = S0 * exp((r_d - r_f) * tau)
                    r = r_d_basis_adj
                    q = r_f
                else:
                    r = r_d
                    q = r_f

                X = heston_price_vanilla_european(
                    S0=S0, tau=tau, r=r, q=q, cp=cp[i], K=K[i], var0=vol_smile_func['var0'],
                    vv=vol_smile_func['vv'], kappa=vol_smile_func['kappa'], theta=vol_smile_func['theta'],
                    rho=vol_smile_func['rho'], lambda_=vol_smile_func['lambda_'],
                    pricing_method=self.smile_interpolation_method.value)

                interp_df.loc[i,'vol'] = garman_kohlhagen_solve_implied_vol(
                    S0=S0, tau=tau, r_d=r, r_f=q, cp=cp[i], K=K[i], X=X,
                    vol_guess=np.sqrt(vol_smile_func['var0']))

        return interp_df


    def price_vanilla_european(self,
                               expiry_dates: pd.DatetimeIndex,
                               K: [float, np.ndarray],
                               cp: [float, np.ndarray],
                               analytical_greeks_flag: bool=False,
                               numerical_greeks_flag: bool=False,
                               intrinsic_time_split_flag: bool=False) -> dict:

        K, cp = map(np.atleast_1d, (K, cp))
        assert expiry_dates.shape == K.shape
        assert expiry_dates.shape == cp.shape

        interp_df = self.interp_vol_surface(expiry_dates=expiry_dates, K=K, cp=cp)

        results = garman_kohlhagen_price(
            S0=self.fx_spot_rate,
            tau=interp_df['expiry_years'].values,
            r_d=interp_df['domestic_zero_rate'].values,
            r_f=interp_df['foreign_zero_rate'].values,
            cp=cp,
            K=K,
            vol=interp_df['vol'].values,
            F=interp_df['fx_forward_rate'].values,
            analytical_greeks=analytical_greeks_flag,
            numerical_greeks=numerical_greeks_flag,
            intrinsic_time_split=intrinsic_time_split_flag
        )

        results['market_data_inputs'] = pd.DataFrame({'vol': interp_df['vol'].values,
                                                      'r_d': interp_df['domestic_zero_rate'].values,
                                                      'r_f': interp_df['foreign_zero_rate'].values,
                                                      'F': interp_df['fx_forward_rate'].values})

        return results


    def simulate_gbm_fx_rate_path(self,
                                  delivery_date_grid: pd.DatetimeIndex,
                                  nb_simulations: Optional[int]=None,
                                  apply_antithetic_variates: bool=True):

        delivery_date_grid = delivery_date_grid.unique().sort_values(ascending=True)
        delivery_date_grid = delivery_date_grid.union(self.spot_date)
        fixing_date_grid = np.busday_offset(delivery_date_grid, offsets=-1*self.spot_offset, roll='preceding', busdaycal=self.cal)
        mask = self.vol_smile_daily_df['expiry_date'].isin(fixing_date_grid)
        vol_smile_daily_df = self.vol_smile_daily_df.loc[mask].copy()

        schedule = pd.DataFrame({
            'fixing_date': fixing_date_grid,
            'delivery_date': delivery_date_grid,
            'fixing_years': year_frac(self.curve_date, delivery_date_grid, self.day_count_basis).values,
            'domestic_zero_rates': vol_smile_daily_df['domestic_zero_rate'].values,
            'foreign_zero_rates': vol_smile_daily_df['foreign_zero_rate'].values,
            'fx_forward_rates:': vol_smile_daily_df['fx_forward_rate'].values,
        })

        if 'atm_forward' in self.vol_smile_daily_df.columns:
            schedule['atm_forward'] = vol_smile_daily_df['atm_forward'].values
        elif 'atm_delta_neutral' in self.vol_smile_daily_df.columns:
            schedule['volatility'] = vol_smile_daily_df['atm_delta_neutral'].values
        else:
            raise ValueError

        schedule['dt'] = schedule['fixing_years'].diff()
        schedule['dt'].iloc[0] = 0.0

        schedule['drift'] = np.log(schedule['fx_forward_rates:'].shift(-1) / schedule['fx_forward_rates:']) / schedule['dt']
        schedule['drift'].iloc[0] = np.nan

        schedule['forward_volatility'] = forward_volatility(
            t1=schedule['fixing_years'].values[:-1],
            vol_t1=schedule['volatility'].values[:-1],
            t2=schedule['fixing_years'].values[1:],
            vol_t2=schedule['volatility'].values[1:]
        )

        results = {'gbm_monte_carlo_market_data_inputs': schedule}

        rand_nbs = generate_rand_nbs(nb_steps=len(delivery_date_grid) - 1,
                                     nb_rand_vars=1,
                                     nb_simulations=nb_simulations,
                                     apply_antithetic_variates=apply_antithetic_variates)

        fx_rate_simulation_paths = simulate_gbm_path(initial_px=self.fx_spot_rate,
                                                     drift=schedule['drift'].values[1:],
                                                     forward_volatility=schedule['forward_volatility'].values[1:],
                                                     timestep_length=schedule['dt'].values[1:],
                                                     rand_nbs=rand_nbs)

        results['fx_rate_simulation_paths'] = fx_rate_simulation_paths

        return results


    def simulate_heston_fx_rate_path(self,
                                     delivery_date_grid,
                                     nb_simulations=None,
                                     flag_apply_antithetic_variates=True):


        delivery_date_grid = delivery_date_grid.unique().sort_values(ascending=True)
        delivery_date_grid = delivery_date_grid.union(self.spot_date)
        fixing_date_grid = np.busday_offset(delivery_date_grid, offsets=-1*self.spot_offset, roll='preceding', busdaycal=self.cal)
        #mask = self.vol_smile_daily_df['expiry_date'].isin(fixing_date_grid)
        #vol_smile_daily_df = self.vol_smile_daily_df.loc[mask].copy()
        self._solve_vol_daily_smile_func(fixing_date_grid)

        # It doesn't really make sense to simulate a rate path using the Heston model.
        # The heston model is designed to for fit the volatility smile at a certain point in time, not the term structure of volatility along the path.

        results = {}

        for i, (fixing_date, delivery_date) in enumerate(zip(fixing_date_grid,delivery_date_grid)):
            print(i, fixing_date, delivery_date)

            expiry_years = year_frac(self.curve_date, fixing_date, self.day_count_basis)
            fx_forward_rate = interp_fx_forward_curve_df(self.fx_forward_curve_df,
                                                         dates=fixing_date,
                                                         date_type='fixing_date').values

            mu = np.log(fx_forward_rate / self.fx_spot_rate) / expiry_years

            var0 = self.vol_smile_daily_func[fixing_date]['var0']
            vv = self.vol_smile_daily_func[fixing_date]['vv']
            kappa = self.vol_smile_daily_func[fixing_date]['kappa']
            theta = self.vol_smile_daily_func[fixing_date]['theta']
            rho = self.vol_smile_daily_func[fixing_date]['rho']
            # lambda_ is not used in the simulation, used in the calibration only.
            # lambda_ = self.daily_volatility_smile_func[fixing_date]['lambda_']

            rand_nbs = generate_rand_nbs(
                nb_steps=100,
                nb_rand_vars=2,
                nb_simulations=10 * 1000,
                apply_antithetic_variates=False)

            sim_results = heston_simulate(
                S0=self.fx_spot_rate,
                mu=mu,
                var0=var0,
                vv=vv,
                kappa=kappa,
                theta=theta,
                rho=rho,
                tau=expiry_years,
                rand_nbs=rand_nbs)

            results[delivery_date] = sim_results

        return results


    def plot_smile(self, expiry_date: pd.Timestamp):

        self._solve_vol_daily_smile_func(pd.DatetimeIndex([expiry_date]))

        vol_daily_smile_df = self.vol_smile_daily_df.loc[self.vol_smile_daily_df['expiry_date'] == expiry_date]

        delta_convention = vol_daily_smile_df['delta_convention'].iloc[0]
        vol_pillar = vol_daily_smile_df[self.quotes_column_names].iloc[0]

        pillar_strikes = garman_kohlhagen_solve_strike_from_delta(S0=self.fx_spot_rate,
                                  tau=vol_daily_smile_df['expiry_years'].iloc[0],
                                  r_d=vol_daily_smile_df['domestic_zero_rate'].iloc[0],
                                  r_f=vol_daily_smile_df['foreign_zero_rate'].iloc[0],
                                  vol=vol_pillar,
                                  signed_delta=self.quotes_signed_delta,
                                  delta_convention=delta_convention,
                                  F=vol_daily_smile_df['fx_forward_rate'].iloc[0],
                                  atm_delta_convention='forward')

        nb_strikes = 100
        strike_delta_neutral = pillar_strikes[self.quotes_column_names == 'atm_delta_neutral'].item()
        strikes_put = np.linspace(np.min(pillar_strikes), strike_delta_neutral, nb_strikes)
        strikes_call = np.linspace(strike_delta_neutral, np.max(pillar_strikes), nb_strikes)
        strikes_array = np.concatenate([strikes_put, strikes_call])
        call_put_array = np.concatenate([-1 * np.ones(nb_strikes), np.ones(nb_strikes)])
        expiry_dates = pd.DatetimeIndex([expiry_date for _ in range(len(strikes_array))])

        result = self.price_vanilla_european(expiry_dates=expiry_dates,
                                             K=strikes_array,
                                             cp=call_put_array,
                                             analytical_greeks_flag=True)

        if delta_convention == 'regular_spot':
            delta_str = 'spot_delta'
        elif delta_convention == 'regular_forward':
            delta_str = 'forward_delta'

        delta = result['analytical_greeks'][delta_str]
        vol = result['market_data_inputs']['vol']

        delta_pillar_for_plot = np.full(self.quotes_signed_delta.shape, np.nan)
        midpoint = len(delta_pillar_for_plot) // 2
        delta_pillar_for_plot[:midpoint] = np.abs(self.quotes_signed_delta[:midpoint])
        delta_pillar_for_plot[midpoint:] = 1.0 - self.quotes_signed_delta[midpoint:]

        delta_for_plot = np.full(delta.shape, np.nan)
        delta_for_plot[:nb_strikes] = np.abs(delta[:nb_strikes])
        delta_for_plot[nb_strikes:] = 1.0 - delta[nb_strikes:]

        # Plot
        plt.plot(100 * delta_pillar_for_plot, vol_pillar, 'x', label='Pillar Volatilities')
        plt.plot(100 * delta_for_plot, vol, label='Fitted Smile')  # X markers

        tenor = np.round(vol_daily_smile_df['expiry_years'].iloc[0], 2)

        plt.title(f'Volatility Smile for {tenor}Y ({expiry_date.date()})\n'
                  f'Smile interpolation method: {self.smile_interpolation_method.value}')
        plt.xlabel('Delta:   Puts: (0,0.5). ATM Δ Neutral: 0.5. Calls: (0.5,1)')
        plt.ylabel('Volatility')
        plt.xlim([0, 100])

        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))  # 1.0 means 100%

        plt.legend()
        plt.show()
