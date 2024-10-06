# -*- coding: utf-8 -*-
import os


from frm.enums.term_structures import FXSmileInterpolationMethod

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 
    

from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.monte_carlo_generic import generate_rand_nbs
from frm.pricing_engine.heston import heston_calibrate_vanilla_smile, heston_price_vanilla_european, simulate_heston
from frm.pricing_engine.garman_kohlhagen import gk_price, gk_solve_strike, gk_solve_implied_volatility
from frm.pricing_engine.geometric_brownian_motion import simulate_gbm_path

from frm.term_structures.fx_volatility_surface_helpers import (clean_and_extract_vol_smile_columns,
                                                               check_delta_convention,
                                                               fx_term_structure_helper,
                                                               fx_forward_curve_helper,
                                                               interp_fx_forward_curve_df,
                                                               resolve_fx_curve_dates,
                                                               validate_ccy_pair,
                                                               forward_volatility,
                                                               flat_forward_interp)
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from frm.utils.daycount import year_fraction
from frm.utils.business_day_calendar import get_busdaycal
from frm.enums.utils import DayCountBasis, CompoundingFrequency

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
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

    vol_quotes_call_put: Optional[pd.DataFrame] = None
    vol_quotes_strategy: Optional[pd.DataFrame] = None # Have a helper function to convert this to vol_quotes_call_put.

    #### Optional initialisation attributes

    # One of curve_date or spot_date must be provided. If spot_date is not provided, it will be calculated from curve_date and spot_offset or set per market convention
    curve_date: Optional[pd.Timestamp] = None
    spot_offset: Optional[int] = None
    spot_date: Optional[pd.Timestamp] = None

    busdaycal: np.busdaycalendar = None
    day_count_basis: DayCountBasis = DayCountBasis.ACT_ACT
    smile_interpolation_method: FXSmileInterpolationMethod = FXSmileInterpolationMethod.HESTON_COSINE

    # Non initialisation attributes set in __post_init__
    fx_spot_rate: float = field(init=False)
    vol_smile_pillar_df: pd.DataFrame = field(init=False)
    vol_smile_daily_df: pd.DataFrame = field(init=False)
    vol_smile_daily_func: dict = field(init=False)

    quotes_column_names: np.ndarray = field(init=False)
    quotes_signed_delta: np.ndarray = field(init=False)
    quotes_call_put_flag: np.ndarray = field(init=False)


    def __post_init__(self):

        self.vol_smile_daily_func = {}

        self.ccy_pair, self.domestic_ccy, self.foreign_ccy =  validate_ccy_pair(self.foreign_ccy + self.domestic_ccy)

        if self.busdaycal is None:
            self.busdaycal = get_busdaycal([self.domestic_ccy, self.foreign_ccy])

        self.curve_date, self.spot_offset, self.spot_date = resolve_fx_curve_dates(self.ccy_pair,
                                                                                   self.busdaycal,
                                                                                   self.curve_date,
                                                                                   self.spot_offset,
                                                                                   self.spot_date)
        assert self.curve_date == self.domestic_zero_curve.curve_date
        assert self.curve_date == self.foreign_zero_curve.curve_date

        self.fx_forward_curve_df = fx_forward_curve_helper(fx_forward_curve_df=self.fx_forward_curve_df,
                                                           curve_date=self.curve_date,
                                                           spot_offset=self.spot_offset,
                                                           busdaycal=self.busdaycal)
        mask = self.fx_forward_curve_df['tenor'] == 'sp'
        self.fx_spot_rate = self.fx_forward_curve_df.loc[mask, 'fx_forward_rate'].values[0]

        # Process volatility input and setup pillar dataframe and daily helper dataframe
        vol_smile_pillar_df, vol_smile_quote_details = clean_and_extract_vol_smile_columns(self.vol_quotes_call_put)
        self.quotes_column_names = np.array(vol_smile_quote_details['quotes_column_names'].to_list())
        self.quotes_signed_delta = np.array(vol_smile_quote_details['quotes_signed_delta'].to_list())
        self.quotes_call_put_flag = np.array(vol_smile_quote_details['quotes_call_put_flag'].to_list())

        self.vol_smile_pillar_df = self._setup_vol_smile_pillar_df(vol_smile_pillar_df)
        self.vol_smile_daily_df = self._setup_vol_smile_daily_df()
        self.strike_pillar_df = self._setup_strike_pillar()



    def _add_fx_ir_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['fx_forward_rate'] = interp_fx_forward_curve_df(fx_forward_curve_df=self.fx_forward_curve_df,
                                                           dates=df['expiry_date'],
                                                           date_type='fixing_date')
        df['domestic_zero_rate'] = self.domestic_zero_curve.get_zero_rate(dates=df['expiry_date'],
                                                                          compounding_frequency=CompoundingFrequency.CONTINUOUS)
        df['foreign_zero_rate'] = self.foreign_zero_curve.get_zero_rate(dates=df['expiry_date'],
                                                                        compounding_frequency=CompoundingFrequency.CONTINUOUS)
        return df

    def _setup_vol_smile_pillar_df(self, vol_smile_pillar_df):
            vol_smile_pillar_df['warnings'] = ''

            vol_smile_pillar_df = check_delta_convention(vol_smile_pillar_df, self.ccy_pair)
            vol_smile_pillar_df = fx_term_structure_helper(df=vol_smile_pillar_df,
                                                           curve_date=self.curve_date,
                                                           spot_offset=self.spot_offset,
                                                           busdaycal=self.busdaycal,
                                                           rate_set_date_str='expiry_date')
            vol_smile_pillar_df['expiry_years'] = year_fraction(start_date=self.curve_date,
                                                                end_date=vol_smile_pillar_df['expiry_date'],
                                                                day_count_basis=self.day_count_basis)
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
        expiry_dates_np = expiry_dates.to_numpy(dtype='datetime64[D]')
        delivery_dates = np.busday_offset(dates=expiry_dates_np, offsets=self.spot_offset, roll='following', busdaycal=self.busdaycal)

        expiry_years = year_fraction(self.curve_date, expiry_dates, self.day_count_basis)

        vol_smile_daily_df = pd.DataFrame({'expiry_date_daily': expiry_dates,
                                           'delivery_date_daily': delivery_dates,
                                           'expiry_years_daily': expiry_years})

        vol_smile_pillar_df = self.vol_smile_pillar_df.copy()
        vol_smile_pillar_df['expiry_years'] = year_fraction(start_date=self.curve_date,
                                                            end_date=vol_smile_pillar_df['expiry_date'],
                                                            day_count_basis=self.day_count_basis)

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
            strikes = gk_solve_strike(S0=self.fx_spot_rate,
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
                    K = gk_solve_strike(S0=S0, tau=tau, r_d=r_d, r_f=r_f, vol=vol, signed_delta=signed_delta, delta_convention=delta_convention, F=F)
                    if self.smile_interpolation_method == FXSmileInterpolationMethod.UNIVARIATE_SPLINE:
                        self.vol_smile_daily_func[expiry_date] = InterpolatedUnivariateSpline(x=K, y=vol)
                    elif self.smile_interpolation_method == FXSmileInterpolationMethod.CUBIC_SPLINE:
                        self.vol_smile_daily_func[expiry_date] = CubicSpline(x=K, y=vol)
                elif self.smile_interpolation_method in [FXSmileInterpolationMethod.HESTON_ANALYTICAL_1993,
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
            elif self.smile_interpolation_method in [FXSmileInterpolationMethod.HESTON_ANALYTICAL_1993,
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
                    S0=S0,
                    tau=tau,
                    r=r,
                    q=q,
                    cp=cp[i],
                    K=K[i],
                    var0=vol_smile_func['var0'],
                    vv=vol_smile_func['vv'],
                    kappa=vol_smile_func['kappa'],
                    theta=vol_smile_func['theta'],
                    rho=vol_smile_func['rho'],
                    lambda_=vol_smile_func['lambda_'],
                    pricing_method=self.smile_interpolation_method.value
                )

                interp_df.loc[i,'vol'] = gk_solve_implied_volatility(
                    S0=self.fx_spot_rate,
                    tau=tau,
                    r_d=r,
                    r_f=q,
                    cp=cp[i],
                    K=K[i],
                    X=X,
                    vol_guess=np.sqrt(vol_smile_func['var0']))

        return interp_df


    def price_vanilla_european(self,
                               expiry_dates: pd.DatetimeIndex,
                               K: [float, np.ndarray],
                               cp: [float, np.ndarray],
                               analytical_greeks_flag: bool=False,
                               numerical_greeks_flag: bool=False,
                               intrinsic_time_split_flag: bool=False) -> dict:

        K = np.atleast_1d(K)
        cp = np.atleast_1d(cp)
        assert expiry_dates.shape == K.shape
        assert expiry_dates.shape == cp.shape

        interp_df = self.interp_vol_surface(expiry_dates=expiry_dates, K=K, cp=cp)

        results = gk_price(
            S0=self.fx_spot_rate,
            tau=interp_df['expiry_years'].values,
            r_d=interp_df['domestic_zero_rate'].values,
            r_f=interp_df['foreign_zero_rate'].values,
            cp=cp,
            K=K,
            vol=interp_df['vol'].values,
            F=interp_df['fx_forward_rate'].values,
            analytical_greeks_flag=analytical_greeks_flag,
            numerical_greeks_flag=numerical_greeks_flag,
            intrinsic_time_split_flag=intrinsic_time_split_flag
        )

        results['market_data_inputs'] = pd.DataFrame({'vol': interp_df['vol'].values,
                                                      'r_d': interp_df['domestic_zero_rate'].values,
                                                      'r_f': interp_df['foreign_zero_rate'].values,
                                                      'F': interp_df['fx_forward_rate'].values})

        return results


    def simulate_gbm_fx_rate_path(self,
                                  delivery_date_grid: pd.DatetimeIndex,
                                  nb_simulations: Optional[int]=None,
                                  flag_apply_antithetic_variates: bool=True):

        delivery_date_grid = delivery_date_grid.unique().sort_values(ascending=True)
        delivery_date_grid = delivery_date_grid.union(self.spot_date)
        fixing_date_grid = np.busday_offset(delivery_date_grid, offsets=-1*self.spot_offset, roll='preceding', busdaycal=self.busdaycal)
        mask = self.vol_smile_daily_df['expiry_date'].isin(fixing_date_grid)
        vol_smile_daily_df = self.vol_smile_daily_df.loc[mask].copy()

        schedule = pd.DataFrame({
            'fixing_date': fixing_date_grid,
            'delivery_date': delivery_date_grid,
            'fixing_years': year_fraction(self.curve_date, delivery_date_grid, self.day_count_basis).values,
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
                                     flag_apply_antithetic_variates=flag_apply_antithetic_variates)

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
        fixing_date_grid = np.busday_offset(delivery_date_grid, offsets=-1*self.spot_offset, roll='preceding', busdaycal=self.busdaycal)
        #mask = self.vol_smile_daily_df['expiry_date'].isin(fixing_date_grid)
        #vol_smile_daily_df = self.vol_smile_daily_df.loc[mask].copy()
        self._solve_vol_daily_smile_func(fixing_date_grid)

        # It doesn't really make sense to simulate a rate path using the Heston model.
        # The heston model is designed to for fit the volatility smile at a certain point in time, not the term structure of volatility along the path.

        results = {}

        for i, (fixing_date, delivery_date) in enumerate(zip(fixing_date_grid,delivery_date_grid)):
            print(i, fixing_date, delivery_date)

            expiry_years = year_fraction(self.curve_date, fixing_date, self.day_count_basis)
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
                flag_apply_antithetic_variates=False)

            sim_results = simulate_heston(
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

        pass




        # self.interp_vol_surface(expiry_dates=pd.DatetimeIndex([expiry_date]),
        #                         K=deltas,
        #                         cp=np.sign(deltas))
        #
        #
        # self.vol_smile_daily_df[]
        #
        # mask = self.vol_smile_daily_df['expiry_date'] == expiry_date
        #
        # def convert_column_name_to_delta_for_plt(input_str):
        #     if 'put' in input_str:
        #         return int(input_str.split('Δ')[0][2:])
        #     elif 'call' in input_str:
        #         return 100 - int(input_str.split('Δ')[0][2:])
        #     elif 'neutral' in input_str:
        #         return 50
        #
        # def convert_column_name_to_delta(input_str):
        #     if 'put' in input_str:
        #         return -1 * int(input_str.split('Δ')[0][2:]) / 100
        #     elif 'call' in input_str:
        #         return int(input_str.split('Δ')[0][2:]) / 100
        #     elif 'neutral' in input_str:
        #         return 0.5
        #
        # i = 0
        # for date, row in self.σ_pillar.iterrows():
        #
        #     # Scalars
        #     tenor_name = row['tenor']
        #     delta_convention = row['delta_convention']
        #     S0 = self.fx_spot_rate
        #     r_f = row['foreign_zero_rate']
        #     r_d = row['domestic_zero_rate']
        #     tau = row['expiry_years']
        #     F = row['fx_forward_rate']
        #
        #     if F is not None:
        #         # Use market forward rate and imply the curry basis-adjusted domestic interest rate
        #         F = np.atleast_1d(F).astype(float)
        #         r_d_basis_adj = np.log(F / S0) / tau + r_f  # from F = S0 * exp((r_d - r_f) * tau)
        #         r = r_d_basis_adj
        #         q = r_f
        #     else:
        #         # By interest rate parity
        #         F = S0 * np.exp((r_d - r_f) * tau)
        #         r = r_d  # noqa: F841
        #         q = r_f  # noqa: F841
        #
        #     # Arrays
        #     Δ = [convert_column_name_to_delta(c) for c in self.σ_pillar.columns if c[0] == 'σ']
        #     Δ_for_plt = [convert_column_name_to_delta_for_plt(c) for c in self.σ_pillar.columns if c[0] == 'σ']
        #     cp = np.sign(Δ)
        #     σ = row[[c for c in self.σ_pillar.columns if c[0] == 'σ']].values
        #
        #     if self.smile_interpolation_method[:6] == 'heston':
        #         var0, vv, kappa, theta, rho, lambda_, IV, SSE = \
        #             heston_calibrate_vanilla_smile(Δ, delta_convention, σ, S0, r_f, r_d, tau, cp,
        #                                            pricing_method=self.smile_interpolation_method)
        #
        #         # Displaying output
        #         print(f'=== {tenor_name} calibration results ===')
        #         print(f'var0, vv, kappa, theta, rho: {var0, vv, kappa, theta, rho}')
        #         print(f'SSE {SSE * 100}')
        #
        #         # Plotting
        #         i += 1
        #         plt.figure(i + 1)
        #         plt.plot(Δ_for_plt, σ * 100, 'ko-', linewidth=1)
        #         plt.plot(Δ_for_plt, IV * 100, 'rs--', linewidth=1)
        #         plt.legend([f'{tenor_name} smile', 'Heston fit'], loc='upper right')
        #         plt.xlabel('Delta [%]')
        #         plt.ylabel('Implied volatility [%]')
        #         plt.xticks(Δ_for_plt)
        #         plt.title(self.smile_interpolation_method)
        #         plt.show()
        #
        #     else:
        #         raise ValueError




# FX Volatility Surface

# At minimum (out of the curve_date, spot_offset & spot_date) the curve_date or spot_date must be specified
# The two parameters can be implied or set per market convention.
# If all three parameters are provided, they will be validated for consistency.
curve_date = pd.Timestamp('2023-06-30') # Friday 30-June-2023

# Market convention is quoted as AUD/USD (1 AUD = x USD)
ccy_pair = 'audusd'
ccy_pair, domestic_ccy, foreign_ccy = validate_ccy_pair(ccy_pair)

# The FX volatility surface is defined as pandas DataFrame.
# Each row of the dataframe defines to the volatility smile for the rows tenor.
# Each column of the dataframe corresponds to a given delta's term structure.
# The dataframe must have
# (i) at least one of 'tenor', 'expiry_date' or 'delivery_date' as columns to define the term structure.
# (ii) the 'delta_convention' column to specify the delta convention for the volatility smile.
# The volatility smile column names are defined by 'X_delta_call' and 'X_delta_put' where X is the delta value with 1<X<50.
# The 'atm_delta_neutral' column is the at-the-money volatility.
vol_surface_data = {
    'tenor': ['1W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y'],
    'delta_convention': ['regular_spot'] * len(['1W', '1M', '2M', '3M', '6M', '9M', '1Y']) \
                      + ['regular_forward'] * len(['2Y', '3Y', '4Y', '5Y', '7Y', '10Y']),
    '5Δ_put': np.array([11.943, 11.145, 11.514, 11.834, 12.402, 12.996, 13.546, 14.159, 14.683, 15.161, 15.477, 16.703, 17.042])/100,
    '10Δ_put': np.array([11.656, 10.786, 10.990, 11.200, 11.599, 12.006, 12.361, 12.824, 13.215, 13.618, 13.875, 14.603, 14.966])/100,
    '15Δ_put': np.array([11.481, 10.568, 10.683, 10.832, 11.141, 11.455, 11.718, 12.123, 12.452, 12.808, 13.032, 13.573, 13.948])/100,
    '20Δ_put': np.array([11.350, 10.405, 10.457, 10.564, 10.812, 11.065, 11.270, 11.645, 11.932, 12.254, 12.455, 12.888, 13.267])/100,
    '25Δ_put': np.array([11.240, 10.271, 10.275, 10.350, 10.550, 10.758, 10.920, 11.273, 11.530, 11.823, 12.005, 12.360, 12.739])/100,
    '30Δ_put': np.array([11.140, 10.152, 10.116, 10.165, 10.326, 10.496, 10.624, 10.961, 11.190, 11.457, 11.620, 11.909, 12.283])/100,
    'atmΔ_neutral': np.array([10.868, 9.814, 9.684, 9.670, 9.745, 9.848, 9.922, 10.150, 10.300, 10.488, 10.600, 10.750, 11.100])/100,
    '30Δ_call': np.array([10.722, 9.598, 9.441, 9.400, 9.440, 9.535, 9.610, 9.831, 9.934, 10.076, 10.166, 10.369, 10.683])/100,
    '25Δ_call': np.array([10.704, 9.559, 9.407, 9.364, 9.404, 9.508, 9.596, 9.833, 9.930, 10.065, 10.155, 10.407, 10.711]) / 100,
    '20Δ_call': np.array([10.683, 9.516, 9.368, 9.323, 9.365, 9.481, 9.585, 9.846, 9.943, 10.071, 10.160, 10.478, 10.774]) / 100,
    '15Δ_call': np.array([10.663, 9.471, 9.331, 9.287, 9.335, 9.470, 9.599, 9.893, 9.998, 10.117, 10.206, 10.615, 10.904]) / 100,
    '10Δ_call': np.array([10.643, 9.421, 9.296, 9.256, 9.318, 9.486, 9.657, 10.004, 10.126, 10.236, 10.325, 10.877, 11.157])/100,
    '5Δ_call': np.array([10.628, 9.365, 9.274, 9.249, 9.349, 9.587, 9.847, 10.306, 10.474, 10.568, 10.660, 11.528, 11.787])/100
}
vol_quotes_call_put = pd.DataFrame(vol_surface_data)

# FX Forward Rates
# The FX forward curve must be specified as a dataframe with
# (i) least one of 'tenor', 'fixing_date' or 'delivery_date' (to define the term structure)
# (ii) the 'fx_forward_rate' column
fx_forward_curve_data = {
    'tenor': ['SP','1D','1W','2W','3W','1M','2M','3M','6M','9M','1Y','15M','18M','2Y','3Y','4Y','5Y','7Y','10Y'],
    'fx_forward_rate': [0.6629, 0.6629, 0.6630, 0.6631, 0.6633, 0.6635, 0.6640, 0.6646,0.6661, 0.6673, 0.6680,
                        0.6681, 0.6679, 0.6668, 0.6631, 0.6591, 0.6525, 0.6358, 0.6084],
}
fx_forward_curve_df = pd.DataFrame(fx_forward_curve_data)


# Interest rates
data = {
    'tenor' : ['1 Day', '1 Week', '2 Week', '3 Week', '1 Month', '2 Month', '3 Month', '6 Month', '9 Month', '1 Year', '15 Month', '18 Month', '2 Year', '3 Year', '4 Year', '5 Year', '7 Year', '10 Year'],
    'domestic_zero_rate': [5.055, 5.059, 5.063, 5.065, 5.142, 5.221, 5.270, 5.390, 5.432, 5.381, 5.248, 5.122, 4.812, 4.373, 4.087, 3.900, 3.690, 3.550],
    'foreign_zero_rate': [4.156, 4.153, 4.150, 4.138, 4.194, 4.274, 4.335, 4.479, 4.595, 4.660, 4.674, 4.673, 4.578, 4.427, 4.295, 4.285, 4.362, 4.493],
}
zero_rate_df = pd.DataFrame(data)
zero_rate_df['foreign_zero_rate'] = zero_rate_df['foreign_zero_rate'] / 100
zero_rate_df['domestic_zero_rate'] = zero_rate_df['domestic_zero_rate'] / 100

zero_rate_domestic_df = zero_rate_df[['tenor', 'domestic_zero_rate']].copy()
zero_rate_domestic_df.rename(columns={'domestic_zero_rate': 'zero_rate'}, inplace=True)

zero_rate_foreign_df = zero_rate_df[['tenor', 'foreign_zero_rate']].copy()
zero_rate_foreign_df.rename(columns={'foreign_zero_rate': 'zero_rate'}, inplace=True)

busdaycal_domestic = get_busdaycal(domestic_ccy)
busdaycal_foreign = get_busdaycal(foreign_ccy)

zero_curve_domestic = ZeroCurve(data=zero_rate_domestic_df, curve_date=pd.Timestamp('2023-06-30'), day_count_basis=DayCountBasis.ACT_360, compounding_frequency=CompoundingFrequency.CONTINUOUS, busdaycal=busdaycal_domestic)
zero_curve_foreign = ZeroCurve(data=zero_rate_foreign_df, curve_date=pd.Timestamp('2023-06-30'), day_count_basis=DayCountBasis.ACT_365, compounding_frequency=CompoundingFrequency.CONTINUOUS, busdaycal=busdaycal_foreign)
del zero_rate_domestic_df, zero_rate_foreign_df, zero_rate_df, busdaycal_domestic, busdaycal_foreign

# If no business day calendar is specified, create it based on holiday calendars of both currencies
busdaycal = None
if busdaycal is None:
    busdaycal = get_busdaycal([domestic_ccy, foreign_ccy])


surf = FXVolatilitySurface(domestic_ccy=domestic_ccy,
                           foreign_ccy=foreign_ccy,
                           fx_forward_curve_df=fx_forward_curve_df,
                           domestic_zero_curve=zero_curve_domestic,
                           foreign_zero_curve=zero_curve_foreign,
                           vol_quotes_call_put=vol_quotes_call_put,
                           curve_date=curve_date,
                           busdaycal=busdaycal,
                           smile_interpolation_method=FXSmileInterpolationMethod.HESTON_COSINE)

expiry_dates = pd.DatetimeIndex(['2025-06-30', '2025-12-31', '2026-06-30','2026-06-30','2026-06-30'])
K = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
cp = np.array([-1, -1, 1, 1, 1])

#df = vol_surface.interp_vol_surface(expiry_dates, K, cp)

surf = vol_surface

# Test the strike solve by comparing the analytical delta to the signed delta
# Excluding, the atm delta neutral quotes, the volatility quotes are given as spot delta for <= 1Y tenors and forward delta for > 1Y tenors.
# The atm delta neutral quotes are all forward delta quotes.

#%%

# Plot smile for any date

expiry_date = pd.Timestamp('2025-06-30')
vol_daily_smile_df = surf.vol_smile_daily_df.loc[surf.vol_smile_daily_df['expiry_date'] == expiry_date]
vol_daily_smile_func = surf.vol_smile_daily_func[expiry_date]

delta_convention = vol_daily_smile_df['delta_convention'].iloc[0]
vol_pillar = vol_daily_smile_df[surf.quotes_column_names].iloc[0]

strikes = gk_solve_strike(S0=surf.fx_spot_rate,
                          tau=vol_daily_smile_df['expiry_years'].iloc[0],
                          r_d=vol_daily_smile_df['domestic_zero_rate'].iloc[0],
                          r_f=vol_daily_smile_df['foreign_zero_rate'].iloc[0],
                          vol=vol_pillar,
                          signed_delta=surf.quotes_signed_delta,
                          delta_convention=delta_convention,
                          F=vol_daily_smile_df['fx_forward_rate'].iloc[0],
                          atm_delta_convention='forward')

nb_strikes = 100
strike_delta_neutral = strikes[surf.quotes_column_names == 'atm_delta_neutral'].item()
strikes_put = np.linspace(np.min(strikes), strike_delta_neutral, nb_strikes)
strikes_call = np.linspace(strike_delta_neutral, np.max(strikes), nb_strikes)
strikes_array = np.concatenate([strikes_put, strikes_call])
call_put_array = np.concatenate([-1 * np.ones(nb_strikes), np.ones(nb_strikes)])
expiry_dates = pd.DatetimeIndex([expiry_date for _ in range(len(strikes_array))])

result = surf.price_vanilla_european(expiry_dates=expiry_dates,
                                     K=strikes_array,
                                     cp=call_put_array,
                                     analytical_greeks_flag=True)

if delta_convention == 'regular_spot':
    delta_str = 'spot_delta'
elif delta_convention == 'regular_forward':
    delta_str = 'forward_delta'

delta = result['analytical_greeks'][delta_str]
vol = result['market_data_inputs']['vol']

delta_pillar_for_plot = np.full(surf.quotes_signed_delta.shape, np.nan)
midpoint = len(delta_pillar_for_plot)//2
delta_pillar_for_plot[:midpoint] = np.abs(surf.quotes_signed_delta[:midpoint])
delta_pillar_for_plot[midpoint:] = 1.0 - surf.quotes_signed_delta[midpoint:]

delta_for_plot = np.full(delta.shape, np.nan)
delta_for_plot[:nb_strikes] = np.abs(delta[:nb_strikes])
delta_for_plot[nb_strikes:] = 1.0 - delta[nb_strikes:]

# Plot
plt.plot(100*delta_pillar_for_plot, vol_pillar, 'x', label='Pillar Volatilities')
plt.plot(100*delta_for_plot, vol, label='Fitted Smile')    # X markers

tenor = np.round(vol_daily_smile_df['expiry_years'].iloc[0],2)

plt.title(f'Volatility Smile for {tenor}Y ({expiry_date.date()})\n'
          f'Smile interpolation method: {surf.smile_interpolation_method.value}')
plt.xlabel('Delta:   Puts: (0,0.5). ATM Δ Neutral: 0.5. Calls: (0.5,1)')
plt.ylabel('Volatility')
plt.xlim([0, 100])

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))  # 1.0 means 100%

plt.legend()
plt.show()








#%%


for i,row in surf.vol_smile_pillar_df.iterrows():
    # Spot delta check

    vols = surf.vol_smile_pillar_df[surf.quotes_column_names].iloc[i]
    signed_delta = surf.quotes_signed_delta
    cp = surf.quotes_call_put_flag
    expiry_date = surf.vol_smile_pillar_df['expiry_date'].iloc[i]
    expiry_dates = pd.DatetimeIndex([expiry_date for _ in range(len(vols))])
    K = surf.strike_pillar_df[surf.quotes_column_names].iloc[i]

    result = surf.price_vanilla_european(expiry_dates=expiry_dates,
                                         K=K,
                                         cp=cp,
                                         analytical_greeks_flag=True)


    mask = surf.quotes_column_names != 'atm_delta_neutral'
    if surf.vol_smile_pillar_df.loc[i,'delta_convention'] == 'regular_spot':
        delta_str = 'spot_delta'
    elif surf.vol_smile_pillar_df.loc[i,'delta_convention'] == 'regular_forward':
        delta_str = 'forward_delta'

    assert (np.abs(result['analytical_greeks'][delta_str][mask] - signed_delta[mask]) < 1e-2).all()
    assert (np.abs(result['analytical_greeks']['forward_delta'][~mask] - signed_delta[~mask]) < 1e-2).all()

#%%

# expiry_date = pd.Timestamp('2025-06-30')
#
# vol_smile = surf.vol_smile_daily_df.loc[surf.vol_smile_daily_df['expiry_date'] == expiry_date]
# surf._solve_vol_daily_smile_func(pd.DatetimeIndex([expiry_date]))
# vol_smile_func = surf.vol_smile_daily_func[expiry_date]
#
# # Pillar data for scatter plot
# pillar_delta = surf.quotes_signed_delta
# pillar_vol = vol_smile[surf.quotes_column_names].values[0]
#
# # Data per smile function for line plot
# min_delta = np.max(pillar_delta[pillar_delta < 0])
# max_delta = np.min(pillar_delta[pillar_delta > 0])
# deltas = np.array(list(np.linspace(min_delta, -0.5, 10)) + list(np.linspace(0.5, max_delta, 10)))


vol_smile = surf.vol_smile_pillar_df.iloc[8]
strike_smile = surf.strike_pillar_df.iloc[5]

vols = vol_smile[surf.quotes_column_names].values
strikes = strike_smile[surf.quotes_column_names].values

strike_delta_neutral = strike_smile['atm_delta_neutral']

strikes_put = np.linspace(min(strikes), strike_delta_neutral, 100)
strikes_call = np.linspace(strike_delta_neutral, max(strikes), 100)
strikes_range = np.concatenate([strikes_put, strikes_call])
call_put_range = np.concatenate([-1 * np.ones(100), np.ones(100)])

expiry_dates = pd.DatetimeIndex([vol_smile['expiry_date'] for _ in range(len(strikes_range))])

result = surf.price_vanilla_european(expiry_dates=expiry_dates,
                                     K=strikes_range,
                                     cp=call_put_range,
                                     analytical_greeks_flag=True)





#%%
# Solve strike for each delta


#
# def clean_and_extract_vol_smile_columns(df: pd.DataFrame,
#                                         call_put_pattern: str = r'^(0?[1-9]|[1-4]\d)[_ ]?(delta|Δ)[_ ]?(call|c|put|p)$',
#                                         atm_delta_neutral_column_pattern: str = r'^atm[_ ]?(delta|Δ)[_ ]?neutral$'):
#     quotes_column_names = []
#     for col_name in df.columns:
#         match = re.match(call_put_pattern, col_name)
#         if match:
#             # Extract components from match groups
#             delta_value = match.group(1)  # 1 or 2 digits
#             option_type = match.group(3)  # "call", "c", "put", or "p"
#
#             # Convert "c" to "call" and "p" to "put"
#             option_type_full = 'call' if option_type in ['call', 'c'] else 'put'
#
#             # Return the column name in the new format
#             new_col_name = f'{delta_value}_delta_{option_type_full}'
#             df.rename(columns={col_name: new_col_name}, inplace=True)
#             quotes_column_names.append(f'{delta_value}_delta_{option_type_full}')
#
# for Δ in Δ_list:
#     atm = 'σ_atmΔneutral'
#     bf = 'σ_' + Δ + 'Δbf'
#     rr = 'σ_' + Δ + 'Δrr'
#
#     for v in ['call', 'put']:
#         column_name = 'σ_' + Δ + 'Δ' + v
#         if column_name not in df.columns:
#             df[column_name] = np.nan
#
#     for i, row in df.iterrows():
#
#         if bf and rr in row.index:
#             if atm in row.index:
#                 if pd.notna(row[bf]) and pd.notna(row[rr]) and pd.notna(row[atm]):
#                     if isinstance(row[bf], (float, int)) and isinstance(row[rr], (float, int)) \
#                             and isinstance(row[atm], (float, int)) and row[atm] > 0:
#                         df.at[i, 'σ_' + Δ + 'Δcall'] = row[bf] + row[atm] + 0.5 * row[rr]
#                         df.at[i, 'σ_' + Δ + 'Δput'] = row[bf] + row[atm] - 0.5 * row[rr]
#
#             if pd.isna(row[bf]) and pd.notna(row[rr]):
#                 df.loc[i, 'errors'] += bf + ' value is absent\n'  # add comment if butterfly is n/a
#             elif pd.isna(row[rr]) and pd.notna(row[bf]):
#                 df.loc[i, 'errors'] += rr + ' value is absent\n'  # add comment if risk reversal is n/a
#             elif (pd.notna(row[bf]) or pd.notna(row[rr])) and pd.isna(row[atm]):
#                 df.loc[i, 'errors'] += atm + ' value is absent\n'  # add comment if at-the-money is n/a
#
#         elif bf in row.index and rr not in row.index:
#             if rr not in row.index and pd.notna(row[bf]):
#                 df.loc[i, 'errors'] += bf + ' value is present but column ' + rr + ' is absent\n'
#             if atm not in row.index and pd.notna(row[bf]):
#                 df.loc[i, 'errors'] += bf + ' value is present but column ' + atm + ' is absent\n'
#
#         elif rr in row.index and bf not in row.index:
#             if bf not in row.index and pd.notna(row[rr]):
#                 df.loc[i, 'errors'] += rr + ' value is present but column ' + bf + ' is absent\n'
#             if atm not in row.index and pd.notna(row[rr]):
#                 df.loc[i, 'errors'] += rr + ' value is present but column ' + atm + ' is absent\n'
#
#             # Drop σ-strategy quote columns
# pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
# cols_to_drop = df.filter(regex=pattern2).columns
# df = df.drop(columns=cols_to_drop)