# -*- coding: utf-8 -*-
import os

from frm.enums.term_structures import FXSmileInterpolationMethod

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from frm.term_structures.zero_curve import ZeroCurve
from frm.term_structures.fx_volatility_surface import FXVolatilitySurface

from frm.utils.business_day_calendar import get_busdaycal
from frm.enums.utils import DayCountBasis, CompoundingFrequency

import numpy as np
import pandas as pd


def test_fx_volatility_surface():
    # FX Volatility Surface

    # At minimum (out of the curve_date, spot_offset & spot_date) the curve_date or spot_date must be specified
    # The two parameters can be implied or set per market convention.
    # If all three parameters are provided, they will be validated for consistency.
    curve_date = pd.Timestamp('2023-06-30')  # Friday 30-June-2023

    # Market convention is quoted as AUD/USD (1 AUD = x USD)
    domestic_ccy = 'usd'
    foreign_ccy = 'aud'

    # The FX volatility surface is defined as pandas DataFrame.
    # Each row of the dataframe defines to the volatility smile for the rows tenor.
    # Each column of the dataframe corresponds to a given delta's term structure.
    # The dataframe must have
    # (i) at least one of 'tenor', 'expiry_date' or 'delivery_date' as columns to define the term structure.
    # (ii) the 'delta_convention' column to specify the delta convention for the volatility smile.
    # The volatility smile column names are defined by 'X_delta_call' and 'X_delta_put' where X is the delta value with 1<X<50.
    # The 'atm_delta_neutral' column is the at-the-money volatility.

    call_put_quotes = {
        'tenor': ['1W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y'],
        'delta_convention': ['regular_spot'] * len(['1W', '1M', '2M', '3M', '6M', '9M', '1Y']) \
                            + ['regular_forward'] * len(['2Y', '3Y', '4Y', '5Y', '7Y', '10Y']),
        '5_delta_put': np.array(
            [11.943, 11.145, 11.514, 11.834, 12.402, 12.996, 13.546, 14.159, 14.683, 15.161, 15.477, 16.703, 17.042]) / 100,
        '10_delta_put': np.array(
            [11.656, 10.786, 10.990, 11.200, 11.599, 12.006, 12.361, 12.824, 13.215, 13.618, 13.875, 14.603, 14.966]) / 100,
        '15_delta_put': np.array(
            [11.481, 10.568, 10.683, 10.832, 11.141, 11.455, 11.718, 12.123, 12.452, 12.808, 13.032, 13.573, 13.948]) / 100,
        '20_delta_put': np.array(
            [11.350, 10.405, 10.457, 10.564, 10.812, 11.065, 11.270, 11.645, 11.932, 12.254, 12.455, 12.888, 13.267]) / 100,
        '25_delta_put': np.array(
            [11.240, 10.271, 10.275, 10.350, 10.550, 10.758, 10.920, 11.273, 11.530, 11.823, 12.005, 12.360, 12.739]) / 100,
        '30_delta_put': np.array(
            [11.140, 10.152, 10.116, 10.165, 10.326, 10.496, 10.624, 10.961, 11.190, 11.457, 11.620, 11.909, 12.283]) / 100,
        'atm_delta_neutral': np.array(
            [10.868, 9.814, 9.684, 9.670, 9.745, 9.848, 9.922, 10.150, 10.300, 10.488, 10.600, 10.750, 11.100]) / 100,
        '30_delta_call': np.array(
            [10.722, 9.598, 9.441, 9.400, 9.440, 9.535, 9.610, 9.831, 9.934, 10.076, 10.166, 10.369, 10.683]) / 100,
        '25_delta_call': np.array(
            [10.704, 9.559, 9.407, 9.364, 9.404, 9.508, 9.596, 9.833, 9.930, 10.065, 10.155, 10.407, 10.711]) / 100,
        '20_delta_call': np.array(
            [10.683, 9.516, 9.368, 9.323, 9.365, 9.481, 9.585, 9.846, 9.943, 10.071, 10.160, 10.478, 10.774]) / 100,
        '15_delta_call': np.array(
            [10.663, 9.471, 9.331, 9.287, 9.335, 9.470, 9.599, 9.893, 9.998, 10.117, 10.206, 10.615, 10.904]) / 100,
        '10_delta_call': np.array(
            [10.643, 9.421, 9.296, 9.256, 9.318, 9.486, 9.657, 10.004, 10.126, 10.236, 10.325, 10.877, 11.157]) / 100,
        '5_delta_call': np.array(
            [10.628, 9.365, 9.274, 9.249, 9.349, 9.587, 9.847, 10.306, 10.474, 10.568, 10.660, 11.528, 11.787]) / 100
    }

    strategy_quotes = {
        'tenor': ['1W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y'],
        'delta_convention': ['regular_spot'] * len(['1W', '1M', '2M', '3M', '6M', '9M', '1Y']) \
                            + ['regular_forward'] * len(['2Y', '3Y', '4Y', '5Y', '7Y', '10Y']),
        '25_delta_risk_reversal': np.array(
            [-0.537, -0.712, -0.868, -0.986, -1.146, -1.250, -1.324, -1.440, -1.600, -1.758, -1.850, -1.953,
             -2.028]) / 100,
        '25_delta_butterfly': np.array(
            [0.104, 0.092, 0.143, 0.170, 0.210, 0.260, 0.310, 0.380, 0.406, 0.430, 0.455, 0.625, 0.625]) / 100,
        'atm_delta_neutral': np.array(
            [10.868, 9.814, 9.684, 9.670, 9.745, 9.848, 9.922, 10.150, 10.300, 10.488, 10.600, 10.750, 11.100]) / 100
    }



    # FX Forward Rates
    # The FX forward curve must be specified as a dataframe with
    # (i) least one of 'tenor', 'fixing_date' or 'delivery_date' (to define the term structure)
    # (ii) the 'fx_forward_rate' column
    fx_forward_curve_data = {
        'tenor': ['SP', '1D', '1W', '2W', '3W', '1M', '2M', '3M', '6M', '9M', '1Y', '15M', '18M', '2Y', '3Y', '4Y', '5Y',
                  '7Y', '10Y'],
        'fx_forward_rate': [0.6629, 0.6629, 0.6630, 0.6631, 0.6633, 0.6635, 0.6640, 0.6646, 0.6661, 0.6673, 0.6680,
                            0.6681, 0.6679, 0.6668, 0.6631, 0.6591, 0.6525, 0.6358, 0.6084],
    }
    fx_forward_curve_df = pd.DataFrame(fx_forward_curve_data)

    # Interest rates
    data = {
        'tenor': ['1 Day', '1 Week', '2 Week', '3 Week', '1 Month', '2 Month', '3 Month', '6 Month', '9 Month', '1 Year',
                  '15 Month', '18 Month', '2 Year', '3 Year', '4 Year', '5 Year', '7 Year', '10 Year'],
        'domestic_zero_rate': [5.055, 5.059, 5.063, 5.065, 5.142, 5.221, 5.270, 5.390, 5.432, 5.381, 5.248, 5.122, 4.812,
                               4.373, 4.087, 3.900, 3.690, 3.550],
        'foreign_zero_rate': [4.156, 4.153, 4.150, 4.138, 4.194, 4.274, 4.335, 4.479, 4.595, 4.660, 4.674, 4.673, 4.578,
                              4.427, 4.295, 4.285, 4.362, 4.493],
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

    zero_curve_domestic = ZeroCurve(data=zero_rate_domestic_df, curve_date=pd.Timestamp('2023-06-30'),
                                    day_count_basis=DayCountBasis.ACT_360,
                                    compounding_frequency=CompoundingFrequency.CONTINUOUS, busdaycal=busdaycal_domestic)
    zero_curve_foreign = ZeroCurve(data=zero_rate_foreign_df, curve_date=pd.Timestamp('2023-06-30'),
                                   day_count_basis=DayCountBasis.ACT_365,
                                   compounding_frequency=CompoundingFrequency.CONTINUOUS, busdaycal=busdaycal_foreign)
    del zero_rate_domestic_df, zero_rate_foreign_df, zero_rate_df, busdaycal_domestic, busdaycal_foreign

    # If no business day calendar is specified, create it based on holiday calendars of both currencies
    busdaycal = None
    if busdaycal is None:
        busdaycal = get_busdaycal([domestic_ccy, foreign_ccy])


    for smile_interpolation_method in [
        FXSmileInterpolationMethod.UNIVARIATE_SPLINE,
     #   FXSmileInterpolationMethod.CUBIC_SPLINE,
        FXSmileInterpolationMethod.HESTON_COSINE,
     #   FXSmileInterpolationMethod.HESTON_1993,
     #   FXSmileInterpolationMethod.HESTON_CARR_MADAN_GAUSS_KRONROD_QUADRATURE,
     #   FXSmileInterpolationMethod.HESTON_CARR_MADAN_FFT_W_SIMPSONS, # The fit is not accurate enough
     #   FXSmileInterpolationMethod.HESTON_LIPTON # The fit is not accurate enough
    ]:

        for vol_data in [strategy_quotes]: # strategy_quotes, call_put_quotes
            vol_quotes = pd.DataFrame(vol_data)
            vol_surface = FXVolatilitySurface(domestic_ccy=domestic_ccy,
                                              foreign_ccy=foreign_ccy,
                                              fx_forward_curve_df=fx_forward_curve_df,
                                              domestic_zero_curve=zero_curve_domestic,
                                              foreign_zero_curve=zero_curve_foreign,
                                              vol_quotes=vol_quotes,
                                              curve_date=curve_date,
                                              busdaycal=busdaycal,
                                              smile_interpolation_method=smile_interpolation_method)

            surf = vol_surface

            # Test the strike solve by comparing the analytical delta to the signed delta
            # Excluding, the atm delta neutral quotes, the volatility quotes are given as spot delta for <= 1Y tenors and forward delta for > 1Y tenors.
            # The atm delta neutral quotes are all forward delta quotes.

            for i, row in surf.vol_smile_pillar_df.iterrows():
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
                if surf.vol_smile_pillar_df.loc[i, 'delta_convention'] == 'regular_spot':
                    delta_str = 'spot_delta'
                elif surf.vol_smile_pillar_df.loc[i, 'delta_convention'] == 'regular_forward':
                    delta_str = 'forward_delta'

                if smile_interpolation_method in [FXSmileInterpolationMethod.UNIVARIATE_SPLINE, FXSmileInterpolationMethod.CUBIC_SPLINE]:
                    # These interpolation methods fit the data exactly
                    tol = 1e-8
                else:
                    # The heston fit is a numerical approximation and the tolerance is higher
                    tol = 1e-2

                assert (np.abs(result['analytical_greeks'][delta_str][mask] - signed_delta[mask]) < tol).all()
                assert (np.abs(result['analytical_greeks']['forward_delta'][~mask] - signed_delta[~mask]) < tol).all()

                surf.plot_smile(expiry_date)


if __name__ == "__main__":
   test_fx_volatility_surface()