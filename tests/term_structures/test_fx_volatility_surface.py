# -*- coding: utf-8 -*-

#%load_ext autoreload
#%autoreload 2

import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.utils.business_day_calendar import get_busdaycal
from frm.term_structures.fx_volatility_surface_helpers_new import *
from frm.pricing_engine.garman_kohlhagen import *
from frm.enums.utils import *
import re


# AUD/USD 30-June-2023
day_count_basis = DayCountBasis.ACT_ACT

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
    '25Δ_call': np.array([10.704, 9.559, 9.407, 9.364, 9.404, 9.508, 9.596, 9.833, 9.930, 10.065, 10.155, 10.407, 10.711])/100,
    '20Δ_call': np.array([10.683, 9.516, 9.368, 9.323, 9.365, 9.481, 9.585, 9.846, 9.943, 10.071, 10.160, 10.478, 10.774])/100,
    '15Δ_call': np.array([10.663, 9.471, 9.331, 9.287, 9.335, 9.470, 9.599, 9.893, 9.998, 10.117, 10.206, 10.615, 10.904])/100,
    '10Δ_call': np.array([10.643, 9.421, 9.296, 9.256, 9.318, 9.486, 9.657, 10.004, 10.126, 10.236, 10.325, 10.877, 11.157])/100,
    '5Δ_call': np.array([10.628, 9.365, 9.274, 9.249, 9.349, 9.587, 9.847, 10.306, 10.474, 10.568, 10.660, 11.528, 11.787])/100
}
vol_smile_pillar = pd.DataFrame(vol_surface_data)

# FX forward data
# The FX forward curve must be specified as a dataframe with
# (i) least one of 'tenor', 'fixing_date' or 'delivery_date' (to define the term structure)
# (ii) the 'fx_forward_rate' column
fx_forward_curve_data = {
    'tenor': ['SP','1D','1W','2W','3W','1M','2M','3M','6M','9M','1Y','15M','18M','2Y','3Y','4Y','5Y','7Y','10Y'],
    'fx_forward_rate': [0.6629, 0.6629, 0.6630, 0.6631, 0.6633, 0.6635, 0.6640, 0.6646,0.6661, 0.6673, 0.6680,
                        0.6681, 0.6679, 0.6668, 0.6631, 0.6591, 0.6525, 0.6358, 0.6084],
}
fx_forward_curve_df = pd.DataFrame(fx_forward_curve_data)

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

# At minimum (out of the curve_date, spot_offset & spot_date) the curve_date or spot_date must be specified
# The two parameters can be implied or set per market convention.
# If all three parameters are provided, they will be validated for consistency.
curve_date = pd.Timestamp('2023-06-30') # Friday 30-June-2023
spot_date = pd.Timestamp('2023-07-05') # Wednesday 5-July-2023, T+2 settlement (4th July is US holiday)
spot_offset = None
curve_date, spot_offset, spot_date = resolve_fx_curve_dates(ccy_pair, busdaycal, curve_date, spot_offset, spot_date)


fx_forward_curve_df = fx_forward_curve_helper(df=fx_forward_curve_df, curve_date=curve_date, spot_offset=spot_offset, busdaycal=busdaycal)
fx_spot_rate = fx_forward_curve_df.loc[fx_forward_curve_df['tenor'] == 'sp', 'fx_forward_rate'].values[0]

vol_smile_pillar['warnings'] = ''
vol_smile_pillar = fx_term_structure_helper(df=vol_smile_pillar, curve_date=curve_date, spot_offset=spot_offset, busdaycal=busdaycal, rate_set_date_str='expiry_date')
vol_smile_pillar = check_delta_convention(vol_smile_pillar, ccy_pair)
vol_smile_pillar, quote_columns = extract_vol_smile_columns(vol_smile_pillar)
vol_smile_pillar = vol_smile_pillar.sort_values('expiry_date').reset_index(drop=True)


vol_smile_pillar['expiry_years'] = year_fraction(start_date=curve_date, end_date=vol_smile_pillar['expiry_date'], day_count_basis=day_count_basis)
vol_smile_pillar['fx_forward_rate'] = interp_fx_forward_curve_df(
    fx_forward_curve_df=fx_forward_curve_df,
    dates=vol_smile_pillar['expiry_date'],
    date_type='fixing_date')

vol_smile_pillar['domestic_zero_rate'] = zero_curve_domestic.get_zero_rate(dates=vol_smile_pillar['expiry_date'], compounding_frequency=CompoundingFrequency.CONTINUOUS)
vol_smile_pillar['foreign_zero_rate'] = zero_curve_foreign.get_zero_rate(dates=vol_smile_pillar['expiry_date'], compounding_frequency=CompoundingFrequency.CONTINUOUS)



vol_smile_daily = create_vol_smile_daily(vol_smile_pillar, quote_columns, curve_date, day_count_basis)

###################################################################
# Calculate the delta-strike surface from the delta-volatility surface           
strike_pillar = vol_smile_pillar.copy()
strike_pillar.loc[:, quote_columns] = np.nan
cp = np.array([1 if 'call' in vol_quote else -1 if 'put' in vol_quote else None for vol_quote in quote_columns])
signed_delta = np.array([0.5 if vol_quote == 'atm_delta_neutral' else cp[i] * float(vol_quote.split('_')[0]) / 100 for i, vol_quote in enumerate(quote_columns)])

for date, row in strike_pillar.iterrows():
    # Scalars
    S0 = fx_spot_rate
    r_f = row['domestic_zero_rate']
    r_d = row['foreign_zero_rate']
    tau = row['expiry_years']
    delta_convention = row['delta_convention']
    F = row['fx_forward_rate']
    # Arrays
    vol_smile = vol_smile_pillar.loc[date, quote_columns].values
    strikes = gk_solve_strike(S0=S0, tau=tau, r_d=r_d, r_f=r_f, vol=vol_smile, signed_delta=signed_delta, delta_convention=delta_convention, F=F)
    strike_pillar.loc[date, quote_columns] = strikes



#%%

#df_interp['delta_convention'] = 'regular_forward_delta'  # need to add a section to calculate forward delta from spot delta
#df_interp.set_index('tenor_date', inplace=True, drop=True)





#%%



#%%






#%%%


# Volatility smile quotes are typically quoted in terms of delta
# 1. call/put/at-the-money quotes
# 2. σ-strategy quotes (risk reversal and butterfly) + atm quote


#%%







# curve_ccy validation











def fx_σ_input_helper(df):
    for i, column_name in enumerate(['errors', 'warnings', 'internal_id']):
        if column_name not in df.columns:
            df.insert(loc=i, column=column_name, value='')

    # % mandatory columns validation
    mandatory_columns = [
        'errors',
        'warnings',
        'internal_id',
        'curve_date',
        'curve_ccy',
        'delta_convention',
    ]
    df = df.dropna(axis=0, subset=mandatory_columns)  # drop rows with blanks in mandatory columns
    missing_mandatory_columns = [col for col in mandatory_columns if col not in df.columns.to_list()]
    if len(missing_mandatory_columns) > 0:
        df['errors'] += f'missing mandatory columns: {missing_mandatory_columns}\n'
        return df

    # Validate  column data
    valid_field_values = {'delta_convention': VALID_DELTA_CONVENTIONS,
                          'day_count_basis': VALID_DAY_COUNT_BASIS + [np.nan, '']}

    # Enforce to list of valid values
    for column_name in mandatory_columns:
        if column_name in valid_field_values.keys():
            bool_cond = df[column_name].isin(valid_field_values[column_name])
            df.loc[np.logical_not(bool_cond), 'errors'] = 'invalid value for ' + column_name

    # fx option specific curve_ccy and delta_convention validation
    field = 'curve_ccy'
    bool_cond = df[field].isin(['usdaud', 'usdeur', 'usdgbp', 'usdnzd'])
    df.loc[bool_cond, 'warnings'] += field + ' value is not per common market convention\n'

    bool_cond = np.logical_and(df['curve_ccy'].isin(['audusd', 'nzdeur', 'gbpusd', 'nzdusd']),
                               df['Δ_convention'].str.contains('premium_adjusted'))
    df.loc[bool_cond, 'warnings'] += 'the regular delta is the market delta_convention for this currency pair\n'

    bool_cond = np.logical_and.reduce([
        np.logical_not(
            df['curve_ccy'].isin(['audusd', 'nzdeur', 'gbpusd', 'nzdusd', 'usdaud', 'usdeur', 'usdgbp', 'usdnzd'])),
        df['curve_ccy'].str.contains('usd'),
        df['Δ_convention'].str.contains('regular')])
    df.loc[bool_cond, 'warnings'] += 'premium adjusted delta is the market delta_convention for this currency pair\n'

    date_columns = ['tenor_name', 'tenor_date']
    optional_columns = ['day_count_basis', 'tenor_years', 'base_ccy', 'quote_ccy']

    for column_name in optional_columns:
        if column_name not in df.columns:
            df[column_name] = np.nan
        else:
            # Enforce to list of valid values
            if column_name in valid_field_values.keys():
                bool_cond = df[column_name].isin(valid_field_values[column_name])
                df.loc[np.logical_not(bool_cond), 'errors'] = 'invalid value for ' + column_name

    user_input_columns = [v for v in df.columns if v not in mandatory_columns + date_columns + optional_columns]

    # drop user input columns if they are all nan
    mask = df[user_input_columns].isna().all()
    cols_to_drop = mask[mask].index.tolist()
    df = df.drop(columns=cols_to_drop)

    # Validate volatility input columns
    not_nan_user_input_columns = [v for v in df.columns if v not in mandatory_columns + date_columns + optional_columns]

    invalid_columns = []
    valid_volatility_input_columns = []
    for i, v in enumerate(not_nan_user_input_columns):
        pattern1 = r'^σ_(\d{1,2})Δ(call|put)$'
        pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
        atm_column_names = ['σ_atmΔneutral', 'σ_atmf']
        if (re.match(pattern1, v) and 1 <= int(re.match(pattern1, v).group(1)) <= 99) \
                or (re.match(pattern2, v) and 1 <= int(re.match(pattern2, v).group(1)) <= 99) \
                or v in atm_column_names:
            valid_volatility_input_columns.append(v)
        else:
            invalid_columns.append(v)

    df = df.drop(columns=invalid_columns)

    for col in invalid_columns:
        msg = 'user added column' + "'" + col + "'" + ' does not ' \
              + 'match regex pattern ' + "'" + pattern1 + "', or pattern " + "'" + pattern2 + "'," \
              + ' and is not in the allowed list (' + ', '.join(atm_column_names) + ')\n'

        bool_cond = df[col].isnotna()
        df.loc[bool_cond, 'errors'] += msg

    # Enforce only one type of σ quote input
    pattern_call_put = r'^σ_(\d{1,2})Δ(call|put)$'
    pattern_strategy = r'^σ_(\d{1,2})Δ(bf|rr)$'
    cols_call_put = df.filter(regex=pattern_call_put).columns
    cols_strategy = df.filter(regex=pattern_strategy).columns
    mask_call_put = df[cols_call_put].apply(lambda x: x.notna().any(), axis=1)
    mask_strategy = df[cols_strategy].apply(lambda x: x.notna().any(), axis=1)
    if 'σ_atmf' in df.columns:
        mask_atmf = df['σ_atmf'].apply(lambda x: x.notna().any(), axis=1)

        mask = np.logical_and.reduce([mask_call_put, mask_strategy, np.logical_not(mask_atmf)])
        df.loc[
            mask, 'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-strategy quotes; specify only one volatility input type per row\n'

        mask = np.logical_and.reduce([mask_call_put, mask_atmf, np.logical_not(mask_strategy)])
        df.loc[
            mask, 'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'

        mask = np.logical_and.reduce([mask_strategy, mask_atmf, np.logical_not(mask_call_put)])
        df.loc[
            mask, 'errors'] = 'row has non-nan values for i) σ-strategy quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'

        mask = np.logical_and.reduce([mask_strategy, mask_atmf, mask_call_put])
        df.loc[
            mask, 'errors'] = 'row has non-nan values for i) σ-strategy quotes and ii) σ-atmf quotes; specify only one volatility input type per row\n'

        mask = mask_call_put & mask_strategy & mask_atmf
        df.loc[
            mask, 'errors'] = 'row has non-nan values for across i) Δ-σ quotes, ii) σ-strategy quotes and iii) σ-atmf quotes; input only one volatility input type per row\n'
    else:
        mask = np.logical_and.reduce([mask_call_put, mask_strategy])
        df.loc[
            mask, 'errors'] = 'row has non-nan values for i) Δ-σ quotes and ii) σ-strategy quotes; input only one volatility input type per row\n'

    # Enforce numeric types for strategy quotes
    for col in valid_volatility_input_columns:
        bool_cond = df[col].apply(lambda x: not isinstance(x, (float, int)) and not pd.isna(x))
        df.loc[bool_cond, 'errors'] += col + ' has an invalid type\n'

        # Call and put quotes validation
    for col in (cols_call_put.to_list() + atm_column_names):
        if col in df.columns:
            bool_cond = df[col].apply(lambda x: isinstance(x, (float, int)) and x <= 0.0)
            df.loc[bool_cond, 'errors'] += col + ' must have a positive value\n'

            bool_cond = df[col].apply(
                lambda x: isinstance(x, (float, int)) and x > 2)  # unlikely for volatility quote to be > 200%
            df.loc[bool_cond, 'warnings'] += col + ' value is unusually large'

    for col in cols_strategy.to_list():
        bool_cond = df[col].apply(
            lambda x: isinstance(x, (float, int)) and abs(x) > 0.25)  # unlikely for strategy spread to be > 25%
        df.loc[bool_cond, 'warnings'] += col + ' value is unusually large'

    def extract_numbers(df, suffix):
        pattern = rf'^σ_(\d{{1,2}})Δ{suffix}$'
        cols = df.filter(regex=pattern).columns
        return [col.split('_')[1].split('Δ')[0] for col in cols]

    Δ_list = list(set(extract_numbers(df, 'rr') + extract_numbers(df, 'bf')))

    for Δ in Δ_list:
        atm = 'σ_atmΔneutral'
        bf = 'σ_' + Δ + 'Δbf'
        rr = 'σ_' + Δ + 'Δrr'

        for v in ['call', 'put']:
            column_name = 'σ_' + Δ + 'Δ' + v
            if column_name not in df.columns:
                df[column_name] = np.nan

        for i, row in df.iterrows():

            if bf and rr in row.index:
                if atm in row.index:
                    if pd.notna(row[bf]) and pd.notna(row[rr]) and pd.notna(row[atm]):
                        if isinstance(row[bf], (float, int)) and isinstance(row[rr], (float, int)) \
                                and isinstance(row[atm], (float, int)) and row[atm] > 0:
                            df.at[i, 'σ_' + Δ + 'Δcall'] = row[bf] + row[atm] + 0.5 * row[rr]
                            df.at[i, 'σ_' + Δ + 'Δput'] = row[bf] + row[atm] - 0.5 * row[rr]

                if pd.isna(row[bf]) and pd.notna(row[rr]):
                    df.loc[i, 'errors'] += bf + ' value is absent\n'  # add comment if butterfly is n/a
                elif pd.isna(row[rr]) and pd.notna(row[bf]):
                    df.loc[i, 'errors'] += rr + ' value is absent\n'  # add comment if risk reversal is n/a
                elif (pd.notna(row[bf]) or pd.notna(row[rr])) and pd.isna(row[atm]):
                    df.loc[i, 'errors'] += atm + ' value is absent\n'  # add comment if at-the-money is n/a

            elif bf in row.index and rr not in row.index:
                if rr not in row.index and pd.notna(row[bf]):
                    df.loc[i, 'errors'] += bf + ' value is present but column ' + rr + ' is absent\n'
                if atm not in row.index and pd.notna(row[bf]):
                    df.loc[i, 'errors'] += bf + ' value is present but column ' + atm + ' is absent\n'

            elif rr in row.index and bf not in row.index:
                if bf not in row.index and pd.notna(row[rr]):
                    df.loc[i, 'errors'] += rr + ' value is present but column ' + bf + ' is absent\n'
                if atm not in row.index and pd.notna(row[rr]):
                    df.loc[i, 'errors'] += rr + ' value is present but column ' + atm + ' is absent\n'

                    # Drop σ-strategy quote columns
    pattern2 = r'^σ_(\d{1,2})Δ(bf|rr)$'
    cols_to_drop = df.filter(regex=pattern2).columns
    df = df.drop(columns=cols_to_drop)

    # Drop all nan columns
    df = df.dropna(axis=1, how='all')

    return df

