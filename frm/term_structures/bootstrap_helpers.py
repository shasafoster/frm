# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Optional

from frm.enums import DayCountBasis
from frm.utils import clean_tenor, tenor_to_date_offset, year_frac, get_busdaycal, workday


def cash_rates_quote_helper(
        df: pd.DataFrame,
        curve_date: pd.Timestamp,
        day_count_basis: DayCountBasis,
        settlement_delay: Optional[int]=None,
        settlement_date: Optional[pd.Timestamp]=None,
        cal: np.busdaycalendar=np.busdaycalendar):

    assert 'rate' in df.columns
    assert 'tenor' in df.columns or ('effective_date' in df.columns and 'maturity_date' in df.columns)

    df['tenor'] = df['tenor'].apply(clean_tenor)

    if settlement_date is None:
        assert settlement_delay is not None
        settlement_date = workday(dates=curve_date, offset=settlement_delay, cal=cal)

    if 'effective_date' not in df.columns:
        df['has_settlement_delay'] = df['tenor'] != 'on'
        df.loc[~df['has_settlement_delay'],'effective_date'] = curve_date
        df.loc[df['has_settlement_delay'],'effective_date'] = settlement_date

    if 'maturity_date' not in df.columns:
        date_offset = df['tenor'].apply(tenor_to_date_offset).values
        df['maturity_date'] = workday(dates=(df['effective_date'] + date_offset), offset=0, cal=cal)

    df['period_years'] = year_frac(df['effective_date'], df['maturity_date'], day_count_basis)

    return df

def bootstrap_cash_rates(
        df: pd.DataFrame,
        curve_date: pd.Timestamp,
        day_count_basis_years: DayCountBasis=DayCountBasis.ACT_ACT):

    df['effective_date_years'] = year_frac(curve_date, df['effective_date'], day_count_basis_years)
    df['maturity_date_years'] = year_frac(curve_date, df['maturity_date'], day_count_basis_years)
    df['effective_date_discount_factor'] = np.nan
    df['maturity_date_discount_factor'] = np.nan

    df.sort_values(by='maturity_date', inplace=True)
    for i, row in df.iterrows():
        if not row['has_settlement_delay']:
            df.at[i, 'effective_date_discount_factor'] = 1.0
            df.at[i, 'maturity_date_discount_factor'] = 1.0 / (1 + row['rate'] * row['period_years'])
        else:
            mask = df['maturity_date'] <= row['effective_date']

            x = row['effective_date_years']
            X = df.loc[mask, 'maturity_date_years'].values
            Y = np.log(df.loc[mask, 'maturity_date_discount_factor'].values)
            interp_func = interp1d(X, Y, kind='linear')
            df.at[i, 'effective_date_discount_factor'] = np.exp(interp_func(x))

            df.at[i, 'maturity_date_discount_factor'] = df.at[i, 'effective_date_discount_factor'] \
                                                                / (1 + row['rate'] * row['period_years'])

    #df = df.drop(columns=['effective_date_years', 'maturity_date_years'], errors='ignore')

    return df



def bootstrap_futures(
        df: pd.DataFrame,
        curve_date: pd.Timestamp,
        cash_rates_df: pd.DataFrame,
        day_count_basis: DayCountBasis,
        day_count_basis_years: DayCountBasis=DayCountBasis.ACT_ACT):

    if 'futures_rate' not in df.columns:
        df['futures_rate'] = (100 - df['price']) / 100
    if 'convexity_bps' not in df.columns:
        df.insert(df.columns.get_loc('price') + 1, 'convexity_bps', 0)
    if 'forward_rate' not in df.columns:
        df['forward_rate'] = df['futures_rate'] - df['convexity_bps'] / 10000

    df['period_years'] = year_frac(df['effective_date'], df['maturity_date'], day_count_basis)
    df['effective_date_years'] = year_frac(curve_date, df['effective_date'], day_count_basis_years)
    df['maturity_date_years'] = year_frac(curve_date, df['maturity_date'], day_count_basis_years)

    X = cash_rates_df['maturity_date_years'].values
    Y = np.log(cash_rates_df['maturity_date_discount_factor'].values)

    df.sort_values(by='effective_date', inplace=True)
    df.reset_index(inplace=True, drop=True)
    for i, row in df.iterrows():
        x = row['effective_date_years']
        interp_func = interp1d(X, Y, kind='linear', fill_value='extrapolate')
        df.at[i, 'effective_date_discount_factor'] = np.exp(interp_func(x))

        df.at[i, 'maturity_date_discount_factor'] = df.at[i, 'effective_date_discount_factor'] \
                                                    / (1 + row['forward_rate'] * row['period_years'])

        X = np.append(X, row['maturity_date_years'])
        Y = np.append(Y, np.log(df.at[i, 'maturity_date_discount_factor']))

    return df


def futures_helper_asx_90day(
        df: pd.DataFrame
    ):
    # Helper function to setup contract details for ASX 90-day bank bill futures.
    # https://www.asx.com.au/content/dam/asx/participants/derivatives-market/90-Day-bank-bill-futures-factsheet.pdf

    # From the contract code, index the effective period of the BBSW 3M fixing of the contract.

    if 'months' not in df.columns or 'years' not in df.columns:

        if 'ticker' in df.columns:

            df['ticker'] = df['ticker'].str.upper().str.strip()
            contract_code = df['ticker'].str[:2]
            assert (contract_code == 'IR').all() # 'IR' for 90-day bank bill futures

            futures_month_code = df['ticker'].str[2]
            assert futures_month_code.isin(['H', 'M', 'U', 'Z']).all() # 'H', 'M', 'U', 'Z' for Mar, Jun, Sep, Dec
            df['months'] = futures_month_code.map({'H': 3, 'M': 6, 'U': 9, 'Z': 12})

            assert df['ticker'].str.len().isin([1, 2, 4]).all()
            futures_year_code = df['ticker'].str[3:].astype(int)  # 1-4-digit year code

            def resolve_year(year_code):
                year_code_str = str(year_code)
                if len(year_code_str) == 1:
                    # Map single-digit year codes to 2020-2029
                    # TODO: does not support cases where the futures schedule spans multiple decades. i.e from 2028-2031.
                    return 2020 + int(year_code_str)
                elif len(year_code_str) == 2:
                    # Map two-digit year codes to current century
                    return 2000 + int(year_code_str)
                elif len(year_code_str) == 4:
                    # Use four-digit year codes as is
                    return int(year_code_str)
                else:
                    raise ValueError(f'Unexpected year code length: {year_code_str}')

            df['years'] = futures_year_code.apply(resolve_year)

        elif 'month_year' in df.columns:
            # SEP23, DEC23, MAR24, JUN24...
            # SEP2024, DEC2024, MAR2025, JUN2025...
            pass



    expiry_dates, settlement_dates = [], []
    for years in range(df['years'].min(), df['years'].max()+1):
        for months in [3, 6, 9, 12]:
            if ((df['years'] == years) & (df['months'] == months)).any():
                # Settlement Day = 2nd Friday of each delivery month (Mar, Jun, Sep, Dec).
                # Start from the 1st of the month and get all Fridays in the month
                month_start = pd.Timestamp(year=years, month=months, day=1)
                fridays = [d for d in pd.date_range(month_start, month_start + pd.offsets.MonthEnd(), freq='W-FRI')]
                settlement_date = fridays[1] # Get the second Friday
                settlement_dates.append(settlement_date)

                # Expiry date is the business day before settlement date
                expiry_dates.append(settlement_date - pd.offsets.BDay(1))

    df['expiry_date'] = expiry_dates
    df['settlement_date'] = settlement_dates

    # Calculate the effective period of the BBSW 3M fixing of the contract.
    df['effective_date'] = df['settlement_date']
    cal = get_busdaycal('AU-SYDNEY')
    df['maturity_date'] = workday(df['settlement_date'] + pd.DateOffset(months=3), 0, cal=cal)

    if 'price' in df.columns:
        # Per the ASX contract definition: Settlement Value = 100 - [BBSW 3M Fixing]
        # Hence: BBSW 3M Fixing implied by the price is 100 - price.
        df['futures_rate'] = (100 - df['price']) * 0.01

    return df


