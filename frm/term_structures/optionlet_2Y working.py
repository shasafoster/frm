# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
from dataclasses import dataclass, field, InitVar
import numpy as np
import pandas as pd
import scipy
import re
from frm.pricing_engine.sabr import fit_sabr_params_to_sln_smile
from frm.pricing_engine.black import black76, bachelier, normal_vol_to_black76_sln, black76_sln_to_normal_vol, black76_ln_to_normal_vol_analytical
from frm.pricing_engine.sabr import solve_alpha_from_sln_vol, calc_sln_vol_for_strike
from frm.term_structures.capfloor_helpers import process_capfloor_quotes
from frm.utils.business_day_calendar import get_busdaycal
from frm.utils.daycount import year_fraction
from frm.enums.utils import DayCountBasis, PeriodFrequency
from frm.enums.term_structures import TermRate
from frm.utils.tenor import clean_tenor, tenor_to_date_offset
from frm.utils.utilities import convert_column_to_consistent_data_type
from frm.utils.schedule import get_schedule, get_payment_dates
from frm.term_structures.zero_curve import ZeroCurve
from typing import Optional

import time

fp = './tests_private/term_structures/ir_vol_surface.xlsm'

vol_ln_df = pd.read_excel(io=fp, sheet_name='CapFloor')
discount_factors = pd.read_excel(io=fp, sheet_name='DFs')


curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')

zero_curve = ZeroCurve(curve_date=curve_date,
                       data=discount_factors,
                       day_count_basis=DayCountBasis.ACT_365,
                       busdaycal=busdaycal,
                       interpolation_method='linear_on_log_of_discount_factors')

vol_ln_df.columns = vol_ln_df.columns.str.lower().str.strip()
vol_ln_df = vol_ln_df.loc[vol_ln_df['field']=='vol_lognormal', :]
del vol_ln_df['field']


@dataclass
class Optionlet:
    curve_date: pd.Timestamp
    capfloor_vol_quote_df: pd.DataFrame
    ln_shift: float
    zero_curve: ZeroCurve
    optionlet_frequency: PeriodFrequency
    settlement_delay: Optional[int]=None
    settlement_date: Optional[pd.Timestamp]=None
    busdaycal: Optional[np.busdaycalendar]=np.busdaycalendar()

    # Attributes set in __post_init__
    day_count_basis: DayCountBasis=field(init=False)
    quote_columns: list = field(init=False)



    def __post_init__(self):

        self.day_count_basis = self.zero_curve.day_count_basis

        if self.settlement_date is None and self.settlement_delay is None:
            raise ValueError('Either settlement_date or settlement_delay must be provided.')
        if self.settlement_date is None and self.settlement_delay is not None:
            self.settlement_date = np.busday_offset(self.curve_date.to_numpy().astype('datetime64[D]'),
                                                    offsets=self.settlement_delay,
                                                    roll='following',
                                                    busdaycal=self.busdaycal)

        effective_date_np = (self.settlement_date + self.optionlet_frequency.date_offset).to_numpy().astype('datetime64[D]')
        effective_date = np.busday_offset(effective_date_np, offsets=0, roll='following', busdaycal=busdaycal)
        self.capfloor_vol_quote_df['tenor'] = self.capfloor_vol_quote_df['tenor'].apply(clean_tenor)
        self.capfloor_vol_quote_df['settlement_date'] = self.settlement_date
        self.capfloor_vol_quote_df['effective_date'] = effective_date

        for i, row in self.capfloor_vol_quote_df.iterrows():
            date_offset = tenor_to_date_offset(row['tenor'])
            termination_date = self.settlement_date + date_offset
            last_optionlet_expiry_date = termination_date - self.optionlet_frequency.date_offset
            last_optionlet_expiry_date_np = last_optionlet_expiry_date.to_numpy().astype('datetime64[D]')
            termination_date_date_np = termination_date.to_numpy().astype('datetime64[D]')
            self.capfloor_vol_quote_df.at[i, 'last_optionlet_expiry_date'] = np.busday_offset(
                last_optionlet_expiry_date_np, offsets=0,roll='following', busdaycal=busdaycal)
            self.capfloor_vol_quote_df.at[i, 'termination_date'] = np.busday_offset(
                termination_date_date_np, offsets=0,roll='following', busdaycal=busdaycal)

        self.capfloor_vol_quote_df['term_years'] = year_fraction(
            self.capfloor_vol_quote_df['effective_date'], self.capfloor_vol_quote_df['termination_date'], self.day_count_basis)
        self.capfloor_vol_quote_df['last_optionlet_expiry_years'] = year_fraction(
            self.curve_date, self.capfloor_vol_quote_df['last_optionlet_expiry_date'],self.day_count_basis)
        self.capfloor_vol_quote_df['F'] = np.nan
        self.capfloor_vol_quote_df['ln_shift'] = self.ln_shift

        self.capfloor_vol_quote_df = convert_column_to_consistent_data_type( self.capfloor_vol_quote_df)

        first_columns = ['tenor', 'settlement_date', 'effective_date', 'last_optionlet_expiry_date', 'termination_date',
                         'term_years', 'last_optionlet_expiry_years', 'F', 'ln_shift']
        column_order = first_columns + [col for col in  self.capfloor_vol_quote_df.columns if col not in first_columns]
        self.capfloor_vol_quote_df =  self.capfloor_vol_quote_df[column_order]
        self.capfloor_vol_quote_df.sort_values(by=['termination_date'], inplace=True)
        self.capfloor_vol_quote_df.reset_index(drop=True, inplace=True)

        # Two methods of specifying quotes
        # 1. Quotes relative to the atm forward rate (e.g. ATM, ATM+/-50bps, ATM+/-100bps...)
        # 2. Absolute quotes (e.g. 2.5%, 3.0%, 3.5%...)

        # Code block for method 1
        quote_str_map = dict()
        for col_name in  self.capfloor_vol_quote_df.columns:

            # Convert column name to common data format
            bps_quote = r'[+-]?\s?\d+\s?(bps|bp)'
            percentage_quote = r'[+-]?\s?\d+(\.\d+)?\s?%'
            atm_quote = '(a|at)[ -]?(t|the)[ -]?(m|money)[ -]?(f|forward)?'

            if re.search(bps_quote, col_name):
                v = round(float(col_name.replace('bps', '').replace('bp', '').replace(' ', '')) / 10000, 8)
                new_col_name = (str(int(v * 10000)) if (round(v * 10000, 8)).is_integer() else str(
                    round(v * 10000, 8))) + 'bps'
                self.capfloor_vol_quote_df =  self.capfloor_vol_quote_df.rename(columns={col_name: new_col_name})
                quote_str_map[new_col_name] = v

            elif re.search(percentage_quote, col_name):
                v = round(float(col_name.replace('%', '').replace(' ', '')) / 100, 8)
                new_col_name = (str(int(v * 100)) if (round(v * 100, 8)).is_integer() else str(
                    round(v * 100, 8))) + 'bps'
                self.capfloor_vol_quote_df =  self.capfloor_vol_quote_df.rename(columns={col_name: new_col_name})
                quote_str_map[new_col_name] = v
            elif re.search(atm_quote, col_name):
                new_col_name = 'atm'
                self.capfloor_vol_quote_df =  self.capfloor_vol_quote_df.rename(columns={col_name: new_col_name})
                quote_str_map[new_col_name] = 0

        self.quote_columns = list(quote_str_map.keys())


        nb_quotes = len(self.capfloor_vol_quote_df) - 1
        optionlet_df = get_schedule(start_date=self.capfloor_vol_quote_df.loc[nb_quotes, 'effective_date'],
                                    end_date=self.capfloor_vol_quote_df.loc[nb_quotes, 'termination_date'],
                                    frequency=self.optionlet_frequency,
                                    busdaycal=self.busdaycal)
        optionlet_df['payment_dates'] = get_payment_dates(schedule=optionlet_df, busdaycal=self.busdaycal)
        optionlet_df['coupon_term'] = year_fraction(optionlet_df['period_start'], optionlet_df['period_end'],
                                                    self.day_count_basis)
        optionlet_df['discount_factors'] = self.zero_curve.get_discount_factors(dates=optionlet_df['payment_dates'])
        optionlet_df['annuity_factor'] = optionlet_df['coupon_term'] * optionlet_df['discount_factors']
        optionlet_df['expiry_years'] = year_fraction(self.curve_date, optionlet_df['period_start'], self.day_count_basis)
        optionlet_df['F'] = self.zero_curve.get_forward_rates(period_start=optionlet_df['period_start'],
                                                         period_end=optionlet_df['period_end'],
                                                         forward_rate_type=TermRate.SIMPLE)
        optionlet_df['vol_n_atm'] = np.nan
        optionlet_df['vol_sln_atm'] = np.nan
        optionlet_df['ln_shift'] = self.ln_shift
        optionlet_df['alpha'] = np.nan
        optionlet_df['beta'] = np.nan
        optionlet_df['rho'] = np.nan
        optionlet_df['volvol'] = np.nan

        optionlet_df.insert(loc=0, column='quote_nb', value=np.nan)
        for i, row in optionlet_df.iterrows():
            mask = row['period_end'] <= self.capfloor_vol_quote_df['termination_date']
            if mask.any():
                last_valid_index = mask.idxmax()
            else:
                last_valid_index = 0
            optionlet_df.at[i, 'quote_nb'] = last_valid_index

        self.optionlet_df = optionlet_df
        del optionlet_df

        # Calculate the forward rate (pre lognormal shift) for the cap/floor quotes
        for i, row in self.capfloor_vol_quote_df.iterrows():
            mask = (self.optionlet_df['period_end'] <= self.capfloor_vol_quote_df.loc[i, 'termination_date'])
            self.capfloor_vol_quote_df.loc[i, 'F'] = \
                (self.optionlet_df.loc[mask, 'F'] * self.optionlet_df.loc[mask, 'annuity_factor']).sum() \
                / self.optionlet_df.loc[mask, 'annuity_factor'].sum()

        # Setup a strike dataframe
        self.strikes_df = self.capfloor_vol_quote_df.copy()
        self.strikes_df[self.quote_columns] = np.nan
        for column in self.strikes_df[quote_str_map.keys()].columns:
            self.strikes_df[column] = self.strikes_df['F'] + quote_str_map[column]

        # Setup a call/put flag dataframe
        self.quote_cp_flag_df = self.capfloor_vol_quote_df.copy()
        self.quote_cp_flag_df[self.quote_columns] = np.nan
        self.quote_cp_flag_df[self.quote_columns] = \
            np.where(self.strikes_df[self.quote_columns].values > self.quote_cp_flag_df['F'].values[:, None], 1, -1)

        # Setup log-normal shift
        self.quote_ln_shift_df = self.quote_cp_flag_df.copy()
        self.quote_ln_shift_df[self.quote_columns] = self.ln_shift #TODO refactor this, ln_shift is a smile parameter, hence should be a column in the capfloor_vol_quote_df

        # Solve the equivalent normal volatilities for the lognormal quotes
        sse_multiplier = 100 # Note, multiplier is only used in return of sse functions
        solve_method = 1 # 1 for analytical, 2 for numerical

        capfloor_pxs = np.full(shape=(len(self.capfloor_vol_quote_df), len(self.quote_columns)), fill_value=np.nan)

        for quote_nb, _ in self.capfloor_vol_quote_df.iterrows():
            K = self.strikes_df.loc[quote_nb, self.quote_columns].astype('float64').values
            vol_sln = self.capfloor_vol_quote_df.loc[quote_nb, self.quote_columns].astype('float64').values
            ln_shift = self.quote_ln_shift_df.loc[quote_nb, self.quote_columns].astype('float64').values
            cp = self.quote_cp_flag_df.loc[quote_nb, self.quote_columns].astype('float64').values
            mask = self.optionlet_df['quote_nb'] <= quote_nb

            optionlet_pxs = np.full(shape=(len(self.optionlet_df[mask]), len(self.quote_columns)), fill_value=np.nan)

            for i, row_optionlet in self.optionlet_df[mask].iterrows():
                black76_optionlet_pxs = black76(
                    F=row_optionlet['F'],
                    tau=row_optionlet['expiry_years'],
                    r=0,
                    cp=cp,
                    K=K,
                    vol_sln=vol_sln,
                    ln_shift=ln_shift)['price'] * row_optionlet['annuity_factor']
                optionlet_pxs[i, :] = black76_optionlet_pxs

            capfloor_pxs[quote_nb,:] = optionlet_pxs.sum(axis=0)

        self.capfloor_pxs = pd.DataFrame(data=capfloor_pxs, columns=self.quote_columns)



        self.capfloor_vol_n_quote_df = self.capfloor_vol_quote_df.copy()
        self.capfloor_vol_n_quote_df[self.quote_columns] = np.nan

        for quote_nb, quote_row in self.capfloor_vol_quote_df.iterrows():
            for quote in self.quote_columns:

                x0 = black76_ln_to_normal_vol_analytical(
                        F=quote_row['F'],
                        tau=quote_row['last_optionlet_expiry_years'],
                        K=self.strikes_df.loc[quote_nb, quote],
                        vol_sln=self.capfloor_vol_quote_df.loc[quote_nb, quote],
                        ln_shift=self.ln_shift)

                if solve_method == 1:
                    self.capfloor_vol_n_quote_df.loc[quote_nb, quote] = x0
                elif solve_method == 2:
                    target = self.capfloor_pxs.loc[quote_nb, quote]
                    res = scipy.optimize.minimize(
                        fun=lambda param: self._capfloor_black76_ln_to_normal_vol_sse_helper(param=param, quote_nb=quote_nb, quote=quote, target=target, sse_multiplier=sse_multiplier),
                        x0=x0,
                        bounds=[(x0*0.8, x0*1.25)])
                    self.capfloor_vol_n_quote_df.loc[quote_nb, quote] = res.x.item()





        # Bootstrap the 1st cap/floor quote
        # We assume the optionlets have a flat normal volatility over the tenor for the first quote

        quote_nb = 0
        mask = self.optionlet_df['quote_nb'] == quote_nb

        self.optionlet_df.loc[mask, 'vol_n_atm'] = self.capfloor_vol_n_quote_df.loc[quote_nb, 'atm']

        # Solve the Black76 volatility for the at-the-money optionlet from the normal volatility.
        for i, row in self.optionlet_df.loc[mask, :].iterrows():
            self.optionlet_df.loc[i, 'vol_sln_atm'] = normal_vol_to_black76_sln(
                F=row['F'],
                tau=row['expiry_years'],
                K=row['F'],
                vol_n=row['vol_n_atm'],
                ln_shift=row['ln_shift'])




        # if solve_alpha_flag:
        #     x0 = np.array([0.10,0.00, 0.10]) if beta_overide is not None else np.array([0.1, 0.0, 0.0, 0.1])
        #     bounds = [(0.0001, None), (-1.0, 1.0), (0.0001, None)] if beta_overide is not None else [(0.0001, None), (-1.0, 1.0), (-1.0, 1.0), (0.0001, None)]
        #
        #     res = scipy.optimize.minimize(
        #         fun=lambda param: self._1st_pillar_vol_sse_plus_alpha(param, quote_nb=quote_nb, sse_multiplier=sse_multiplier, beta_overide=beta_overide),
        #         x0=x0,
        #         bounds=bounds)
        #
        #     beta, alpha, rho, volvol = (beta_overide, *res.x) if beta_overide is not None else res.x
        #
        #     for i, row in self.optionlet_df.loc[mask, :].iterrows():
        #         alpha_ = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta, rho=rho, volvol=volvol,
        #                             vol_sln_atm=row['vol_sln_atm'], ln_shift=row['ln_shift'])
        #         print(alpha, alpha_)
        #         self.optionlet_df.loc[i, 'alpha'] = alpha
        #         self.optionlet_df.loc[i, 'beta'] = beta
        #         self.optionlet_df.loc[i, 'rho'] = rho
        #         self.optionlet_df.loc[i, 'volvol'] = volvol

        # params = (beta), rho, volvol
        # alpha has a valid range of 0≤α≤∞
        # beta has a valid range of 0≤β≤1
        # rho has a valid range of -1≤ρ≤1
        # volvol has a valid range of 0<v≤∞
        beta_overide = 1.0
        solve_alpha_flag = True

        if solve_alpha_flag:
            x0 = np.array([0.00, 0.10]) if beta_overide is not None else np.array([0.0, 0.0, 0.1])
            bounds = [(-1.0, 1.0), (0.0001, None)] if beta_overide is not None else [(-1.0, 1.0), (-1.0, 1.0), (0.0001, None)]

            target = self.capfloor_pxs.loc[quote_nb, self.quote_columns].values

            res = scipy.optimize.minimize(
                fun=lambda param: self._1st_pillar_vol_sse(param, quote_nb=quote_nb, target=target, sse_multiplier=1e8, beta_overide=beta_overide),
                x0=x0,
                bounds=bounds)

            beta, rho, volvol = (beta_overide, *res.x) if beta_overide is not None else res.x

            for i,row in self.optionlet_df.loc[mask, :].iterrows():
                alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta, rho=rho, volvol=volvol, vol_sln_atm=row['vol_sln_atm'], ln_shift=row['ln_shift'])
                self.optionlet_df.loc[i, 'alpha'] = alpha
                self.optionlet_df.loc[i, 'beta'] = beta
                self.optionlet_df.loc[i, 'rho'] = rho
                self.optionlet_df.loc[i, 'volvol'] = volvol


            print(res)


        # Now for bootstraping the 2nd quote, the 2Y
        # At the 2Y pillar point we have a defined smile from 0Y to 1Y and an undefined smile from 1Y to 2Y.

        # The (known) scalar 2Y atm quote maps to a (known) 2Y price, priced by optionlets on the scalar vol.

        # This must be consistent with (a) + (b) where:
        # a = [0Y, 1Y] optionlets, priced with the caplet specific vol, calcluated by interpolating the SABR smile with the atm strike/forward.
        # b = [1Y, 2Y] optionlets, priced with the caplet specific vol.
        #              - the 2Y caplet vol (for the strike) is unknown, the 1Y vol is known (calculated prior)
        #              - we solve on the 2Y vol pillar point, and interpolate the 1Y-2Y vols.
        # Normal vols are the vols used for this interpolation, hence we use bacheliers pricing formulae.

        quote_nb = 1
        K_2Y_atm = self.strikes_df.loc[quote_nb, 'atm']
        px_2Y_atm = self.capfloor_pxs.loc[quote_nb, 'atm'] # Target to solve for
        mask = self.optionlet_df['quote_nb'] < quote_nb

        optionlet_black_vol = []
        optionlet_px = []
        optionlet_normal_vol = []

        for i,row in self.optionlet_df.loc[mask, :].iterrows():

            optionlet_black_vol.append(calc_sln_vol_for_strike(
                    tau=row['expiry_years'],
                    F=row['F'],
                    alpha=row['alpha'],
                    beta=row['beta'],
                    rho=row['rho'],
                    volvol=row['volvol'],
                    K=K_2Y_atm,
                    ln_shift=row['ln_shift']))

            optionlet_px.append(black76(
                F=row['F'],
                tau=row['expiry_years'],
                r=0,
                cp=self.quote_cp_flag_df.loc[quote_nb, 'atm'],
                K=K_2Y_atm,
                vol_sln=optionlet_black_vol[i],
                ln_shift=row['ln_shift']
            )['price'] * row['annuity_factor'])

            optionlet_normal_vol.append(black76_sln_to_normal_vol(
                F=row['F'],
                tau=row['expiry_years'],
                K=K_2Y_atm,
                vol_sln=optionlet_black_vol[i],
                ln_shift=row['ln_shift']
            ))

        diff_to_solve_to_zero = px_2Y_atm - sum(optionlet_px)
        print('diff_to_solve_to_zero', diff_to_solve_to_zero)
        print('px_2Y_atm', px_2Y_atm)
        print('sum(optionlet_px)', sum(optionlet_px))
        print('optionlet_black_vol', optionlet_black_vol)
        print('optionlet_normal_vol', optionlet_normal_vol)
        print('##############################')


        # Solve for the 2Y atm vol by finding the value of caplets that equal's
        res = scipy.optimize.minimize(
            fun=lambda param: self._sse_2Y_vol_n_atm(param=param, quote_nb=quote_nb, target=diff_to_solve_to_zero, normal_vol_prior_pillar=optionlet_normal_vol[-1], sse_multiplier=1e6),
            x0=0.01,
            bounds=[(0.0001, None)])

        vol_2Y_atm_N = res.x.item()

        print('optionlet_vol',optionlet_normal_vol)
        print('vol_2Y_atm_N', vol_2Y_atm_N)

        last_optionlet_for_this_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == quote_nb)[::-1].idxmax()

        vol_black_atm_2Y = normal_vol_to_black76_sln(
            F=self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'F'],
            tau=self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'expiry_years'],
            K=self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'F'],
            vol_n=vol_2Y_atm_N,
            ln_shift=self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'ln_shift'])

        print('vol_black_atm_2Y', vol_black_atm_2Y)

        mask = self.optionlet_df['quote_nb'] == quote_nb
        last_optionlet_for_prior_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == (quote_nb - 1))[::-1].idxmax()
        last_optionlet_for_this_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == quote_nb)[::-1].idxmax()
        X = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'expiry_years'],
             self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'expiry_years']]
        Y = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'vol_n_atm'], vol_2Y_atm_N]
        x = self.optionlet_df.loc[mask, 'expiry_years']
        vol_n = np.interp(x, X, Y)

        self.optionlet_df.loc[mask, 'vol_n_atm'] = vol_n

        # Solve the Black76 volatility for the at-the-money optionlet from the normal volatility.
        for i, row in self.optionlet_df.iterrows():
            self.optionlet_df.loc[i, 'vol_sln_atm'] = normal_vol_to_black76_sln(
                F=row['F'],
                tau=row['expiry_years'],
                K=row['F'],
                vol_n=row['vol_n_atm'],
                ln_shift=row['ln_shift'])


        solve_2Y_sabr_params = True
        if solve_2Y_sabr_params:
            # Now solve SABR params
            # Now need to carefully consider the use of masks.
            # I am now solving the delta between the caplets with defined SABR params (<=1Y) and the 1Y-2Y caplets with undefined SABR params.
            #

            # Step 1. Iterate over the optionlets and calculate the prices for the optionlets with defined SABR params.
            # Hence, we get the prices for the <=1Y optionlets.
            # Step 2. Solve, the SABR params, linterping rho, beta, volvol between the pillar points for the 1Y-2Y optionlets.
            #


            quote_nb = 1
            cp = self.quote_cp_flag_df.loc[quote_nb, self.quote_columns].astype('float64').values
            K = self.strikes_df.loc[quote_nb, self.quote_columns].astype('float64').values

            # Step 1. Solve the prices of the optionlets with defined SABR params
            mask_solved_quotes = self.optionlet_df['quote_nb'] < quote_nb
            optionlet_pxs = np.full(shape=(len(self.optionlet_df[mask_solved_quotes]), len(self.quote_columns)), fill_value=np.nan)
            for i,row in self.optionlet_df.loc[mask_solved_quotes, :].iterrows():

                vol_sln = calc_sln_vol_for_strike(
                        tau=row['expiry_years'],
                        F=row['F'],
                        alpha=row['alpha'],
                        beta=row['beta'],
                        rho=row['rho'],
                        volvol=row['volvol'],
                        K=K,
                        ln_shift=row['ln_shift'])

                optionlet_pxs[i,:] = black76(
                    F=row['F'],
                    tau=row['expiry_years'],
                    r=0,
                    cp=cp,
                    K=K,
                    vol_sln=vol_sln,
                    ln_shift=row['ln_shift']
                )['price'] * row['annuity_factor']
            optionlet_pxs = optionlet_pxs.sum(axis=0)

            target = self.capfloor_pxs.loc[quote_nb, self.quote_columns] - optionlet_pxs

            print('target', target)


            beta_overide = 0.0

            x0 = np.array([0.00, 0.10]) if beta_overide is not None else np.array([0.0, 0.0, 0.1])
            bounds = [(-1.0, 1.0), (0.0001, None)] if beta_overide is not None else [(-1.0, 1.0), (-1.0, 1.0),(0.0001, None)]

            res = scipy.optimize.minimize(
                fun=lambda param: self._2nd_pillar_vol_sse_helper(param=param, quote_nb=quote_nb, target=target, sse_multiplier=1e8, beta_overide=beta_overide),
                x0=x0,
                bounds=bounds)

            beta_pillar, rho_pillar, volvol_pillar = (beta_overide, *res.x) if beta_overide is not None else res.x

            last_optionlet_for_prior_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == (quote_nb - 1))[::-1].idxmax()
            last_optionlet_for_this_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == quote_nb)[::-1].idxmax()
            X = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'expiry_years'],
                 self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'expiry_years']]
            x = self.optionlet_df.loc[mask, 'expiry_years'].values

            beta = np.interp(x, X, [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'beta'], beta_pillar])
            rho = np.interp(x, X, [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'rho'], rho_pillar])
            volvol = np.interp(x, X, [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'volvol'], volvol_pillar])



            for idx, (i, row) in enumerate(self.optionlet_df.loc[mask, :].iterrows()):
                alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta[idx], rho=rho[idx], volvol=volvol[idx],vol_sln_atm=row['vol_sln_atm'], ln_shift=row['ln_shift'])
                self.optionlet_df.loc[i, 'alpha'] = alpha
                self.optionlet_df.loc[i, 'beta'] = beta[idx]
                self.optionlet_df.loc[i, 'rho'] = rho[idx]
                self.optionlet_df.loc[i, 'volvol'] = volvol[idx]


    def _sse_2Y_vol_n_atm(self,
                       param,
                       quote_nb,
                       target,
                       normal_vol_prior_pillar,
                       sse_multiplier):

        mask = self.optionlet_df['quote_nb'] == quote_nb
        K = self.strikes_df.loc[quote_nb, 'atm']

        last_optionlet_for_prior_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == (quote_nb-1))[::-1].idxmax()
        last_optionlet_for_this_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == quote_nb)[::-1].idxmax()

        X = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'expiry_years'],
             self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'expiry_years']]
        Y = [normal_vol_prior_pillar, param.item()]
        x = self.optionlet_df.loc[mask,'expiry_years']


        vol_n = np.interp(x, X, Y)

        print(X,Y)
        print(vol_n)

        optionlet_pxs = bachelier(F=self.optionlet_df.loc[mask, 'F'],
                                 tau=self.optionlet_df.loc[mask, 'expiry_years'],
                                 r=0,
                                 cp=self.quote_cp_flag_df.loc[quote_nb, 'atm'],
                                 K=K,
                                 vol_n=vol_n)['price'] * self.optionlet_df.loc[mask, 'annuity_factor']
        optionlet_px = sum(optionlet_pxs)

        return sse_multiplier * sum((target - optionlet_px) ** 2)


    def _2nd_pillar_vol_sse_helper(self, param, quote_nb, target, sse_multiplier, beta_overide=None):
        if beta_overide is None:
            beta_pillar, rho_pillar, volvol_pillar = param
        else:
            rho_pillar, volvol_pillar = param
            beta_pillar = beta_overide

        mask = self.optionlet_df['quote_nb'] == quote_nb
        optionlet_pxs = np.full(shape=(len(self.optionlet_df[mask]), len(self.quote_columns)), fill_value=np.nan)
        cp = self.quote_cp_flag_df.loc[quote_nb, self.quote_columns].astype('float64').values,
        K = self.strikes_df.loc[quote_nb, self.quote_columns].astype('float64').values

        # The SABR params define the pillar optionlet for the quote tenor.
        # Linearly interpolate the SABR params between the prior and current quote.
        last_optionlet_for_prior_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == (quote_nb - 1))[::-1].idxmax()
        last_optionlet_for_this_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == quote_nb)[::-1].idxmax()
        X = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'expiry_years'],
             self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'expiry_years']]
        x = self.optionlet_df.loc[mask, 'expiry_years'].values

        beta = np.interp(x, X, [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'beta'], beta_pillar])
        rho = np.interp(x, X, [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'rho'], rho_pillar])
        volvol = np.interp(x, X, [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'volvol'], volvol_pillar])

        for idx, (i,row) in enumerate(self.optionlet_df.loc[mask, :].iterrows()):
            alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta[idx], rho=rho[idx], volvol=volvol[idx], vol_sln_atm=row['vol_sln_atm'], ln_shift=row['ln_shift'])
            vols_sln = calc_sln_vol_for_strike(tau=row['expiry_years'], F=row['F'], alpha=alpha, beta=beta[idx], rho=rho[idx], volvol=volvol[idx], K=K, ln_shift=row['ln_shift'])

            optionlet_pxs[idx, :] = black76(
                F=row['F'],
                tau=row['expiry_years'],
                r=0,
                cp=cp,
                K=K,
                vol_sln=vols_sln,
                ln_shift=row['ln_shift'])['price'] * row['annuity_factor']

        optionlet_pxs = optionlet_pxs.sum(axis=0)

        return sse_multiplier * sum((target - optionlet_pxs) ** 2)



    # def _1st_pillar_vol_sse_plus_alpha(self, param, quote_nb, sse_multiplier, beta_overide=None):
    #     if beta_overide is None:
    #         alpha, beta, rho, volvol = param
    #     else:
    #         alpha, rho, volvol = param
    #         beta = beta_overide
    #
    #     print(alpha, beta, rho, volvol)
    #
    #     mask = self.optionlet_df['quote_nb'] == quote_nb
    #     optionlet_pxs = np.full(shape=(len(self.optionlet_df[mask]), len(self.quote_columns)), fill_value=np.nan)
    #     cp = self.quote_cp_flag_df.loc[quote_nb, self.quote_columns].astype('float64').values,
    #     K = self.strikes_df.loc[quote_nb, self.quote_columns].astype('float64').values
    #
    #     for i,row in self.optionlet_df.loc[mask, :].iterrows():
    #         #alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta, rho=rho, volvol=volvol, vol_sln_atm=row['vol_sln_atm'], ln_shift=row['ln_shift'])
    #         sabr_vols = calc_sln_vol_for_strike(tau=row['expiry_years'], F=row['F'], alpha=alpha, beta=beta, rho=rho, volvol=volvol, K=K, ln_shift=row['ln_shift'])
    #
    #         optionlet_pxs[i, :] = black76(
    #             F=row['F'],
    #             tau=row['expiry_years'],
    #             r=0,
    #             cp=cp,
    #             K=K,
    #             vol_sln=sabr_vols,
    #             ln_shift=row['ln_shift'])['price'] * row['annuity_factor']
    #
    #     capfloor_pxs = optionlet_pxs.sum(axis=0) * sse_multiplier
    #
    #     return sum((self.capfloor_pxs.loc[quote_nb, self.quote_columns].values - capfloor_pxs) ** 2)

    def _1st_pillar_vol_sse(self, param, quote_nb, target, sse_multiplier, beta_overide=None):
        if beta_overide is None:
            beta, rho, volvol = param
        else:
            rho, volvol = param
            beta = beta_overide

        mask = self.optionlet_df['quote_nb'] == quote_nb
        optionlet_pxs = np.full(shape=(len(self.optionlet_df[mask]), len(self.quote_columns)), fill_value=np.nan)
        cp = self.quote_cp_flag_df.loc[quote_nb, self.quote_columns].astype('float64').values,
        K = self.strikes_df.loc[quote_nb, self.quote_columns].astype('float64').values

        for i,row in self.optionlet_df.loc[mask, :].iterrows():
            alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta, rho=rho, volvol=volvol, vol_sln_atm=row['vol_sln_atm'], ln_shift=row['ln_shift'])
            sabr_vols = calc_sln_vol_for_strike(tau=row['expiry_years'], F=row['F'], alpha=alpha, beta=beta, rho=rho, volvol=volvol, K=K, ln_shift=row['ln_shift'])

            optionlet_pxs[i, :] = black76(
                F=row['F'],
                tau=row['expiry_years'],
                r=0,
                cp=cp,
                K=K,
                vol_sln=sabr_vols,
                ln_shift=row['ln_shift'])['price'] * row['annuity_factor']

        optionlet_pxs = optionlet_pxs.sum(axis=0)

        return sse_multiplier * sum((target - optionlet_pxs) ** 2)




    def _capfloor_black76_ln_to_normal_vol_sse_helper(self,
                        param: np.array,
                        quote_nb: int,
                        quote: str,
                        target: float,
                        sse_multiplier: int):
        """Helper function for solving the equivalent normal (bachelier) volatility for a cap/floor quote"""

        mask = self.optionlet_df['quote_nb'] <= quote_nb

        bachelier_optionlet_pxs = bachelier(
            F=self.optionlet_df[mask]['F'],
            tau=self.optionlet_df[mask]['expiry_years'],
            r=0,
            cp=self.quote_cp_flag_df.loc[quote_nb, quote],
            K=self.strikes_df.loc[quote_nb, quote],
            vol_n=param.item())['price'] * self.optionlet_df[mask]['annuity_factor']

        bachelier_px = bachelier_optionlet_pxs.sum(axis=0)
        return sse_multiplier * (target - bachelier_px) ** 2





    def _vol_n_atm_sse(self,
                param: np.array,
                quote_nb: int,
                target: float,
                sse_multiplier: int = 1e6):
        """SSE function for solving the pillar point normal (bachelier) volatility for the optionlet term structure"""

        bachelier_atm_cap_px = 0
        mask = self.optionlet_df['quote_nb'] <= quote_nb

        for i, row in self.optionlet_df[mask].iterrows():
            if quote_nb == 0:
                # For the first pillar point, use a flat normal volatility over the tenor.
                vol_n = param
            elif row['quote_nb'] < quote_nb:
                # For optionlets that are already solved, use the solved normal volatility.
                vol_n = row['vol_n_atm']
            elif row['quote_nb'] == quote_nb:
                # The normal volatility for the final optionlet is the solved parameter.
                # The normal volatility for the optionlets between the penultimate and final pillar points are linearly interpolated.
                last_optionlet_for_prior_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == (row['quote_nb']-1))[::-1].idxmax()
                last_optionlet_for_this_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == row['quote_nb'])[::-1].idxmax()
                X = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'expiry_years'],
                     self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'expiry_years']]
                Y = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'vol_n_atm'], param.item()]
                x = row['expiry_years']
                vol_n = np.interp(x, X, Y)
            else:
                raise ValueError(f'Invalid quote_nb {quote_nb}')

            bachelier_forward_px = bachelier(F=row['F'],
                                             tau=row['expiry_years'],
                                             r=0,
                                             cp=self.quote_cp_flag_df.loc[quote_nb, 'atm'],
                                             K=self.strikes_df.loc[quote_nb, 'F'],
                                             vol_n=vol_n)['price']

            bachelier_atm_cap_px += bachelier_forward_px * row['coupon_term'] * row['discount_factors']

        return sse_multiplier * (target - bachelier_atm_cap_px) ** 2






optionlet = Optionlet(
    curve_date=curve_date,
    capfloor_vol_quote_df=vol_ln_df,
    ln_shift=0.02,
    zero_curve=zero_curve,
    optionlet_frequency=PeriodFrequency.QUARTERLY,
    settlement_delay=1,
    busdaycal=busdaycal
)






