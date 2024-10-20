# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
from dataclasses import dataclass, field, InitVar
import numpy as np
import pandas as pd
import scipy
import re
from frm.pricing_engine.black import black76, bachelier, normal_vol_to_black76_sln, black76_sln_to_normal_vol, black76_ln_to_normal_vol_analytical
from frm.pricing_engine.sabr import solve_alpha_from_sln_vol, calc_sln_vol_for_strike
from frm.utils.daycount import year_fraction
from frm.enums.utils import DayCountBasis, PeriodFrequency
from frm.enums.term_structures import TermRate
from frm.utils.tenor import clean_tenor, tenor_to_date_offset
from frm.utils.utilities import convert_column_to_consistent_data_type
from frm.utils.schedule import get_schedule, get_payment_dates
from frm.term_structures.zero_curve import ZeroCurve
from typing import Optional
import time
import concurrent.futures as cf


@dataclass
class Optionlet:
    curve_date: pd.Timestamp
    quote_vol_sln: pd.DataFrame
    ln_shift: float
    zero_curve: ZeroCurve
    optionlet_frequency: PeriodFrequency
    settlement_delay: Optional[int]=None
    settlement_date: Optional[pd.Timestamp]=None
    busdaycal: Optional[np.busdaycalendar]=np.busdaycalendar()


    #### Attributes set in __post_init__
    day_count_basis: DayCountBasis=field(init=False)

    quote_columns: list = field(init=False)
    quote_strikes: pd.DataFrame = field(init=False)
    quote_cp: pd.DataFrame = field(init=False)
    quote_pxs: pd.DataFrame = field(init=False)
    quote_vol_n: pd.DataFrame = field(init=False)

    term_structure: pd.DataFrame = field(init=False)



    def __post_init__(self):


        t = time.time()
        self.day_count_basis = self.zero_curve.day_count_basis

        if self.settlement_date is None and self.settlement_delay is None:
            raise ValueError('Either settlement_date or settlement_delay must be provided.')
        if self.settlement_date is None and self.settlement_delay is not None:
            self.settlement_date = np.busday_offset(self.curve_date.to_numpy().astype('datetime64[D]'),
                                                    offsets=self.settlement_delay,
                                                    roll='following',
                                                    busdaycal=self.busdaycal)

        effective_date_np = (self.settlement_date + self.optionlet_frequency.date_offset).to_numpy().astype('datetime64[D]')
        effective_date = np.busday_offset(effective_date_np, offsets=0, roll='following', busdaycal=self.busdaycal)
        self.quote_vol_sln['tenor'] = self.quote_vol_sln['tenor'].apply(clean_tenor)
        self.quote_vol_sln['settlement_date'] = self.settlement_date
        self.quote_vol_sln['effective_date'] = effective_date

        for i, row in self.quote_vol_sln.iterrows():
            date_offset = tenor_to_date_offset(row['tenor'])
            termination_date = self.settlement_date + date_offset
            last_optionlet_expiry_date = termination_date - self.optionlet_frequency.date_offset
            last_optionlet_expiry_date_np = last_optionlet_expiry_date.to_numpy().astype('datetime64[D]')
            termination_date_date_np = termination_date.to_numpy().astype('datetime64[D]')
            self.quote_vol_sln.at[i, 'last_optionlet_expiry_date'] = np.busday_offset(
                last_optionlet_expiry_date_np, offsets=0,roll='following', busdaycal=self.busdaycal)
            self.quote_vol_sln.at[i, 'termination_date'] = np.busday_offset(
                termination_date_date_np, offsets=0,roll='following', busdaycal=self.busdaycal)

        self.quote_vol_sln['term_years'] = year_fraction(
            self.quote_vol_sln['effective_date'], self.quote_vol_sln['termination_date'], self.day_count_basis)
        self.quote_vol_sln['last_optionlet_expiry_years'] = year_fraction(
            self.curve_date, self.quote_vol_sln['last_optionlet_expiry_date'],self.day_count_basis)
        self.quote_vol_sln['F'] = np.nan
        self.quote_vol_sln['ln_shift'] = self.ln_shift

        self.quote_vol_sln = convert_column_to_consistent_data_type( self.quote_vol_sln)

        first_columns = ['tenor', 'settlement_date', 'effective_date', 'last_optionlet_expiry_date', 'termination_date',
                         'term_years', 'last_optionlet_expiry_years', 'F', 'ln_shift']
        column_order = first_columns + [col for col in  self.quote_vol_sln.columns if col not in first_columns]
        self.quote_vol_sln =  self.quote_vol_sln[column_order]
        self.quote_vol_sln.sort_values(by=['termination_date'], inplace=True)
        self.quote_vol_sln.reset_index(drop=True, inplace=True)

        # Two methods of specifying quotes
        # 1. Quotes relative to the atm forward rate (e.g. ATM, ATM+/-50bps, ATM+/-100bps...)
        # 2. Absolute quotes (e.g. 2.5%, 3.0%, 3.5%...)

        # Code block for method 1
        quote_str_map = dict()
        for col_name in  self.quote_vol_sln.columns:

            # Convert column name to common data format
            bps_quote = r'[+-]?\s?\d+\s?(bps|bp)'
            percentage_quote = r'[+-]?\s?\d+(\.\d+)?\s?%'
            atm_quote = '(a|at)[ -]?(t|the)[ -]?(m|money)[ -]?(f|forward)?'

            if re.search(bps_quote, col_name):
                v = round(float(col_name.replace('bps', '').replace('bp', '').replace(' ', '')) / 10000, 8)
                new_col_name = (str(int(v * 10000)) if (round(v * 10000, 8)).is_integer() else str(
                    round(v * 10000, 8))) + 'bps'
                self.quote_vol_sln =  self.quote_vol_sln.rename(columns={col_name: new_col_name})
                quote_str_map[new_col_name] = v

            elif re.search(percentage_quote, col_name):
                v = round(float(col_name.replace('%', '').replace(' ', '')) / 100, 8)
                new_col_name = (str(int(v * 100)) if (round(v * 100, 8)).is_integer() else str(
                    round(v * 100, 8))) + 'bps'
                self.quote_vol_sln =  self.quote_vol_sln.rename(columns={col_name: new_col_name})
                quote_str_map[new_col_name] = v
            elif re.search(atm_quote, col_name):
                new_col_name = 'atm'
                self.quote_vol_sln =  self.quote_vol_sln.rename(columns={col_name: new_col_name})
                quote_str_map[new_col_name] = 0

        self.quote_cols = list(quote_str_map.keys())
    
        nb_quotes = len(self.quote_vol_sln) - 1
        self.term_structure = get_schedule(start_date=self.quote_vol_sln.loc[nb_quotes, 'effective_date'],
                                           end_date=self.quote_vol_sln.loc[nb_quotes, 'termination_date'],
                                           frequency=self.optionlet_frequency,
                                           busdaycal=self.busdaycal)
        self.term_structure['payment_dates'] = get_payment_dates(schedule=self.term_structure,
                                                                 busdaycal=self.busdaycal)
        self.term_structure['coupon_term'] = year_fraction(start_date=self.term_structure['period_start'],
                                                           end_date=self.term_structure['period_end'],
                                                           day_count_basis=self.day_count_basis)
        self.term_structure['discount_factors'] = self.zero_curve.get_discount_factors(dates=self.term_structure['payment_dates'])
        self.term_structure['annuity_factor'] = self.term_structure['coupon_term'] * self.term_structure['discount_factors']
        self.term_structure['expiry_years'] = year_fraction(self.curve_date, self.term_structure['period_start'], self.day_count_basis)
        self.term_structure['F'] = self.zero_curve.get_forward_rates(period_start=self.term_structure['period_start'],
                                                                     period_end=self.term_structure['period_end'],
                                                                     forward_rate_type=TermRate.SIMPLE)

        self.term_structure[['vol_n_atm', 'vol_sln_atm', 'ln_shift', 'alpha', 'beta', 'rho', 'volvol']] = np.nan
        self.term_structure['ln_shift'] = self.ln_shift

        self.term_structure.insert(loc=0, column='quote_nb', value=np.nan)
        for i, row in self.term_structure.iterrows():
            mask = row['period_end'] <= self.quote_vol_sln['termination_date']
            if mask.any():
                last_valid_index = mask.idxmax()
            else:
                last_valid_index = 0
            self.term_structure.at[i, 'quote_nb'] = last_valid_index


        # Calculate the forward rate (pre lognormal shift) for the cap/floor quotes
        for i, row in self.quote_vol_sln.iterrows():
            mask = (self.term_structure['period_end'] <= self.quote_vol_sln.loc[i, 'termination_date'])
            self.quote_vol_sln.loc[i, 'F'] = \
                (self.term_structure.loc[mask, 'F'] * self.term_structure.loc[mask, 'annuity_factor']).sum() \
                / self.term_structure.loc[mask, 'annuity_factor'].sum()

        # Setup a strikes dataframe for the cap/floor quotes
        self.quote_strikes = self.quote_vol_sln.copy()
        self.quote_strikes[self.quote_cols] = np.nan
        for col_name in self.quote_strikes[self.quote_cols].columns:
            self.quote_strikes[col_name] = self.quote_strikes['F'] + quote_str_map[col_name]

        # Setup a call/put flag dataframe for the cap/floor quotes
        self.quote_cp = self.quote_vol_sln.copy()
        self.quote_cp[self.quote_cols] = np.nan
        self.quote_cp[self.quote_cols] = \
            np.where(self.quote_strikes[self.quote_cols].values > self.quote_cp['F'].values[:, None], 1, -1)

        # Price the cap/floors per the lognormal volatility quotes
        self.quote_pxs = pd.DataFrame(
            data=np.full(shape=(len(self.quote_vol_sln), len(self.quote_cols)), fill_value=np.nan),
            columns=self.quote_cols)

        for quote_nb, _ in self.quote_vol_sln.iterrows():
            mask = self.term_structure['quote_nb'] <= quote_nb
            optionlet_pxs = np.full(shape=(len(self.term_structure[mask]), len(self.quote_cols)), fill_value=np.nan)
            cp = self.quote_cp.loc[quote_nb, self.quote_cols].astype('float64').values
            K = self.quote_strikes.loc[quote_nb, self.quote_cols].astype('float64').values
            vol_sln = self.quote_vol_sln.loc[quote_nb, self.quote_cols].astype('float64').values
            ln_shift = self.quote_vol_sln.loc[quote_nb, 'ln_shift']

            for i, row in self.term_structure[mask].iterrows():
                optionlet_pxs[i, :] = black76(
                    F=row['F'],
                    tau=row['expiry_years'],
                    r=0,
                    cp=cp,
                    K=K,
                    vol_sln=vol_sln,
                    ln_shift=ln_shift)['price'] * row['annuity_factor']
            self.quote_pxs.loc[quote_nb, self.quote_cols] = optionlet_pxs.sum(axis=0)


        # Solve the equivalent normal volatilities for to the lognormal cap/floor volatility quotes.
        self.quote_vol_n = self.quote_vol_sln.copy()
        self.quote_vol_n[self.quote_cols] = np.nan

        for quote_nb, row in self.quote_vol_sln.iterrows():
            for quote_col in self.quote_cols:

                x0 = np.atleast_1d(black76_ln_to_normal_vol_analytical(
                        F=row['F'],
                        tau=row['last_optionlet_expiry_years'],
                        K=self.quote_strikes.loc[quote_nb, quote_col],
                        vol_sln=self.quote_vol_sln.loc[quote_nb, quote_col],
                        ln_shift=row['ln_shift']))

                # For speed, index the parameters of minimisation objective function to variables outside the function.
                target = self.quote_pxs.loc[quote_nb, quote_col]
                mask = self.term_structure['quote_nb'] <= quote_nb
                F = self.term_structure[mask]['F']
                tau = self.term_structure[mask]['expiry_years']
                cp = self.quote_cp.loc[quote_nb, quote_col]
                K = self.quote_strikes.loc[quote_nb, quote_col]
                annuity_factor = self.term_structure[mask]['annuity_factor']
                quote_terms = (F, tau, cp, K, annuity_factor)

                res = scipy.optimize.minimize(
                    fun=lambda param: self._capfloor_black76_ln_to_normal_vol_error_function(param=param, quote_terms=quote_terms, target=target),
                    x0=x0,
                    bounds=[(x0*0.8, x0*1.25)],
                    tol=1e-4 # Prices are within 0.01% - i.e. if price is 10,000, acceptable range is (9999, 10001)
                )
                self.quote_vol_n.loc[quote_nb, quote_col] = res.x.item()



        solve_vol_n_t = []
        solve_sabr_t = []

        # Bootstrap the optionlet volatilities and fit the SABR smiles
        for quote_nb in range(len(self.quote_vol_sln)):

            if 'beta' in self.quote_vol_sln.columns and not pd.isnull(self.quote_vol_sln.loc[quote_nb, 'beta']):
                beta_overide = self.quote_vol_sln.loc[quote_nb, 'beta']
            else:
                beta_overide = None
            t1 = time.time()
            self._solve_vol_n_atm(quote_nb=quote_nb)
            t2 = time.time()
            self._solve_sabr_params(quote_nb=quote_nb, beta_overide=beta_overide)
            t3 = time.time()

            solve_vol_n_t.append(t2-t1)
            solve_sabr_t.append(t3-t2)

        print('time to solve_vol_n_atm:', np.sum(solve_vol_n_t))
        print('time to solve_sabr_params:', np.sum(solve_sabr_t))


    def _solve_vol_n_atm(self, quote_nb):
        """
        Function for bootstrapping the normal at-the-money forward (ATMF) volatilities for the optionlets term structure.

        When bootstrapping an optionlet term structure, one methodology decision is whether to
        (a) calculate alpha from the other SABR parameters, or
        (b) include alpha in the calibration of the SABR parameters.

        This function is a helper function for the optionlet bootstrapping process under decision (a).
        Method (a) requires the optionlet ATMF volatilities to be bootstrapped first,
        as the optionlet ATMF is an input to the analytical formula for alpha.

        Assumptions:
        For term of the first quote, the ATMF normal volatility of the component optionlets is set to be equal to the quote.
        For intra-quote optionlets, the ATMF normal volatility is linearly interpolated between pillar optionlets.
        """

        mask_prior = self.term_structure['quote_nb'] < quote_nb
        mask_current = self.term_structure['quote_nb'] == quote_nb
        prior_pillar_idx = pd.Series(mask_prior)[::-1].idxmax()
        current_pillar_idx = pd.Series(mask_current)[::-1].idxmax()

        if quote_nb == 0:
            # Set the ATMF normal volatility of the component optionlets of the 1st ATMF quote to be equal to the quote.
            self.term_structure.loc[mask_current, 'vol_n_atm'] = self.quote_vol_n.loc[quote_nb, 'atm']
        elif quote_nb > 0:
            # For 2nd, 3rd... ATMF quotes, an optimisation is required to solve the optionlets normal ATMF volatilities.
            #
            # We must solve (b) to equal (a) where:
            # (a) is the optionlets priced with the flat/scalar ATMF normal volatilities quote, and
            # (b) is the optionlets priced with a term structure of ATMF normal volatilities, interpolated of the SABR smile.
            #
            # (b) is composed of:
            # (bi) pricing the optionlets with the SABR smile already solved.
            # (bii) numerically solve the normal ATMF volatility at the quotes terminal 'pillar' optionlet in order for (bi) + (bii) to match (a)

            # (bi): Price the optionlets where with the SABR smile already solved.

            optionlets = self.term_structure.loc[mask_prior, :].copy()

            vol_sln = calc_sln_vol_for_strike(
                    tau=optionlets['expiry_years'].values,
                    F=optionlets['F'].values,
                    alpha=optionlets['alpha'].values,
                    beta=optionlets['beta'].values,
                    rho=optionlets['rho'].values,
                    volvol=optionlets['volvol'].values,
                    K=self.quote_strikes.loc[quote_nb, 'atm'],
                    ln_shift=optionlets['ln_shift'].values)

            optionlet_px = black76(
                F=optionlets['F'].values,
                tau=optionlets['expiry_years'].values,
                r=0,
                cp=self.quote_cp.loc[quote_nb, 'atm'],
                K=self.quote_strikes.loc[quote_nb, 'atm'],
                vol_sln=vol_sln,
                ln_shift=optionlets['ln_shift'].values
            )['price'] * optionlets['annuity_factor'].values

            N = len(optionlets) - 1
            normal_vol_prior_pillar = black76_sln_to_normal_vol(
                F=optionlets.loc[N,'F'],
                tau=optionlets.loc[N,'expiry_years'],
                K=self.quote_strikes.loc[quote_nb, 'atm'],
                vol_sln=vol_sln[-1],
                ln_shift=optionlets.loc[N,'ln_shift'],
            )

            # The target of the numerical solve is (a) - (bi)
            target = self.quote_pxs.loc[quote_nb, 'atm'] - sum(optionlet_px)

            # Index and define the parameters of the numerical solve to variables outside the function for speed.
            X = [self.term_structure.loc[prior_pillar_idx, 'expiry_years'],
                 self.term_structure.loc[current_pillar_idx, 'expiry_years']]
            x = self.term_structure.loc[mask_current, 'expiry_years'].values
            F = self.term_structure.loc[mask_current, 'F'].values
            tau = self.term_structure.loc[mask_current, 'expiry_years'].values
            cp = self.quote_cp.loc[quote_nb, 'atm']
            K = self.quote_strikes.loc[quote_nb, 'atm']
            annuity_factor = self.term_structure.loc[mask_current, 'annuity_factor'].values

            def vol_n_atm_obj_func(param):
                """"Error function for the optimisation of the normal at-the-money volatility for pillar optionlet."""
                Y = [normal_vol_prior_pillar, param.item()]
                vol_n = np.interp(x, X, Y)
                optionlet_pxs = bachelier(F=F, tau=tau, r=0, cp=cp, K=K, vol_n=vol_n)['price'] * annuity_factor

                # We want an error function that is invariant to the price.
                # The price is a function of the term & money-ness of the cap/floor quote.
                # Hence, we have used the absolute relative error between the prices.
                relative_error = np.abs(target - optionlet_pxs.sum(axis=0)) / target
                return relative_error

            # (bii): Numerically solve the normal ATMF volatility for the terminal 'pillar' optionlet.
            res = scipy.optimize.minimize(
                fun=lambda param: vol_n_atm_obj_func(param=param),
                x0=np.atleast_1d(normal_vol_prior_pillar),
                bounds=[(0.0001, None)],
                tol=1e-4) # Prices are within 0.01% - i.e. if price is 10,000, acceptable range is (9999, 10001)

            if res.success:
                vol_n_atm_pillar = res.x.item()
            else:
                print(res)
                raise ValueError('Optimisation failed to converge')

            # For the intra-pillar optionlets, interpolate the normal ATMF volatility between the pillar points.
            # Linear interpolation is used, this is an assumption. Other interpolation methods could be used.
            X = [self.term_structure.loc[prior_pillar_idx, 'expiry_years'],
                 self.term_structure.loc[current_pillar_idx, 'expiry_years']]
            Y = [self.term_structure.loc[prior_pillar_idx, 'vol_n_atm'], vol_n_atm_pillar]
            x = self.term_structure.loc[mask_current, 'expiry_years']
            self.term_structure.loc[mask_current, 'vol_n_atm'] = np.interp(x, X, Y)

        # Solve the equivalent log-normal volatilities for each optionlet.
        for i, row in self.term_structure.loc[mask_current,:].iterrows():
            self.term_structure.loc[i, 'vol_sln_atm'] = normal_vol_to_black76_sln(
                F=row['F'],
                tau=row['expiry_years'],
                K=row['F'],
                vol_n=row['vol_n_atm'],
                ln_shift=row['ln_shift'])


    def _solve_sabr_params(self, quote_nb, beta_overide=None):
        """
        Function for calibrating the SABR parameters for the optionlet term structure based on cap/floor lognormal volatility quotes.
        """

        def sse_sabr_for_pillar_optionlet(self, param):
            """
            Helper function producing the sum of squared errors between
            (a) cap/floor prices based on the quoted flat/scalar lognormal volatility quotes
            (b) repricing where each component optionlet uses the volatility interploted of its SABR smiles.
            """

            if beta_overide is None:
                beta_pillar, rho_pillar, volvol_pillar = param
            else:
                rho_pillar, volvol_pillar = param
                beta_pillar = beta_overide

            optionlet_pxs = np.full(shape=(len(self.term_structure[mask_current]), len(self.quote_cols)), fill_value=np.nan)
            K = self.quote_strikes.loc[quote_nb, self.quote_cols].astype('float64').values

            if quote_nb == 0:
                # For term of the 1st quote, solve for a flat beta, rho & volvol.
                N = len(self.term_structure[mask_current])
                beta = [beta_pillar] * N
                rho = [rho_pillar] * N
                volvol = [volvol_pillar] * N
            else:
                # The solved SABR parameters define the terminal/'pillar' optionlet for the current quote.
                # For the intra-quote optionlets, interpolate the SABR parameters between the prior and current SABR parameters.
                # Linear interpolation is applied, this is an assumption. Other interpolation methods could be used.
                X = [self.term_structure.loc[prior_pillar_idx, 'expiry_years'],
                     self.term_structure.loc[current_pillar_idx, 'expiry_years']]
                x = self.term_structure.loc[mask_current, 'expiry_years'].values

                beta = np.interp(x, X, [self.term_structure.loc[prior_pillar_idx, 'beta'], beta_pillar])
                rho = np.interp(x, X, [self.term_structure.loc[prior_pillar_idx, 'rho'], rho_pillar])
                volvol = np.interp(x, X,[self.term_structure.loc[prior_pillar_idx, 'volvol'], volvol_pillar])

            for idx, (i, row) in enumerate(self.term_structure.loc[mask_current, :].iterrows()):
                alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta[idx], rho=rho[idx], volvol=volvol[idx], vol_sln_atm=row['vol_sln_atm'],ln_shift=row['ln_shift'])
                vols_sln = calc_sln_vol_for_strike(tau=row['expiry_years'], F=row['F'], alpha=alpha, beta=beta[idx], rho=rho[idx], volvol=volvol[idx], K=K, ln_shift=row['ln_shift'])

                optionlet_pxs[idx, :] = black76(
                    F=row['F'],
                    tau=row['expiry_years'],
                    r=0,
                    cp=self.quote_cp.loc[quote_nb, self.quote_cols].astype('float64').values,
                    K=K,
                    vol_sln=vols_sln,
                    ln_shift=row['ln_shift'])['price'] * row['annuity_factor']

            relative_error_of_pxs = (target - optionlet_pxs.sum(axis=0)) / target
            return sum((relative_error_of_pxs ** 2) / len(relative_error_of_pxs))

        mask_prior = self.term_structure['quote_nb'] < quote_nb
        mask_current = self.term_structure['quote_nb'] == quote_nb
        prior_pillar_idx = pd.Series(mask_prior)[::-1].idxmax()
        current_pillar_idx = pd.Series(mask_current)[::-1].idxmax()

        # We solve the SABR smile for the pillar optionlet in order to match
        # (a) the pricing using the scalar (flat) volatility quote.
        # (b) the cap/floor quote pricing using a volatility optionlet term structure with SABR smile.

        # (b) is composed of:
        # (bi) pricing the optionlets where the SABR smile has already be solved in prior iterations.
        # (bii) numerically solve the SABR parameters at the quotes terminal 'pillar' optionlet in order for (bi) + (bii) to match (a)

        # (bi): Price the optionlets where with the SABR smile already solved.
        if quote_nb == 0:
            # For the 1st pillar point (bi)=0 so we just solve (bii) to equal (a).
            target = self.quote_pxs.loc[quote_nb, self.quote_cols]
        else:
            # For the 2nd, 3rd... pillar points, we first must solve (bi).
            prior_optionlet_pxs = np.full(shape=(len(self.term_structure[mask_prior]), len(self.quote_cols)), fill_value=np.nan)
            for i, row in self.term_structure.loc[mask_prior, :].iterrows():
                vol_sln = calc_sln_vol_for_strike(
                    tau=row['expiry_years'],
                    F=row['F'],
                    alpha=row['alpha'],
                    beta=row['beta'],
                    rho=row['rho'],
                    volvol=row['volvol'],
                    K=self.quote_strikes.loc[quote_nb, self.quote_cols].astype('float64').values,
                    ln_shift=row['ln_shift'])

                prior_optionlet_pxs[i, :] = black76(
                    F=row['F'],
                    tau=row['expiry_years'],
                    r=0,
                    cp=self.quote_cp.loc[quote_nb, self.quote_cols].astype('float64').values,
                    K=self.quote_strikes.loc[quote_nb, self.quote_cols].astype('float64').values,
                    vol_sln=vol_sln,
                    ln_shift=row['ln_shift']
            )['price'] * row['annuity_factor']

            target = self.quote_pxs.loc[quote_nb, self.quote_cols] - prior_optionlet_pxs.sum(axis=0)

        # alpha has a valid range of 0≤α≤∞, though it is analytically calculated from the other SABR parameters.
        # beta has a valid range of 0≤β≤1 and is either solved or specified.
        # rho has a valid range of -1≤ρ≤1
        # volvol has a valid range of 0<v≤∞
        x0 = np.array([0.00, 0.10]) if beta_overide is not None else np.array([0.0, 0.0, 0.1])
        bounds = [(-1.0, 1.0), (0.0001, None)] if beta_overide is not None else [(-1.0, 1.0), (-1.0, 1.0),(0.0001, None)]
        res = scipy.optimize.minimize(fun=lambda param: sse_sabr_for_pillar_optionlet(self=self, param=param),
                                      x0=x0,
                                      bounds=bounds,
                                      tol=1e-5)
        if res.success:
            beta_pillar, rho_pillar, volvol_pillar = (beta_overide, *res.x) if beta_overide is not None else res.x
        else:
            print(res)
            raise ValueError('Optimisation failed to converge')

        if quote_nb == 0:
            # For term of the 1st cap/floor quote, we assume a flat beta, rho & volvol over the term.
            N = len(self.term_structure[mask_current])
            beta = [beta_pillar] * N
            rho = [rho_pillar] * N
            volvol = [volvol_pillar] * N
        else:
            # For the 2nd, 3rd... cap/floor quotes, we interpolate the SABR parameters of intra-pillar between the prior and current pillar points.
            X = [self.term_structure.loc[prior_pillar_idx, 'expiry_years'],
                 self.term_structure.loc[current_pillar_idx, 'expiry_years']]
            x = self.term_structure.loc[mask_current, 'expiry_years'].values
            beta = np.interp(x, X, [self.term_structure.loc[prior_pillar_idx, 'beta'], beta_pillar])
            rho = np.interp(x, X, [self.term_structure.loc[prior_pillar_idx, 'rho'], rho_pillar])
            volvol = np.interp(x, X, [self.term_structure.loc[prior_pillar_idx, 'volvol'], volvol_pillar])

        # Store the optionlets SABR params in 'term_structure'
        for idx, (i, row) in enumerate(self.term_structure.loc[mask_current, :].iterrows()):
            # Alpha is analytically solved from the other SABR parameters, the ATMF optionlet volatility and the forward rate.
            alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta[idx], rho=rho[idx], volvol=volvol[idx], vol_sln_atm=row['vol_sln_atm'], ln_shift=row['ln_shift'])
            self.term_structure.loc[i, 'alpha'] = alpha
            self.term_structure.loc[i, 'beta'] = beta[idx]
            self.term_structure.loc[i, 'rho'] = rho[idx]
            self.term_structure.loc[i, 'volvol'] = volvol[idx]


    def _capfloor_black76_ln_to_normal_vol_error_function(self,
                        param: np.array,
                        quote_terms: tuple,
                        target: float):
        """Helper function for solving the equivalent normal (bachelier) volatility for a cap/floor quote"""

        F, tau, cp, K, annuity_factor = quote_terms
        bachelier_optionlet_pxs = bachelier(F=F, tau=tau, r=0, cp=cp, K=K, vol_n=param.item())['price'] * annuity_factor

        # Given we are solving on price, we want an error function that is invariant to the price.
        # The price is a function of the term & money-ness of the cap/floor quote.
        # Hence, we have used the absolute relative error between the prices.
        bachelier_px = bachelier_optionlet_pxs.sum(axis=0)
        relative_error_of_pxs = abs(target - bachelier_px) / target
        return relative_error_of_pxs





