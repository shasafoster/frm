# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy
from frm.pricing_engine.black76_bachelier import black76_price, bachelier_price, black76_solve_implied_vol, bachelier_solve_implied_vol, normal_vol_to_black76_sln, black76_sln_to_normal_vol, black76_sln_to_normal_vol_analytical, normal_vol_atm_to_black76_sln_atm, VOL_SLN_BOUNDS, VOL_N_BOUNDS
from frm.pricing_engine.sabr import solve_alpha_from_sln_vol, calc_sln_vol_for_strike_from_sabr_params
from frm.utils import year_frac, clean_tenor, tenor_to_date_offset, convert_column_to_consistent_data_type, Schedule, get_schedule
from frm.enums import DayCountBasis, PeriodFreq, TermRate
from frm.term_structures.zero_curve import ZeroCurve
from typing import Optional, Union, List
import time
import numbers
from frm.term_structures.interest_rate_option_helpers import standardise_relative_quote_col_names, standardise_atmf_quote_col_names



@dataclass
class Optionlet:
    curve_date: pd.Timestamp
    quote_vol_sln: pd.DataFrame
    ln_shift: float
    zero_curve: ZeroCurve
    optionlet_freq: PeriodFreq
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

        # Identical for Cap/Floor and IRS bootstrapping
        if self.settlement_date is None and self.settlement_delay is None:
            raise ValueError('Either settlement_date or settlement_delay must be provided.')
        elif self.settlement_date is None and self.settlement_delay is not None:
            self.settlement_date = np.busday_offset(self.curve_date.to_numpy().astype('datetime64[D]'),
                                                    offsets=self.settlement_delay,
                                                    roll='following',
                                                    busdaycal=self.busdaycal)

        # TODO This should be a setting - settlement_date + freq (as below) or effective at the settlement date.
        effective_date_np = (self.settlement_date + self.optionlet_freq.date_offset).to_numpy().astype('datetime64[D]')
        effective_date = np.busday_offset(effective_date_np, offsets=0, roll='following', busdaycal=self.busdaycal)
        self.quote_vol_sln['tenor'] = self.quote_vol_sln['tenor'].apply(clean_tenor)
        self.quote_vol_sln['settlement_date'] = self.settlement_date
        self.quote_vol_sln['effective_date'] = effective_date
        self.quote_vol_sln['last_optionlet_expiry_date'] = np.nan
        self.quote_vol_sln['termination_date'] = np.nan

        for i, row in self.quote_vol_sln.iterrows():
            date_offset = tenor_to_date_offset(row['tenor'])
            termination_date = self.settlement_date + date_offset
            last_optionlet_expiry_date = termination_date - self.optionlet_freq.date_offset
            last_optionlet_expiry_date_np = last_optionlet_expiry_date.to_numpy().astype('datetime64[D]')
            termination_date_date_np = termination_date.to_numpy().astype('datetime64[D]')
            self.quote_vol_sln.at[i, 'last_optionlet_expiry_date'] = np.busday_offset(
                last_optionlet_expiry_date_np, offsets=0,roll='following', busdaycal=self.busdaycal)
            self.quote_vol_sln.at[i, 'termination_date'] = np.busday_offset(
                termination_date_date_np, offsets=0,roll='following', busdaycal=self.busdaycal)

        self.quote_vol_sln['term_years'] = year_frac(
            self.quote_vol_sln['effective_date'], self.quote_vol_sln['termination_date'], self.day_count_basis)
        self.quote_vol_sln['last_optionlet_expiry_years'] = year_frac(
            self.curve_date, self.quote_vol_sln['last_optionlet_expiry_date'],self.day_count_basis)
        self.quote_vol_sln['F'] = np.nan

        self.quote_vol_sln = convert_column_to_consistent_data_type( self.quote_vol_sln)

        first_columns = ['tenor', 'settlement_date', 'effective_date', 'last_optionlet_expiry_date', 'termination_date',
                         'term_years', 'last_optionlet_expiry_years', 'F']
        column_order = first_columns + [col for col in  self.quote_vol_sln.columns if col not in first_columns]
        self.quote_vol_sln =  self.quote_vol_sln[column_order]
        self.quote_vol_sln.sort_values(by=['termination_date'], inplace=True)
        self.quote_vol_sln.reset_index(drop=True, inplace=True)

        # Two methods of specifying quotes
        # 1. Quotes relative to the atm forward rate (e.g. ATM, ATM+/-50bps, ATM+/-100bps...)
        # 2. Absolute quotes (e.g. 2.5%, 3.0%, 3.5%...)

        # Method 1
        col_name_update, quote_str_map = standardise_relative_quote_col_names(col_names=list(self.quote_vol_sln.columns))
        self.quote_vol_sln.rename(columns=col_name_update, inplace=True)

        self.quote_vol_sln.reset_index(drop=True, inplace=True)
        self.quote_cols = list(quote_str_map.keys())
    
        nb_quotes = len(self.quote_vol_sln) - 1
        self.term_structure = get_schedule(start_date=self.quote_vol_sln.loc[nb_quotes, 'effective_date'],
                                           end_date=self.quote_vol_sln.loc[nb_quotes, 'termination_date'],
                                           freq=self.optionlet_freq,
                                           cal=self.cal)
        self.term_structure.add_period_yearfrac(day_count_basis=self.day_count_basis)

        self.term_structure['discount_factors'] = self.zero_curve.get_discount_factors(dates=self.term_structure['payment_date'])
        self.term_structure['annuity_factor'] = self.term_structure['period_yearfrac'] * self.term_structure['discount_factors']
        self.term_structure['expiry_years'] = year_frac(self.curve_date, self.term_structure['period_start'], self.day_count_basis)
        self.term_structure['F'] = self.zero_curve.get_forward_rates(period_start=self.term_structure['period_start'],
                                                                     period_end=self.term_structure['period_end'],
                                                                     forward_rate_type=TermRate.SIMPLE)

        self.term_structure[['vol_n_atm', 'vol_sln_atm', 'alpha', 'beta', 'rho', 'volvol']] = np.nan

        self.term_structure.insert(loc=0, column='quote_nb', value=np.nan)
        for i, row in self.term_structure.iterrows():
            mask = row['period_end'] <= self.quote_vol_sln['termination_date']
            if mask.any():
                last_valid_index = mask.idxmax()
            else:
                last_valid_index = 0
            self.term_structure.at[i, 'quote_nb'] = last_valid_index

        # Calculate the forward rate (pre lognormal shift) for the term structure
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
            ln_shift = self.ln_shift

            for i, row in self.term_structure[mask].iterrows():
                optionlet_pxs[i, :] = black76_price(
                    F=row['F'],
                    tau=row['expiry_years'],
                    cp=cp,
                    K=K,
                    vol_sln=vol_sln,
                    ln_shift=ln_shift)['price'] * row['annuity_factor']
            try:
                self.quote_pxs.loc[quote_nb, self.quote_cols] = optionlet_pxs.sum(axis=0)
            except:
                print(quote_nb, self.quote_cols, optionlet_pxs)
                raise

        t1 = time.time()

        # Solve the equivalent normal volatilities for to the lognormal cap/floor volatility quotes.
        # Can be parallelized as each solve is independent
        self.quote_vol_n = self.quote_vol_sln.copy()
        self.quote_vol_n[self.quote_cols] = np.nan

        def obj_func_relative_px_error(vol_n):
            """Helper function for solving the equivalent normal (bachelier) volatility for a cap/floor quote"""
            bachelier_optionlet_pxs = bachelier_price(F=F, tau=tau, cp=cp, K=K, vol_n=vol_n.item(), annuity_factor=annuity_factor)['price']
            # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
            # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
            return ((target - bachelier_optionlet_pxs.sum(axis=0)) / target) ** 2

        # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
        # Hence set obj function tol 'ftol' to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
        # We set gtol to zero, so that the optimisation is terminated based on ftol.
        obj_func_tol = 1e-5 ** 2  # 0.001%^2
        options = {'ftol': obj_func_tol, 'gtol': 0}

        for quote_nb, row in self.quote_vol_sln.iterrows():
            mask = self.term_structure['quote_nb'] <= quote_nb
            F = self.term_structure.loc[mask, 'F'].values
            tau = self.term_structure.loc[mask, 'expiry_years'].values
            annuity_factor = self.term_structure.loc[mask, 'annuity_factor'].values

            for quote_col in self.quote_cols:
                cp = self.quote_cp.loc[quote_nb, quote_col]
                K = self.quote_strikes.loc[quote_nb, quote_col]
                target = self.quote_pxs.loc[quote_nb, quote_col]

                x0 = np.atleast_1d(black76_sln_to_normal_vol_analytical(
                        F=row['F'],
                        tau=row['last_optionlet_expiry_years'],
                        K=K,
                        vol_sln=self.quote_vol_sln.loc[quote_nb, quote_col],
                        ln_shift=self.ln_shift))

                res = scipy.optimize.minimize(
                    fun=obj_func_relative_px_error,
                    x0=x0,
                    bounds=[(x0*0.8, x0*1.25)],
                    method='L-BFGS-B',
                    options=options)

                # 2nd condition required as we have overridden options to terminate based on ftol
                if res.success or abs(res.fun) < obj_func_tol:
                    self.quote_vol_n.loc[quote_nb, quote_col] = res.x.item()
                else:
                    print(res)
                    raise ValueError('Optimisation of lognormal volatility quotes, normal volatility equivalent, failed to converge')

        print('time to solve quotes equivalent normal vol:', time.time() - t1)

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
            self.term_structure.loc[mask_current, 'vol_n_atm'] = self.quote_vol_n.loc[quote_nb, 'atmf']
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

            vol_sln = calc_sln_vol_for_strike_from_sabr_params(
                    tau=optionlets['expiry_years'].values,
                    F=optionlets['F'].values,
                    alpha=optionlets['alpha'].values,
                    beta=optionlets['beta'].values,
                    rho=optionlets['rho'].values,
                    volvol=optionlets['volvol'].values,
                    K=self.quote_strikes.loc[quote_nb, 'atmf'],
                    ln_shift=self.ln_shift)

            optionlet_px = black76_price(
                F=optionlets['F'].values,
                tau=optionlets['expiry_years'].values,
                cp=self.quote_cp.loc[quote_nb, 'atmf'],
                K=self.quote_strikes.loc[quote_nb, 'atmf'],
                vol_sln=vol_sln,
                ln_shift=self.ln_shift)['price'] * optionlets['annuity_factor'].values

            N = len(optionlets) - 1
            normal_vol_prior_pillar = black76_sln_to_normal_vol(
                F=optionlets.loc[N,'F'],
                tau=optionlets.loc[N,'expiry_years'],
                K=self.quote_strikes.loc[quote_nb, 'atmf'],
                vol_sln=vol_sln[-1],
                ln_shift=self.ln_shift)

            # The target of the numerical solve is (a) - (bi)
            target = self.quote_pxs.loc[quote_nb, 'atmf'] - sum(optionlet_px)

            # Index and define the parameters of the numerical solve to variables outside the function for speed.
            X = [self.term_structure.loc[prior_pillar_idx, 'expiry_years'],
                 self.term_structure.loc[current_pillar_idx, 'expiry_years']]
            x = self.term_structure.loc[mask_current, 'expiry_years'].values
            F = self.term_structure.loc[mask_current, 'F'].values
            tau = self.term_structure.loc[mask_current, 'expiry_years'].values
            cp = self.quote_cp.loc[quote_nb, 'atmf']
            K = self.quote_strikes.loc[quote_nb, 'atmf']
            annuity_factor = self.term_structure.loc[mask_current, 'annuity_factor'].values

            def vol_n_atm_obj_func(param):
                """"Error function for the optimisation of the normal at-the-money volatility for pillar optionlet."""
                try:
                    Y = [normal_vol_prior_pillar, param.item()]
                    vol_n = np.interp(x, X, Y)
                except ValueError:
                    print(f'X: {X}, Y: {Y}, x: {x}')
                    raise
                optionlet_pxs = bachelier_price(F=F, tau=tau, cp=cp, K=K, vol_n=vol_n)['price'] * annuity_factor
                # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
                # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
                return ((target - optionlet_pxs.sum(axis=0)) / target)**2

            # (bii): Numerically solve the normal ATMF volatility for the terminal 'pillar' optionlet.
            # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
            # Hence set obj function tol 'ftol' to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
            # We set gtol to zero, so that the optimisation is terminated based on ftol.
            obj_func_tol = 1e-5**2 # 0.001%^2
            options = {'ftol': obj_func_tol, 'gtol': 0}
            res = scipy.optimize.minimize(
                fun=vol_n_atm_obj_func,
                x0=np.atleast_1d(normal_vol_prior_pillar),
                bounds=[VOL_N_BOUNDS],
                method='L-BFGS-B',
                options=options)

            # 2nd condition required as we have overridden options to terminate based on ftol
            if res.success or abs(res.fun) < obj_func_tol:
                vol_n_atm_pillar = res.x.item()
            else:
                print(res)
                raise ValueError('Optimisation of pillar optionlet normal volatility failed to converge')

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
                ln_shift=self.ln_shift)


    def _solve_sabr_params(self, quote_nb, beta_overide=None):
        """
        Function for calibrating the SABR parameters for the optionlet term structure based on cap/floor lognormal volatility quotes.
        """

        def sse_sabr_for_pillar_optionlet(self, param):
            """
            Helper function producing the sum of squared errors between
            (a) cap/floor prices based on the quoted flat/scalar lognormal volatility quotes
            (b) repricing where each component optionlet uses the volatility interpolated of its SABR smiles.
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
                alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta[idx], rho=rho[idx], volvol=volvol[idx], vol_sln_atm=row['vol_sln_atm'],ln_shift=self.ln_shift)
                vols_sln = calc_sln_vol_for_strike_from_sabr_params(tau=row['expiry_years'], F=row['F'], alpha=alpha, beta=beta[idx], rho=rho[idx], volvol=volvol[idx], K=K, ln_shift=self.ln_shift)

                optionlet_pxs[idx, :] = black76_price(
                    F=row['F'],
                    tau=row['expiry_years'],
                    cp=self.quote_cp.loc[quote_nb, self.quote_cols].astype('float64').values,
                    K=K,
                    vol_sln=vols_sln,
                    ln_shift=self.ln_shift)['price'] * row['annuity_factor']

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
                vol_sln = calc_sln_vol_for_strike_from_sabr_params(
                    tau=row['expiry_years'],
                    F=row['F'],
                    alpha=row['alpha'],
                    beta=row['beta'],
                    rho=row['rho'],
                    volvol=row['volvol'],
                    K=self.quote_strikes.loc[quote_nb, self.quote_cols].astype('float64').values,
                    ln_shift=self.ln_shift)

                prior_optionlet_pxs[i, :] = black76_price(
                    F=row['F'],
                    tau=row['expiry_years'],
                    cp=self.quote_cp.loc[quote_nb, self.quote_cols].astype('float64').values,
                    K=self.quote_strikes.loc[quote_nb, self.quote_cols].astype('float64').values,
                    vol_sln=vol_sln,
                    ln_shift=self.ln_shift)['price'] * row['annuity_factor']

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
            raise ValueError('Optimisation of SABR parameters failed to converge')

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
            alpha = solve_alpha_from_sln_vol(tau=row['expiry_years'], F=row['F'], beta=beta[idx], rho=rho[idx], volvol=volvol[idx], vol_sln_atm=row['vol_sln_atm'], ln_shift=self.ln_shift)
            self.term_structure.loc[i, 'alpha'] = alpha
            self.term_structure.loc[i, 'beta'] = beta[idx]
            self.term_structure.loc[i, 'rho'] = rho[idx]
            self.term_structure.loc[i, 'volvol'] = volvol[idx]


    def interpolate_sabr_term_structure(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """ Interpolates the SABR term structure for a schedule."""
        #TODO - Add a check to ensure the effective date is within the term structure
        # Currently we are using the objects daycountbasis, calendar etc.
        #TODO - consider how to support stubs, non-default day count basis, payment delays, etc.
        # TODO - how to consider rolled effective dates be rolled if weekends or holidays

        F = self.zero_curve.get_forward_rates(period_start=schedule_df['period_start'],
                                              period_end=schedule_df['period_end'],
                                              forward_rate_type=TermRate.SIMPLE)

        # Selecting the specific columns for linear interpolation
        cols_to_interpolate = ['beta', 'rho', 'volvol', 'vol_n_atm']

        # Linearly interpolate SABR parameters for each date in 'period_start'
        interp_sabr_params = (
            pd.concat([self.term_structure, pd.DataFrame({'period_start': schedule_df['period_start']})])
            .set_index('period_start')
            .sort_index()[cols_to_interpolate]
            .interpolate(method='index')
        ).loc[schedule_df['period_start']]

        # Remove any duplicate (occurs if duplicate across term structure and effective_dates)
        interp_sabr_params = interp_sabr_params[~interp_sabr_params.index.duplicated(keep='first')]

        interp_sabr_params.loc[:,'expiry_years'] = year_frac(self.curve_date, schedule_df['period_start'], self.day_count_basis)
        interp_sabr_params.loc[:,'F'] = F
        interp_sabr_params.loc[:,['vol_sln_atm', 'alpha']] = np.nan

        interp_sabr_params.loc[:,'vol_sln_atm'] = normal_vol_atm_to_black76_sln_atm(
            F=interp_sabr_params['F'].values,
            tau=interp_sabr_params['expiry_years'].values,
            vol_n_atm=interp_sabr_params['vol_n_atm'].values,
            ln_shift=self.ln_shift)

        for i, (period_start, row) in enumerate(interp_sabr_params.iterrows()):
            interp_sabr_params.loc[period_start, 'alpha'] = solve_alpha_from_sln_vol(
                tau=row['expiry_years'], F=row['F'], beta=row['beta'], rho=row['rho'], volvol=row['volvol'],
                vol_sln_atm=row['vol_sln_atm'], ln_shift=self.ln_shift)

        column_order = [col for col in self.term_structure.columns if col in interp_sabr_params.columns]
        interp_sabr_params = interp_sabr_params[column_order]

        # Reindex to the dates & order in schedule_df
        result = pd.concat([interp_sabr_params.loc[row['period_start']] for _, row in schedule_df.iterrows()], axis=1).T
        result.reset_index(inplace=True, drop=True)

        # Horizontal concatenation of the schedule_df and the interpolated SABR parameters
        result = pd.concat([schedule_df.reset_index(drop=True), result], axis=1)

        return result


    def _validate_strike_and_option_type(self, K, cp, schedule_length):
        # Ensure that 'K' is either a scalar or an array/list of the same length as 'effective_dates'
        if isinstance(K, (list, np.ndarray)):
            assert len(K) == schedule_length, \
                "'K' must have the same length as 'effective_dates' if it's a list or array."
        else:
            assert isinstance(K, numbers.Real), "'K' must be a numeric scalar."

        # Ensure that 'cp' is either a scalar or an array/list of the same length as 'effective_dates' and that its values are -1 or 1
        if isinstance(cp, (list, np.ndarray)):
            assert len(cp) == schedule_length, \
                "'cp' must have the same length as 'effective_dates' if it's a list or array."
            assert all(val in [-1, 1] for val in cp), "'cp' must only contain values of -1 or 1."
        else:
            assert isinstance(cp, int) and cp in [-1, 1], "'cp' must be a scalar and must be either -1 or 1."

        # Ensure both 'K' and 'cp' are either both scalars or both arrays/lists of the same length
        assert isinstance(K, (list, np.ndarray)) == isinstance(cp, (list, np.ndarray)), \
            "'K' and 'cp' must both be scalars or both arrays/lists of the same length."


    def _price_optionlets(self,
                          schedule: Schedule,
                          K: Union[float, List[float], np.ndarray],
                          cp: Union[int, List[int], np.ndarray],
                          day_count_basis: DayCountBasis,
                          vol_sln_override: Union[float, np.ndarray] = None,
                          vol_n_override: Union[float, np.ndarray] = None,
                          use_term_structure: bool = False,
                          solve_equivalent_flat_vol=False) -> pd.DataFrame:

        schedule.add_period_yearfrac(day_count_basis=day_count_basis)
        schedule_length = len(schedule.df)

        # Validate inputs
        self._validate_strike_and_option_type(K, cp, schedule_length)

        detail_df = schedule.df.copy()

        detail_df = self.interpolate_sabr_term_structure(schedule_df=detail_df)

        # Compute volatilities
        if use_term_structure:
            detail_df['vol_sln'] = calc_sln_vol_for_strike_from_sabr_params(
                tau=detail_df['expiry_years'].values,
                F=detail_df['F'].values,
                alpha=detail_df['alpha'].values,
                beta=detail_df['beta'].values,
                rho=detail_df['rho'].values,
                volvol=detail_df['volvol'].values,
                K=K,
                ln_shift=self.ln_shift
            )
        elif vol_sln_override is not None:
            if isinstance(vol_sln_override, np.ndarray) and vol_sln_override.size == 1:
                vol_sln_override = vol_sln_override.item()
            detail_df['vol_sln'] = vol_sln_override
        elif vol_n_override is not None:
            if isinstance(vol_n_override, np.ndarray) and vol_n_override.size == 1:
                vol_n_override = vol_n_override.item()
            detail_df['vol_n'] = vol_n_override

        detail_df['K'] = K
        detail_df['cp'] = cp
        detail_df['discount_factor'] = self.zero_curve.get_discount_factors(dates=detail_df['payment_date'])
        detail_df['annuity_factor'] = detail_df['period_yearfrac'] * detail_df['discount_factor']

        # Choose the appropriate pricing model
        if vol_n_override is not None:
            # Use Bachelier model
            optionlet_pricing = bachelier_price(
                F=detail_df['F'].values,
                tau=detail_df['expiry_years'].values,
                cp=detail_df['cp'].values,
                K=detail_df['K'].values,
                vol_n=detail_df['vol_n'].values,
                annuity_factor=detail_df['annuity_factor'].values,
                intrinsic_time_split=True,
                analytical_greeks=True)
        else:
            # Use Black76 model
            optionlet_pricing = black76_price(
                F=detail_df['F'].values,
                tau=detail_df['expiry_years'].values,
                cp=detail_df['cp'].values,
                K=detail_df['K'].values,
                vol_sln=detail_df['vol_sln'].values,
                ln_shift=self.ln_shift,
                annuity_factor=detail_df['annuity_factor'].values,
                intrinsic_time_split=True,
                analytical_greeks=True)

        # Assign results
        detail_df['forward_price'] = optionlet_pricing['price'] / detail_df['annuity_factor']
        detail_df['price'] = optionlet_pricing['price']
        detail_df['intrinsic'] = optionlet_pricing['intrinsic']
        detail_df['time'] = optionlet_pricing['time']
        detail_df['delta'] = optionlet_pricing['analytical_greeks']['delta']
        detail_df['vega'] = optionlet_pricing['analytical_greeks']['vega']

        if use_term_structure and solve_equivalent_flat_vol:
            target = detail_df['price'].sum()

            # TODO for extremely out the money strikes, with short tenors, this solve does not work.
            # TODO for detailed output, include the optionlet specific vol_n that matches the vol_sln.
            #  This also allows for including the normal vega at the optionlet detail.

            vol_sln_flat = black76_solve_implied_vol(
                F=detail_df['F'].values,
                tau=detail_df['expiry_years'].values,
                cp=detail_df['cp'].values,
                K=detail_df['K'].values,
                ln_shift=self.ln_shift,
                X=target,
                vol_sln_guess=detail_df['vol_sln'].mean(),
                annuity_factor=detail_df['annuity_factor'].values)

            vol_n_flat = bachelier_solve_implied_vol(
                F=detail_df['F'].values,
                tau=detail_df['expiry_years'].values,
                cp=detail_df['cp'].values,
                K=detail_df['K'].values,
                X=target,
                vol_n_guess=detail_df['vol_n_atm'].mean(),
                annuity_factor=detail_df['annuity_factor'].values)

            return detail_df, vol_sln_flat, vol_n_flat
        else:
            return detail_df

    def price_optionlets_per_term_structure(self,
                                            schedule: Schedule,
                                            K: Union[float, List[float], np.ndarray],
                                            cp: Union[int, List[int], np.ndarray],
                                            day_count_basis: DayCountBasis,
                                            solve_equivalent_flat_vol=False) -> pd.DataFrame:
        return self._price_optionlets(
            schedule=schedule,
            K=K,
            cp=cp,
            day_count_basis=day_count_basis,
            use_term_structure=True,
            solve_equivalent_flat_vol=solve_equivalent_flat_vol
        )

    def price_optionlets_per_flat_vol_sln(self,
                                          schedule: Schedule,
                                          K: Union[float, List[float], np.ndarray],
                                          cp: Union[int, List[int], np.ndarray],
                                          day_count_basis: DayCountBasis,
                                          vol_sln_override: Union[float, np.ndarray]) -> pd.DataFrame:
        return self._price_optionlets(
            schedule=schedule,
            K=K,
            cp=cp,
            day_count_basis=day_count_basis,
            vol_sln_override=vol_sln_override
        )

    def price_optionlets_per_flat_vol_n(self,
                                        schedule: Schedule,
                                        K: Union[float, List[float], np.ndarray],
                                        cp: Union[int, List[int], np.ndarray],
                                        day_count_basis: DayCountBasis,
                                        vol_n_override: Union[float, np.ndarray]) -> pd.DataFrame:
        return self._price_optionlets(
            schedule=schedule,
            K=K,
            cp=cp,
            day_count_basis=day_count_basis,
            vol_n_override=vol_n_override
        )























