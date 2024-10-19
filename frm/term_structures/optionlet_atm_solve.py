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


        # Bootstrap optionlet term structure

        for quote_nb in range(len(self.capfloor_vol_quote_df)):

            mask_optionlets_to_solve = self.optionlet_df['quote_nb']== quote_nb

            # Price the cap/floor using the scalar lognormal at-the-money volatility quote.
            capfloor_atm_px = self._capfloor_px_from_sln_quote(quote_nb=quote_nb, quote_columns=['atm']).at[0, 'atm']

            # Solve the normal volatility term structures pillar point.
            x0 = np.atleast_1d(black76_ln_to_normal_vol_analytical(
                F=self.capfloor_vol_quote_df.loc[quote_nb, 'F'],
                tau=self.capfloor_vol_quote_df.loc[quote_nb, 'last_optionlet_expiry_years'],
                K=self.capfloor_vol_quote_df.loc[quote_nb, 'F'],
                vol_sln=self.capfloor_vol_quote_df.loc[quote_nb, 'atm'],
                ln_shift=self.capfloor_vol_quote_df.loc[quote_nb, 'ln_shift']))

            sse_multiplier = 10e6
            res = scipy.optimize.minimize(
                fun=lambda param: self._vol_n_atm_sse(param=param, quote_nb=quote_nb, target=capfloor_atm_px, sse_multiplier=sse_multiplier),
                x0=x0,
                bounds=[(0, None)])

            while res.success is False:
                sse_multiplier /= 10
                res = scipy.optimize.minimize(
                    fun=lambda param: self._vol_n_atm_sse(param=param, quote_nb=quote_nb, target=capfloor_atm_px, sse_multiplier=sse_multiplier),
                    x0=x0,
                    bounds=[(0, None)])


            if res.success:
                vol_n_atm = res.x.item()
                if quote_nb == 0:
                    self.optionlet_df.loc[mask_optionlets_to_solve, 'vol_n_atm'] = vol_n_atm
                else:
                    # The pillar point optionlet normal volatility is set the solved value.
                    # Intra-pillar dates are linearly interpolated between the pillar points.
                    # The normal volatility for the final optionlet is the solved parameter.
                    # The normal volatility for the optionlets between the penultimate and final pillar points are linearly interpolated.
                    last_optionlet_for_prior_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == (quote_nb - 1))[::-1].idxmax()
                    last_optionlet_for_this_quote_nb = pd.Series(self.optionlet_df['quote_nb'] == quote_nb)[::-1].idxmax()
                    X = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'expiry_years'],
                         self.optionlet_df.loc[last_optionlet_for_this_quote_nb, 'expiry_years']]
                    Y = [self.optionlet_df.loc[last_optionlet_for_prior_quote_nb, 'vol_n_atm'], vol_n_atm]
                    x = self.optionlet_df.loc[mask_optionlets_to_solve, 'expiry_years'].values
                    self.optionlet_df.loc[mask_optionlets_to_solve, 'vol_n_atm'] = np.interp(x, X, Y)
            else:
                raise ValueError(f'Optimisation for quote_nb {quote_nb} failed to converge.')

            # Solve the Black76 volatility for the at-the-money optionlet from the normal volatility.
            for i, row in self.optionlet_df.loc[mask_optionlets_to_solve, :].iterrows():
                self.optionlet_df.loc[i, 'vol_sln_atm'] = normal_vol_to_black76_sln(
                    F=row['F'],
                    tau=row['expiry_years'],
                    K=row['F'],
                    vol_n=row['vol_n_atm'],
                    ln_shift=row['ln_shift'])

            print(f'quote_nb: {quote_nb}')
            print(res)
            print(self.optionlet_df.loc[mask_optionlets_to_solve, ['expiry_years','quote_nb', 'vol_n_atm', 'vol_sln_atm']])


    def _capfloor_px_from_sln_quote(self,
                                    quote_nb: int,
                                    quote_columns: Optional[list]=None) -> pd.DataFrame:
        """Get the (Black76) price of the cap/floor scalar lognormal volatility quote"""

        if quote_columns is None:
            quote_columns = self.quote_columns

        mask = self.optionlet_df['quote_nb'] <= quote_nb

        optionlet_pxs = np.full(shape=(len(self.optionlet_df[mask]), len(quote_columns)), fill_value=np.nan)

        for i, row in self.optionlet_df[mask].iterrows():
            black76_forward_px = black76(F=row['F'],
                                         tau=row['expiry_years'],
                                         r=0,  # Annuity factor is applied later
                                         cp=self.quote_cp_flag_df.loc[quote_nb, quote_columns].astype('float64').values,
                                         K=self.strikes_df.loc[quote_nb, quote_columns].astype('float64').values,
                                         vol_sln=self.capfloor_vol_quote_df.loc[quote_nb, quote_columns].astype('float64').values,
                                         ln_shift=self.ln_shift)['price']
            optionlet_pxs[i, :] = black76_forward_px * row['annuity_factor']

        capfloor_pxs = optionlet_pxs.sum(axis=0)
        return pd.DataFrame(data=capfloor_pxs.reshape(1, -1), columns=quote_columns)


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
                                             K=self.capfloor_vol_quote_df.loc[quote_nb, 'F'],
                                             vol_n=vol_n)['price']

            bachelier_atm_cap_px += bachelier_forward_px * row['coupon_term'] * row['discount_factors']

        SSE = sse_multiplier * (target - bachelier_atm_cap_px) ** 2
        return SSE




optionlet = Optionlet(
    curve_date=curve_date,
    capfloor_vol_quote_df=vol_ln_df,
    ln_shift=0.02,
    zero_curve=zero_curve,
    optionlet_frequency=PeriodFrequency.QUARTERLY,
    settlement_delay=1,
    busdaycal=busdaycal
)


NOTE = """
THIS IS SETUP TO ITERATIVELY BUILD UP THE ATM OPTIONLET VOLATILITY TERM STRUCTURE

It is functional but it seems this methodology is incorrect, observable 20Y+. 

We did it this way as an initial step, rather than solving the smile, so to not need to solve for the smile at each pillar point.

However it seems that the smile is required to be solved at each pillar point, as the ATM volatility is not constant over the tenor.




"""



#% Price the at-the-money option based on the lognormal volatility quote and solve the equivalent normal volatility.
#
# # ToDo #########################################################################
# # ToDo #########################################################################
# # This script is a completo mess
# # Create a dict that has all the quote info,
# # If lognormal_vols_df, K_df, cp_df, quote_columns where a dict, the code would be much cleaner
# # Move the setup to a new script.
# # Calc F/stikes from the zero curve
# # ToDo #########################################################################
# # ToDo #########################################################################
#
#

