# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy 
from scipy.stats import norm

from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.monte_carlo_generic import generate_rand_nbs
from frm.utils.daycount import DayCountBasis

from dataclasses import dataclass, field, InitVar
from typing import Optional
from frm.enums.utils import CompoundingFrequency
from prettytable import PrettyTable

# Notes on calibration
# The level of mean reversion should be set, so the implied volatility of the term structure is flatish per Figure 3 of [3]

# [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
#     In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
# [2] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
# [3] Gurrieri, Sebastien and Nakabayashi, Masaki and Wong, Tony, Calibration Methods of Hull-White Model (November 27, 2009).pdf


@dataclass
class HullWhite1Factor():
    zero_curve: ZeroCurve
    mean_rev_lvl: float # Mean reversion level of the short rate
    vol: float # Volatility of the short rate
    dt: Optional[float]=1e-4 # Time step for numerical differentiation. Smaller is not always better.
    num: Optional[int]=1000 # Granularity of the theta grid between pillar points. Smaller is not always better.

    # Attributes set in __post_init__
    r0: str=field(init=False)
    theta_grid: np.array=field(init=False)
    theta_spline: tuple=field(init=False) # tuple (t,c,k) used by scipy.interpolate.splev

    def __post_init__(self):
        assert self.zero_curve.interpolation_method == 'cubic_spline_on_zero_rates'

        if self.zero_curve.data['years'].min() == 0:
            self.r0 = self.zero_curve.data['nacc'].iloc[0]
            print(f"r0 set to: {round(100 * self.r0, 4)}% per the t=0, zero curve data point.\n")
        else:
            self.r0 = self.zero_curve.get_zero_rates(years=0, compounding_frequency=CompoundingFrequency.CONTINUOUS)[0]
            print(f"r0 set to: {round(100 * self.r0, 4)}% by extrapolating the zero curve data, for t=0.\n")

    def fit_theta(self,
                  dts=(1e-3,1e-4,1e-5,1e-6),
                  nums=(10, 100, 1000, 10000)):
        """Simple grid search to find the best fit for theta grid granularity and dt for numerical differentiation."""

        results = []
        for dt in dts:
            for num in nums:
                self.dt = dt
                self.num = num
                self.setup_theta()
                average_error_bps = self.calc_error_for_theta_fit(print_results=False)
                results.append([dt, num, average_error_bps])

        # Sort results by error
        results = sorted(results, key=lambda x: x[2])

        # Print error results
        print('')
        table = PrettyTable()
        table.add_column("dt", [res[0] for res in results])
        table.add_column("nums", [res[1] for res in results])
        table.add_column("errors", [round(res[2],4) for res in results])
        print(table)

        # Best fit
        dt, num, _ = results[0]
        self.dt = dt
        self.num = num


    def calc_error_for_theta_fit(self, print_results=False):
        """Calculate the basis point error between the pillar zero rates and the theta fit."""
        years_pillars = self.zero_curve.data['years'].values
        cczr_pillars = self.zero_curve.data['nacc'].values

        cczr_recalcs = np.array([self.get_zero_rate(0, yr) for yr in years_pillars])
        diff_bps = 1e4 * (cczr_pillars - cczr_recalcs)
        average_error_bps = np.sqrt(np.mean(diff_bps ** 2))

        if print_results:
            table = PrettyTable()
            print(f'\nDifferences b/t pillar zero rates and theta fit for: dt={self.dt} and num={self.num}')
            print(f'Average error (bps): {average_error_bps:.4g}')
            table.add_column('Years', [f'{yr:.3g}' for yr in years_pillars])
            table.add_column('Pillar CCZR (%)', np.round(100 * cczr_pillars, 4).tolist())
            table.add_column('Recalc CCZR (%)', np.round(100 * cczr_recalcs, 4).tolist())
            table.add_column('Diff. (bps)', np.round(diff_bps, 2).tolist())
            print(table)

        return average_error_bps


    def setup_theta(self, years_grid=None, num=None):
        """
        Setups up the theta cubic spline definition used by calc_A() per the granularity defined by n, default is 100.
        To consider - how to support a term structure of α, σ.
        """

        if years_grid is None:
            if num is None:
                num = self.num

            years_pillars = self.zero_curve.data['years'].values

            # Same number of grid points between each pillar point
            interp_values = [np.linspace(years_pillars[i], years_pillars[i + 1], num=num, endpoint=False)
                             for i in range(len(years_pillars) - 1)]
            years_grid = np.concatenate(interp_values)

        self.theta_spline = scipy.interpolate.splrep(years_grid, self.get_thetas(years=years_grid))


    def get_thetas(self, years):
        """
        Calculates the theta values for specified years per the term structure of rates in the ZeroCurve object.

        References:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 6/41
        [2] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
            In Section 3.3.1 'The Short-Rate Dynamics' on page 73 of [1] (page 121/1007 of the pdf)
        """
        α = self.mean_rev_lvl
        σ = self.vol

        # Calculate the derivative of the instantaneous forward rate  by numerical differentiation
        f = self.get_instantaneous_forward_rate(years=years)
        f_plus_dt = self.get_instantaneous_forward_rate(years=years+self.dt)
        f_minus_dt = self.get_instantaneous_forward_rate(years=years-self.dt)
        df_dt = (f_plus_dt - f_minus_dt) / (2 * self.dt)

        return df_dt \
               + α * f \
               + (σ**2 / (2*α)) * (1 - np.exp(-2 * α * years))


    def get_instantaneous_forward_rate(self, years):
        return self.zero_curve.get_instantaneous_forward_rate(years=years)


    def calc_B(self, t, T):
        """
        Calculate the ordinary differential equation (ODE) B(t,T) for the Hull-White 1-factor model.

        References:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
        [2] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
             In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
        """
        α = self.mean_rev_lvl
        return (1/α) *(1-np.exp(-α*(T- t)))


    def calc_A(self, t, T):
        """
        Calculate the ordinary differential equation (ODE) A(t,T) for the Hull-White 1-factor model.

        References:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
        [2] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
            In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
        """
        def integrand_1(t):
            return self.calc_B(t, T)**2
        def integrand_2(t):
            theta_values = scipy.interpolate.splev(t, self.theta_spline)
            return theta_values * self.calc_B(t, T)

        integrand_1_res = scipy.integrate.quad(func=integrand_1, a=t, b=T)[0]
        integrand_2_res = scipy.integrate.quad(func=integrand_2, a=t, b=T)[0]

        return 0.5 * self.vol**2 * integrand_1_res - integrand_2_res
        

    def get_discount_factor(self, t, T):
        """
        Calculates the discount factor (i.e. the zero coupon bond price).

        Reference:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
        """
        B = self.calc_B(t=t, T=T)
        A = self.calc_A(t=t, T=T)
        return np.exp(A - self.r0*B)

    def get_zero_rate(self, t, T):
        return -np.log(self.get_discount_factor(t=t, T=T)) / (T-t)


    def get_forward_rate(self, t, T):
        df_t = self.get_discount_factor(0, t)
        df_T = self.get_discount_factor(0, T)
        return (df_t / df_T - 1) / (T - t)


    def get_yield(self, t, T, rate):
        α = self.mean_rev_lvl
        tau = T - t
        return -self.calc_A(t, T) / tau + rate * (1 - np.exp(-2*α*tau)) / (α * tau)


    def get_forward_rate_by_integration(self,t, T):
        # Useful to validate the instantaneous forward rate is valid, otherwise no identifiable use case.
        return scipy.integrate.quad(func=self.zero_curve.get_instantaneous_forward_rate, a=t, b=T)[0]


    def simulate(self,
                 tau: float,
                 nb_steps: int,
                 nb_simulations: int,
                 flag_apply_antithetic_variates: bool=True,
                 random_seed: int=None):
        """
        Simulate the Δt rate, R. Note this is not the short-rate, r.
        For a small Δt, we assume R follows the same dynamics os r, i.e. dR = (θ(t) - αR)dt + σdW.
        From the rates R, we integrate to get the discount factors and zero rates for each simulation.

        Results: dict with keys:
            R: np.array
                Simulated rates for each simulation and each time step.
            years_grid: np.array
                The time steps in years.
            sim_dsc_factors: np.array
                Simulated discount factors for each simulation and year grid.
            sim_cczrs: np.array
                Simulated continuously compounded zero rates for each simulation and year grid
            averages_df: pd.DataFrame
                Averages of the discount factors and zero rates across all simulations. Should align to the term structure.

        References:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 7/41
        """

        rand_nbs = generate_rand_nbs(nb_steps=nb_steps,
                                     nb_rand_vars=1,
                                     nb_simulations=nb_simulations,
                                     flag_apply_antithetic_variates=flag_apply_antithetic_variates,
                                     random_seed=random_seed)

        Δt = tau / float(nb_steps)

        R = np.zeros((nb_steps + 1, nb_simulations))
        years_grid = np.linspace(start=0, stop=tau, num=nb_steps + 1)
        thetas = self.get_thetas(years_grid)

        # Simulate the Δt rate, R
        R[0, :] = self.r0
        for i in range(nb_steps):
            R[i + 1, :] = R[i, :] \
                          + (thetas[i] - self.mean_rev_lvl * R[i, :]) * Δt \
                          + self.vol * np.sqrt(Δt) * rand_nbs[i, :, :]

        # Integration R to get simulation discount factors.
        # Integration optimised by being is vectorised and cumulative.
        # Cumulative by using prior period integration, so only the current step is integrated in each iteration.
        sim_dsc_factors = np.full(R.shape, np.nan)
        sim_dsc_factors[0, :] = 1.0
        cumulative_integrated_R = np.full(R.shape, np.nan)

        # Initial integration at step 1
        step_nb = 1
        cumulative_integrated_R[step_nb] = scipy.integrate.simpson(
            y=R[(step_nb - 1):(step_nb + 1), :], x=years_grid[:(step_nb + 1)], axis=0)
        sim_dsc_factors[step_nb, :] = np.exp(-cumulative_integrated_R[step_nb])

        # Integration from step 2 to nb_steps
        for step_nb in range(2, nb_steps + 1):
            cumulative_integrated_R[step_nb] = cumulative_integrated_R[step_nb - 1] + scipy.integrate.simpson(
                y=R[(step_nb - 1):(step_nb + 1), :], x=years_grid[(step_nb - 1):(step_nb + 1)], axis=0)
            sim_dsc_factors[step_nb, :] = np.exp(-cumulative_integrated_R[step_nb])

        # Transform the simulated discount factors to continuously compounded zero rates
        sim_cczrs = -1 * np.log(sim_dsc_factors) / years_grid[:, np.newaxis]

        # Averages - these should align to the discount factor / zero rates term structure
        avg_sim_dsc_factors = np.mean(sim_dsc_factors, axis=1)
        avg_cczrs = -1 * np.log(avg_sim_dsc_factors) / years_grid
        averages_df = pd.DataFrame({'years': years_grid, 'discount_factor': avg_sim_dsc_factors, 'cczr': avg_cczrs})

        results = {'R': R,
                   'years_grid': years_grid,
                   'sim_dsc_factors': sim_dsc_factors,
                   'sim_cczrs': sim_cczrs,
                   'averages_df': averages_df}

        return results


    def price_zero_coupon_bond_option(self, expiry_years, maturity_years, K, cp):
        """
        Price (at time t) a European option on a zero-coupon bond using the Hull-White model.

        Parameters:
        T : float
            expiry (in years) of the option.
        S : float
            Maturity (in years) of the underlying zero-coupon bond.
        K : float
            Strike price of the bond option.
        cp : int
            1 for call, -1 for put

        Returns:
        float
            Price of the zero-coupon bond option.

        References:
        [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
            In section 3.3.2 'Bond and Option Pricing, formulae 3.40 and 3.41, page 76 (124/1007 of the pdf)
        """

        t = 0
        σ = self.vol
        α = self.mean_rev_lvl
        T = expiry_years # Per notation in [1
        S = maturity_years # Per notation in [1]

        assert S > T
        P_t_T = self.get_discount_factor(t, T)  # DF(t,T)
        P_t_S = self.get_discount_factor(t, S)  # DF(t,S)

        # Calculate bond price volatility between T and S
        σP = σ * np.sqrt((1 - np.exp(-2 * α * (T-t))) / (2 * α)) * self.calc_B(T, S)
        h = (1/σP) * np.log(P_t_S / (K * P_t_T)) + 0.5 * σP

        price = cp * (P_t_S * norm.cdf(cp*h) - K * P_t_T * norm.cdf(cp*(h - σP)))
        return price


    def get_model_analytical_price(self, t, r, capflr):
        """ Calculates the price of a Cap or Floor at valuation time t
        Args:
            t: valuation time
            r: instantaneous short rate at t
            capflr: a dictionary with Cap or Floor info
        """
        dt = capflr.end_t - capflr.start_t
        K = 1 + capflr.strike * dt

        captlets_floorlets = capflr.notional * K * self.ZCBO(t, T=capflr.start_t, S=capflr.end_t,
                                                             X=1 / K, r=r, type=capflr.capflr_type)

        return np.sum(captlets_floorlets)


    def price_optionlet(self, effective_years, termination_years, K, cp):
        """
        Prices a European optionlet (caplet/floorlet) using the HW1F model.

        Parameters:
        t1 : float
            Start of the forward rate period (caplet expiry).
        t2 : float
            End of the forward rate period (caplet payment date).
        K : float
            Cap rate or strike price of the caplet.
        cp : int
            1 for call, -1 for put
        annuity_factor : float, optional
            Multiplier to adjust the optionlet forward price to present value (default is 1).

        Returns:
        float
            Price of the optionlet.

        References:
        [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
            In section 2.6 'The Fundamental Pricing Formulas, page 41 (124&125/1007 of the pdf)

        """

        # (2.26) in [1]
        K_ = 1 / (1 + K * (termination_years - effective_years))
        px = self.price_zero_coupon_bond_option(expiry_years=effective_years,
                                                maturity_years=termination_years,
                                                K=K_,
                                                cp=cp) / K_
        return px


    def price_swaption(self):
        # TODO - follow on in Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
        pass