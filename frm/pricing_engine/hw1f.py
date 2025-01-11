# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy 
from scipy.stats import norm

from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.monte_carlo_generic import generate_rand_nbs

from dataclasses import dataclass, field
from typing import Optional
from frm.enums import CompoundingFreq
from prettytable import PrettyTable

# Notes on calibration
# The level of mean reversion should be set, so the implied volatility of the term structure is flatish per Figure 3 of [3]

# [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
#     In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
# [2] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
# [3] Gurrieri, Sebastien and Nakabayashi, Masaki and Wong, Tony, Calibration Methods of Hull-White Model (November 27, 2009).pdf


@dataclass
class HullWhite1Factor:
    zero_curve: ZeroCurve
    mean_rev_lvl: float # Mean reversion level of the short rate
    vol: float # Volatility of the short rate. As mean_rev_lvl → 0, short rate vol → bachelier vol x T.
    dt: Optional[float]=1e-4 # Time step for numerical differentiation. Smaller is not always better.
    num: Optional[int]=1000 # Granularity of the theta grid between pillar points. Smaller is not always better.

    # Attributes set in __post_init__
    r0: float=field(init=False)
    theta_spline: tuple=field(init=False) # tuple (t,c,k) used by scipy.interpolate.splev

    def __post_init__(self):
        assert self.zero_curve.interp_method in ('cubic_spline_on_ln_discount', 'cubic_spline_on_cczr')

        # Set to t=0 datapoint if available, otherwise extrapolate cubic spline to t=0.
        if self.zero_curve.pillar_df['years'].min() == 0:
            self.r0 = float(self.zero_curve.pillar_df['cczr'].iloc[0])
        else:
            self.r0 = float(self.zero_curve.get_zero_rates(years=1e-8, compounding_freq=CompoundingFreq.CONTINUOUS)[0])

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
        years_pillars = self.zero_curve.pillar_df['years'].values
        cczr_pillars = self.zero_curve.pillar_df['cczr'].values

        cczr_recalcs = np.array([-1*np.log(self.calc_discount_factor_by_solving_ode_1(0, yr))/yr for yr in years_pillars])
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

            years_pillars = self.zero_curve.pillar_df['years'].values

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
        f_t = self.get_instantaneous_forward_rate(years=years)
        f_t_plus_dt = self.get_instantaneous_forward_rate(years=years+self.dt)
        f_t_minus_dt = self.get_instantaneous_forward_rate(years=years-self.dt)
        df_dt = (f_t_plus_dt - f_t_minus_dt) / (2 * self.dt)

        return df_dt \
               + α * f_t \
               + (σ**2 / (2*α)) * (1 - np.exp(-2 * α * years))


    def get_instantaneous_forward_rate(self, years):
        return self.zero_curve.get_instantaneous_forward_rate(years=years)


    def get_forward_rate_by_integration(self,t, T):
        # Only useful for validating the instantaneous forward rate per the cubic spline is valid.
        return scipy.integrate.quad(func=self.zero_curve.get_instantaneous_forward_rate, a=t, b=T)[0]


    def simulate(self,
                 tau: float,
                 nb_steps: int,
                 nb_simulations: int,
                 flag_apply_antithetic_variates: bool=True,
                 random_seed: int=None):
        """
        Euler simulation of the Δt rate, R. Note this is not the short-rate, r.
        For a small Δt, we assume R follows the same dynamics os r, i.e. dR = (θ(t) - αR)dt + σdW.
        From the rates R, we integrate to get the discount factors and zero rates for each simulation.

       Parameters:
        ----------
        tau : float
            Total time horizon for the simulation in years.
        nb_steps : int
            Number of time steps to discretize the simulation.
        nb_simulations : int
            Number of simulation paths to generate.
        flag_apply_antithetic_variates : bool, optional
            If True, applies antithetic variates to reduce variance in the simulation. Default is True.
        random_seed : int, optional
            Seed for random number generation to ensure reproducibility. Default is None.


        Returns:
        -------
        dict
            Contains the following keys:
            - 'R': np.array
                Simulated rates for each simulation across time steps.
            - 'years_grid': np.array
                Time steps in years, from 0 to tau.
            - 'sim_dsc_factors': np.array
                Simulated discount factors for each simulation along the time grid.
            - 'sim_cczrs': np.array
                Simulated continuously compounded zero rates for each simulation along the time grid.
            - 'averages_df': pd.DataFrame
                DataFrame with average discount factors and zero rates across simulations,
                aligned to the term structure.

       Notes:
        ------
        * Uses Simpson's rule for numerical integration to compute cumulative integrals of R,
          which are then used to calculate discount factors and zero rates.
        * The average discount factors are computed across simulations and converted to zero rates
          to approximate the term structure.

        References:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 7/41
        """
        # TODO at later date: helper function to wrap euler simulation to support 1m+ simulations (which would caused memory issues if done in one go)

        rand_nbs = generate_rand_nbs(nb_steps=nb_steps,
                                     nb_rand_vars=1,
                                     nb_simulations=nb_simulations,
                                     apply_antithetic_variates=flag_apply_antithetic_variates,
                                     random_seed=random_seed)

        Δt = tau / float(nb_steps)

        R = np.zeros((nb_steps + 1, nb_simulations))
        years_grid = np.linspace(start=0, stop=tau, num=nb_steps + 1)
        thetas = self.get_thetas(years_grid)

        # Simulate the Δt rate, R through the Euler discretization of the HW1F SDE.
        # Note, R is not technically the short rate, r, but should be similar for small Δt.
        R[0, :] = self.r0
        for i in range(nb_steps):
            R[i + 1, :] = R[i, :] \
                          + (thetas[i] - self.mean_rev_lvl * R[i, :]) * Δt \
                          + self.vol * np.sqrt(Δt) * rand_nbs[i, :, :]

        # We integrate R over each simulation path, and calculate the cumulative path discount factor from t=0 to t=t.
        # The integration is optimised by being is vectorised and 'cumulative'.
        # By cumulative, we only calculate a periods integration once, and store it for future uses.
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

        # Averages - these should align to the discount factor / zero rates term structure if our discretization is effective.
        # Note the average must be done over the average discount factors (vs averaging the zero rates in each simulation).
        avg_sim_dsc_factors = np.mean(sim_dsc_factors, axis=1)
        avg_cczrs = -1 * np.log(avg_sim_dsc_factors) / years_grid
        averages_df = pd.DataFrame({'years': years_grid, 'discount_factor': avg_sim_dsc_factors, 'cczr': avg_cczrs})

        results = {'R': R,
                   'years_grid': years_grid,
                   'sim_dsc_factor': sim_dsc_factors,
                   'sim_cczr': sim_cczrs,
                   'averages_df': averages_df}

        return results


    def price_zero_coupon_bond_option(self, expiry_years, maturity_years, K, cp):
        """
        Price (at time t) a European option on a zero-coupon bond using the Hull-White model.

        Parameters:
        T : float
            expiry (in years) of the option, and the start of the underlying zero-coupon bond.
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
        # TODO at later date: test if we need to use the HW1F DF/ZC bond price function - why not just call on the ZeroCurve object.
        P_t_T = self.zero_curve.get_discount_factors(T)  # DF(t,T).
        P_t_S = self.zero_curve.get_discount_factors(S)  # DF(t,S).

        # Calculate bond price volatility between T and S
        σP = σ * np.sqrt((1 - np.exp(-2 * α * (T-t))) / (2 * α)) * self.calc_b(t0=T, T=S)
        h = (1/σP) * np.log(P_t_S / (K * P_t_T)) + 0.5 * σP

        price = cp * (P_t_S * norm.cdf(cp*h) - K * P_t_T * norm.cdf(cp*(h - σP)))
        return price


    def price_optionlet(self, effective_years, termination_years, K, cp):
        """
        Prices a European optionlet (caplet/floorlet) with using the HW1F model.
        Assumes fixing at the start and payment at the end of the effective period.

        Parameters:
        t1 : float
            Start of the forward rate period (optionlet expiry).
        t2 : float
            End of the forward rate period (optionlet payment date).
        K : float
            Strike price (cap rate / floor rate) of the optionlet.
        cp : int
            1 for caplet, -1 for floorlet
        annuity_factor : float, optional
            Multiplier to adjust the optionlet forward price to present value (default is 1).

        Returns:
        float
            Price of the optionlet.

        References:
        [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
            In section 2.6 'The Fundamental Pricing Formulas, page 41 (124&125/1007 of the pdf)

        """

        # Flip call/put perspective as:
        # - Cap = Put option on a zero-coupon bond
        # - Floor = Call option on a zero-coupon bond
        cp_ = -1 * cp

        # (2.26) in [1]
        K_ = 1 / (1 + K * (termination_years - effective_years))
        px = self.price_zero_coupon_bond_option(expiry_years=effective_years,
                                                maturity_years=termination_years,
                                                K=K_,
                                                cp=cp_) / K_
        return px


    def calc_b(self, t0, T):
        """
        Calculate b(t,T) used in the ODE's for the ZC bond price (i.e. the discount factor).

        Args:
            t0: Time t at which to evaluate the discount factor
            T: Maturity time T

        References:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
        [2] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
             In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
        """
        α = self.mean_rev_lvl
        return (1/α) *(1-np.exp(-α*(T- t0)))


    def calc_discount_factor_by_solving_ode_1(self, t0, T, r=None):
        """
        Calculates the discount factor (i.e. the zero coupon bond price), for the ODE:
        DF(t,T) = exp(a(t,T)-b(t,T) * r(t))

       Args:
            T: Maturity time T
            t0: Time t at which to evaluate the discount factor
            r: Optional short rate to use. If None and t=0, uses self.r0.
                For t>0, the short rate must be provided.

        This is not used in pricings / simulations, only for validation.
        (We call the ZeroCurve object to get discount factors / zero rates)

        Returns:
            float: The discount factor P(t,T,r)

        Reference:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
        """
        def calc_a():
            def integrand_1(t_): return self.calc_b(t_, T) ** 2
            def integrand_2(t_): return scipy.interpolate.splev(t_, self.theta_spline) * self.calc_b(t_, T)

            integrand_1_res = scipy.integrate.quad(func=integrand_1, a=t0, b=T)[0]
            # The 2nd integral is trickier. Increase limit to 100 (from default of 50) for better accuracy.
            integrand_2_res = scipy.integrate.quad(func=integrand_2, a=t0, b=T, limit=100)[0]

            return 0.5 * self.vol ** 2 * integrand_1_res - integrand_2_res

        b = self.calc_b(t0=t0, T=T)
        a = calc_a()

        if t0 == 0:
            r = self.r0
        else:
            raise ValueError('For t>0, the short rate must be provided.')

        return np.exp(a - r*b)


    def calc_discount_factor_by_solving_ode_2(self, t0, T, r=None):
        """
        Calculates the discount factor (i.e. the zero coupon bond price), for another ODE:
        DF(t,T) = exp(a(t,T)-b(t,T) * r(t))

        Reference:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 7/41
        """

        def calc_a():
            df_t = self.zero_curve.get_discount_factors(years=t0)[0]
            df_T = self.zero_curve.get_discount_factors(years=T)[0]
            f_t = self.zero_curve.get_instantaneous_forward_rate(years=t0)
            B_t_T = self.calc_b(t0, T)
            α = self.mean_rev_lvl
            σ = self.vol

            return (df_T / df_t) \
                * np.exp(
                    B_t_T * f_t - (σ ** 2 / (4 * α)) * (1 - np.exp(-2 * α * t0)) * B_t_T ** 2
                )

        if t0 == 0:
            r = self.r0
        else:
            raise ValueError('For t>0, the short rate must be provided.')

        b = self.calc_b(t0=t0, T=T)
        a2 = calc_a()
        return a2 * np.exp(- b * r)

    # def price_swaption(self):
    #     """
    #     References:
    #     [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
    #         In section 3.3 'The Hull-White Extended Vasicek Model, page 77 (125&126/1007 of the pdf)
    #     """
    #
    #     # Use Jamshidian's (1989) decomposition.
    #     # Steps:
    #     # 1. Solve the short rate, as at swaption expiry,
    #     #    that makes the underlying swap of the swaption (per the given strike), a par swap.
    #     # 2. For each fixing date of the swaption, calculate the discount factor as at the fixing date, per this short rate.
    #     #    This discount factor is used to price a zero-coupon bond option (ZCBO) for the fixing period.
    # #     # 3. Sum the ZCBO prices to get the swaption price.
    #
    #     pass
    #
    # Hull - White
    # Jamshidian
    # Decomposition
    # Implementation



    def jamshidian_decomposition(self,
                                 swaption_expiry: float,
                                 coupon_payment_times: np.ndarray,
                                 coupon_year_fractions: np.ndarray,
                                 notional_payment_time: float,
                                 fixed_rate: float) -> np.ndarray:
        """
        Implements Jamshidian decomposition for swaption pricing under Hull-White model.
        Valid for European swaptions that are:
         - vanilla (flat fixed rate, zero spread, non-amortizing)
         - have the same forward & discount curve on the floating leg of the underlying swap

        This implements equation 10.22 from Andersen & Piterbarg (2010) "Interest Rate
        Modeling", Volume II, Section 10.1 to find the critical rate x* where:
        P(T₀,Tₙ,x*) + c∑τᵢP(T₀,Tᵢ₊₁,x*) = 1

        Args:
            swaption_expiry: T₀, time to swaption expiry in years
            coupon_payment_times: Array [T₁,...,Tₙ] of the coupon payment times
            coupon_year_fractions: Array [τ₁,...,τₙ] of the year fractions for the coupon periods
            notional_payment_time: Tₙ, the time of the terminal notional payment in years (can be different to the last coupon payment)
            fixed_rate: c, the fixed rate of the underlying swap

        Returns:
            Array of critical bond prices (strikes) Kᵢ = P(T₀,Tᵢ,x*)
        """

        def _zero_function(r):
            """
            Function to find r* (x* in paper) that solves equation 10.22:
            P(T₀,Tₙ,x*) + c∑τᵢP(T₀,Tᵢ₊₁,x*) = 1
            Fixed Leg Notional + Fixed Leg Coupons = Floating Leg (floating leg is 1 if same discount & forward curves)
            """
            # P(T₀,Tₙ,x*)
            terminal_notional = self.calc_discount_factor_by_solving_ode_1(
                t0=swaption_expiry, T=notional_payment_time, r=r)

            # c∑τᵢP(T₀,Tᵢ₊₁,x*)
            fixed_leg_coupons = sum(
                fixed_rate * dcf * self.calc_discount_factor_by_solving_ode_1(
                    t0=swaption_expiry, T=T, r=r)
                for T, dcf in zip(coupon_payment_times, coupon_year_fractions)
            )

            fixed_leg = terminal_notional + fixed_leg_coupons
            floating_leg = 1 # Assume the same discount & forward curves

            return fixed_leg - floating_leg

        # Input validation
        if len(coupon_payment_times) != len(coupon_year_fractions):
            raise ValueError("Length mismatch: coupon_payment_times and coupon_year_fractions "
                             "must have the same length")

        if not all(x < y for x, y in zip(coupon_payment_times[:-1], coupon_payment_times[1:])):
            raise ValueError("coupon_payment_times must be strictly increasing")

        if notional_payment_time < coupon_payment_times[-2]:
            raise ValueError("notional_payment_time must not be earlier than the 2nd to last coupon payment")

        # Find critical rate r* (x* in paper) by solving _zero_function(r*) = 0
        from scipy.optimize import brentq
        r_star = brentq(f=_zero_function, a=-1.0, b=1.0)

        # Calculate critical zero coupon bond prices (i.e. discount factors) at the critical rate r*
        # First for all coupon payments
        coupon_strikes = np.array([
            self.calc_discount_factor_by_solving_ode_1(
                t0=swaption_expiry, T=T, r=r_star)
            for T in coupon_payment_times
        ])

        # Then for notional payment if it's different from last coupon
        if notional_payment_time != coupon_payment_times[-1]:
            notional_strike = self.calc_discount_factor_by_solving_ode_1(
                t0=swaption_expiry, T=notional_payment_time, r=r_star)
            critical_bond_prices = np.append(coupon_strikes, notional_strike)
        else:
            critical_bond_prices = coupon_strikes

        return critical_bond_prices


    def price_swaption(self,
                       swaption_expiry: float,
                       coupon_payment_times: np.ndarray,
                       coupon_year_fractions: np.ndarray,
                       notional_payment_time: float,
                       fixed_rate: float,
                       is_payer: bool = True,
                       notional: float = 1.0) -> float:
        """
        Prices a European swaption using the Hull-White model via Jamshidian decomposition.
        Valid for European swaptions that are:
         - vanilla (flat fixed rate, zero spread, non-amortizing)
         - have the same forward & discount curve on the floating leg of the underlying swap

        Args:
            swaption_expiry: T₀, time to swaption expiry in years
            coupon_payment_times: Array [T₁,...,Tₙ] of the coupon payment times
            coupon_year_fractions: Array [τ₁,...,τₙ] of the year fractions for the coupon periods
            notional_payment_time: Tₙ, the time of the terminal notional payment in years
            fixed_rate: c, the fixed rate of the underlying swap
            is_payer: True for payer swaption, False for receiver swaption
            notional: Notional amount of the underlying swap

        Returns:
            float: Price of the swaption

        References:
            [1] Andersen & Piterbarg (2010) "Interest Rate Modeling", Volume II, Section 10.1
        """
        # Get critical bond prices (strikes) from Jamshidian decomposition
        critical_bond_prices = self.jamshidian_decomposition(
            swaption_expiry=swaption_expiry,
            coupon_payment_times=coupon_payment_times,
            coupon_year_fractions=coupon_year_fractions,
            notional_payment_time=notional_payment_time,
            fixed_rate=fixed_rate
        )

        # Determine strikes and coefficients for each component bond option
        coupon_strikes = critical_bond_prices[:-1]
        notional_strike = critical_bond_prices[-1]

        # For a payer swaption:
        # - We receive the floating leg (1 at T₀)
        # - We pay the fixed leg (coupons + notional)
        # Therefore, for a payer:
        # - We're short the bond options (buying bonds at strike K)
        # For a receiver, it's the opposite
        cp = -1 if is_payer else 1

        # Price the bond options for coupon payments
        coupon_option_values = np.array([
            self.price_zero_coupon_bond_option(
                expiry_years=swaption_expiry,
                maturity_years=payment_time,
                K=strike,
                cp=cp
            ) for payment_time, strike in zip(coupon_payment_times, coupon_strikes)
        ])

        # Price the bond option for notional payment
        notional_option_value = self.price_zero_coupon_bond_option(
            expiry_years=swaption_expiry,
            maturity_years=notional_payment_time,
            K=notional_strike,
            cp=cp
        )

        # Sum up the components with their coefficients:
        # 1. Coupon payments: multiply by fixed_rate * year_fraction
        coupon_values = fixed_rate * np.sum(
            coupon_year_fractions * coupon_option_values
        )

        # 2. Add notional exchange
        swaption_value = coupon_values + notional_option_value

        return notional * swaption_value



    #####################################################################################
    # Validation functions to demonstrate the mathematical correctness of the HW1F model.
    # These are not used in the pricing / simulation functions.
    #####################################################################################


