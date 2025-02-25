# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy
from scipy import integrate
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional
import warnings
from frm.enums import CompoundingFreq
from prettytable import PrettyTable

from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.monte_carlo_generic import generate_rand_nbs
from frm.enums import ZeroCurveInterpMethod
from frm.utils import DEFAULT_NB_SIMULATIONS

# Notes on calibration
# The level of mean reversion should be set, so the implied volatility of the term structure is flatish per Figure 3 of [3]

# [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
#     In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
# [2] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
# [3] Gurrieri, Sebastien and Nakabayashi, Masaki and Wong, Tony, Calibration Methods of Hull-White Model (November 27, 2009).pdf



option_expiries = 1.0
bond_maturities = 5.0
strikes = np.exp(-0.01 * bond_maturities) / np.exp(-0.01 * option_expiries)
cp = 1 # 1 for call, -1 for put
mean_reversion_scalar = 0.03
volatility_scalar = 0.02
num_samples = 500000
time_step = 0.1
seed = None
skip = 0

# Create flat 1% zero curve for your implementation
years = np.linspace(0, 10, 11)
rates = np.full_like(years, 0.01)
zero_curve = ZeroCurve(
    curve_date=pd.Timestamp('2024-04-01'),
    pillar_df=pd.DataFrame({'years': years, 'zero_rate': rates}),
    compounding_freq=CompoundingFreq.CONTINUOUS,
    interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR
)


strikes = np.atleast_1d(strikes).astype(np.float64)
option_expiries = np.atleast_1d(option_expiries).astype(np.float64)
bond_maturities = np.atleast_1d(bond_maturities).astype(np.float64)
cp =  np.atleast_1d(cp).astype(np.int_)

sim_times = np.arange(time_step, np.max(option_expiries), time_step)
sim_times = np.concatenate([sim_times, np.array([np.max(option_expiries)])])

tau = bond_maturities - option_expiries
curve_times = np.unique(np.reshape(tau, -1))
curve_times.sort()

mean_reversion = np.full_like(sim_times, mean_reversion_scalar)
volatility = np.full_like(sim_times, volatility_scalar)

t = sim_times
mr_t = mean_reversion
sigma_t = volatility


def y_integral_inner_term(t0, t, vol, k):
    # TODO see last Claude Chat -
    #  e⁻²∫ᵗᵘx(s)ds ≈ e⁻²ᵏ⁽ᵗ⁻ᵘ⁾
    #  e⁻²ᵏᵗ · e²ᵏᵘ
    # y(t) = e⁻²ᵏᵗ · ∫₀ᵗ e²ᵏᵘ σ² du
    # k = χ, chi = mean reversion

    """Computes int_t0^t sigma(u)^2 exp(2*k*u) du."""
    result = (vol**2) / (2 * k) * (np.exp(2 * k * t) - np.exp(2 * k * t0))
    return result

# Note, this calculation is for a scalar volatility.
# It needs to be extended to support a term structure of volatilities.
y_t = np.exp(-2 * mr_t * t) * y_integral_inner_term(t0=0, t=t, vol=sigma_t, k=mr_t)


def ex_integral(t0, t, vol, k, y_t0):
    """
    Compute the analytical solution for the integral used in drift calculation.

    Evaluates ∫(t0→t) [exp(k*s)*y(s)] ds, where:
    y(s) = y(t0) + ∫(t0→s) [exp(-2*(s-u))*vol(u)^2] du

    This is used in models with mean reversion, such as Hull-White or extended Vasicek.

    Parameters
    ----------
    t0 : float
        Initial time point.
    t : float
        Terminal time point.
    vol : float
        Constant volatility parameter.
    k : float
        Mean reversion rate.
    y_t0 : float
        Initial value of y at time t0.

    Returns
    -------
    float
        Value of the integral.

    Notes
    -----
    Assumes constant volatility. For time-dependent volatility, a numerical
    integration approach would be required.
    """
    value = (np.exp(k * t) - np.exp(k * t0) + np.exp(2 * k * t0) * (np.exp(-k * t) - np.exp(-k * t0)))
    value = value * vol**2 / (2 * k * k) + y_t0 * (np.exp(-k * t0) - np.exp(-k * t)) / k
    return value

def conditional_mean_x_scalar(t, mr_t, sigma_t):
    # Note, this calculation is for a scalar volatility.
    # It needs to be extended to support a term structure of volatilities.
    exp_x_t = ex_integral(t0=0, t=t, vol=sigma_t, k=mr_t, y_t0=0)
    exp_x_t = (exp_x_t[1:] - exp_x_t[:-1]) * np.exp(-np.broadcast_to(mr_t, t.shape)[1:] * t[1:])
    return exp_x_t

def variance_int(t0, t, vol, k):
    """Computes int_t0^t exp(2*k*s) vol(s)^2 ds."""
    return vol * vol / (2 * k) * (np.exp(2 * k * t) - np.exp(2 * k * t0))

def conditional_variance_x_scalar(t, mr_t, sigma_t):
    # Note, this calculation is for a scalar volatility.
    # It needs to be extended to support a term structure of volatilities.
    var_x_t = variance_int(t0=0, t=t, vol=sigma_t, k=mr_t)
    var_x_t = (var_x_t[1:] - var_x_t[:-1]) * np.exp(-2 * np.broadcast_to(mr_t, t.shape)[1:] * t[1:])
    return var_x_t



sim_grid = np.concatenate((np.array([0]), sim_times))
mean_reversion = np.full_like(sim_grid, mean_reversion_scalar)
volatility = np.full_like(sim_grid, volatility_scalar)

exp_x_t = conditional_mean_x_scalar(sim_grid, mean_reversion, volatility)
var_x_t = conditional_variance_x_scalar(sim_grid, mean_reversion, volatility)

exp_x_t_check = np.array([[1.99401055e-06, 5.96413033e-06, 9.91050071e-06, 1.38332638e-05, 1.77325607e-05,
                           2.16085319e-05, 2.54613169e-05, 2.92910544e-05, 3.30978823e-05, 3.68819318e-05]])

var_x_t_check = np.array([[3.98802402e-05, 3.98802402e-05, 3.98802402e-05, 3.98802402e-05, 3.98802402e-05,
                           3.98802402e-05, 3.98802402e-05, 3.98802402e-05, 3.98802402e-05, 3.98802343e-05]])

assert np.all((np.abs(exp_x_t - exp_x_t_check) / exp_x_t_check) < 1e-6)
assert np.all((np.abs(var_x_t - var_x_t_check) / var_x_t_check) < 1e-6)

initial_x = np.zeros((num_samples, 1))
f0_t = zero_curve.get_instantaneous_forward_rate(years=sim_times[0])


dt = sim_grid[1:] - sim_grid[:-1]
nb_iterations = dt.shape[0]
keep_mask = np.array([False] + [True] * len(sim_times))
normal_draws = [np.random.multivariate_normal(mean=np.zeros(1), cov=np.eye(1), size=num_samples) for _ in range(len(dt))]

rate_paths = np.full((nb_iterations, 1, num_samples), np.nan)
rate_paths[0, 0, :] = (f0_t + initial_x).flatten()



i = 0
current_x = initial_x

while i < nb_iterations:
   normals = normal_draws[i]
   vol_x_t = np.sqrt(np.maximum(var_x_t.T[i], 0))
   vol_x_t = np.where(vol_x_t > 0.0, vol_x_t, 0.0)
   next_x = (np.exp(-mean_reversion.T[i + 1] * dt[i])
             * current_x
             + exp_x_t.T[i]
             + vol_x_t * normals)
   f_0_t = zero_curve.get_instantaneous_forward_rate(years=sim_grid[i + 1])
   rate_paths[i,0,:] = (next_x + f_0_t).flatten()
   i += 1
   current_x = next_x


average_rate = np.mean(rate_paths[-1,0,:])
print("Average of last rate: ", average_rate)



short_rate = rate_paths
num_curve_nodes = 1
num_sim_steps = len(sim_times)
mean_reversion = np.full_like(sim_times, mean_reversion_scalar)


# Compute discount bond prices using Eq. 10.18 in


# Compute the discount bond price per Equation 10.18 in
# Leif B. G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling. Volume II: Term Structure Models.
# P(t,T) = (P(0,T) / P(0,t)) * exp(-x(t) G(t,T) - 0.5 * y(t) G(t,T)^2)
# x(t) = r(t) - f(0,t)


f_0_t = zero_curve.get_instantaneous_forward_rate(years=sim_times)
f_0_t = f_0_t[:, np.newaxis, np.newaxis]  # reshapes (N,) to (N,1,1)

x_t = rate_paths - f_0_t

p_0_t = zero_curve.get_discount_factors(years=sim_times)
p_0_t_tau = zero_curve.get_discount_factors(years=(sim_times + tau)) / p_0_t
g_t_tau = (1. - np.exp(-mean_reversion * tau)) / mean_reversion

term1 = x_t * g_t_tau[:, np.newaxis, np.newaxis]
term2 = y_t * g_t_tau ** 2
term2 = term2[:, np.newaxis, np.newaxis]

p_t_tau = p_0_t_tau[:, np.newaxis, np.newaxis] * np.exp(-term1 - 0.5 * term2)




r_t = rate_paths

print("Average:", np.mean(p_t_tau[-1,:,:]))
print("SD:", np.std(p_t_tau[-1,:,:]))


dt_ = np.concat(([0.], dt[1:]))
period_discount_factors = np.exp(-r_t * dt_[:, np.newaxis, np.newaxis])
cumulative_discount_factors = np.cumprod(period_discount_factors, axis=0)

bond_option_prices = cp * np.maximum(cp * (p_t_tau[-1,:,:] - strikes), 0.) * cumulative_discount_factors[-1,:,:]
print(np.mean(bond_option_prices))


#%%

@dataclass
class HullWhite1Factor2:
    zero_curve: ZeroCurve
    mean_rev_lvl: float # Mean reversion level of the short rate
    vol: float # Volatility of the short rate. As mean_rev_lvl → 0, short rate vol → bachelier vol x T.
    dt: Optional[float]=1e-4 # Time step for numerical differentiation. Smaller is not always better.
    num: Optional[int]=1000 # Granularity of the theta grid between pillar points. Smaller is not always better.

    # Attributes set in __post_init__
    r0: float=field(init=False)
    theta_spline: tuple=field(init=False) # tuple (t,c,k) used by scipy.interpolate.splev

    def __post_init__(self):
        assert self.zero_curve.interp_method in [ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR, ZeroCurveInterpMethod.CUBIC_SPLINE_ON_LN_DISCOUNT]

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
            table.add_column('Pillar CCZR (%)', np.round(100 * cczr_pillars.astype(float), 4).tolist())
            table.add_column('Recalc CCZR (%)', np.round(100 * cczr_recalcs.astype(float), 4).tolist())
            table.add_column('Diff. (bps)', np.round(diff_bps.astype(float), 2).tolist())
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
            In Section 3.3.1 'The Short-Rate Dynamics' on page 73 of [1] (page 121/1007 of the pdf), Eq, 3.34.
        """
        α = self.mean_rev_lvl
        σ = self.vol

        # Calculate the derivative of the instantaneous forward rate  by numerical differentiation
        f_t = self.zero_curve.get_instantaneous_forward_rate(years=years)
        f_t_plus_dt = self.zero_curve.get_instantaneous_forward_rate(years=years+self.dt)
        f_t_minus_dt = self.zero_curve.get_instantaneous_forward_rate(years=years-self.dt)
        df_dt = (f_t_plus_dt - f_t_minus_dt) / (2 * self.dt)

        if True:
            f_t = np.full_like(years, self.r0)
            return α * f_t + (σ**2 / (2*α)) * (1 - np.exp(-2 * α * years))
        else:

            return df_dt \
                   + α * f_t \
                   + (σ**2 / (2*α)) * (1 - np.exp(-2 * α * years))


    def get_forward_rate_by_integration(self,t, T):
        # Only useful for validating the instantaneous forward rate per the cubic spline is valid.
        return scipy.integrate.quad(func=self.zero_curve.get_instantaneous_forward_rate, a=t, b=T)[0]


    def simulate(self,
                 tau: float,
                 nb_steps: int,
                 nb_simulations: int=DEFAULT_NB_SIMULATIONS,
                 flag_apply_antithetic_variates: bool=True,
                 random_seed: int=None,
                 method: str='euler'):
        """
        Euler simulation of the Δt rate, R, from t=0. Note this is not the short-rate, r.
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
        method : str, optional
            Method to use for simulation. Either 'euler' or 'exact'. Default is 'euler'.

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
        # As HW1F has linear drift and volatility, the higher order terms of the Milstein method are unnecessary.
        # Note, R is not technically the short rate, r, but should be similar for small Δt.
        R[0, :] = self.r0
        α = self.mean_rev_lvl
        σ = self.vol
        if method == 'euler':
            for i in range(nb_steps):
                R[i + 1, :] = R[i, :] + (thetas[i] - α * R[i, :]) * Δt + σ * np.sqrt(Δt) * rand_nbs[i, :, :]
        elif method == 'rk4':
            for i in range(nb_steps):
                k1 = Δt * (thetas[i] - α * R[i, :])
                k2 = Δt * (thetas[i] - α * (R[i, :] + 0.5 * k1))
                k3 = Δt * (thetas[i] - α * (R[i, :] + 0.5 * k2))
                k4 = Δt * (thetas[i] - α * (R[i, :] + k3))
                R[i + 1, :] = R[i, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6 + σ * np.sqrt(Δt) * rand_nbs[i, :]

        elif method == 'exact':

            def integrand(s, t1, α, theta_func):
                return theta_func(s) * np.exp(-α * (t1 - s))

            for i in range(nb_steps):
                # Calculate conditional mean
                t0 = years_grid[i]
                t1 = years_grid[i + 1]

                # Conditional mean: E[r(t1) | r(t0)] = r(t0)exp(-α(t1-t0)) + ∫[t0,t1] θ(s)exp(-α(t1-s))ds
                deterministic = R[i, :] * np.exp(-α * Δt)

                # Approximate the integral of theta using trapezoidal rule
                # s_grid = np.linspace(t0, t1, 10)  # Use 10 points for integration
                # ds = (t1 - t0) / 9
                # theta_vals = self.get_thetas(s_grid)
                # exp_vals = np.exp(-α * (t1 - s_grid))
                # theta_integral = np.trapz(theta_vals * exp_vals, dx=ds)

                theta_integral, _ = integrate.quad(integrand, t0, t1,
                                                   args=(t1, α, lambda s: np.interp(s, years_grid, thetas)))


                conditional_mean = deterministic + theta_integral
                # conditional_mean = 0.01

                # Conditional variance
                conditional_var = (σ ** 2 / (2 * α)) * (1 - np.exp(-2 * α * Δt))

                # Generate next value
                R[i + 1, :] = conditional_mean + np.sqrt(conditional_var) * rand_nbs[i, :, :]

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
        # Note the average must be done over the average discount factors (averaging zero rates does not get the same result).
        avg_sim_dsc_factors = np.mean(sim_dsc_factors, axis=1)
        avg_cczrs = -1 * np.log(avg_sim_dsc_factors) / years_grid
        averages_df = pd.DataFrame({'years': years_grid, 'discount_factor': avg_sim_dsc_factors, 'cczr': avg_cczrs})

        results = {'R': R,
                   'years_grid': years_grid,
                   'sim_dsc_factor': sim_dsc_factors,
                   'sim_cczr': sim_cczrs,
                   'averages_df': averages_df}

        return results


    def price_zero_coupon_bond_option(self,
                                      option_expiry: float,
                                      bond_maturity: float,
                                      K: float,
                                      cp: int,
                                      t0: float=0):
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
        t : float, optional
            Time at which to evaluate the bond option price. Default is 0.

        Returns:
        float
            Price of the zero-coupon bond option.

        References:
        [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
            In section 3.3.2 'Bond and Option Pricing, formulae 3.40 and 3.41, page 76 (124/1007 of the pdf)
        """

        σ = self.vol
        α = self.mean_rev_lvl
        T = np.atleast_1d(option_expiry) # Aligning to the notation in [1]
        S = np.atleast_1d(bond_maturity) # Aligning to the notation in [1]
        K = np.atleast_1d(K)
        cp = np.atleast_1d(cp)

        max_dim = np.maximum.reduce([x.size for x in [T, S, K, cp]])
        assert all((x.size == 1) or (x.size == max_dim) for x in [T, S, K, cp]), "Arrays must be of size 1 or match maximum dimension"

        if not np.all(S > T):
            warnings.warn("Bond maturities should be after the bond option expiries")

        if t0 == 0:
            P_t_T = self.zero_curve.get_discount_factors(years=T)  # Discount Bond Price (t,T) = Discount Factor(t,T)
            P_t_S = self.zero_curve.get_discount_factors(years=S) # Discount Bond Price (t,T) = DiscountFactor(t,S)
        else:
            raise ValueError('Need to assess if different functionality (i.e. calc_discount_factor_by_solving_ode_1/2) needed for P_t_T and P_t_S when t>0')

        # Calculate bond price volatility between T and S
        σP = σ * np.sqrt((1 - np.exp(-2 * α * (T-t0))) / (2 * α)) * self.calc_b(t1=T, t2=S)

        zero_σ_mask = σP <= 0.0
        h = (1/σP) * np.log(P_t_S / (K * P_t_T)) + 0.5 * σP

        price = np.zeros(max_dim)
        price[~zero_σ_mask] = (cp * (P_t_S * norm.cdf(cp*h) - K * P_t_T * norm.cdf(cp*(h - σP))))[~zero_σ_mask]

        # Handle zero variance case (set to intrinsic value)
        if np.any(zero_σ_mask):
            F = P_t_S / P_t_T # forward bond price
            price[zero_σ_mask] = np.maximum(cp * (F - K), 0.0)[zero_σ_mask]

        # Set option value to zero if maturity is before expiry
        price[bond_maturity < option_expiry] = 0.0

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
        K_ = 1 / (1 + K * (termination_years - effective_years)) # Strike for the zero-coupon bond option
        px = self.price_zero_coupon_bond_option(expiry_years=effective_years,
                                                maturity_years=termination_years,
                                                K=K_,
                                                cp=cp_) / K_
        return px


    def calc_b(self, t1, t2):
        """
        Calculate b(t1,t2) used in the ODE's for the ZC bond price (i.e. the discount factor).

        Args:
            t1: Time t (in years) at which to evaluate the discount factor
            t2: Maturity time (in years)

        References:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
        [2] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer)
             In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
        [3] QuantLib - https://rkapl123.github.io/QLAnnotatedSource/d6/df1/vasicek_8cpp_source.html

        """
        α = self.mean_rev_lvl

        if α < 1e-8:
            return t2 - t1
        else:
            return (1/α) *(1-np.exp(-α*(t2- t1)))


    def calc_discount_factor_by_solving_ode_1(self, t1, t2, r_t1=None):
        """
        Calculates the discount factor (i.e. the zero coupon bond price), for the ODE:
        DF(t1,t2) = exp(a(t1,t2)-b(t1,t2) * r(t1))
        Note the "a(t1,t2)" function is different to calc_discount_factor_by_solving_ode_2.
        This calculation numerically integrates over the period t1, t2 and the theta spline function.

        Args:
            t1: Time t at which to evaluate the discount factor
            t2: Maturity time T
            r_t1: The short rate for time t1. Can be None if t1=0 as the short rate is assumed to be r0.

        Returns:
            float: The discount factor (i.e. the discount bond price) DF(t1,t2,r_t1)

        Reference:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
        """
        def calc_a():
            def integrand_1(t):
                # Integration of b(t,t2) ** 2 over t1 to t2
                return self.calc_b(t, t2) ** 2

            def integrand_2(t):
                # Integration of b(t,t2) * θ(t) over t1 to t2
                return scipy.interpolate.splev(t, self.theta_spline) * self.calc_b(t, t2)

            integrand_1_res = scipy.integrate.quad(func=integrand_1, a=t1, b=t2)[0]
            # The 2nd integral is trickier. Increase limit to 100 (from default of 50) for better accuracy.
            integrand_2_res = scipy.integrate.quad(func=integrand_2, a=t1, b=t2, limit=100)[0]

            return 0.5 * self.vol ** 2 * integrand_1_res - integrand_2_res

        assert t2 > t1, 't2 must be greater than t1'

        if t1 == 0:
            r_t1 = self.r0
        else:
            assert r_t1 is not None, 'For t1>0, the short rate must be provided.'

        b = self.calc_b(t1, t2)
        a = calc_a()

        return np.exp(a - r_t1*b)


    def calc_discount_factor_by_solving_ode_2(self, t1, t2, r_t1=None):
        """
        Calculates the discount factor (i.e. the zero coupon bond price), for another ODE:
        DF(t1,t2) = a(t1,t2) * exp(-b(t1,t2) * r(t1))
        Note the "a(t1,t2)" function is different to calc_discount_factor_by_solving_ode_1.
        This calculation uses the zero curve object to get the instantaneous forward rate and discount factors.

        Args:
            t1: Time t at which to evaluate the discount factor
            t2: Maturity time T
            r_t1: The short rate for time t1. Can be None if t1=0 as the short rate is assumed to be r0.

        Returns:
            float: The discount factor (i.e. the discount bond price) DF(t1,t2,r_t1)

        Reference:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 7/41 for a * np.exp(- b * r_t1)
        [2] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer) Page 75, 3.39
        """

        def calc_a():
            df_t = self.zero_curve.get_discount_factors(years=t1)
            df_T = self.zero_curve.get_discount_factors(years=t2)
            f_t = self.zero_curve.get_instantaneous_forward_rate(years=t1)
            B_t_T = self.calc_b(t1, t2)
            α = self.mean_rev_lvl
            σ = self.vol

            return (df_T / df_t) \
                * np.exp(
                    B_t_T * f_t - (σ ** 2 / (4 * α)) * (1 - np.exp(-2 * α * t1)) * B_t_T ** 2
                )

        assert t2 > t1, 't2 must be greater than t1'

        if t1 == 0:
            r_t1 = self.r0
        else:
            assert r_t1 is not None, 'For t1>0, the short rate must be provided.'

        b = self.calc_b(t1, t2)
        a = calc_a()
        return a * np.exp(- b * r_t1)


    def calc_discount_factor_by_solving_ode_3(self, t1, t2, r_t1=None):
        """
        Calculates the discount factor (i.e. the zero coupon bond price), for another ODE:
        DF(t1,t2) = a(t1,t2) * exp(-b(t1,t2) * r(t1))
        Note the "a(t1,t2)" function is different to calc_discount_factor_by_solving_ode_1.
        This calculation uses the zero curve object to get the instantaneous forward rate and discount factors.

        Args:
            t1: Time t at which to evaluate the discount factor
            t2: Maturity time T
            r_t1: The short rate for time t1. Can be None if t1=0 as the short rate is assumed to be r0.

        Returns:
            float: The discount factor (i.e. the discount bond price) DF(t1,t2,r_t1)

        Reference:
        [1] MAFS525 – Computational Methods for Pricing Structured Products, Slide 7/41 for a * np.exp(- b * r_t1)
        [2] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer) Page 75, 3.39
        """




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

        def _zero_function(r_t1):
            """
            Function to find r* (x* in paper) that solves equation 10.22:
            P(T₀,Tₙ,x*) + c∑τᵢP(T₀,Tᵢ₊₁,x*) = 1
            Fixed Leg Notional + Fixed Leg Coupons = Floating Leg (floating leg is 1 if same discount & forward curves)
            """
            # P(T₀,Tₙ,x*)
            terminal_notional = self.calc_discount_factor_by_solving_ode_1(t1=swaption_expiry, t2=notional_payment_time, r_t1=r_t1)

            # c∑τᵢP(T₀,Tᵢ₊₁,x*)
            fixed_leg_coupons = sum(
                fixed_rate * dcf * self.calc_discount_factor_by_solving_ode_1(t1=swaption_expiry, t2=T, r_t1=r_t1)
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
            self.calc_discount_factor_by_solving_ode_1(t1=swaption_expiry, t2=T, r_t1=r_star)
            for T in coupon_payment_times
        ])

        # Then for notional payment if it's different from last coupon
        if notional_payment_time != coupon_payment_times[-1]:
            notional_strike = self.calc_discount_factor_by_solving_ode_1(t1=swaption_expiry, t2=notional_payment_time, r_t1=r_star)
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




