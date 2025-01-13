# This is a script version of the HW1F notebook, hw1f.ipynb
import time

#%% Imports
import numpy as np
import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine import HullWhite1Factor
from frm.utils import year_frac
from frm.enums import DayCountBasis, ZeroCurveInterpMethod, CompoundingFreq
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# ESTR swap curve on 1 April 2024 per https://github.com/YANJINI/One-Factor-Hull-White-Model-Calibration-with-CAF
curve_date = pd.Timestamp('2024-04-01')
df = pd.DataFrame({
    'tenor': ['ON', 'SW', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '12M', '15M', '18M', '21M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '11Y', '12Y', '15Y', '20Y', '25Y', '30Y'],
    'date': pd.to_datetime(['2-Apr-2024', '10-Apr-2024', '17-Apr-2024', '24-Apr-2024', '3-May-2024', '3-Jun-2024', '3-Jul-2024', '5-Aug-2024', '3-Sep-2024', '3-Oct-2024', '4-Nov-2024', '3-Dec-2024', '3-Jan-2025', '3-Feb-2025', '3-Mar-2025', '3-Apr-2025', '3-Jul-2025', '3-Oct-2025', '5-Jan-2026', '7-Apr-2026', '5-Apr-2027', '3-Apr-2028', '3-Apr-2029', '3-Apr-2030', '3-Apr-2031', '5-Apr-2032', '4-Apr-2033', '3-Apr-2034', '3-Apr-2035', '3-Apr-2036', '4-Apr-2039', '4-Apr-2044', '5-Apr-2049', '3-Apr-2054']),
    'discount_factor': [0.999892, 0.999026, 0.998266, 0.997514, 0.996546, 0.993222, 0.99014, 0.98688, 0.984079, 0.981287, 0.978453, 0.975944, 0.973358, 0.970875, 0.968705, 0.966373, 0.959921, 0.954107, 0.948336, 0.942805, 0.922607, 0.903406, 0.884216, 0.864765, 0.845061, 0.824882, 0.804566, 0.783991, 0.763235, 0.742533, 0.683701, 0.605786, 0.54803, 0.500307]
})
df['years'] = year_frac(curve_date, df['date'], DayCountBasis.ACT_ACT)

zero_curve = ZeroCurve(curve_date=curve_date,
                       pillar_df=df[['years','discount_factor']],
                       interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR)

#%% HW1F model parameters
short_rate_mean_rev_lvl = 0.05 # Standard values are 1%-10% annualized
short_rate_vol = 0.0196 # Standard values are 1%-10% annualized
hw1f = HullWhite1Factor(zero_curve=zero_curve,
                        mean_rev_lvl=short_rate_mean_rev_lvl,
                        vol=short_rate_vol)

#%% Price a zero coupon bond option

# Test parameters
expiry = 1
maturity = 5
K = hw1f.zero_curve.get_discount_factors(years=maturity) / hw1f.zero_curve.get_discount_factors(years=expiry)
cp = 1 # Call option

# Price call option
analytical_price = hw1f.price_zero_coupon_bond_option(
    expiry_years=expiry,
    maturity_years=maturity,
    K=K,
    cp=cp
)
print(f"Analytical ZCBO Price: {100 * analytical_price[0]:.6f}")

#%% Price a zero coupon bond option using simulation

# Run simulation up to expiry (95% of the run time is spent here)
sim_results = hw1f.simulate(
    tau=expiry,
    nb_steps=maturity * 100, # daily steps
    nb_simulations=100_000,
    flag_apply_antithetic_variates=True,
    method='exact'
)

# Extract simulated rates at expiry
r_T = sim_results['R'][-1, :]
bond_prices = hw1f.calc_discount_factor_by_solving_ode_2(t0=expiry, T=maturity, r=r_T)
# Calculate payoffs for the option
payoffs = np.maximum(cp * (bond_prices - K), 0)
# Discount payoffs back to t=0
discount_factor_to_expiry = hw1f.zero_curve.get_discount_factors(years=expiry)
present_values = payoffs * discount_factor_to_expiry
# Calculate Monte Carlo price estimate
mc_price = np.mean(present_values)

mc_std_error = np.std(present_values) / np.sqrt(len(present_values))

print(f"Analytical ZCBO Price: {100 * analytical_price[0]:.6f}")
print(f"\nMonte Carlo Results:")
print(f"MC Price: {100 * mc_price:.6f}")
print(f"MC Std Error: {100 * mc_std_error:.6f}")
print(f"MC 95% CI: [{100 * (mc_price - 1.96 * mc_std_error):.6f}, {100 * (mc_price + 1.96 * mc_std_error):.6f}]")
print(f"Difference from Analytical: {100 * (mc_price - analytical_price[0]):.6f}")


# Analytical result: 0.02444878
# Result with 5_000_000 simulations with ODE 1: 0.024131199310247316
# Result with 5_000_000 simulations with ODE 2: 0.024141808579278154


# Calculate cumulative statistics
cumulative_means = np.cumsum(present_values) / np.arange(1, len(present_values) + 1)
cumulative_stds = np.array([np.std(present_values[:i+1]) for i in range(len(present_values))])
cumulative_se = cumulative_stds / np.sqrt(np.arange(1, len(present_values) + 1))
ci_lower = cumulative_means - 1.96 * cumulative_se
ci_upper = cumulative_means + 1.96 * cumulative_se

# Create plot
# Ignore 1st 30 averages to allow for burn-in so plot isn't dominated by early values
min_samples = 1000
plt.figure(figsize=(10, 6))
plt.plot(range(min_samples, len(cumulative_means)), 100 * cumulative_means[min_samples:], label='MC Price')
plt.fill_between(range(min_samples, len(cumulative_means)),
               100 * ci_lower[min_samples:],
               100 * ci_upper[min_samples:],
               alpha=0.2,
               label='95% CI')
plt.axhline(y=100 * analytical_price[0], color='r', linestyle='--', label='Analytical Price')
plt.xlabel('Number of Simulations')
plt.ylabel('Option Price (%)')
plt.title('Monte Carlo Price Convergence')
plt.legend()
plt.grid(True)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def simulate_hw1f(
        a: float,
        sigma: float,
        theta: float,
        r0: float,
        T: float,
        N: int,
        n_paths: int,
        method: str = "euler"
) -> Tuple[np.ndarray, np.ndarray]:
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    t = np.linspace(0, T, N + 1)
    dW = np.random.normal(0, sqrt_dt, (n_paths, N))

    r = np.zeros((n_paths, N + 1))
    r[:, 0] = r0

    for i in range(N):
        if method == "euler":
            dr = (theta - a * r[:, i]) * dt + sigma * dW[:, i]
        else:  # milstein
            dr = (theta - a * r[:, i]) * dt + sigma * dW[:, i] + \
                 0.5 * sigma * sigma * (dW[:, i] ** 2 - dt)
        r[:, i + 1] = r[:, i] + dr

    return t, r


# Example usage
t, r_euler = simulate_hw1f(0.1, 0.02, 0.05, 0.03, 5, 500, 1_000_000, "euler")
_, r_milstein = simulate_hw1f(0.1, 0.02, 0.05, 0.03, 5, 500, 1_000_000, "milstein")
#%%

print(f"Mean terminal rate (Euler): {100*r_euler[:, -1].mean():.4f}")
print(f"Mean terminal rate (Milstein): {100*r_milstein[:, -1].mean():.4f}")


#%%

#
# nb_steps = 5
# nb_rand_vars = 1
# nb_simulations = 10
# apply_antithetic_variates = True
# random_seed = 0
#
# np.random.seed(random_seed)
#
# assert isinstance(nb_steps, int), type(nb_steps)
# assert isinstance(nb_rand_vars, int), type(nb_rand_vars)
# assert isinstance(nb_simulations, int), type(nb_simulations)
# assert isinstance(apply_antithetic_variates, bool)
# assert nb_steps >= 1, nb_steps
# assert nb_rand_vars >= 1, nb_rand_vars
# assert nb_simulations >= 1, nb_simulations
#
#
#
# if apply_antithetic_variates and nb_simulations == 1:
#     raise ValueError("Antithetic variates requiries >=2 simulations")
#
# if apply_antithetic_variates:
#     nb_pairs = nb_simulations // 2
#     nb_normal_simulations = nb_simulations - nb_pairs
#     rand_nbs_normal = np.random.normal(0, 1, (
#     nb_steps, nb_rand_vars, nb_normal_simulations))  # standard normal random numbers
#     rand_nbs_antithetic_variate = -1 * rand_nbs_normal[:, :, :nb_pairs]
#     rand_nbs = np.concatenate([rand_nbs_normal, rand_nbs_antithetic_variate], axis=2)
#
#     # Reindex to pair normal with antithetic variates
#     idx = np.column_stack((np.arange(nb_pairs), np.arange(nb_pairs) + nb_pairs)).flatten()
#     rand_nbs = rand_nbs[:, :, idx]
# else:
#     rand_nbs = np.random.normal(0, 1, (nb_steps, nb_rand_vars, nb_simulations))
#
#
# for i in range(nb_simulations):
#     print(list(rand_nbs[:, :, i]))

#%%

#%% Fit the model to the zero curve
grid_length = 50
hw1f.setup_theta(num=grid_length)

# Demonstrate the fit of the model to the zero curve with a table of the errors (in basis points ) for each source data point
avg_error_bps = hw1f.calc_error_for_theta_fit(print_results=True)

#%% Demonstrate the fit of the model with plot of the zero curve and the model zero curve
years_grid = np.linspace(zero_curve.pillar_df['years'].min(), zero_curve.pillar_df['years'].max(),grid_length)
hw1f_instantaneous_forward_rate  = [hw1f.zero_curve.get_instantaneous_forward_rate(t) for t in years_grid]
hw1f_discount_factors = [hw1f.calc_discount_factor_by_solving_ode_1(0, t) for t in years_grid]
hw1f_zero_rates = [-1*np.log(df)/t for t,df in zip(years_grid,hw1f_discount_factors)]

plt.figure()
plt.plot(years_grid, 100 * np.array(hw1f_instantaneous_forward_rate), label ='HW1F model instantaneous forward rates')
plt.plot(years_grid, 100 * np.array(hw1f_zero_rates), label='HW1F model zero rates')
plt.plot(zero_curve.pillar_df['years'], 100 * zero_curve.pillar_df['cczr'].values, marker='.', linestyle='None', label='Source data zero rates')
plt.ylim(1.5,4.5)

plt.xlabel(r'Years')
plt.title(r'Plot of fit of HW1F interest rate term structure to source data')
plt.grid(True)
plt.legend()
plt.ylabel('Rate (%)')
plt.show()

#%% Demonstrate the HW1F simulations average zero rate matches the source data's zero rates
nb_simulations = 5000
nb_steps = pd.date_range(start=curve_date, end=df['date'].max()).shape[0]-1 # Daily Steps

results = hw1f.simulate(tau=df['years'].max(),
                        nb_steps=nb_steps,
                        nb_simulations=nb_simulations,
                        flag_apply_antithetic_variates=True,
                        random_seed=1500)

plt.figure()
sim_avg_zero_rates = 100 * results['averages_df']['cczr'].values
years_grid = results['averages_df']['years'].values
plt.plot(years_grid, sim_avg_zero_rates, label='Simulation average zero rates')
plt.plot(zero_curve.pillar_df['years'], 100 * zero_curve.pillar_df['cczr'].values, marker='.', linestyle='None', label='Source data zero rates')
plt.ylim(1.5,4.5)
plt.xlabel(r'Years')
plt.title(r"Plot of HW1F simulations average and source data's zero rates")
plt.grid(True)
plt.legend()
plt.ylabel('Rate (%)')
plt.show()