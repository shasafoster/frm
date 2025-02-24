# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import pandas as pd
import numpy as np
import time
from frm.pricing_engine import HullWhite1Factor
from frm.term_structures import ZeroCurve
from frm.enums import CompoundingFreq, ZeroCurveInterpMethod
import matplotlib.pyplot as plt

# Tests for test_zero_coupon_bond_option_pricing()
# Test details copied from tf_quant_finance
# https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/hull_white/zero_coupon_bond_option_test.py

#% Setup ZeroCurve and HW1F object

mean_reversion = 0.03
volatility = 0.02

# Create flat zero curve at 1%
curve_date = pd.Timestamp('2024-04-01')
years = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
rates = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
pillar_df = pd.DataFrame(np.array([years, rates]).T, columns=['years', 'zero_rate'])

zero_curve = ZeroCurve(curve_date=curve_date,
                       pillar_df=pillar_df,
                       compounding_freq=CompoundingFreq.CONTINUOUS,
                       interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR)

# Initialize model
hw1f = HullWhite1Factor(
    zero_curve=zero_curve,
    mean_rev_lvl=mean_reversion,
    vol=volatility
)

hw1f.setup_theta()

effective = 0
maturity = 1
N = 100

#%% Test the discount factor produces at the 1Y point

print(f"Analytical discount factor at 1Y: {hw1f.zero_curve.get_discount_factors(years=maturity):.10f}")
print(f"ODE 1 discount factor at 1Y:      {hw1f.calc_discount_factor_by_solving_ode_1(t1=effective, t2=maturity):.10f}")
print(f"ODE 2 discount factor at 1Y:      {hw1f.calc_discount_factor_by_solving_ode_2(t1=effective, t2=maturity):.10f}")


#%%




#%% Scalar test case

# Test parameters
expiry = 10
maturity = 15
K = np.exp(-0.01 * maturity) / np.exp(-0.01 * expiry)  # Strike calculation from test
cp = 1 # call option

# Price call option
analytical_price = hw1f.price_zero_coupon_bond_option(
    option_expiry=expiry,
    bond_maturity=maturity,
    K=K,
    cp=cp
)

expected_price = 8.720313 / 100

print("Price: ", analytical_price[0])
print("Expected Price: ", expected_price)
print("Difference: ", analytical_price[0] - expected_price)

print(f"Analytical ZCBO Price (%): {100 * analytical_price[0]:.6f}")

#%%%

# Simulate to maturity
sim_result = hw1f.simulate(
    tau=expiry,
    nb_steps=expiry * 365,
    nb_simulations=20_000,
    flag_apply_antithetic_variates=True,
    method='exact'
)
print(sim_result['averages_df'].iloc[-1],'\n')

R_expiry = sim_result['R'][-1,:] # discount factors at expiry

#%%

bond_prices_expiry = sim_result['sim_dsc_factor'][-1, :]
print(f"Average simulated discount factor at expiry: {np.mean(bond_prices_expiry):.6f}")
print(f"Average simulated CCZR at expiry: {-1 * np.log(np.mean(bond_prices_expiry)) / expiry:.6f}")
print(f"Expected discount factor at expiry: {np.exp(-0.01 * expiry):.6f}")

bond_prices = hw1f.calc_discount_factor_by_solving_ode_1(t1=expiry, t2=maturity, r_t1=R_expiry)
print(f"Average simulated discount factor at maturity: {np.mean(bond_prices):.6f}")
print(f"Expected bond price: {K}")


print(f"Average bond price: {np.mean(bond_prices)}")

print(f"Expected bond price: {1/K}")

payoffs = np.maximum(cp * (bond_prices - K), 0)

print(f"Analytical ZCBO Price (%): {100 * analytical_price[0]:.6f}")
print(f"MC Price (%): {100 * np.mean(payoffs):.6f}")




#%%



# Run simulation up to expiry (95% of the run time is spent here)
sim_results = hw1f.simulate(
    tau=expiry,
    nb_steps=maturity * 365, # daily steps
    nb_simulations=10_000,
    flag_apply_antithetic_variates=True,
    method='euler'
)

# Extract simulated rates at expiry
r_T = sim_results['R'][-1, :]
bond_prices = hw1f.calc_discount_factor_by_solving_ode_2(t1=expiry, t2=maturity, r_t1=r_T)
# Calculate payoffs for the option
payoffs = np.maximum(cp * (bond_prices - K), 0)
# Discount payoffs back to t=0
discount_factor_to_expiry = hw1f.zero_curve.get_discount_factors(years=expiry)
present_values = payoffs #* discount_factor_to_expiry
# Calculate Monte Carlo price estimate
mc_price = np.mean(present_values)

mc_std_error = np.std(present_values) / np.sqrt(len(present_values))

print(f"Analytical ZCBO Price: {100 * analytical_price[0]:.6f}")
print(f"\nMonte Carlo Results:")
print(f"MC Price: {100 * mc_price:.6f}")
print(f"MC Std Error: {100 * mc_std_error:.6f}")
print(f"MC 95% CI: [{100 * (mc_price - 1.96 * mc_std_error):.6f}, {100 * (mc_price + 1.96 * mc_std_error):.6f}]")
print(f"Difference from Analytical: {100 * (mc_price - analytical_price[0]):.6f}")

#%%

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
min_samples = 100
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

#
#
#
#
# #%% Vector test case
#
# # Test multiple options
# expiries = np.array([1.0, 2.0])
# maturities = np.array([5.0, 4.0])
# strikes = hw1f.zero_curve.get_zero_rates(maturities) / hw1f.zero_curve.get_zero_rates(expiries)
#
# prices = hw1f.price_zero_coupon_bond_option(
#     expiry_years=expiries,
#     maturity_years=maturities,
#     K=strikes,
#     cp=1
# )
#
# # Expected results from reference implementation
# expected_prices = [0.02817777, 0.02042677]
# print("Prices: ", prices)
# print("Expected Prices: ", expected_prices)
# print("Difference: ", prices - expected_prices)